# scripts/06_oof_predictions.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Utils
# -----------------------------
def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(12):
        if (p / ".git").exists() or (p / "README.md").exists():
            return p
        p = p.parent
    raise RuntimeError("Repo root not found.")


def make_onehot_encoder():
    # sklearn compatibility: sparse vs sparse_output
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# -----------------------------
# MLP model (same spirit as step 05)
# -----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class MLPConfig:
    hidden: List[int] = None
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4096
    max_epochs: int = 25
    patience: int = 4
    device: str = "auto"


def train_mlp_one_fold(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    cfg: MLPConfig,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Train on (Xtr,ytr) and predict on Xva.
    Early stopping on val RMSE.
    Returns: predictions on Xva, val metrics
    """
    set_seed(seed)
    device = get_device(cfg.device)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    model = MLPRegressor(input_dim=Xtr.shape[1], hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_t.to(device)).cpu().numpy()
        m = evaluate(yva, pred_va)

        if m["rmse"] < best_rmse - 1e-4:
            best_rmse = m["rmse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            break

    # restore best and predict
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_va = model(Xva_t.to(device)).cpu().numpy()

    return pred_va, evaluate(yva, pred_va)


def main() -> None:
    t0 = time.time()
    repo = find_repo_root(Path(__file__).parent)

    data_path = repo / "data" / "processed" / "features_sample.parquet"
    if not data_path.exists():
        data_path = repo / "data" / "processed" / "features_sample.csv"
    if not data_path.exists():
        raise FileNotFoundError("features_sample.(parquet|csv) not found. Run scripts/03_build_features.py")

    print(f"📄 Loading features: {data_path}")
    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} rows")

    target = "win"
    group_col = "shoe_id"

    cat_cols = ["dealer_up", "card1", "card2", "player_total", "is_soft", "is_pair", "insurance_offered", "insurance_no"]
    num_cols = ["cards_remaining", "run_count", "true_count"]
    feat_cols = cat_cols + num_cols

    # minimal sanity
    for c in [target, group_col] + feat_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    y = df[target].astype(float).to_numpy()
    groups = df[group_col].to_numpy()
    X_df = df[feat_cols].copy()

    # Preprocessor (fit per fold)
    def make_preproc():
        return ColumnTransformer(
            transformers=[
                ("cat", make_onehot_encoder(), cat_cols),
                ("num", StandardScaler(), num_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )

    # HistGB best params from step 04
    hgb_params = dict(
        learning_rate=0.1,
        max_depth=6,
        min_samples_leaf=50,
        random_state=42,
    )

    # MLP config (same as step 05)
    mlp_cfg = MLPConfig(
        hidden=[256, 128],
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=4096,
        max_epochs=25,
        patience=4,
        device="auto",
    )
    print("🖥️ MLP device:", get_device(mlp_cfg.device))

    # OOF arrays
    oof_hgb = np.full(shape=len(df), fill_value=np.nan, dtype=np.float32)
    oof_mlp = np.full(shape=len(df), fill_value=np.nan, dtype=np.float32)

    gkf = GroupKFold(n_splits=5)

    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_df, y, groups=groups), start=1):
        print(f"\n🚀 Fold {fold}/5")

        Xtr_df = X_df.iloc[tr_idx]
        Xva_df = X_df.iloc[va_idx]
        ytr = y[tr_idx]
        yva = y[va_idx]

        preproc = make_preproc()
        Xtr = preproc.fit_transform(Xtr_df)
        Xva = preproc.transform(Xva_df)

        Xtr = np.asarray(Xtr, dtype=np.float32)
        Xva = np.asarray(Xva, dtype=np.float32)

        # ---- HistGB
        hgb = HistGradientBoostingRegressor(**hgb_params)
        hgb.fit(Xtr, ytr)
        pred_hgb = hgb.predict(Xva)
        oof_hgb[va_idx] = pred_hgb.astype(np.float32)
        m_hgb = evaluate(yva, pred_hgb)

        # ---- MLP
        pred_mlp, m_mlp = train_mlp_one_fold(
            Xtr, ytr, Xva, yva, mlp_cfg, seed=1000 + fold
        )
        oof_mlp[va_idx] = pred_mlp.astype(np.float32)

        print("HistGB Val:", {k: round(v, 6) for k, v in m_hgb.items()},
              "| MLP Val:", {k: round(v, 6) for k, v in m_mlp.items()})

        fold_metrics.append({
            "fold": fold,
            "histgb_mae": m_hgb["mae"],
            "histgb_rmse": m_hgb["rmse"],
            "histgb_r2": m_hgb["r2"],
            "mlp_mae": m_mlp["mae"],
            "mlp_rmse": m_mlp["rmse"],
            "mlp_r2": m_mlp["r2"],
        })

    # Build OOF dataframe (keep some cols useful for analysis)
    out_cols = [
        "shoe_id",
        "cards_remaining",
        "run_count",
        "true_count",
        "dealer_up",
        "card1",
        "card2",
        "player_total",
        "is_soft",
        "is_pair",
        "insurance_offered",
        "insurance_no",
        "win",
    ]
    oof_df = df[out_cols].copy()
    oof_df["pred_histgb"] = oof_hgb
    oof_df["pred_mlp"] = oof_mlp

    # Safety check: no NaNs left
    if oof_df["pred_histgb"].isna().any() or oof_df["pred_mlp"].isna().any():
        raise RuntimeError("OOF predictions contain NaNs. Something went wrong in fold filling.")

    # Aggregate CV metrics from OOF (global)
    m_global_hgb = evaluate(oof_df["win"].to_numpy(), oof_df["pred_histgb"].to_numpy())
    m_global_mlp = evaluate(oof_df["win"].to_numpy(), oof_df["pred_mlp"].to_numpy())

    print("\n=== Global OOF metrics (all rows) ===")
    print("HistGB:", {k: round(v, 6) for k, v in m_global_hgb.items()})
    print("MLP   :", {k: round(v, 6) for k, v in m_global_mlp.items()})

    # Save artifacts
    reports_dir = repo / "reports" / "tables"
    models_dir = repo / "outputs" / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    oof_path = reports_dir / "oof_predictions_histgb_vs_mlp.csv"
    oof_df.to_csv(oof_path, index=False)

    fold_path = reports_dir / "oof_fold_metrics_histgb_vs_mlp.csv"
    pd.DataFrame(fold_metrics).to_csv(fold_path, index=False)

    summary_path = reports_dir / "oof_global_metrics_histgb_vs_mlp.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "histgb_params": hgb_params,
                "mlp_config": mlp_cfg.__dict__,
                "global_oof": {"histgb": m_global_hgb, "mlp": m_global_mlp},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save model configs (optional)
    joblib.dump(hgb_params, models_dir / "histgb_best_params.joblib")

    print("\n💾 Saved:")
    print(" -", oof_path)
    print(" -", fold_path)
    print(" -", summary_path)

    print("\n✅ DONE. Total runtime:", time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)))


if __name__ == "__main__":
    main()
