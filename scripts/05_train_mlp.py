# scripts/05_train_mlp.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "y_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
    }


def make_onehot_encoder():
    """
    sklearn changed param name from sparse -> sparse_output.
    This helper keeps compatibility.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Model
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
class TrainConfig:
    hidden: List[int] = None
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4096
    max_epochs: int = 25
    patience: int = 4  # early stopping
    device: str = "auto"


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# -----------------------------
# Training loop
# -----------------------------
def train_one_fold(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    cfg: TrainConfig,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float], dict]:
    """
    Returns:
      - metrics_val (MAE/RMSE/R2 on val)
      - metrics_train (same on train)
      - best_state (state_dict + training info)
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

    best_val_rmse = float("inf")
    best_epoch = -1
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

        # ---- eval
        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_t.to(device)).cpu().numpy()
            pred_tr = model(Xtr_t.to(device)).cpu().numpy()

        m_va = evaluate(yva, pred_va)
        # early stopping on RMSE
        if m_va["rmse"] < best_val_rmse - 1e-4:
            best_val_rmse = m_va["rmse"]
            best_epoch = epoch
            best_state = {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "best_epoch": best_epoch,
                "best_val_metrics": m_va,
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            break

    # restore best for final metrics
    model.load_state_dict(best_state["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred_va = model(Xva_t.to(device)).cpu().numpy()
        pred_tr = model(Xtr_t.to(device)).cpu().numpy()

    metrics_val = evaluate(yva, pred_va)
    metrics_train = evaluate(ytr, pred_tr)

    best_state["final_val_metrics"] = metrics_val
    best_state["final_train_metrics"] = metrics_train
    return metrics_val, metrics_train, best_state


def main() -> None:
    t0 = time.time()
    repo = find_repo_root(Path(__file__).parent)

    data_path = repo / "data" / "processed" / "features_sample.parquet"
    if not data_path.exists():
        data_path = repo / "data" / "processed" / "features_sample.csv"
    if not data_path.exists():
        raise FileNotFoundError("Run scripts/03_build_features.py first (features_sample.* not found).")

    print(f"📄 Loading features: {data_path}")
    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} rows")

    target = "win"
    group_col = "shoe_id"

    # same columns as your step 04
    cat_cols = ["dealer_up", "card1", "card2", "player_total", "is_soft", "is_pair", "insurance_offered", "insurance_no"]
    num_cols = ["cards_remaining", "run_count", "true_count"]
    feature_cols = cat_cols + num_cols

    for c in [target, group_col] + feature_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    X_df = df[feature_cols].copy()
    y = df[target].astype(float).to_numpy()
    groups = df[group_col].to_numpy()

    print("Categorical cols:", cat_cols)
    print("Numeric cols:", num_cols)
    print("X shape:", X_df.shape)

    # Preprocess: one-hot for categorical, standardize numeric
    preproc = ColumnTransformer(
        transformers=[
            ("cat", make_onehot_encoder(), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    # Config (simple, strong baseline)
    cfg = TrainConfig(
        hidden=[256, 128],
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=4096,
        max_epochs=25,
        patience=4,
        device="auto",
    )
    device = get_device(cfg.device)
    print("🖥️ Device:", device)

    # GroupKFold CV
    gkf = GroupKFold(n_splits=5)
    fold_rows = []
    fold_states = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_df, y, groups=groups), start=1):
        print(f"\n🚀 Fold {fold}/5")

        Xtr_df = X_df.iloc[tr_idx]
        Xva_df = X_df.iloc[va_idx]
        ytr = y[tr_idx]
        yva = y[va_idx]

        # Fit preprocessor on train fold ONLY
        Xtr = preproc.fit_transform(Xtr_df)
        Xva = preproc.transform(Xva_df)

        # Ensure numpy float32
        Xtr = np.asarray(Xtr, dtype=np.float32)
        Xva = np.asarray(Xva, dtype=np.float32)

        metrics_val, metrics_train, best_state = train_one_fold(Xtr, ytr, Xva, yva, cfg, seed=42 + fold)

        row = {
            "fold": fold,
            "val_mae": metrics_val["mae"],
            "val_rmse": metrics_val["rmse"],
            "val_r2": metrics_val["r2"],
            "train_mae": metrics_train["mae"],
            "train_rmse": metrics_train["rmse"],
            "train_r2": metrics_train["r2"],
            "best_epoch": int(best_state["best_epoch"]),
        }
        fold_rows.append(row)
        fold_states.append(best_state)

        print("Val:", {k: round(metrics_val[k], 6) for k in ["mae", "rmse", "r2"]},
              "| Train:", {k: round(metrics_train[k], 6) for k in ["mae", "rmse", "r2"]},
              "| best_epoch:", best_state["best_epoch"])

    res = pd.DataFrame(fold_rows)

    # Aggregate
    summary = {
        "val_mae_mean": float(res["val_mae"].mean()),
        "val_mae_std": float(res["val_mae"].std(ddof=1)),
        "val_rmse_mean": float(res["val_rmse"].mean()),
        "val_rmse_std": float(res["val_rmse"].std(ddof=1)),
        "val_r2_mean": float(res["val_r2"].mean()),
        "val_r2_std": float(res["val_r2"].std(ddof=1)),
    }

    print("\n=== MLP CV Summary ===")
    print(res.to_string(index=False))
    print("\nMEAN±STD:",
          f"MAE {summary['val_mae_mean']:.6f}±{summary['val_mae_std']:.6f} |",
          f"RMSE {summary['val_rmse_mean']:.6f}±{summary['val_rmse_std']:.6f} |",
          f"R2 {summary['val_r2_mean']:.6f}±{summary['val_r2_std']:.6f}")

    # Save artifacts
    reports_dir = repo / "reports" / "tables"
    models_dir = repo / "outputs" / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    res_path = reports_dir / "mlp_cv_results.csv"
    res.to_csv(res_path, index=False)

    summary_path = reports_dir / "mlp_cv_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg.__dict__, "summary": summary}, f, ensure_ascii=False, indent=2)

    # Fit preprocessor on full data and save it (useful for final train later)
    preproc.fit(X_df)
    preproc_path = models_dir / "mlp_preprocessor.joblib"
    joblib.dump(preproc, preproc_path)

    # Save last fold best model weights as a starting artifact (simple)
    # (For a "final model", we will train on all data later.)
    state_path = models_dir / "mlp_lastfold_state.pt"
    torch.save({"config": cfg.__dict__, "state": fold_states[-1]["model_state_dict"]}, state_path)

    print("\n💾 Saved:")
    print(" -", res_path)
    print(" -", summary_path)
    print(" -", preproc_path)
    print(" -", state_path)

    print("\n✅ DONE. Total runtime:", time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)))


if __name__ == "__main__":
    main()
