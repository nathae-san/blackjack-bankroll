# scripts/04_train_classic.py
from __future__ import annotations

from pathlib import Path
import warnings
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

import joblib


warnings.filterwarnings("ignore", category=UserWarning)


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(12):
        if (p / ".git").exists() or (p / "README.md").exists():
            return p
        p = p.parent
    raise RuntimeError("Repo root not found.")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fmt_seconds(s: float) -> str:
    s = int(round(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def main() -> None:
    t0 = time.perf_counter()

    repo = find_repo_root(Path(__file__).parent)

    data_path = repo / "data" / "processed" / "features_sample.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run scripts/03_build_features.py first."
        )

    out_models = repo / "outputs" / "models"
    out_tables = repo / "reports" / "tables"
    out_models.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    print("📄 Loading features:", data_path)
    df = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df):,} rows | elapsed {fmt_seconds(time.perf_counter()-t0)}")

    # --- Target and groups
    y = df["win"].astype(float).values
    groups = df["shoe_id"].astype(int).values

    # --- Features (IMPORTANT: do NOT use shoe_id as input)
    X = df.drop(columns=["win", "shoe_id"]).copy()

    # Identify column types
    categorical_cols = [
        "dealer_up", "card1", "card2", "player_total",
        "is_soft", "is_pair", "insurance_offered", "insurance_no"
    ]
    numeric_cols = ["cards_remaining", "run_count", "true_count"]

    # Ensure columns exist
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in numeric_cols if c in X.columns]

    print("Categorical cols:", categorical_cols)
    print("Numeric cols:", numeric_cols)
    print("X shape:", X.shape)

    # ✅ Preprocess for linear models (Ridge): OneHot + scaling
    pre_linear = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    # ✅ Preprocess for tree models (RF / HistGB): keep numeric as-is (no OneHot)
    pre_tree = ColumnTransformer(
        transformers=[
            ("cat", "passthrough", categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

    cv = GroupKFold(n_splits=5)

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": make_scorer(rmse, greater_is_better=False),
        "R2": "r2",
    }

    # --- Define models + grids
    candidates = []

    # 1) Ridge (fast)
    ridge = Pipeline([("pre", pre_linear), ("model", Ridge(random_state=42))])
    ridge_grid = {"model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]}
    candidates.append(("Ridge", ridge, ridge_grid))

    # 2) RandomForest (use pre_tree to avoid OneHot explosion)
    rf = Pipeline(
        [
            ("pre", pre_tree),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )
    rf_grid = {
        "model__n_estimators": [120],
        "model__max_depth": [None, 16],
        "model__min_samples_leaf": [20, 50],
        "model__max_features": ["sqrt"],
    }
    candidates.append(("RandomForest", rf, rf_grid))

    # 3) HistGradientBoosting
    hgb = Pipeline(
        [
            ("pre", pre_tree),
            ("model", HistGradientBoostingRegressor(random_state=42)),
        ]
    )
    # Original grid (kept here for reference), but we will override in-loop to avoid WinError 1450
    hgb_grid = {
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [3, 6, None],
        "model__max_leaf_nodes": [31, 63],
        "model__min_samples_leaf": [20, 50, 100],
    }
    candidates.append(("HistGB", hgb, hgb_grid))

    all_rows = []
    best_overall = None
    best_score = -np.inf  # select by best CV R2

    for name, pipe, grid in candidates:
        # ✅ Fix for Windows resource exhaustion on HistGB:
        # - reduce grid
        # - limit parallelism
        # - tolerate failures with error_score=np.nan
        if name == "HistGB":
            grid = {
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [3, 6],
                "model__min_samples_leaf": [50, 100],
            }
            local_n_jobs = 1
        else:
            local_n_jobs = -1

        t_model = time.perf_counter()
        n_cfg = len(list(ParameterGrid(grid))) if isinstance(grid, dict) else 0
        print(f"\n🚀 Training: {name} | total elapsed {fmt_seconds(time.perf_counter()-t0)}")
        print(f"Grid configs: {n_cfg}")
        print(f"CV: {cv.get_n_splits()} folds | total fits: {n_cfg * cv.get_n_splits()}")

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scoring,
            refit="R2",
            cv=cv,
            n_jobs=local_n_jobs,
            verbose=3,
            error_score=np.nan,  # ✅ don't crash the whole run if one fit fails
        )

        gs.fit(X, y, groups=groups)

        print(f"✅ Finished {name} in {fmt_seconds(time.perf_counter()-t_model)} | total {fmt_seconds(time.perf_counter()-t0)}")

        # Best CV metrics
        best_idx = gs.best_index_
        row = {
            "model": name,
            "best_params": gs.best_params_,
            "cv_R2": float(gs.cv_results_["mean_test_R2"][best_idx]),
            "cv_MAE": float(-gs.cv_results_["mean_test_MAE"][best_idx]),
            "cv_RMSE": float(-gs.cv_results_["mean_test_RMSE"][best_idx]),
        }
        all_rows.append(row)

        print("Best params:", gs.best_params_)
        print("CV R2:", row["cv_R2"])
        print("CV MAE:", row["cv_MAE"])
        print("CV RMSE:", row["cv_RMSE"])

        if row["cv_R2"] > best_score:
            best_score = row["cv_R2"]
            best_overall = (name, gs.best_estimator_, row)

    results_df = pd.DataFrame(all_rows).sort_values("cv_R2", ascending=False)
    out_csv = out_tables / "cv_results.csv"
    results_df.to_csv(out_csv, index=False)
    print("\n✅ Saved CV table:", out_csv)
    print(results_df.to_string(index=False))

    # Save best model
    assert best_overall is not None
    best_name, best_estimator, best_row = best_overall
    model_path = out_models / "best_model.joblib"
    joblib.dump(best_estimator, model_path)
    print(f"\n🏆 Best model: {best_name}")
    print("Saved model to:", model_path)
    print("Best row:", best_row)

    # Quick sanity (NOT a report metric)
    y_pred = best_estimator.predict(X)
    print("\n=== In-sample sanity (NOT a report metric) ===")
    print("MAE:", mean_absolute_error(y, y_pred))
    print("RMSE:", rmse(y, y_pred))
    print("R2:", r2_score(y, y_pred))

    print("\n✅ DONE. Total runtime:", fmt_seconds(time.perf_counter() - t0))


if __name__ == "__main__":
    main()
