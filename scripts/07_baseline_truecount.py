# scripts/07_baseline_truecount.py
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(12):
        if (p / ".git").exists() or (p / "README.md").exists():
            return p
        p = p.parent
    raise RuntimeError("Repo root not found.")


def ev_by_deciles(df: pd.DataFrame, score_col: str, y_col: str = "win", n_bins: int = 10) -> pd.DataFrame:
    tmp = df[[score_col, y_col]].copy()
    tmp["decile"] = pd.qcut(tmp[score_col], q=n_bins, labels=False, duplicates="drop") + 1
    agg = tmp.groupby("decile")[y_col].agg(["count", "mean"]).reset_index()
    agg = agg.rename(columns={"mean": "ev"})
    agg["model"] = score_col
    return agg


def ev_topk(df: pd.DataFrame, score_col: str, ks: List[float], y_col: str = "win") -> pd.DataFrame:
    tmp = df[[score_col, y_col]].copy()
    tmp = tmp.sort_values(score_col, ascending=False)
    n = len(tmp)
    rows = []
    for k in ks:
        m = max(1, int(round(k * n)))
        ev = float(tmp.iloc[:m][y_col].mean())
        rows.append({"model": score_col, "top_k": k, "n_rows": m, "ev": ev})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    repo = find_repo_root(Path(__file__).parent)

    # On part du fichier OOF de l'étape 6, car il contient win + true_count + pred_histgb + pred_mlp
    oof_path = repo / "reports" / "tables" / "oof_predictions_histgb_vs_mlp.csv"
    if not oof_path.exists():
        raise FileNotFoundError(
            "OOF file not found. Run scripts/06_oof_predictions.py first."
        )

    df = pd.read_csv(oof_path)

    required = ["win", "true_count", "run_count", "pred_histgb", "pred_mlp"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {oof_path.name}")

    # Scores baselines
    df["score_true_count"] = df["true_count"].astype(float)
    df["score_run_count"] = df["run_count"].astype(float)

    reports_tables = repo / "reports" / "tables"
    reports_figs = repo / "reports" / "figures"
    reports_tables.mkdir(parents=True, exist_ok=True)
    reports_figs.mkdir(parents=True, exist_ok=True)

    # ----- DECILES
    models_for_deciles = ["pred_histgb", "pred_mlp", "score_true_count", "score_run_count"]
    dec = pd.concat([ev_by_deciles(df, m) for m in models_for_deciles], ignore_index=True)
    dec_out = reports_tables / "ev_by_deciles_models_vs_count.csv"
    dec.to_csv(dec_out, index=False)

    # Plot deciles
    plt.figure()
    label_map = {
        "pred_histgb": "HISTGB",
        "pred_mlp": "MLP",
        "score_true_count": "TRUE_COUNT",
        "score_run_count": "RUN_COUNT",
    }
    for m in models_for_deciles:
        sub = dec[dec["model"] == m].sort_values("decile")
        plt.plot(sub["decile"], sub["ev"], marker="o", label=label_map.get(m, m))
    plt.xlabel("Decile (1=lowest score, 10=highest score)")
    plt.ylabel("Realized EV (mean win)")
    plt.title("EV by decile (OOF): Models vs Counting baselines")
    plt.legend()
    fig_dec = reports_figs / "ev_by_decile_models_vs_count.png"
    plt.savefig(fig_dec, dpi=200, bbox_inches="tight")
    plt.close()

    # ----- TOP-K
    ks = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01]
    models_for_topk = ["pred_histgb", "pred_mlp", "score_true_count", "score_run_count"]
    top = pd.concat([ev_topk(df, m, ks) for m in models_for_topk], ignore_index=True)
    top_out = reports_tables / "ev_topk_models_vs_count.csv"
    top.to_csv(top_out, index=False)

    # Plot top-k
    plt.figure()
    for m in models_for_topk:
        sub = top[top["model"] == m].sort_values("top_k", ascending=False)
        plt.plot(sub["top_k"], sub["ev"], marker="o", label=label_map.get(m, m))
    plt.xlabel("Top-k fraction kept (higher=less selective)")
    plt.ylabel("Realized EV (mean win)")
    plt.title("EV on Top-k% (OOF): Models vs Counting baselines")
    plt.legend()
    fig_top = reports_figs / "topk_ev_curve_models_vs_count.png"
    plt.savefig(fig_top, dpi=200, bbox_inches="tight")
    plt.close()

    print("💾 Saved:")
    print(" -", dec_out)
    print(" -", top_out)
    print(" -", fig_dec)
    print(" -", fig_top)
    print("\n✅ DONE. Total runtime:", time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)))


if __name__ == "__main__":
    main()
