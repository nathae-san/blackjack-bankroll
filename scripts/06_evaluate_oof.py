# scripts/06_evaluate_oof.py
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


def ev_by_deciles(df: pd.DataFrame, pred_col: str, y_col: str = "win", n_bins: int = 10) -> pd.DataFrame:
    tmp = df[[pred_col, y_col]].copy()
    # qcut can fail if too many identical values; duplicates='drop' helps
    tmp["decile"] = pd.qcut(tmp[pred_col], q=n_bins, labels=False, duplicates="drop") + 1
    agg = tmp.groupby("decile")[y_col].agg(["count", "mean"]).reset_index()
    agg = agg.rename(columns={"mean": "ev"})
    agg["model"] = pred_col
    return agg


def ev_topk(df: pd.DataFrame, pred_col: str, ks: List[float], y_col: str = "win") -> pd.DataFrame:
    tmp = df[[pred_col, y_col]].copy()
    tmp = tmp.sort_values(pred_col, ascending=False)
    rows = []
    n = len(tmp)
    for k in ks:
        m = max(1, int(round(k * n)))
        ev = float(tmp.iloc[:m][y_col].mean())
        rows.append({"model": pred_col, "top_k": k, "n_rows": m, "ev": ev})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    repo = find_repo_root(Path(__file__).parent)

    oof_path = repo / "reports" / "tables" / "oof_predictions_histgb_vs_mlp.csv"
    if not oof_path.exists():
        raise FileNotFoundError("Run scripts/06_oof_predictions.py first (OOF file not found).")

    df = pd.read_csv(oof_path)
    for col in ["win", "pred_histgb", "pred_mlp"]:
        if col not in df.columns:
            raise KeyError(f"Missing {col} in OOF file.")

    reports_tables = repo / "reports" / "tables"
    reports_figs = repo / "reports" / "figures"
    reports_tables.mkdir(parents=True, exist_ok=True)
    reports_figs.mkdir(parents=True, exist_ok=True)

    # ---- Deciles
    dec_hgb = ev_by_deciles(df, "pred_histgb")
    dec_mlp = ev_by_deciles(df, "pred_mlp")
    dec = pd.concat([dec_hgb, dec_mlp], ignore_index=True)

    dec_out = reports_tables / "ev_by_deciles_histgb_vs_mlp.csv"
    dec.to_csv(dec_out, index=False)

    # plot
    plt.figure()
    for model in ["pred_histgb", "pred_mlp"]:
        sub = dec[dec["model"] == model].sort_values("decile")
        plt.plot(sub["decile"], sub["ev"], marker="o", label=model.replace("pred_", "").upper())
    plt.xlabel("Decile (1=lowest score, 10=highest score)")
    plt.ylabel("Realized EV (mean win)")
    plt.title("EV by decile of model score (OOF)")
    plt.legend()
    fig1 = reports_figs / "ev_by_decile_histgb_vs_mlp.png"
    plt.savefig(fig1, dpi=200, bbox_inches="tight")
    plt.close()

    # ---- Top-k
    ks = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01]
    top_hgb = ev_topk(df, "pred_histgb", ks)
    top_mlp = ev_topk(df, "pred_mlp", ks)
    top = pd.concat([top_hgb, top_mlp], ignore_index=True)

    top_out = reports_tables / "ev_topk_histgb_vs_mlp.csv"
    top.to_csv(top_out, index=False)

    # plot
    plt.figure()
    for model in ["pred_histgb", "pred_mlp"]:
        sub = top[top["model"] == model].sort_values("top_k", ascending=False)
        plt.plot(sub["top_k"], sub["ev"], marker="o", label=model.replace("pred_", "").upper())
    plt.xlabel("Top-k fraction kept (higher=less selective)")
    plt.ylabel("Realized EV (mean win)")
    plt.title("EV on Top-k% by model score (OOF)")
    plt.legend()
    fig2 = reports_figs / "topk_ev_curve_histgb_vs_mlp.png"
    plt.savefig(fig2, dpi=200, bbox_inches="tight")
    plt.close()

    print("💾 Saved:")
    print(" -", dec_out)
    print(" -", top_out)
    print(" -", fig1)
    print(" -", fig2)
    print("\n✅ DONE. Total runtime:", time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)))


if __name__ == "__main__":
    main()