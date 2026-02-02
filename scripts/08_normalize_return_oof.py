# scripts/08_normalize_return_oof.py
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


def safe_parse_actions(x: str) -> List[str]:
    """
    actions_taken looks like:
      "[['H','S']]" or "[['N','S']]" or "[['S']]" or "[]"
    We convert to flat list of tokens: ["N","S"] etc.
    """
    if not isinstance(x, str) or x.strip() == "":
        return []
    s = x.strip()

    # Remove brackets and quotes in a simple robust way
    # Keep letters like H,S,D,P,R,N,I
    tokens = []
    cur = ""
    for ch in s:
        if ch.isalpha():
            cur += ch
        else:
            if cur:
                tokens.append(cur)
                cur = ""
    if cur:
        tokens.append(cur)
    return tokens


def estimate_wager(tokens: List[str]) -> float:
    """
    Approx wager estimation:
    - base bet = 1
    - each 'D' (double) adds +1 (bet becomes 2 on that hand)
    - each 'P' (split) adds +1 (a new hand with a new base bet)
    Note: if split + double(s) happen, this is still approximate but reasonable.
    Insurance ignored (dataset follows basic strategy; 'I' rarely used).
    """
    base = 1.0
    n_double = sum(t == "D" for t in tokens)
    n_split = sum(t == "P" for t in tokens)
    return base + float(n_double) + float(n_split)


def ev_by_deciles(df: pd.DataFrame, score_col: str, y_col: str, n_bins: int = 10) -> pd.DataFrame:
    tmp = df[[score_col, y_col]].copy()
    tmp["decile"] = pd.qcut(tmp[score_col], q=n_bins, labels=False, duplicates="drop") + 1
    agg = tmp.groupby("decile")[y_col].agg(["count", "mean"]).reset_index()
    agg = agg.rename(columns={"mean": "ev"})
    agg["model"] = score_col
    return agg


def ev_topk(df: pd.DataFrame, score_col: str, ks: List[float], y_col: str) -> pd.DataFrame:
    tmp = df[[score_col, y_col]].copy().sort_values(score_col, ascending=False)
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

    oof_path = repo / "reports" / "tables" / "oof_predictions_histgb_vs_mlp.csv"
    sample_path = repo / "data" / "processed" / "blackjack_sample.csv"

    if not oof_path.exists():
        raise FileNotFoundError("Missing OOF file. Run scripts/06_oof_predictions.py first.")
    if not sample_path.exists():
        raise FileNotFoundError("Missing sample csv. Run scripts/01_make_sample.py first.")

    print("📄 Loading OOF:", oof_path)
    oof = pd.read_csv(oof_path)

    print("📄 Loading sample:", sample_path)
    sample = pd.read_csv(sample_path)

    if "actions_taken" not in sample.columns:
        raise KeyError("actions_taken not found in blackjack_sample.csv")

    # Parse actions + wager + normalized return on sample
    sample = sample.copy()
    sample["tokens"] = sample["actions_taken"].map(safe_parse_actions)
    sample["wager_est"] = sample["tokens"].map(estimate_wager)
    sample["return_norm"] = sample["win"] / sample["wager_est"].replace(0, np.nan)

    # Merge OOF with sample to recover wager_est / return_norm
    # Use a rich key to minimize collisions
    merge_keys = ["shoe_id", "cards_remaining", "dealer_up", "run_count", "true_count", "win"]
    for k in merge_keys:
        if k not in oof.columns:
            raise KeyError(f"Missing '{k}' in OOF file. It must contain these columns.")
        if k not in sample.columns:
            raise KeyError(f"Missing '{k}' in sample file. It must contain these columns.")

    merged = oof.merge(
        sample[merge_keys + ["wager_est", "return_norm"]],
        on=merge_keys,
        how="left",
        validate="m:1",  # many OOF rows to 1 sample row is expected; if it fails, it will tell us
    )

    merge_rate = merged["wager_est"].notna().mean()
    print(f"🔎 Merge success rate: {merge_rate*100:.2f}%")
    if merge_rate < 0.98:
        print("⚠️ Merge rate < 98%. There might be key collisions or mismatches.")
        print("   If needed, we can strengthen the merge key (add card1/card2/player_total...).")

    # Drop rows without return_norm (should be rare)
    merged = merged.dropna(subset=["return_norm"]).copy()

    reports_tables = repo / "reports" / "tables"
    reports_figs = repo / "reports" / "figures"
    reports_tables.mkdir(parents=True, exist_ok=True)
    reports_figs.mkdir(parents=True, exist_ok=True)

    # Compare: models + counting baselines (same as step 7)
    merged["score_true_count"] = merged["true_count"].astype(float)
    merged["score_run_count"] = merged["run_count"].astype(float)

    score_cols = ["pred_histgb", "pred_mlp", "score_true_count", "score_run_count"]
    label_map = {
        "pred_histgb": "HISTGB",
        "pred_mlp": "MLP",
        "score_true_count": "TRUE_COUNT",
        "score_run_count": "RUN_COUNT",
    }

    # DECILES on normalized return
    dec = pd.concat([ev_by_deciles(merged, c, y_col="return_norm") for c in score_cols], ignore_index=True)
    dec_out = reports_tables / "return_by_deciles_models_vs_count.csv"
    dec.to_csv(dec_out, index=False)

    plt.figure()
    for c in score_cols:
        sub = dec[dec["model"] == c].sort_values("decile")
        plt.plot(sub["decile"], sub["ev"], marker="o", label=label_map.get(c, c))
    plt.xlabel("Decile (1=lowest score, 10=highest score)")
    plt.ylabel("Realized return (mean win / wager_est)")
    plt.title("Return by decile (OOF): Models vs Counting baselines")
    plt.legend()
    fig_dec = reports_figs / "return_by_decile_models_vs_count.png"
    plt.savefig(fig_dec, dpi=200, bbox_inches="tight")
    plt.close()

    # TOP-K on normalized return
    ks = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01]
    top = pd.concat([ev_topk(merged, c, ks, y_col="return_norm") for c in score_cols], ignore_index=True)
    top_out = reports_tables / "return_topk_models_vs_count.csv"
    top.to_csv(top_out, index=False)

    plt.figure()
    for c in score_cols:
        sub = top[top["model"] == c].sort_values("top_k", ascending=False)
        plt.plot(sub["top_k"], sub["ev"], marker="o", label=label_map.get(c, c))
    plt.xlabel("Top-k fraction kept (higher=less selective)")
    plt.ylabel("Realized return (mean win / wager_est)")
    plt.title("Return on Top-k% (OOF): Models vs Counting baselines")
    plt.legend()
    fig_top = reports_figs / "return_topk_curve_models_vs_count.png"
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
