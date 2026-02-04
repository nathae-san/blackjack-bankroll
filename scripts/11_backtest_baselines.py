# scripts/11_backtest_baselines.py
from __future__ import annotations

from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(12):
        if (p / ".git").exists() or (p / "README.md").exists():
            return p
        p = p.parent
    raise RuntimeError("Repo root not found. Run from inside the repo.")


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Input must be .parquet or .csv")

    # Required columns
    required = {"win", "true_count"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure win is numeric
    df["win"] = pd.to_numeric(df["win"], errors="coerce")
    df["true_count"] = pd.to_numeric(df["true_count"], errors="coerce")
    df = df.dropna(subset=["win", "true_count"]).reset_index(drop=True)

    return df


# ----------------------------
# Betting rules (strategies)
# ----------------------------
def bet_flat(tc: float) -> float:
    return 1.0


def bet_wong_out(tc: float) -> float:
    # play only when tc>=1
    return 1.0 if tc >= 1.0 else 0.0


def bet_tc_spread_wong_out(tc: float) -> float:
    # 0 if tc<1; 1 if [1,3); 2 if [3,5); 4 if tc>=5
    if tc < 1.0:
        return 0.0
    if tc < 3.0:
        return 1.0
    if tc < 5.0:
        return 2.0
    return 4.0


STRATEGIES = {
    "flat_bet": {
        "tag": "flat_bet",
        "strategy": "FLAT_BET",
        "rule": "1 unit always",
        "bet_fn": bet_flat,
    },
    "wong_out": {
        "tag": "wong_out",
        "strategy": "WONG_OUT",
        "rule": "0 if tc<1 else 1",
        "bet_fn": bet_wong_out,
    },
    "truecount_spread": {
        "tag": "truecount_spread",
        "strategy": "TC_SPREAD_WONG_OUT",
        "rule": "0 if tc<1; 1 if [1,3); 2 if [3,5); 4 if tc>=5",
        "bet_fn": bet_tc_spread_wong_out,
    },
}


# ----------------------------
# Backtest core
# ----------------------------
def run_backtest(
    df: pd.DataFrame,
    score_col: str,
    bet_fn,
    initial_bankroll: float = 1000.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest with:
      pnl_t = bet_t * win_t
      equity_t = equity_{t-1} + pnl_t
      drawdown_t = equity_t/peak_t - 1
    """
    n = len(df)
    tc = df[score_col].to_numpy(dtype=float)
    win = df["win"].to_numpy(dtype=float)

    bet = np.fromiter((bet_fn(x) for x in tc), dtype=float, count=n)
    pnl = bet * win

    equity = np.empty(n, dtype=float)
    peak = np.empty(n, dtype=float)
    drawdown = np.empty(n, dtype=float)

    eq = initial_bankroll
    pk = initial_bankroll

    for i in range(n):
        eq = eq + pnl[i]
        pk = max(pk, eq)
        equity[i] = eq
        peak[i] = pk
        drawdown[i] = (eq / pk) - 1.0  # <= 0

    curve = pd.DataFrame(
        {
            "t": np.arange(n, dtype=int),
            "equity": equity,
            "drawdown": drawdown,
            "pnl": pnl,
            "stake": bet,  # keeping same naming style as your earlier files
            "bet": bet,
        }
    )

    mask_bets = bet > 0
    n_bets = int(mask_bets.sum())
    total_pnl = float(pnl.sum())
    final_bankroll = float(initial_bankroll + total_pnl)

    avg_bet_size = float(bet[mask_bets].mean()) if n_bets > 0 else 0.0
    avg_pnl_per_bet = float(total_pnl / n_bets) if n_bets > 0 else 0.0
    hit_rate = float((pnl[mask_bets] > 0).mean()) if n_bets > 0 else 0.0
    max_dd = float(drawdown.min()) if n > 0 else 0.0
    time_in_market = float(n_bets / n) if n > 0 else 0.0

    stats = {
        "initial_bankroll": float(initial_bankroll),
        "final_bankroll": float(final_bankroll),
        "total_pnl": float(total_pnl),
        "n_rows_simulated": int(n),
        "n_bets": int(n_bets),
        "avg_bet_size": float(avg_bet_size),
        "avg_pnl_per_bet": float(avg_pnl_per_bet),
        "hit_rate": float(hit_rate),
        "max_drawdown": float(max_dd),
        "time_in_market": float(time_in_market),
    }

    return curve, stats


# ----------------------------
# Plotting
# ----------------------------
def save_plots(curve: pd.DataFrame, title_tag: str, out_fig_dir: Path) -> None:
    # Bankroll
    plt.figure()
    plt.plot(curve["t"], curve["equity"])
    plt.title(f"Bankroll backtest: {title_tag}")
    plt.xlabel("Time (hand index)")
    plt.ylabel("Bankroll")
    plt.tight_layout()
    plt.savefig(out_fig_dir / f"bankroll_backtest_{title_tag}.png", dpi=160)
    plt.close()

    # Drawdown
    plt.figure()
    plt.plot(curve["t"], curve["drawdown"])
    plt.title(f"Drawdown: {title_tag}")
    plt.xlabel("Time (hand index)")
    plt.ylabel("Drawdown (relative to peak)")
    plt.tight_layout()
    plt.savefig(out_fig_dir / f"drawdown_{title_tag}.png", dpi=160)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/features_sample.parquet",
        help="Path to features (.parquet or .csv)",
    )
    parser.add_argument("--initial_bankroll", type=float, default=1000.0)
    parser.add_argument(
        "--score_col", type=str, default="true_count", help="Column used for betting rule"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="flat_bet,wong_out,truecount_spread",
        help="Comma-separated keys: flat_bet,wong_out,truecount_spread",
    )
    args = parser.parse_args()

    repo = find_repo_root(Path(__file__).parent)
    in_path = (repo / args.input).resolve()

    out_tables = repo / "reports" / "tables"
    out_figs = repo / "reports" / "figures"
    ensure_dirs(out_tables, out_figs)

    t0 = time.time()
    print(f"📄 Loading features: {in_path}")
    df = load_features(in_path)
    print(f"✅ Loaded {len(df):,} rows")

    chosen = [s.strip() for s in args.strategies.split(",") if s.strip()]
    summary_rows = []

    for key in chosen:
        if key not in STRATEGIES:
            raise ValueError(f"Unknown strategy key: {key}. Available: {list(STRATEGIES.keys())}")

        meta = STRATEGIES[key]
        tag = meta["tag"]
        strategy_name = meta["strategy"]
        rule = meta["rule"]
        bet_fn = meta["bet_fn"]

        print(f"\n🚀 Backtest: {strategy_name} ({tag}) | rule: {rule}")
        curve, stats = run_backtest(
            df=df,
            score_col=args.score_col,
            bet_fn=bet_fn,
            initial_bankroll=args.initial_bankroll,
        )

        # Save curve + plots
        curve_path = out_tables / f"bankroll_curve_{tag}.csv"
        curve.to_csv(curve_path, index=False)
        save_plots(curve, tag, out_figs)

        # Save stats row (same schema style as what you pasted)
        row = {
            "tag": tag,
            "score_col": args.score_col,
            "strategy": strategy_name,
            "rule": rule,
            **stats,
        }
        stats_path = out_tables / f"bankroll_stats_{tag}.csv"
        pd.DataFrame([row]).to_csv(stats_path, index=False)

        summary_rows.append(row)

        print(f"✅ Saved:")
        print(f" - {stats_path}")
        print(f" - {curve_path}")
        print(f" - {out_figs / f'bankroll_backtest_{tag}.png'}")
        print(f" - {out_figs / f'drawdown_{tag}.png'}")
        print(
            f"   Stats: final={row['final_bankroll']:.2f}, pnl={row['total_pnl']:.2f}, "
            f"n_bets={row['n_bets']}, time_in_market={row['time_in_market']:.3f}, "
            f"max_dd={row['max_drawdown']:.3f}, avg_pnl/bet={row['avg_pnl_per_bet']:.4f}"
        )

    # Save global summary
    summary = pd.DataFrame(summary_rows).sort_values(by="final_bankroll", ascending=False)
    summary_path = out_tables / "bankroll_baselines_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n📌 Saved baseline summary: {summary_path}")
    print(summary[["tag", "strategy", "final_bankroll", "total_pnl", "n_bets", "time_in_market", "max_drawdown", "avg_pnl_per_bet"]])

    print(f"\n✅ DONE. Total runtime: {time.time() - t0:0.1f}s")


if __name__ == "__main__":
    main()
