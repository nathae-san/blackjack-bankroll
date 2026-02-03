# scripts/10_backtest_truecount_spread.py
from __future__ import annotations

import argparse
from pathlib import Path
import time

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


def bet_size_from_true_count(tc: float) -> int:
    """
    Option A (report-friendly):
    - Wong-out if tc < 1
    - Spread: 1 / 2 / 4 units as tc increases
    """
    if pd.isna(tc):
        return 0
    if tc < 1:
        return 0
    if tc < 3:
        return 1
    if tc < 5:
        return 2
    return 4


def compute_equity_curve(df: pd.DataFrame, initial_bankroll: float = 1000.0) -> tuple[pd.DataFrame, dict]:
    """
    df must contain at least: true_count, win
    Optional: shoe_id (used only for info / ordering if desired)
    """

    # Apply betting rule (pre-cards only)
    bet = df["true_count"].apply(bet_size_from_true_count).astype(int)
    stake = (bet > 0).astype(int)

    # PnL per row: win is "per 1 unit". We scale by bet size.
    # (win can be -1, 0, 1, 1.5, 2, -2, etc.)
    pnl = df["win"].astype(float) * bet.astype(float)

    equity = np.empty(len(df), dtype=float)
    drawdown = np.empty(len(df), dtype=float)

    eq = float(initial_bankroll)
    peak = float(initial_bankroll)

    for i in range(len(df)):
        eq += float(pnl.iloc[i])
        if eq > peak:
            peak = eq
        equity[i] = eq
        # drawdown as relative drop from peak (<=0)
        drawdown[i] = (eq / peak) - 1.0 if peak > 0 else 0.0

    curve = pd.DataFrame(
        {
            "t": np.arange(len(df)),
            "equity": equity,
            "drawdown": drawdown,
            "pnl": pnl.values,
            "stake": stake.values,
            "bet": bet.values,
        }
    )

    # Stats
    n_rows = len(df)
    n_bets = int(stake.sum())
    total_pnl = float(pnl.sum())
    final_bankroll = float(equity[-1]) if n_rows > 0 else float(initial_bankroll)

    # hit rate among bets: % of pnl>0 conditional on bet>0
    if n_bets > 0:
        hit_rate = float((pnl[bet > 0] > 0).mean())
        avg_pnl_per_bet = float((pnl[bet > 0]).mean())
        avg_bet_size = float(bet[bet > 0].mean())
    else:
        hit_rate = float("nan")
        avg_pnl_per_bet = float("nan")
        avg_bet_size = float("nan")

    stats = {
        "tag": "truecount_spread",
        "score_col": "true_count",
        "strategy": "TC_SPREAD_WONG_OUT",
        "rule": "0 if tc<1; 1 if [1,3); 2 if [3,5); 4 if tc>=5",
        "initial_bankroll": float(initial_bankroll),
        "final_bankroll": final_bankroll,
        "total_pnl": total_pnl,
        "n_rows_simulated": int(n_rows),
        "n_bets": int(n_bets),
        "avg_bet_size": avg_bet_size,
        "avg_pnl_per_bet": avg_pnl_per_bet,
        "hit_rate": hit_rate,
        "max_drawdown": float(curve["drawdown"].min()) if n_rows > 0 else float("nan"),
        "time_in_market": float(n_bets / n_rows) if n_rows > 0 else float("nan"),
    }

    return curve, stats


def plot_equity_and_drawdown(curve: pd.DataFrame, title_suffix: str, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Equity
    fig1 = plt.figure()
    plt.plot(curve["t"], curve["equity"])
    plt.title(f"Bankroll backtest: {title_suffix}")
    plt.xlabel("Time (hand index)")
    plt.ylabel("Bankroll")
    equity_path = out_dir / f"bankroll_{title_suffix}.png"
    plt.tight_layout()
    plt.savefig(equity_path, dpi=150)
    plt.close(fig1)

    # Drawdown
    fig2 = plt.figure()
    plt.plot(curve["t"], curve["drawdown"])
    plt.title(f"Drawdown: {title_suffix}")
    plt.xlabel("Time (hand index)")
    plt.ylabel("Drawdown (relative to peak)")
    dd_path = out_dir / f"drawdown_{title_suffix}.png"
    plt.tight_layout()
    plt.savefig(dd_path, dpi=150)
    plt.close(fig2)

    return equity_path, dd_path


def main():
    parser = argparse.ArgumentParser(description="Step 10: realistic bankroll backtest using True Count spread (Wong out + bet spread).")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to features file (parquet/csv). Default: data/processed/features_sample.parquet",
    )
    parser.add_argument("--initial_bankroll", type=float, default=1000.0)
    parser.add_argument("--use_shoe_order", action="store_true", help="If set, sorts by shoe_id then original index (more shoe-consistent).")
    args = parser.parse_args()

    t0 = time.time()
    repo = find_repo_root(Path(__file__).parent)

    in_path = Path(args.input) if args.input else repo / "data" / "processed" / "features_sample.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    print(f"📄 Loading features: {in_path}")
    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    # Minimal required columns
    required = {"true_count", "win"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in features: {sorted(missing)}")

    # Optional ordering
    if args.use_shoe_order and "shoe_id" in df.columns:
        df = df.reset_index(drop=False).rename(columns={"index": "_row"})
        df = df.sort_values(["shoe_id", "_row"]).reset_index(drop=True)
        print("🔁 Using shoe-consistent ordering: sorted by (shoe_id, original_row_index).")
    else:
        df = df.reset_index(drop=True)

    curve, stats = compute_equity_curve(df, initial_bankroll=args.initial_bankroll)

    # Save outputs
    reports_tables = repo / "reports" / "tables"
    reports_figs = repo / "reports" / "figures"
    reports_tables.mkdir(parents=True, exist_ok=True)
    reports_figs.mkdir(parents=True, exist_ok=True)

    tag = stats["tag"]
    curve_path = reports_tables / f"bankroll_curve_{tag}.csv"
    stats_path = reports_tables / f"bankroll_stats_{tag}.csv"

    curve.to_csv(curve_path, index=False)
    pd.DataFrame([stats]).to_csv(stats_path, index=False)

    eq_fig, dd_fig = plot_equity_and_drawdown(curve, title_suffix=tag, out_dir=reports_figs)

    # Console summary
    print("\n=== TRUE COUNT SPREAD BACKTEST (REALISTIC) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\n💾 Saved:")
    print(f" - {curve_path}")
    print(f" - {stats_path}")
    print(f" - {eq_fig}")
    print(f" - {dd_fig}")

    elapsed = time.time() - t0
    print(f"\n✅ DONE. Total runtime: {elapsed:0.2f}s")


if __name__ == "__main__":
    main()
