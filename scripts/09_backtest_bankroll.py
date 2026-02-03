# scripts/09_backtest_bankroll.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


@dataclass
class BacktestConfig:
    # Input
    oof_path: str = "reports/tables/oof_predictions_histgb_vs_mlp.csv"

    # Sorting (approx chronological inside each shoe)
    sort_by_shoe: bool = True

    # Strategy
    score_col: str = "pred_histgb"   # "pred_histgb" | "pred_mlp" | "true_count" | etc.
    strategy: str = "TOP_QUANTILE"  # "ALWAYS_BET" | "TOP_QUANTILE"
    top_quantile: float = 0.05      # if TOP_QUANTILE: bet only if score >= quantile(1-top_quantile)

    # Betting
    initial_bankroll: float = 1000.0
    base_bet: float = 1.0           # stake unit
    bet_sizing: str = "FLAT"        # "FLAT" | "LINEAR_SCORE"
    score_center: str = "MEDIAN"    # "MEDIAN" | "ZERO" (only for LINEAR_SCORE)
    score_scale: float = 1.0        # larger => more aggressive sizing
    bet_cap: float = 10.0           # max stake (in units of base_bet)
    allow_negative_bankroll: bool = False  # if False: stop betting when bankroll < bet

    # Output naming
    tag: str = "histgb_top5pct"


def compute_drawdown(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak == 0, 1.0, peak)
    return dd


def sharpe(x: np.ndarray) -> float:
    # Simple daily-like Sharpe on per-bet PnL series (no annualization)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    mu = np.mean(x)
    sd = np.std(x, ddof=1)
    if sd == 0:
        return float("nan")
    return float(mu / sd)


def main() -> None:
    t0 = time.time()
    cfg = BacktestConfig()

    repo = find_repo_root(Path(__file__).parent)
    oof_path = repo / cfg.oof_path
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}. Run step 6 first.")

    print("📄 Loading OOF:", oof_path)
    df = pd.read_csv(oof_path)

    # Ensure required columns
    for col in ["win", "shoe_id", "cards_remaining", "true_count", "run_count"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in OOF file.")

    # Build score column if user wants true_count directly
    if cfg.score_col not in df.columns:
        if cfg.score_col == "true_count":
            df["true_count"] = df["true_count"].astype(float)
        elif cfg.score_col == "run_count":
            df["run_count"] = df["run_count"].astype(float)
        else:
            raise KeyError(f"score_col='{cfg.score_col}' not found in OOF file.")

    # Sort to approximate chronology
    if cfg.sort_by_shoe:
        # cards_remaining is higher earlier in shoe, so DESC to simulate time
        df = df.sort_values(["shoe_id", "cards_remaining"], ascending=[True, False]).reset_index(drop=True)

    # Selection threshold
    score = df[cfg.score_col].astype(float).values
    if cfg.strategy == "ALWAYS_BET":
        threshold = -np.inf
    elif cfg.strategy == "TOP_QUANTILE":
        q = 1.0 - float(cfg.top_quantile)
        q = min(max(q, 0.0), 1.0)
        threshold = float(np.quantile(score, q))
    else:
        raise ValueError("strategy must be ALWAYS_BET or TOP_QUANTILE")

    # Betting sizing reference
    if cfg.bet_sizing == "FLAT":
        center_val = 0.0
    elif cfg.bet_sizing == "LINEAR_SCORE":
        if cfg.score_center == "MEDIAN":
            center_val = float(np.median(score))
        elif cfg.score_center == "ZERO":
            center_val = 0.0
        else:
            raise ValueError("score_center must be MEDIAN or ZERO")
    else:
        raise ValueError("bet_sizing must be FLAT or LINEAR_SCORE")

    bankroll = float(cfg.initial_bankroll)
    equity = []
    pnl_series = []          # realized PnL on bets (0 if no bet)
    bet_series = []
    bet_flag = []

    # Backtest loop
    for i in range(len(df)):
        s = float(df.at[i, cfg.score_col])
        y = float(df.at[i, "win"])  # profit in "units" per hand in this dataset

        do_bet = (s >= threshold)

        # stake sizing
        if do_bet:
            if cfg.bet_sizing == "FLAT":
                stake = cfg.base_bet
            else:
                # Linear sizing: stake increases with (score - center)
                raw = 1.0 + cfg.score_scale * (s - center_val)
                # Prevent negative or tiny stakes
                stake = cfg.base_bet * max(0.0, raw)
                # Cap
                stake = min(stake, cfg.base_bet * cfg.bet_cap)
                # Ensure minimal stake if selected
                stake = max(stake, cfg.base_bet)

            # bankroll constraint
            if (not cfg.allow_negative_bankroll) and (bankroll < stake):
                do_bet = False
                stake = 0.0
        else:
            stake = 0.0

        # PnL
        pnl = stake * y if do_bet else 0.0
        bankroll = bankroll + pnl

        equity.append(bankroll)
        pnl_series.append(pnl)
        bet_series.append(stake)
        bet_flag.append(1 if do_bet else 0)

        # If bankroll cannot go negative and hits 0, you can stop early (optional)
        if (not cfg.allow_negative_bankroll) and bankroll <= 0:
            # keep recording flat equity afterwards? here we stop
            break

    equity = np.array(equity, dtype=float)
    pnl_series = np.array(pnl_series, dtype=float)
    bet_series = np.array(bet_series, dtype=float)
    bet_flag = np.array(bet_flag, dtype=int)

    dd = compute_drawdown(equity)
    max_dd = float(np.min(dd)) if len(dd) else float("nan")

    n_bets = int(bet_flag.sum())
    total_pnl = float(np.sum(pnl_series))
    avg_pnl_per_bet = float(np.sum(pnl_series) / n_bets) if n_bets > 0 else float("nan")
    hit_rate = float(np.mean(pnl_series[pnl_series != 0] > 0)) if n_bets > 0 else float("nan")

    stats = {
        "tag": cfg.tag,
        "score_col": cfg.score_col,
        "strategy": cfg.strategy,
        "top_quantile": cfg.top_quantile if cfg.strategy == "TOP_QUANTILE" else np.nan,
        "bet_sizing": cfg.bet_sizing,
        "initial_bankroll": cfg.initial_bankroll,
        "final_bankroll": float(equity[-1]) if len(equity) else cfg.initial_bankroll,
        "total_pnl": total_pnl,
        "n_rows_simulated": int(len(equity)),
        "n_bets": n_bets,
        "avg_pnl_per_bet": avg_pnl_per_bet,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
        "sharpe_pnl": sharpe(pnl_series[pnl_series != 0]) if n_bets > 1 else float("nan"),
    }

    # Save outputs
    reports_tables = repo / "reports" / "tables"
    reports_figs = repo / "reports" / "figures"
    reports_tables.mkdir(parents=True, exist_ok=True)
    reports_figs.mkdir(parents=True, exist_ok=True)

    curve_df = pd.DataFrame({
        "t": np.arange(len(equity)),
        "equity": equity,
        "drawdown": dd,
        "pnl": pnl_series,
        "stake": bet_series,
        "bet": bet_flag,
    })
    curve_path = reports_tables / f"bankroll_curve_{cfg.tag}.csv"
    curve_df.to_csv(curve_path, index=False)

    stats_path = reports_tables / f"bankroll_stats_{cfg.tag}.csv"
    pd.DataFrame([stats]).to_csv(stats_path, index=False)

    # Plot equity curve
    plt.figure()
    plt.plot(curve_df["t"], curve_df["equity"])
    plt.xlabel("Time (hand index in sorted OOF)")
    plt.ylabel("Bankroll")
    plt.title(f"Bankroll backtest: {cfg.tag}")
    fig_equity = reports_figs / f"bankroll_curve_{cfg.tag}.png"
    plt.savefig(fig_equity, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot drawdown
    plt.figure()
    plt.plot(curve_df["t"], curve_df["drawdown"])
    plt.xlabel("Time (hand index in sorted OOF)")
    plt.ylabel("Drawdown")
    plt.title(f"Drawdown: {cfg.tag}")
    fig_dd = reports_figs / f"drawdown_{cfg.tag}.png"
    plt.savefig(fig_dd, dpi=200, bbox_inches="tight")
    plt.close()

    print("✅ Backtest summary")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\n💾 Saved:")
    print(" -", curve_path)
    print(" -", stats_path)
    print(" -", fig_equity)
    print(" -", fig_dd)
    print("\n✅ DONE. Total runtime:", time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)))


if __name__ == "__main__":
    main()
