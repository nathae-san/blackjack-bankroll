# scripts/03_build_features.py
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional

import pandas as pd


PLAY_ACTIONS = {"H", "S", "D", "P", "R"}
INSURANCE_ACTIONS = {"N", "I"}  # in practice: N only in your dataset


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(12):
        if (p / ".git").exists() or (p / "README.md").exists():
            return p
        p = p.parent
    raise RuntimeError("Repo root not found. Run this script from inside the repo.")


def safe_literal_eval(x: Any):
    if pd.isna(x):
        return None
    if isinstance(x, (list, dict, int, float)):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x.strip())
        except Exception:
            return None
    return None


def parse_initial_hand(x: Any) -> Optional[list[int]]:
    obj = safe_literal_eval(x)
    if isinstance(obj, list) and len(obj) == 2:
        try:
            return [int(obj[0]), int(obj[1])]
        except Exception:
            return None
    return None


def parse_actions_taken(x: Any) -> list[str]:
    obj = safe_literal_eval(x)
    if obj is None:
        return []
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], list):
            return [str(t) for t in obj[0]]
        return [str(t) for t in obj]
    return []


def has_token(tokens: list[str], token_set: set[str]) -> bool:
    return any(t in token_set for t in tokens)


def main() -> None:
    repo = find_repo_root(Path(__file__).parent)
    sample_path = repo / "data" / "processed" / "blackjack_sample.csv"
    out_dir = repo / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_path}")

    print("📄 Loading sample:", sample_path)
    df = pd.read_csv(sample_path)

    # --- Parse initial hand
    hand = df["initial_hand"].map(parse_initial_hand)
    if hand.isna().any():
        bad = hand.isna().mean() * 100
        raise ValueError(f"initial_hand parsing failed for {bad:.2f}% rows. Unexpected format.")

    df["card1"] = hand.map(lambda h: h[0])
    df["card2"] = hand.map(lambda h: h[1])

    df["is_pair"] = (df["card1"] == df["card2"]).astype(int)
    df["has_ace"] = ((df["card1"] == 11) | (df["card2"] == 11)).astype(int)
    df["player_total"] = (df["card1"] + df["card2"]).astype(int)
    # "soft" means at least one Ace counted as 11 (with only 2 cards here)
    df["is_soft"] = df["has_ace"].astype(int)

    # --- Parse actions for insurance info (optional feature)
    tokens = df["actions_taken"].map(parse_actions_taken)
    df["insurance_offered"] = (df["dealer_up"] == 11).astype(int)
    df["insurance_no"] = tokens.map(lambda t: int("N" in t))  # should match insurance_offered in your data

    # --- Keep only PRE-ROUND features (no leakage)
    feature_cols = [
        "shoe_id",            # kept for grouped CV split only (NOT as model input)
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
    ]

    if "win" not in df.columns:
        raise KeyError("Column 'win' not found in sample.")

    out = df[feature_cols + ["win"]].copy()

    # Sanity: no missing values expected
    na = out.isna().mean().sort_values(ascending=False)
    if na.iloc[0] > 0:
        print("\n⚠️ Missing values detected (top):")
        print((na[na > 0] * 100).round(3).to_string())

    # Save
    parquet_path = out_dir / "features_sample.parquet"
    csv_path = out_dir / "features_sample.csv"

    out.to_parquet(parquet_path, index=False)
    # CSV optional but handy
    out.to_csv(csv_path, index=False)

    print("\n✅ Saved:")
    print(" -", parquet_path)
    print(" -", csv_path)

    print("\nPreview:")
    print(out.head(10).to_string(index=False))
    print("\nShape:", out.shape)
    print("\nWin mean (EV):", out["win"].mean())


if __name__ == "__main__":
    main()
