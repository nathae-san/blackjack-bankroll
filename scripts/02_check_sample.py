# scripts/02_check_sample.py
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional, Iterable

import pandas as pd


# Action codes from dataset doc
PLAY_ACTIONS = {"H", "S", "D", "P", "R"}  # Hit, Stand, Double, Split, Surrender
INSURANCE_ACTIONS = {"N", "I"}           # No Insurance, Insurance (I may be absent)


def find_repo_root(start: Path) -> Path:
    """Find repo root by looking for .git or README.md."""
    p = start.resolve()
    for _ in range(12):
        if (p / ".git").exists() or (p / "README.md").exists():
            return p
        p = p.parent
    raise RuntimeError("Repo root not found. Run this script from inside the repo.")


def safe_literal_eval(x: Any):
    """Safely parse Python-literal-like strings, e.g. '[7, 5]' or "[['H','S']]"."""
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
    """initial_hand is expected to be like '[11, 9]'."""
    obj = safe_literal_eval(x)
    if isinstance(obj, list) and len(obj) == 2:
        try:
            return [int(obj[0]), int(obj[1])]
        except Exception:
            return None
    return None


def parse_actions_taken(x: Any) -> list[str]:
    """
    actions_taken looks like: "[['H','S']]" or "[['S']]" or "[[]]"
    Return a flat list of tokens (first inner list if nested).
    """
    obj = safe_literal_eval(x)
    if obj is None:
        return []
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        # Nested list typical
        if isinstance(obj[0], list):
            return [str(t) for t in obj[0]]
        # Already flat
        return [str(t) for t in obj]
    return []


def extract_insurance_decision(tokens: Iterable[str]) -> Optional[str]:
    """Return 'N' or 'I' if present in tokens, else None."""
    for t in tokens:
        if t in INSURANCE_ACTIONS:
            return t
    return None


def extract_first_play_action(tokens: Iterable[str]) -> Optional[str]:
    """Return first real play action among H,S,D,P,R (ignoring insurance tokens)."""
    for t in tokens:
        if t in PLAY_ACTIONS:
            return t
    return None


def parse_player_final_value(x: Any) -> Optional[list[int]]:
    """player_final_value is like '[20]' or '[22]'."""
    obj = safe_literal_eval(x)
    if isinstance(obj, list):
        try:
            return [int(v) for v in obj]
        except Exception:
            return None
    return None


def parse_int_str(x: Any) -> Optional[int]:
    """dealer_final_value often is string '17' etc."""
    if pd.isna(x):
        return None
    try:
        return int(str(x).strip())
    except Exception:
        return None


def main() -> None:
    repo = find_repo_root(Path(__file__).parent)
    sample_path = repo / "data" / "processed" / "blackjack_sample.csv"

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample not found: {sample_path}\n"
            f"Generate it first with scripts/01_make_sample.py"
        )

    print("📄 Loading sample:", sample_path)
    df = pd.read_csv(sample_path)

    print("\n=== BASIC INFO ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    print("\n=== DTYPES ===")
    print(df.dtypes)

    print("\n=== HEAD (5) ===")
    print(df.head(5).to_string(index=False))

    print("\n=== MISSING VALUES (%) ===")
    print((df.isna().mean() * 100).round(3).to_string())

    # Dealer upcard distribution
    if "dealer_up" in df.columns:
        print("\n=== Dealer upcard distribution ===")
        print(df["dealer_up"].value_counts().to_string())

    # Win distribution
    if "win" in df.columns:
        print("\n=== WIN distribution ===")
        print(df["win"].value_counts(dropna=False).to_string())
        print("\nWIN summary:")
        print(df["win"].describe().to_string())

    # Parse initial hand sanity
    if "initial_hand" in df.columns:
        hand = df["initial_hand"].map(parse_initial_hand)
        fail_rate = hand.isna().mean() * 100
        print("\n=== initial_hand parsing ===")
        print(f"Parse failure rate: {fail_rate:.3f}%")
        if fail_rate < 1e-6:
            lengths = hand.map(len)
            print("Length distribution:")
            print(lengths.value_counts().to_string())

            # quick check: Ace encoding appears?
            has_ace = hand.map(lambda h: 11 in h if isinstance(h, list) else False).mean() * 100
            print(f"Hands containing an Ace (11): {has_ace:.2f}%")

    # Parse actions & split insurance vs play
    if "actions_taken" in df.columns:
        tokens = df["actions_taken"].map(parse_actions_taken)

        insurance = tokens.map(extract_insurance_decision)
        first_play = tokens.map(extract_first_play_action)

        print("\n=== ACTIONS parsing ===")
        # Tokens quality
        empty_seq = tokens.map(len).eq(0).mean() * 100
        print(f"Empty action sequences: {empty_seq:.2f}%")

        # Insurance decisions
        print("\nInsurance decision distribution (N/I/None):")
        print(insurance.value_counts(dropna=False).to_string())

        # First play action
        print("\nFirst PLAY action distribution (H/S/D/P/R/None):")
        print(first_play.value_counts(dropna=False).to_string())

        # Most common full sequences
        seq_str = tokens.map(lambda x: ",".join(x))
        print("\nTop 15 raw action sequences:")
        print(seq_str.value_counts().head(15).to_string())

        # Examples
        cols = [c for c in ["dealer_up", "initial_hand", "actions_taken", "win"] if c in df.columns]
        ex = df[cols].head(12).copy()
        ex["tokens"] = tokens.head(12).tolist()
        ex["insurance"] = insurance.head(12).tolist()
        ex["first_play"] = first_play.head(12).tolist()
        print("\nExamples:")
        print(ex.to_string(index=False))

    # Parse dealer_final_value and player_final_value for sanity (NOT for features)
    if "dealer_final_value" in df.columns:
        dealer_val = df["dealer_final_value"].map(parse_int_str)
        print("\n=== dealer_final_value sanity ===")
        print("Parse failure (%):", round(dealer_val.isna().mean() * 100, 3))
        print("Unique parsed values (sorted, first 20):", sorted(dealer_val.dropna().unique())[:20])

    if "player_final_value" in df.columns:
        player_val = df["player_final_value"].map(parse_player_final_value)
        print("\n=== player_final_value sanity ===")
        print("Parse failure (%):", round(player_val.isna().mean() * 100, 3))
        # Show common first element (often 20/21/22 etc.)
        first_elem = player_val.map(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else None)
        print("Top player_final_value[0]:")
        print(first_elem.value_counts(dropna=False).head(15).to_string())

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
