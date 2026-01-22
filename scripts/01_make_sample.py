# scripts/01_make_sample.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

SEED = 42

def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    raw_dir = repo / "data" / "raw" / "blackjack"
    out_dir = repo / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # adapte le nom exact du fichier
    csv_path = raw_dir / "blackjack_simulator.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    target_n = 500_000  # 500k mains (à ajuster)
    chunk_size = 200_000

    rng = np.random.default_rng(SEED)
    kept = []
    seen = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        seen += len(chunk)

        # On garde une fraction approximative pour atteindre target_n
        # (simple + rapide)
        remaining = max(target_n - sum(len(c) for c in kept), 0)
        if remaining <= 0:
            break

        frac = min(1.0, remaining / len(chunk))
        mask = rng.random(len(chunk)) < frac
        kept.append(chunk.loc[mask])

    sample = pd.concat(kept, ignore_index=True).sample(frac=1, random_state=SEED)  # shuffle
    sample = sample.head(target_n)

    out_path = out_dir / "blackjack_sample.csv"
    sample.to_csv(out_path, index=False)
    print("✅ Sample saved to:", out_path, "shape:", sample.shape)

if __name__ == "__main__":
    main()
