from __future__ import annotations

import shutil
from pathlib import Path
import kagglehub


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(12):
        if (p / ".git").exists() or (p / "README.md").exists():
            return p
        p = p.parent
    raise RuntimeError("Repo root not found.")


def main() -> None:
    repo = find_repo_root(Path(__file__).parent)
    out_dir = repo / "data" / "raw" / "blackjack"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(kagglehub.dataset_download("dennisho/blackjack-hands"))

    for item in src_path.iterdir():
        dest = out_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print("✅ Dataset copied to:", out_dir)
    print("Files:")
    for f in sorted([p.relative_to(out_dir) for p in out_dir.rglob("*") if p.is_file()]):
        print(" -", f)


if __name__ == "__main__":
    main()
