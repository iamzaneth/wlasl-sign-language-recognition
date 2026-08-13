"""Build the reusable fixed-length NPZ landmark cache for encoder training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python scripts/<file>.py`` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import DATA_DIR, PROCESSED_LANDMARK_DIR
from src.training.encoder_data import build_feature_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True, choices=("wlasl100", "wlasl300", "wlasl1000", "wlasl2000"))
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--landmark-dir", type=Path, default=PROCESSED_LANDMARK_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing cache directory.")
    args = parser.parse_args()
    cache_dir = args.cache_dir or DATA_DIR / "processed" / "encoder_features" / args.subset / f"seq{args.seq_len}"
    result = build_feature_cache(args.subset, seq_len=args.seq_len, cache_dir=cache_dir, landmark_dir=args.landmark_dir, overwrite=args.overwrite)
    print(f"Feature cache ready: {result}")


if __name__ == "__main__":
    main()
