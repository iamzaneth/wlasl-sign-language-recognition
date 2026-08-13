"""Run the resumable WLASL audit-to-processed-landmark pipeline for one subset.

Examples:
    python -m src.pipeline.run_data_pipeline --subset wlasl100
    python -m src.pipeline.run_data_pipeline --subset wlasl2000 --refresh-audit
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config.paths import INTERIM_MANIFEST_DIR, MEDIAPIPE_MODEL_DIR, SUBSETS
from src.data.audit_dataset import audit_dataset
from src.data.build_splits import build_splits
from src.preprocessing.download_mediapipe_models import MODELS, download, valid_asset
from src.preprocessing.extract_landmarks import ids_from_split, run_extraction
from src.preprocessing.preprocess_landmarks import run_preprocessing


def ensure_models(model_dir: Path, overwrite: bool = False) -> None:
    """Fetch missing task bundles only; valid local assets are always reused."""
    for filename, url in MODELS.items():
        target = model_dir / filename
        print(f"{filename}: {download(url, target, overwrite)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True, choices=SUBSETS, help="WLASL supports 100, 300, 1000 and 2000 classes; there is no WLASL3000.")
    parser.add_argument("--model-dir", type=Path, default=MEDIAPIPE_MODEL_DIR)
    parser.add_argument("--refresh-audit", action="store_true", help="Force a complete raw-video audit even when valid manifests already exist.")
    parser.add_argument("--skip-model-download", action="store_true", help="Require existing valid task bundles instead of downloading missing ones.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing landmark outputs instead of resuming.")
    parser.add_argument("--max-gap-frames", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--input-is-mirrored", action="store_true")
    parser.add_argument("--show-mediapipe-logs", action="store_true", help="Show native MediaPipe/TFLite diagnostics for troubleshooting.")
    args = parser.parse_args()

    manifest = INTERIM_MANIFEST_DIR / "valid_videos.csv"
    if args.refresh_audit or not manifest.exists():
        print("\n[1/5] Auditing raw WLASL data...")
        audit_dataset()
    else:
        print("\n[1/5] Reusing existing valid-video manifest (pass --refresh-audit to rebuild it).")

    print(f"\n[2/5] Building official usable {args.subset} splits...")
    print(build_splits(args.subset))

    print("\n[3/5] Ensuring MediaPipe task bundles...")
    if args.skip_model_download:
        missing = [name for name in MODELS if not valid_asset(args.model_dir / name)]
        if missing:
            parser.error("Missing required task bundle(s): " + ", ".join(missing))
        print("Using existing valid task bundles.")
    else:
        ensure_models(args.model_dir)

    video_ids = ids_from_split(args.subset)
    print(f"\n[4/5] Extracting raw MediaPipe landmarks for {len(video_ids):,} usable videos...")
    extraction = run_extraction(video_ids, overwrite=args.overwrite, model_dir=args.model_dir, input_is_mirrored=args.input_is_mirrored, report_every=args.report_every, quiet_native_logs=not args.show_mediapipe_logs)

    print(f"\n[5/5] Preprocessing XYZ landmarks for {len(video_ids):,} usable videos...")
    preprocessing = run_preprocessing(video_ids, max_gap_frames=args.max_gap_frames, overwrite=args.overwrite)
    print(f"\nPipeline complete for {args.subset}: extraction={extraction}; preprocessing={preprocessing}")


if __name__ == "__main__":
    main()
