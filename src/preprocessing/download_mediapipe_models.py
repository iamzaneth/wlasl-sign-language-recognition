"""Download the three official MediaPipe Tasks assets required by extraction.

The assets are intentionally kept outside Git. Downloads stream to a temporary
file and are atomically renamed only after a basic size check succeeds.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from src.config.paths import MEDIAPIPE_MODEL_DIR

MODELS = {
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}
MIN_BYTES = 100_000


def valid_asset(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= MIN_BYTES


def download(url: str, destination: Path, overwrite: bool) -> str:
    if valid_asset(destination) and not overwrite:
        return "skipped_existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = Request(url, headers={"User-Agent": "wlasl-sign-language-recognition/1.0"})
    with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if not valid_asset(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded asset is unexpectedly small: {destination.name}")
    temporary.replace(destination)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MEDIAPIPE_MODEL_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for filename, url in MODELS.items():
        target = args.model_dir / filename
        print(f"{filename}: {download(url, target, args.overwrite)} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
