"""Render a lightweight landmark overlay for raw or processed NPZ inspection."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config.paths import INTERIM_LANDMARK_DIR, PROCESSED_LANDMARK_DIR
from src.data.wlasl import video_path
from src.preprocessing.landmark_io import load_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True); parser.add_argument("--processed", action="store_true"); parser.add_argument("--frame", type=int, default=0); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("Visualization requires opencv-python; install requirements.txt first.") from exc
    source = (PROCESSED_LANDMARK_DIR if args.processed else INTERIM_LANDMARK_DIR) / f"{args.video_id}.npz"
    payload = load_npz(source, processed=args.processed)
    capture = cv2.VideoCapture(str(video_path(args.video_id))); capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, image = capture.read(); capture.release()
    if not ok:
        raise SystemExit("Could not decode requested RGB frame.")
    height, width = image.shape[:2]
    palettes = {"pose_xyz": (0, 255, 0), "left_hand_xyz": (255, 80, 80), "right_hand_xyz": (80, 80, 255), "mouth_xyz": (0, 200, 255), "left_eye_xyz": (255, 255, 0), "right_eye_xyz": (255, 0, 255), "left_eyebrow_xyz": (200, 255, 0), "right_eyebrow_xyz": (0, 140, 255)}
    scale = float(payload.get("shoulder_scale", np.asarray(np.nan)).item())
    center = payload.get("normalization_center_xy")
    for key, color in palettes.items():
        points = payload[key][args.frame]
        for point_index, (x, y, z) in enumerate(points):
            if args.processed and np.isfinite(scale) and center is not None:
                x, y = x * scale + center[args.frame, 0], y * scale + center[args.frame, 1]
            if np.isfinite((x, y)).all(): cv2.circle(image, (round(x * width), round(y * height)), 2, color, -1)
    if args.processed:
        for side, color in (("left", (0, 255, 255)), ("right", (255, 255, 0))):
            if payload[f"{side}_hand_imputed"][args.frame]:
                cv2.putText(image, f"{side} hand: imputed", (10, 25 if side == "left" else 50), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    output = args.output or Path("reports") / "landmark_debug" / f"{args.video_id}_{'processed' if args.processed else 'raw'}_{args.frame:05d}.jpg"
    output.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(output), image)
    print(f"Wrote {output}. Z can be inspected directly in {source} (not rendered on RGB).")


if __name__ == "__main__":
    main()
