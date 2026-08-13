"""Extract raw, unnormalised MediaPipe XYZ landmarks for selected WLASL videos."""
from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.config.paths import INTERIM_LANDMARK_DIR, MEDIAPIPE_MODEL_DIR, METADATA_DIR, SUBSETS
from src.data.wlasl import video_path
from src.preprocessing.landmark_io import is_valid_npz, load_npz, save_npz
from src.preprocessing.landmark_schema import (COORDINATE_ORDER, HAND_LANDMARK_COUNT, LANDMARK_SCHEMA_VERSION, LEFT_EYE_INDICES, LEFT_EYEBROW_INDICES, MOUTH_INDICES, POSE_SOURCE_INDICES, RIGHT_EYE_INDICES, RIGHT_EYEBROW_INDICES, empty_xyz)

REPORT_FIELDS = ["video_id", "num_frames", "fps", "pose_detected_frames", "pose_detection_rate", "left_hand_detected_frames", "left_hand_detection_rate", "right_hand_detected_frames", "right_hand_detection_rate", "face_detected_frames", "face_detection_rate", "left_hand_missing_frames", "right_hand_missing_frames", "left_hand_short_gap_count", "right_hand_short_gap_count", "left_hand_imputed_frames", "right_hand_imputed_frames", "left_hand_unresolved_frames", "right_hand_unresolved_frames", "normalization_invalid_frames", "extraction_success", "error_message"]


def _dependencies() -> tuple[Any, Any]:
    # Set before importing MediaPipe so its native TFLite/glog runtime suppresses
    # noisy INFO/WARNING diagnostics while retaining genuine ERROR/FATAL output.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Landmark extraction requires opencv-python and mediapipe; install requirements.txt first.") from exc
    if not hasattr(mp, "tasks") or not hasattr(mp.tasks, "vision"):
        raise RuntimeError("This pipeline requires MediaPipe Tasks Vision API (mp.tasks.vision).")
    return cv2, mp


@contextlib.contextmanager
def _quiet_native_stderr(enabled: bool = True):
    """Hide noisy MediaPipe/TFLite native diagnostics without swallowing Python errors.

    Some Windows builds bypass GLOG_minloglevel and write directly to file
    descriptor 2. Python exceptions still propagate and are recorded per video.
    """
    if not enabled:
        yield
        return
    saved = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as null:
            os.dup2(null.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def _points(landmarks: Any, indices: tuple[int, ...]) -> np.ndarray:
    output = np.full((len(indices), 3), np.nan, dtype=np.float32)
    if landmarks is None:
        return output
    points = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    for target, source in enumerate(indices):
        point = points[source]
        output[target] = (point.x, point.y, point.z)
    return output


def _all_points(landmarks: Any, count: int) -> np.ndarray:
    return _points(landmarks, tuple(range(count)))


def _task_landmarkers(mp: Any, model_dir: Path) -> tuple[Any, Any, Any]:
    """Construct Tasks API landmarkers in VIDEO mode from user-provided assets."""
    required = {"pose": model_dir / "pose_landmarker.task", "hand": model_dir / "hand_landmarker.task", "face": model_dir / "face_landmarker.task"}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing MediaPipe Tasks model asset(s): " + ", ".join(missing) + ". Download them from MediaPipe's official model pages; the pipeline deliberately does not download models automatically.")
    vision = mp.tasks.vision
    options = lambda path: mp.tasks.BaseOptions(model_asset_path=str(path))
    mode = vision.RunningMode.VIDEO
    return (
        vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=options(required["pose"]), running_mode=mode, num_poses=1)),
        vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=options(required["hand"]), running_mode=mode, num_hands=2)),
        vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=options(required["face"]), running_mode=mode, num_faces=1)),
    )


def _hand_sides(result: Any, *, input_is_mirrored: bool) -> dict[str, Any]:
    """Map Tasks handedness to the signer side without changing XYZ coordinates.

    Tasks labels assume a mirrored selfie image. WLASL videos are normally
    externally recorded, so labels are reversed by default; callers can opt
    out for mirrored input.
    """
    sides: dict[str, Any] = {}
    for hand, handedness in zip(result.hand_landmarks, result.handedness):
        if not handedness:
            continue
        label = handedness[0].category_name.lower()
        if not input_is_mirrored:
            label = {"left": "right", "right": "left"}.get(label, label)
        if label in {"left", "right"} and label not in sides:
            sides[label] = hand
    return sides


def extract_video(video_id: str, *, overwrite: bool = False, model_dir: Path = MEDIAPIPE_MODEL_DIR, input_is_mirrored: bool = False, quiet_native_logs: bool = True) -> dict[str, object]:
    """Extract one video. Existing schema-valid output is returned as skipped."""
    target = INTERIM_LANDMARK_DIR / f"{video_id}.npz"
    if target.exists() and is_valid_npz(target) and not overwrite:
        return report_row(load_npz(target), success=True, error="skipped_existing")
    cv2, mp = _dependencies()
    capture = cv2.VideoCapture(str(video_path(video_id)))
    if not capture.isOpened():
        return {"video_id": video_id, "extraction_success": False, "error_message": "cannot_open"}
    fps = float(capture.get(cv2.CAP_PROP_FPS)); width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    arrays = {"pose_xyz": [], "left_hand_xyz": [], "right_hand_xyz": [], "mouth_xyz": [], "left_eye_xyz": [], "right_eye_xyz": [], "left_eyebrow_xyz": [], "right_eyebrow_xyz": []}
    masks = {"pose_observed": [], "left_hand_observed": [], "right_hand_observed": [], "face_observed": []}
    landmarkers: tuple[Any, Any, Any] | None = None
    try:
        with _quiet_native_stderr(quiet_native_logs):
            landmarkers = _task_landmarkers(mp, model_dir)
            pose_task, hand_task, face_task = landmarkers
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = round(frame_index * 1000.0 / fps)
                pose_result = pose_task.detect_for_video(image, timestamp_ms)
                hand_result = hand_task.detect_for_video(image, timestamp_ms)
                face_result = face_task.detect_for_video(image, timestamp_ms)
                pose = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
                sides = _hand_sides(hand_result, input_is_mirrored=input_is_mirrored); left, right = sides.get("left"), sides.get("right")
                face = face_result.face_landmarks[0] if face_result.face_landmarks else None
                arrays["pose_xyz"].append(_points(pose, POSE_SOURCE_INDICES)); arrays["left_hand_xyz"].append(_all_points(left, HAND_LANDMARK_COUNT)); arrays["right_hand_xyz"].append(_all_points(right, HAND_LANDMARK_COUNT))
                arrays["mouth_xyz"].append(_points(face, MOUTH_INDICES)); arrays["left_eye_xyz"].append(_points(face, LEFT_EYE_INDICES)); arrays["right_eye_xyz"].append(_points(face, RIGHT_EYE_INDICES)); arrays["left_eyebrow_xyz"].append(_points(face, LEFT_EYEBROW_INDICES)); arrays["right_eyebrow_xyz"].append(_points(face, RIGHT_EYEBROW_INDICES))
                masks["pose_observed"].append(pose is not None); masks["left_hand_observed"].append(left is not None); masks["right_hand_observed"].append(right is not None); masks["face_observed"].append(face is not None)
                frame_index += 1
    except Exception as exc:
        return {"video_id": video_id, "extraction_success": False, "error_message": f"mediapipe_exception:{type(exc).__name__}:{exc}"}
    finally:
        capture.release()
        if landmarkers is not None:
            for landmarker in landmarkers:
                landmarker.close()
    frames = len(masks["pose_observed"])
    if frames == 0 or not fps > 0 or width <= 0 or height <= 0:
        return {"video_id": video_id, "extraction_success": False, "error_message": "zero_frames_or_invalid_metadata"}
    payload: dict[str, object] = {key: np.asarray(values, dtype=np.float32) for key, values in arrays.items()}
    payload.update({key: np.asarray(values, dtype=bool) for key, values in masks.items()})
    payload.update({"video_id": np.asarray(video_id), "fps": np.asarray(fps, dtype=np.float32), "num_frames": np.asarray(frames, dtype=np.int32), "width": np.asarray(width, dtype=np.int32), "height": np.asarray(height, dtype=np.int32), "schema_version": np.asarray(LANDMARK_SCHEMA_VERSION), "coordinate_order": np.asarray(COORDINATE_ORDER)})
    save_npz(target, payload)
    return report_row(payload, success=True)


def _gap_count(observed: np.ndarray, max_gap: int = 2) -> int:
    from src.preprocessing.hand_recovery import internal_missing_gaps
    return sum(gap.length <= max_gap for gap in internal_missing_gaps(observed))


def report_row(payload: dict[str, object], *, success: bool, error: str = "") -> dict[str, object]:
    total = int(np.asarray(payload.get("num_frames", 0)).item())
    output: dict[str, object] = {"video_id": str(np.asarray(payload.get("video_id", "")).item()), "num_frames": total, "fps": float(np.asarray(payload.get("fps", 0)).item()) if total else ""}
    for name in ("pose", "left_hand", "right_hand", "face"):
        observed = np.asarray(payload.get(f"{name}_observed", np.zeros(total, dtype=bool)), dtype=bool)
        output[f"{name}_detected_frames"] = int(observed.sum()); output[f"{name}_detection_rate"] = round(float(observed.mean()), 6) if total else 0.0
    left, right = np.asarray(payload.get("left_hand_observed", np.zeros(total, dtype=bool))), np.asarray(payload.get("right_hand_observed", np.zeros(total, dtype=bool)))
    output.update({"left_hand_missing_frames": int((~left).sum()), "right_hand_missing_frames": int((~right).sum()), "left_hand_short_gap_count": _gap_count(left), "right_hand_short_gap_count": _gap_count(right), "left_hand_imputed_frames": 0, "right_hand_imputed_frames": 0, "left_hand_unresolved_frames": int((~left).sum()), "right_hand_unresolved_frames": int((~right).sum()), "normalization_invalid_frames": "", "extraction_success": success, "error_message": error})
    return output


def ids_from_split(subset: str) -> list[str]:
    ids: set[str] = set()
    from src.config.paths import split_dir
    for split in ("train", "val", "test"):
        path = split_dir(subset) / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} is absent; run audit then build_splits first.")
        with path.open(newline="", encoding="utf-8") as handle:
            ids.update(row["video_id"] for row in csv.DictReader(handle))
    return sorted(ids)


def write_report(rows: Iterable[dict[str, object]]) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    path = METADATA_DIR / "landmark_extraction_report.csv"
    existing: dict[str, dict[str, object]] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = {row["video_id"]: row for row in csv.DictReader(handle)}
    for row in rows:
        existing[str(row["video_id"])] = row
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS); writer.writeheader(); writer.writerows(existing[key] for key in sorted(existing))


def _progress(done: int, total: int, successful: int, failed: int, skipped: int, started: float, *, stage: str = "Extracting") -> None:
    """Render one non-dependency terminal progress line with speed and ETA."""
    elapsed = time.monotonic() - started
    rate = done / elapsed if elapsed >= 0.01 else 0.0
    percent = 100 * done / total if total else 100.0
    speed_eta = "estimating..." if not rate else f"{rate:.2f} video/s | ETA {(total - done) / rate / 60:.1f} min"
    line = (f"\r{stage}: {done:,}/{total:,} ({percent:6.2f}%) | "
            f"ok={successful:,} failed={failed:,} skipped={skipped:,} | {speed_eta}")
    print(line, end="", flush=True)


def run_extraction(video_ids: Iterable[str], *, overwrite: bool = False, model_dir: Path = MEDIAPIPE_MODEL_DIR, input_is_mirrored: bool = False, report_every: int = 25, show_progress: bool = True, quiet_native_logs: bool = True) -> dict[str, int]:
    """Extract a concrete ID list with progress and periodic resumable reports."""
    if report_every < 1:
        raise ValueError("report_every must be at least 1")
    ids = list(video_ids)
    total, rows, successful, failed, skipped = len(ids), [], 0, 0, 0
    started = time.monotonic()
    print(f"Starting landmark extraction for {total:,} video(s). Existing schema-valid outputs will be skipped.")
    try:
        for done, video_id in enumerate(ids, start=1):
            row = extract_video(video_id, overwrite=overwrite, model_dir=model_dir, input_is_mirrored=input_is_mirrored, quiet_native_logs=quiet_native_logs)
            rows.append(row)
            if row["extraction_success"]:
                successful += 1
                skipped += row.get("error_message") == "skipped_existing"
            else:
                failed += 1
            if done % report_every == 0:
                write_report(rows)
            if show_progress:
                _progress(done, total, successful, failed, skipped, started)
    finally:
        write_report(rows)
    if show_progress:
        print()
    print(f"Finished {total:,} video(s): ok={successful:,}, failed={failed:,}, skipped={skipped:,}. Report: {METADATA_DIR / 'landmark_extraction_report.csv'}")
    return {"total": total, "successful": successful, "failed": failed, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subset", choices=SUBSETS); group.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--limit", type=int); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--model-dir", type=Path, default=MEDIAPIPE_MODEL_DIR, help="Folder containing pose_landmarker.task, hand_landmarker.task and face_landmarker.task."); parser.add_argument("--input-is-mirrored", action="store_true", help="Use only for selfie-mirrored video input; WLASL defaults to externally recorded input."); parser.add_argument("--report-every", type=int, default=25, help="Persist the merged CSV report every N videos (default: 25)."); parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress output."); parser.add_argument("--show-mediapipe-logs", action="store_true", help="Show native MediaPipe/TFLite diagnostics for troubleshooting.")
    args = parser.parse_args()
    ids = ids_from_split(args.subset) if args.subset else args.video_ids
    if args.limit is not None: ids = ids[:args.limit]
    try:
        run_extraction(ids, overwrite=args.overwrite, model_dir=args.model_dir, input_is_mirrored=args.input_is_mirrored, report_every=args.report_every, show_progress=not args.no_progress, quiet_native_logs=not args.show_mediapipe_logs)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
