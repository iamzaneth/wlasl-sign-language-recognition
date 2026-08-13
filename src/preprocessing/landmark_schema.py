"""Versioned landmark selections and NPZ schema checks.

Face groups are explicit, duplicate-free ordered indices derived from MediaPipe
Face Mesh connection groups (the same topology used by Holistic in MediaPipe
0.10.x). They intentionally exclude iris landmarks.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

LANDMARK_SCHEMA_VERSION = "mediapipe_v1"
COORDINATE_ORDER = ("x", "y", "z")
POSE_SOURCE_INDICES = (0, 11, 12, 13, 14, 15, 16, 23, 24)
POSE_NAMES = ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip")
LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX = 1, 2
LEFT_WRIST_INDEX, RIGHT_WRIST_INDEX = 5, 6
HAND_LANDMARK_COUNT = 21

MOUTH_INDICES = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415)
LEFT_EYE_INDICES = (263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398)
RIGHT_EYE_INDICES = (33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173)
LEFT_EYEBROW_INDICES = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
RIGHT_EYEBROW_INDICES = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)

XYZ_KEYS = ("pose_xyz", "left_hand_xyz", "right_hand_xyz", "mouth_xyz", "left_eye_xyz", "right_eye_xyz", "left_eyebrow_xyz", "right_eyebrow_xyz")
RAW_REQUIRED_KEYS = XYZ_KEYS + ("pose_observed", "left_hand_observed", "right_hand_observed", "face_observed", "video_id", "fps", "num_frames", "width", "height", "schema_version", "coordinate_order")
PROCESSED_REQUIRED_KEYS = RAW_REQUIRED_KEYS + ("normalization_valid", "left_hand_imputed", "right_hand_imputed", "left_hand_unresolved", "right_hand_unresolved", "xy_normalized", "z_normalized", "normalization", "hand_recovery")


def empty_xyz(frames: int, points: int) -> np.ndarray:
    return np.full((frames, points, 3), np.nan, dtype=np.float32)


def validate_landmark_payload(payload: Mapping[str, np.ndarray], *, processed: bool = False) -> None:
    required = PROCESSED_REQUIRED_KEYS if processed else RAW_REQUIRED_KEYS
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing NPZ fields: {', '.join(missing)}")
    frames: int | None = None
    for key in XYZ_KEYS:
        value = payload[key]
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"{key} must have shape [T, points, 3], received {value.shape}")
        if value.dtype.kind != "f":
            raise ValueError(f"{key} must use a floating dtype")
        frames = value.shape[0] if frames is None else frames
        if value.shape[0] != frames:
            raise ValueError("All modalities must have the same T")
    for key in ("pose_observed", "left_hand_observed", "right_hand_observed", "face_observed"):
        if payload[key].shape != (frames,):
            raise ValueError(f"{key} must have shape [T]")
    if str(payload["schema_version"].item()) != LANDMARK_SCHEMA_VERSION:
        raise ValueError("Unsupported landmark schema version")
    order = tuple(str(value) for value in payload["coordinate_order"].tolist())
    if order != COORDINATE_ORDER:
        raise ValueError("coordinate_order must be [x, y, z]")
