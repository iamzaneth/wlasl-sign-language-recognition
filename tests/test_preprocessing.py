from __future__ import annotations

import numpy as np

from src.preprocessing.hand_recovery import internal_missing_gaps, recover_hand
from src.preprocessing.landmark_io import load_npz, save_npz
from src.preprocessing.landmark_schema import (COORDINATE_ORDER, LANDMARK_SCHEMA_VERSION, LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX, XYZ_KEYS)
from src.preprocessing.normalize_landmarks import shoulder_normalize


def payload(frames: int = 4) -> dict[str, np.ndarray]:
    points = {"pose_xyz": 9, "left_hand_xyz": 21, "right_hand_xyz": 21, "mouth_xyz": 40, "left_eye_xyz": 16, "right_eye_xyz": 16, "left_eyebrow_xyz": 10, "right_eyebrow_xyz": 10}
    result = {key: np.full((frames, count, 3), 3, dtype=np.float32) for key, count in points.items()}
    result["pose_xyz"][:, LEFT_SHOULDER_INDEX] = (0, 0, 7); result["pose_xyz"][:, RIGHT_SHOULDER_INDEX] = (2, 0, 8)
    result.update({"pose_observed": np.ones(frames, bool), "left_hand_observed": np.ones(frames, bool), "right_hand_observed": np.ones(frames, bool), "face_observed": np.ones(frames, bool), "video_id": np.asarray("sample"), "fps": np.asarray(25, np.float32), "num_frames": np.asarray(frames, np.int32), "width": np.asarray(100, np.int32), "height": np.asarray(80, np.int32), "schema_version": np.asarray(LANDMARK_SCHEMA_VERSION), "coordinate_order": np.asarray(COORDINATE_ORDER)})
    return result


def test_shoulder_normalization_is_translation_and_scale_invariant_and_preserves_z() -> None:
    first = payload(); second = payload()
    second["pose_xyz"][:, :, :2] = second["pose_xyz"][:, :, :2] * 4 + np.asarray((9, -2), np.float32)
    first_normalized, first_valid, _ = shoulder_normalize(first); second_normalized, second_valid, _ = shoulder_normalize(second)
    assert first_valid.all() and second_valid.all()
    np.testing.assert_allclose(first_normalized["pose_xyz"][..., :2], second_normalized["pose_xyz"][..., :2], atol=1e-6)
    midpoint = (first_normalized["pose_xyz"][:, LEFT_SHOULDER_INDEX, :2] + first_normalized["pose_xyz"][:, RIGHT_SHOULDER_INDEX, :2]) / 2
    np.testing.assert_allclose(midpoint, 0, atol=1e-6)
    assert np.allclose(np.linalg.norm(first_normalized["pose_xyz"][:, LEFT_SHOULDER_INDEX, :2] - first_normalized["pose_xyz"][:, RIGHT_SHOULDER_INDEX, :2], axis=1), 1)
    np.testing.assert_allclose(first_normalized["pose_xyz"][..., 2], first["pose_xyz"][..., 2])


def test_gap_detection_and_wrist_anchored_xyz_recovery() -> None:
    observed = np.asarray([True, True, False, False, True])
    assert [(gap.start, gap.length) for gap in internal_missing_gaps(observed)] == [(2, 2)]
    hand = np.full((5, 21, 3), np.nan, np.float32); wrist = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], np.float32)
    hand[0] = (0, 0, 0); hand[1] = (1, 0, 1); hand[4] = (4, 0, 4)
    recovered, imputed, unresolved = recover_hand(hand, observed, wrist, max_gap_frames=2)
    assert imputed.tolist() == [False, False, True, True, False]; assert not unresolved.any()
    np.testing.assert_allclose(recovered[2, :, 0], 2); np.testing.assert_allclose(recovered[3, :, 2], 3)


def test_long_and_boundary_gaps_are_not_fabricated() -> None:
    hand = np.full((5, 21, 3), np.nan, np.float32); wrist = np.zeros((5, 2), np.float32)
    hand[0] = 0; hand[4] = 4
    _, imputed, unresolved = recover_hand(hand, np.asarray([True, False, False, False, True]), wrist, 2)
    assert not imputed.any() and unresolved[1:4].all()
    _, imputed, _ = recover_hand(hand, np.asarray([False, False, True, True, True]), wrist, 2)
    assert not imputed.any()


def test_npz_roundtrip_keeps_xyz_schema(tmp_path) -> None:
    source = payload(3); path = tmp_path / "raw.npz"; save_npz(path, source)
    loaded = load_npz(path)
    assert all(loaded[key].shape[-1] == 3 for key in XYZ_KEYS)
    assert tuple(loaded["coordinate_order"].tolist()) == COORDINATE_ORDER
