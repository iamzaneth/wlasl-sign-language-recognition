import csv

import numpy as np

from src.preprocessing.landmark_io import save_npz
from src.training.encoder_data import build_feature_cache, feature_columns, feature_layout, landmark_subset_components, landmark_subset_modalities, load_cache, parse_feature_set


def test_feature_sets_are_canonical_and_have_expected_widths() -> None:
    assert parse_feature_set("hands,pose") == ("pose", "hand")
    assert parse_feature_set("month") == ("mouth",)
    assert landmark_subset_modalities("pose_hands_eyes") == ("pose", "hand", "eye")
    assert landmark_subset_modalities("pose_hands_face") == ("pose", "hand", "mouth", "eye", "eyebrow")
    assert landmark_subset_components("pose_hands_eyes") == ("pose", "left_hand", "right_hand", "left_eye", "right_eye")
    assert parse_feature_set("pose+hand+mouth") == ("pose", "hand", "mouth")
    assert len(feature_columns(("pose",))) == 9 * 3
    assert len(feature_columns(("hand",))) == 42 * 3
    assert len(feature_columns(("mouth", "eye", "eyebrow"))) == (40 + 32 + 20) * 3


def test_feature_layout_is_contiguous() -> None:
    spans = list(feature_layout().values())
    assert spans[0][0] == 0
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
    all_columns = feature_columns(("pose", "hand", "mouth", "eye", "eyebrow"))
    np.testing.assert_array_equal(all_columns, np.arange(429))


def test_build_feature_cache_preserves_splits_and_pads_sequences(tmp_path) -> None:
    landmarks = tmp_path / "landmarks"
    splits = tmp_path / "splits" / "wlasl100"
    splits.mkdir(parents=True)
    for split, video_id in (("train", "00001"), ("val", "00002"), ("test", "00003")):
        with (splits / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("video_id", "gloss", "label", "split"))
            writer.writeheader(); writer.writerow({"video_id": video_id, "gloss": "test", "label": 0, "split": split})
        payload = _payload(frames=2)
        payload["video_id"] = np.asarray(video_id)
        save_npz(landmarks / f"{video_id}.npz", payload, processed=True)
    cache_dir = build_feature_cache("wlasl100", seq_len=4, cache_dir=tmp_path / "cache", landmark_dir=landmarks, splits_dir=tmp_path / "splits")
    cache = load_cache(cache_dir)
    assert cache["features"].shape == (3, 4, 429)
    assert cache["frame_mask"].tolist() == [[True, True, False, False]] * 3
    assert [row["split"] for row in cache["samples"]] == ["train", "val", "test"]


def _payload(frames: int) -> dict[str, np.ndarray]:
    points = {"pose_xyz": 9, "left_hand_xyz": 21, "right_hand_xyz": 21, "mouth_xyz": 40, "left_eye_xyz": 16, "right_eye_xyz": 16, "left_eyebrow_xyz": 10, "right_eyebrow_xyz": 10}
    payload = {key: np.ones((frames, point_count, 3), dtype=np.float32) for key, point_count in points.items()}
    payload.update({
        "pose_observed": np.ones(frames, dtype=bool), "left_hand_observed": np.ones(frames, dtype=bool), "right_hand_observed": np.ones(frames, dtype=bool), "face_observed": np.ones(frames, dtype=bool),
        "video_id": np.asarray("sample"), "fps": np.asarray(25, dtype=np.float32), "num_frames": np.asarray(frames, dtype=np.int32), "width": np.asarray(100, dtype=np.int32), "height": np.asarray(80, dtype=np.int32),
        "schema_version": np.asarray("mediapipe_v1"), "coordinate_order": np.asarray(("x", "y", "z")),
        "normalization_valid": np.ones(frames, dtype=bool), "normalization_center_xy": np.ones((frames, 2), dtype=np.float32), "left_hand_imputed": np.zeros(frames, dtype=bool), "right_hand_imputed": np.zeros(frames, dtype=bool), "left_hand_unresolved": np.zeros(frames, dtype=bool), "right_hand_unresolved": np.zeros(frames, dtype=bool),
        "xy_normalized": np.asarray(True), "z_normalized": np.asarray(False), "normalization": np.asarray("shoulder_norm_v1"), "hand_recovery": np.asarray("wrist_anchored_linear"),
    })
    return payload
