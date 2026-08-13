from src.config.paths import INTERIM_LANDMARK_DIR, METADATA_DIR, PROCESSED_LANDMARK_DIR, RAW_DIR, RAW_VIDEO_DIR, SPLITS_DIR
from src.data.wlasl import canonical_split, official_subset


def test_path_contract_is_central_and_semantic() -> None:
    assert RAW_VIDEO_DIR == RAW_DIR / "videos"
    assert INTERIM_LANDMARK_DIR.as_posix().endswith("interim/landmarks/mediapipe_v1")
    assert PROCESSED_LANDMARK_DIR.as_posix().endswith("processed/landmarks/shoulder_norm_v1")
    assert METADATA_DIR.name == "metadata" and SPLITS_DIR.name == "splits"


def test_official_split_assignment_is_not_reinterpreted() -> None:
    official = official_subset("wlasl100")
    assert canonical_split(official["05237"]["subset"]) == "train"
    assert canonical_split(official["69422"]["subset"]) == "val"
