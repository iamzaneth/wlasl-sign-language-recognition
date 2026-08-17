from __future__ import annotations

from src.config.paths import (
    PROJECT_ROOT,
    normalize_project_paths,
    project_relative_path,
    resolve_project_path,
)


def test_project_paths_are_portable_and_old_clone_paths_are_remapped() -> None:
    expected = PROJECT_ROOT / "output" / "encoder_only" / "run"
    old_clone = PROJECT_ROOT.anchor + "old\\wlasl-sign-language-recognition\\output\\encoder_only\\run"

    assert project_relative_path(expected) == "output/encoder_only/run"
    assert resolve_project_path("output/encoder_only/run") == expected
    assert resolve_project_path(old_clone) == expected
    assert normalize_project_paths({"run_dir": str(expected)}) == {
        "run_dir": "output/encoder_only/run"
    }
