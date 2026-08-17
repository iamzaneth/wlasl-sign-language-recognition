from __future__ import annotations

from src.training.adaptive_tuning import (
    DEFAULT_CAMPAIGN_NAME,
    LEGACY_TRIAL_NAMESPACE,
    campaign_identity,
    create_round_trials,
    load_round_results,
    round_is_complete,
    write_round_results,
)
from src.training.encoder_data import VALID_LANDMARK_SUBSETS
from src.training.experiment_results import PUBLISHED_ARTIFACTS, rebuild_best


def completed_result(round_number: int) -> dict:
    return {
        "campaign_id": "test",
        "round": round_number,
        "status": "complete",
        "completed_at": "2026-01-01T00:00:00",
        "subsets": [{
            "landmark_subset": subset,
            "selection_metric": "mean_val_macro_f1_over_five_seeds",
            "test_metrics_used_for_selection": False,
            "profiles": [{
                "profile": f"profile_{profile}",
                "config": {"d_model": 256 + profile * 64, "dropout": 0.2},
                "replicates": 5,
                "val_macro_f1_mean": 0.5 + profile / 100,
                "val_macro_f1_stdev": 0.01,
                "val_macro_f1_median": 0.5,
                "val_macro_f1_max": 0.55,
                "test_macro_f1_mean_reporting_only": 0.49,
                "best_epoch_median": 42,
            } for profile in range(5)],
        } for subset in VALID_LANDMARK_SUBSETS],
    }


def test_round_results_are_stored_in_one_flat_csv(tmp_path) -> None:
    campaign = tmp_path / DEFAULT_CAMPAIGN_NAME
    write_round_results(campaign, completed_result(1))

    loaded = load_round_results(campaign, 1)

    assert round_is_complete(campaign, 1)
    assert len(loaded["subsets"]) == len(VALID_LANDMARK_SUBSETS)
    assert len(loaded["subsets"][0]["profiles"]) == 5
    assert (campaign / "results.csv").exists()
    assert not list(campaign.glob("round_*"))


def test_round_one_plan_has_25_unique_trials_per_subset(tmp_path) -> None:
    base_config = {
        "d_model": 384,
        "num_heads": 8,
        "num_layers": 2,
        "dim_feedforward": 1024,
        "dropout": 0.25,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
    }
    baselines = {
        subset: {"config": dict(base_config)} for subset in VALID_LANDMARK_SUBSETS
    }

    trials, rationales, plan = create_round_trials(
        tmp_path / DEFAULT_CAMPAIGN_NAME,
        LEGACY_TRIAL_NAMESPACE,
        1,
        baselines,
    )

    assert len(trials) == 25 * len(VALID_LANDMARK_SUBSETS)
    assert len({trial["name"] for trial in trials}) == len(trials)
    assert len(rationales) == len(plan) == len(VALID_LANDMARK_SUBSETS)


def test_default_campaign_preserves_existing_trial_namespace_for_resume() -> None:
    assert campaign_identity(DEFAULT_CAMPAIGN_NAME) == (
        DEFAULT_CAMPAIGN_NAME,
        DEFAULT_CAMPAIGN_NAME,
    )
    assert campaign_identity(LEGACY_TRIAL_NAMESPACE) == (
        DEFAULT_CAMPAIGN_NAME,
        LEGACY_TRIAL_NAMESPACE,
    )


def test_rebuild_best_publishes_highest_validation_run(tmp_path) -> None:
    dataset_root = tmp_path / "wlasl100"
    for subset in VALID_LANDMARK_SUBSETS:
        for run_name, score in (("low", 0.4), ("high", 0.6)):
            run_dir = dataset_root / subset / "runs" / run_name
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                '{"landmark_subset":"' + subset + '"}', encoding="utf-8"
            )
            (run_dir / "summary.json").write_text(
                '{"best_val_macro_f1":' + str(score) + ',"test_macro_f1":0.5}',
                encoding="utf-8",
            )
            for artifact in PUBLISHED_ARTIFACTS:
                if artifact not in {"config.json", "summary.json"}:
                    (run_dir / artifact).write_text(run_name, encoding="utf-8")

    winners = rebuild_best(dataset_root)

    assert all(winner["best_val_macro_f1"] == 0.6 for winner in winners)
    for subset in VALID_LANDMARK_SUBSETS:
        published = dataset_root / "best_models" / subset
        assert (published / "best.pt").read_text(encoding="utf-8") == "high"
