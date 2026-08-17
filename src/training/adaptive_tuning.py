"""Run resumable adaptive tuning rounds for every WLASL100 landmark subset.

Each round evaluates five profiles over five matched seeds (25 configurations
per landmark subset, 175 per round).  Validation macro-F1 means, never test
metrics, determine the search center for the next round. After the initial
five-stage search, later rounds repeat the adaptive capacity, learning-rate,
regularization, and portfolio cycle around the latest evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config.paths import PROJECT_ROOT, project_relative_path, resolve_project_path
from src.training.encoder_data import VALID_LANDMARK_SUBSETS
from src.training.experiment_results import TUNABLE_KEYS, read_json, rebuild_best, write_json


PROFILE_COUNT = 5
SEEDS_PER_PROFILE = 5
SEED_OFFSETS = (17, 149, 613, 1229, 2521)
DEFAULT_CAMPAIGN_NAME = "adaptive_large_model_tuning"
LEGACY_TRIAL_NAMESPACE = "large5"
CONFIG_ID_KEYS = (
    "d_model", "num_heads", "num_layers", "dim_feedforward", "dropout",
    "batch_size", "learning_rate", "weight_decay", "label_smoothing",
    "scheduler", "warmup_epochs", "use_class_weight", "balanced_sampler",
    "augment_train", "augment_noise_std", "standardize",
)
RESULT_CONFIG_KEYS = tuple(key for key in TUNABLE_KEYS if key != "seed")
RESULT_FIELDS = (
    "campaign_id", "round", "completed_at", "landmark_subset", "profile",
    "profile_rank", "replicates", "val_macro_f1_mean", "val_macro_f1_stdev",
    "val_macro_f1_median", "val_macro_f1_max",
    "test_macro_f1_mean_reporting_only", "best_epoch_median",
    *RESULT_CONFIG_KEYS, "config_json",
)
PROFILE_TIE_ORDER = {
    1: ("wide512", "wide640", "wide768", "wide1024", "deep768"),
    2: ("capacity_ref", "ff_down", "ff_up", "depth_up", "depth_down", "width_up", "depth_up2", "dropout_fill", "lr_fill", "wd_fill"),
    3: ("lr_0.65", "lr_0.80", "lr_1.00", "lr_1.20", "lr_1.40"),
    4: ("regularization_ref", "dropout_low", "dropout_high", "wd_low", "wd_high"),
}
STATUS_FIELDS = (
    "campaign_id", "round", "status", "trial_count",
    "completed_trial_count", "started_at", "paused_at", "completed_at",
    "stdout_log", "stderr_log", "resume_command",
)


def tunable(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in TUNABLE_KEYS if key in config and key != "seed"}


def config_identity(config: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(config.get(key) for key in CONFIG_ID_KEYS)


def campaign_identity(requested: str) -> tuple[str, str]:
    """Return the storage name and stable trial namespace for a campaign."""
    if requested == LEGACY_TRIAL_NAMESPACE:
        return DEFAULT_CAMPAIGN_NAME, LEGACY_TRIAL_NAMESPACE
    return requested, requested


def current_best(dataset_root: Path, subset: str) -> dict[str, Any]:
    best_dir = dataset_root / "best_models" / subset
    selection = read_json(best_dir / "selection.json")
    source = resolve_project_path(selection["source_run_dir"])
    summary = read_json(source / "summary.json")
    return {
        "source_run_dir": project_relative_path(source),
        "source_name": read_json(source / "config.json")["name"],
        "best_val_macro_f1": float(summary["best_val_macro_f1"]),
        "best_epoch": int(summary["best_epoch"]),
        "config": tunable(read_json(source / "config.json")),
    }


def large_architecture_profiles(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Expand width substantially while retaining a safe 6 GB GPU envelope."""
    depth = int(base["num_layers"])
    deep = min(6, max(3, depth + 2))
    return [
        ("wide512", {**base, "d_model": 512, "num_heads": 8, "num_layers": depth, "dim_feedforward": 2048, "batch_size": 16}),
        ("wide640", {**base, "d_model": 640, "num_heads": 8, "num_layers": depth, "dim_feedforward": 2560, "batch_size": 12}),
        ("wide768", {**base, "d_model": 768, "num_heads": 12, "num_layers": depth, "dim_feedforward": 3072, "batch_size": 8}),
        ("wide1024", {**base, "d_model": 1024, "num_heads": 16, "num_layers": depth, "dim_feedforward": 4096, "batch_size": 6}),
        ("deep768", {**base, "d_model": 768, "num_heads": 12, "num_layers": deep, "dim_feedforward": 3072, "batch_size": 6}),
    ]


def capacity_profiles(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    d_model = int(base["d_model"])
    depth = int(base["num_layers"])
    feedforward = int(base["dim_feedforward"])
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("capacity_ref", dict(base)),
        ("ff_down", {**base, "dim_feedforward": max(d_model * 2, feedforward - 512)}),
        ("ff_up", {**base, "dim_feedforward": min(4096, max(d_model * 2, feedforward + 512))}),
        ("depth_up", {**base, "num_layers": min(6, depth + 1), "batch_size": min(int(base["batch_size"]), 8)}),
    ]
    if depth > 1:
        candidates.append(("depth_down", {**base, "num_layers": depth - 1, "batch_size": min(16, int(base["batch_size"]) + 2)}))
    elif d_model < 1024:
        wider = min(1024, d_model + 128)
        heads = 16 if wider == 1024 else 8
        candidates.append(("width_up", {**base, "d_model": wider, "num_heads": heads, "dim_feedforward": min(4096, max(feedforward, wider * 3)), "batch_size": min(int(base["batch_size"]), 8)}))
    else:
        candidates.append(("depth_up2", {**base, "num_layers": min(6, depth + 2), "batch_size": min(int(base["batch_size"]), 6)}))
    return ensure_five_unique(candidates, base)


def learning_rate_profiles(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    lr = float(base["learning_rate"])
    return [(f"lr_{multiplier:.2f}", {**base, "learning_rate": lr * multiplier}) for multiplier in (0.65, 0.80, 1.00, 1.20, 1.40)]


def regularization_profiles(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    dropout = float(base["dropout"])
    weight_decay = float(base["weight_decay"])
    return [
        ("regularization_ref", dict(base)),
        ("dropout_low", {**base, "dropout": max(0.10, dropout - 0.08)}),
        ("dropout_high", {**base, "dropout": min(0.50, dropout + 0.08)}),
        ("wd_low", {**base, "weight_decay": max(1e-4, weight_decay * 0.40)}),
        ("wd_high", {**base, "weight_decay": min(1e-2, weight_decay * 2.50)}),
    ]


def ensure_five_unique(candidates: list[tuple[str, dict[str, Any]]], base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    unique: list[tuple[str, dict[str, Any]]] = []
    identities: set[tuple[Any, ...]] = set()
    for name, config in candidates:
        identity = config_identity(config)
        if identity not in identities:
            unique.append((name, config))
            identities.add(identity)
    fillers = [
        ("dropout_fill", {**base, "dropout": min(0.50, float(base["dropout"]) + 0.05)}),
        ("lr_fill", {**base, "learning_rate": float(base["learning_rate"]) * 0.90}),
        ("wd_fill", {**base, "weight_decay": min(1e-2, float(base["weight_decay"]) * 1.75)}),
    ]
    for name, config in fillers:
        identity = config_identity(config)
        if len(unique) < PROFILE_COUNT and identity not in identities:
            unique.append((name, config))
            identities.add(identity)
    if len(unique) != PROFILE_COUNT:
        raise RuntimeError(f"Could not construct {PROFILE_COUNT} unique profiles")
    return unique


def results_csv_path(campaign_root: Path) -> Path:
    return campaign_root / "results.csv"


def status_csv_path(campaign_root: Path) -> Path:
    return campaign_root / "status.csv"


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_round_results(campaign_root: Path, results: dict[str, Any]) -> None:
    """Insert or replace one round in the campaign-wide results CSV."""
    path = results_csv_path(campaign_root)
    round_number = int(results["round"])
    rows = _read_csv(path)
    rows = [row for row in rows if int(row["round"]) != round_number]
    for subset in results["subsets"]:
        for profile_rank, profile in enumerate(subset["profiles"], start=1):
            config = dict(profile["config"])
            rows.append({
                "campaign_id": results["campaign_id"],
                "round": round_number,
                "completed_at": results["completed_at"],
                "landmark_subset": subset["landmark_subset"],
                "profile": profile["profile"],
                "profile_rank": profile_rank,
                "replicates": profile["replicates"],
                "val_macro_f1_mean": profile["val_macro_f1_mean"],
                "val_macro_f1_stdev": profile["val_macro_f1_stdev"],
                "val_macro_f1_median": profile["val_macro_f1_median"],
                "val_macro_f1_max": profile["val_macro_f1_max"],
                "test_macro_f1_mean_reporting_only": profile["test_macro_f1_mean_reporting_only"],
                "best_epoch_median": profile["best_epoch_median"],
                **{key: config.get(key, "") for key in RESULT_CONFIG_KEYS},
                "config_json": json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            })
    rows.sort(key=lambda row: (int(row["round"]), str(row["landmark_subset"]), str(row["profile"])))
    _write_csv(path, RESULT_FIELDS, rows)


def load_round_results(campaign_root: Path, round_number: int) -> dict[str, Any]:
    """Load one round from flat CSV, with fallback for pre-migration JSON."""
    path = results_csv_path(campaign_root)
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["round"]) == round_number]
    if not rows:
        legacy = campaign_root / f"round_{round_number:02d}" / "results.json"
        if legacy.exists():
            return read_json(legacy)
        raise FileNotFoundError(f"No completed results for Round {round_number}")

    subsets: list[dict[str, Any]] = []
    for subset in VALID_LANDMARK_SUBSETS:
        subset_rows = [row for row in rows if row["landmark_subset"] == subset]
        if not subset_rows:
            continue
        tie_order = PROFILE_TIE_ORDER.get(round_number, ())
        subset_rows.sort(key=lambda row: (
            int(row["profile_rank"]) if row.get("profile_rank") else 10_000,
            -float(row["val_macro_f1_mean"]),
            tie_order.index(row["profile"]) if row["profile"] in tie_order else len(tie_order),
        ) if row.get("profile_rank") else (
            10_000,
            -float(row["val_macro_f1_mean"]),
            tie_order.index(row["profile"]) if row["profile"] in tie_order else len(tie_order),
        ))
        profiles = [{
            "profile": row["profile"],
            "config": json.loads(row["config_json"]),
            "replicates": int(row["replicates"]),
            "val_macro_f1_mean": float(row["val_macro_f1_mean"]),
            "val_macro_f1_stdev": float(row["val_macro_f1_stdev"]),
            "val_macro_f1_median": float(row["val_macro_f1_median"]),
            "val_macro_f1_max": float(row["val_macro_f1_max"]),
            "test_macro_f1_mean_reporting_only": float(row["test_macro_f1_mean_reporting_only"]),
            "best_epoch_median": float(row["best_epoch_median"]),
        } for row in subset_rows]
        subsets.append({
            "landmark_subset": subset,
            "selection_metric": "mean_val_macro_f1_over_five_seeds",
            "test_metrics_used_for_selection": False,
            "profiles": profiles,
        })
    return {
        "campaign_id": rows[0]["campaign_id"],
        "round": round_number,
        "status": "complete",
        "completed_at": rows[0]["completed_at"],
        "subsets": subsets,
    }


def round_is_complete(campaign_root: Path, round_number: int) -> bool:
    try:
        results = load_round_results(campaign_root, round_number)
    except FileNotFoundError:
        return False
    return (
        len(results["subsets"]) == len(VALID_LANDMARK_SUBSETS)
        and all(
            len(subset["profiles"]) == PROFILE_COUNT
            and all(profile["replicates"] == SEEDS_PER_PROFILE for profile in subset["profiles"])
            for subset in results["subsets"]
        )
    )


def update_round_status(campaign_root: Path, status: dict[str, Any]) -> None:
    path = status_csv_path(campaign_root)
    round_number = int(status["round"])
    rows = _read_csv(path)
    rows = [row for row in rows if int(row["round"]) != round_number]
    rows.append({key: status.get(key, "") for key in STATUS_FIELDS})
    rows.sort(key=lambda row: int(row["round"]))
    _write_csv(path, STATUS_FIELDS, rows)


def winning_config_from_round(campaign_root: Path, round_number: int, subset: str) -> dict[str, Any]:
    results = load_round_results(campaign_root, round_number)
    subset_result = next(row for row in results["subsets"] if row["landmark_subset"] == subset)
    return dict(subset_result["profiles"][0]["config"])


def portfolio_profiles(campaign_root: Path, round_number: int, subset: str) -> list[tuple[str, dict[str, Any]]]:
    """Return the five strongest distinct profiles from all prior rounds."""
    candidates: list[dict[str, Any]] = []
    for source_round in range(1, round_number):
        results = load_round_results(campaign_root, source_round)
        subset_result = next(row for row in results["subsets"] if row["landmark_subset"] == subset)
        for profile in subset_result["profiles"]:
            candidates.append({**profile, "source_round": source_round})
    candidates.sort(key=lambda row: float(row["val_macro_f1_mean"]), reverse=True)
    selected: list[tuple[str, dict[str, Any]]] = []
    identities: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        identity = config_identity(candidate["config"])
        if identity in identities:
            continue
        selected.append((f"portfolio_r{candidate['source_round']}_{candidate['profile']}", dict(candidate["config"])))
        identities.add(identity)
        if len(selected) == PROFILE_COUNT:
            break
    if len(selected) != PROFILE_COUNT:
        raise RuntimeError(f"Only found {len(selected)} distinct portfolio profiles for {subset}")
    return selected


def profiles_for_round(campaign_root: Path, round_number: int, subset: str, baseline: dict[str, Any]) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    if round_number == 1:
        return "Expand model width from the current global winner and include one substantially deeper 768-wide profile.", large_architecture_profiles(dict(baseline["config"]))
    previous = winning_config_from_round(campaign_root, round_number - 1, subset)
    phase = (round_number - 2) % 4
    if phase == 0:
        return "Center on the previous round's best mean and resolve local feed-forward width and depth.", capacity_profiles(previous)
    if phase == 1:
        return "Center on the previous round's best mean and resolve the learning-rate optimum.", learning_rate_profiles(previous)
    if phase == 2:
        return "Center on the previous round's best mean and resolve dropout and weight decay.", regularization_profiles(previous)
    return (
        "Re-evaluate the five strongest distinct mean-performing profiles from all prior rounds on five fresh matched seeds.",
        portfolio_profiles(campaign_root, round_number, subset),
    )


def create_round_trials(
    campaign_root: Path,
    campaign_id: str,
    round_number: int,
    baseline_by_subset: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    rationales: dict[str, str] = {}
    plan: list[dict[str, Any]] = []
    for subset_index, subset in enumerate(VALID_LANDMARK_SUBSETS):
        rationale, profiles = profiles_for_round(campaign_root, round_number, subset, baseline_by_subset[subset])
        if len(profiles) != PROFILE_COUNT:
            raise RuntimeError(f"Expected {PROFILE_COUNT} profiles for {subset}, found {len(profiles)}")
        rationales[subset] = rationale
        plan.append({
            "landmark_subset": subset,
            "rationale": rationale,
            "profiles": [{"profile": name, "config": config} for name, config in profiles],
        })
        epochs = 145 if round_number == 5 else 130
        patience = 35 if round_number == 5 else 30
        for profile_name, profile_config in profiles:
            for seed_index, seed_offset in enumerate(SEED_OFFSETS, start=1):
                matched_seed = 200_000 + round_number * 100_000 + subset_index * 10_000 + seed_offset
                trials.append({
                    **profile_config,
                    "name": f"{subset}__{campaign_id}_r{round_number}_{profile_name}_seed{seed_index}",
                    "landmark_subset": subset,
                    "seed": matched_seed,
                    "epochs": epochs,
                    "early_stopping_patience": patience,
                    "tuning_basis": {
                        "campaign": campaign_id,
                        "round": round_number,
                        "profile": profile_name,
                        "replicate": seed_index,
                        "matched_seed_within_subset": matched_seed,
                        "selection_metric": "val_macro_f1",
                        "test_metrics_used_for_selection": False,
                        "rationale": rationale,
                    },
                })
    expected = PROFILE_COUNT * SEEDS_PER_PROFILE * len(VALID_LANDMARK_SUBSETS)
    assert len(trials) == expected
    assert len({trial["name"] for trial in trials}) == expected
    return trials, rationales, plan


def aggregate_round(dataset_root: Path, campaign_id: str, round_number: int, plan: list[dict[str, Any]]) -> dict[str, Any]:
    subsets: list[dict[str, Any]] = []
    for subset_plan in plan:
        subset = subset_plan["landmark_subset"]
        planned_configs = {row["profile"]: row["config"] for row in subset_plan["profiles"]}
        grouped: dict[str, list[dict[str, float]]] = {profile: [] for profile in planned_configs}
        for summary_path in (dataset_root / subset / "runs").glob("*/summary.json"):
            config_path = summary_path.parent / "config.json"
            if not config_path.exists():
                continue
            config = read_json(config_path)
            basis = config.get("tuning_basis", {})
            if basis.get("campaign") != campaign_id or int(basis.get("round", -1)) != round_number:
                continue
            summary = read_json(summary_path)
            grouped[str(basis["profile"])].append({
                "val": float(summary["best_val_macro_f1"]),
                "test": float(summary["test_macro_f1"]),
                "best_epoch": float(summary["best_epoch"]),
            })
        profiles: list[dict[str, Any]] = []
        for profile, rows in grouped.items():
            if len(rows) != SEEDS_PER_PROFILE:
                raise RuntimeError(f"{subset}/{profile} has {len(rows)} completed replicates; expected {SEEDS_PER_PROFILE}")
            val_scores = [row["val"] for row in rows]
            profiles.append({
                "profile": profile,
                "config": planned_configs[profile],
                "replicates": len(rows),
                "val_macro_f1_mean": statistics.mean(val_scores),
                "val_macro_f1_stdev": statistics.stdev(val_scores),
                "val_macro_f1_median": statistics.median(val_scores),
                "val_macro_f1_max": max(val_scores),
                "test_macro_f1_mean_reporting_only": statistics.mean(row["test"] for row in rows),
                "best_epoch_median": statistics.median(row["best_epoch"] for row in rows),
            })
        profiles.sort(key=lambda row: float(row["val_macro_f1_mean"]), reverse=True)
        subsets.append({
            "landmark_subset": subset,
            "selection_metric": "mean_val_macro_f1_over_five_seeds",
            "test_metrics_used_for_selection": False,
            "profiles": profiles,
        })
    return {
        "campaign_id": campaign_id,
        "round": round_number,
        "status": "complete",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "subsets": subsets,
    }


def run_round(
    args: argparse.Namespace,
    campaign_root: Path,
    dataset_root: Path,
    baseline_by_subset: dict[str, dict[str, Any]],
    round_number: int,
) -> dict[str, Any]:
    if round_is_complete(campaign_root, round_number):
        print(f"Round {round_number}/{args.end_round} already complete; using saved results.", flush=True)
        return load_round_results(campaign_root, round_number)

    trials, _rationales, plan = create_round_trials(
        campaign_root, args.trial_namespace, round_number, baseline_by_subset
    )
    config_path = campaign_root / "active_trials.json"
    if config_path.exists():
        saved_trials = read_json(config_path).get("trials", [])
        if saved_trials != trials:
            raise RuntimeError(
                f"Saved active trial plan does not match regenerated Round {round_number}; "
                "refusing to overwrite resumable state."
            )
    else:
        write_json(config_path, {"defaults": {}, "trials": trials})
    started_at = datetime.now().isoformat(timespec="seconds")
    update_round_status(campaign_root, {
        "campaign_id": args.campaign_id,
        "round": round_number,
        "status": "running",
        "trial_count": len(trials),
        "started_at": started_at,
    })

    log_root = dataset_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{args.campaign_id}_round{round_number}.stdout.log"
    stderr_path = log_root / f"{args.campaign_id}_round{round_number}.stderr.log"
    command = [
        sys.executable, "-u", "scripts/train_encoder_only.py",
        "--subset", "wlasl100", "--config", str(config_path),
        "--device", args.device, "--output-dir", str(args.output_root),
        "--skip-completed",
    ]
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr, check=True)

    results = aggregate_round(dataset_root, args.trial_namespace, round_number, plan)
    results["campaign_id"] = args.campaign_id
    results["stdout_log"] = str(stdout_path)
    results["stderr_log"] = str(stderr_path)
    write_round_results(campaign_root, results)
    rebuild_best(dataset_root)
    update_round_status(campaign_root, {
        "campaign_id": args.campaign_id,
        "round": round_number,
        "status": "complete",
        "trial_count": len(trials),
        "completed_trial_count": len(trials),
        "started_at": started_at,
        "completed_at": results["completed_at"],
        "stdout_log": project_relative_path(stdout_path),
        "stderr_log": project_relative_path(stderr_path),
    })
    config_path.unlink(missing_ok=True)
    print(f"Round {round_number}/{args.end_round} complete; analyzed profile means and updated the next-round search center.", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output" / "encoder_only")
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--end-round", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.start_round <= args.end_round:
        raise ValueError("Require 1 <= start-round <= end-round")

    args.campaign_id, args.trial_namespace = campaign_identity(args.campaign_id)
    dataset_root = args.output_root / "wlasl100"
    campaign_root = dataset_root / "experiments" / args.campaign_id
    campaign_root.mkdir(parents=True, exist_ok=True)
    baseline_path = campaign_root / "baseline_best.json"
    if baseline_path.exists():
        baseline_rows = read_json(baseline_path)
    else:
        baseline_rows = [current_best(dataset_root, subset) | {"landmark_subset": subset} for subset in VALID_LANDMARK_SUBSETS]
        write_json(baseline_path, baseline_rows)
    baseline_by_subset = {row["landmark_subset"]: row for row in baseline_rows}
    write_json(campaign_root / "manifest.json", {
        "campaign_id": args.campaign_id,
        "trial_namespace": args.trial_namespace,
        "dataset": "wlasl100",
        "planned_through_round": args.end_round,
        "configurations_per_subset_per_round": PROFILE_COUNT * SEEDS_PER_PROFILE,
        "subsets_per_round": len(VALID_LANDMARK_SUBSETS),
        "trials_per_round": PROFILE_COUNT * SEEDS_PER_PROFILE * len(VALID_LANDMARK_SUBSETS),
        "planned_trials_through_round": args.end_round * PROFILE_COUNT * SEEDS_PER_PROFILE * len(VALID_LANDMARK_SUBSETS),
        "selection_metric": "mean_val_macro_f1_over_five_seeds",
        "test_metrics_used_for_selection": False,
    })

    if args.dry_run:
        if args.start_round != 1 and not round_is_complete(campaign_root, args.start_round - 1):
            raise RuntimeError("The previous round must be complete before dry-running this round")
        trials, _, plan = create_round_trials(campaign_root, args.trial_namespace, args.start_round, baseline_by_subset)
        print(f"Validated Round {args.start_round}: {len(trials)} trials, {len(plan)} subsets, {PROFILE_COUNT} profiles x {SEEDS_PER_PROFILE} seeds each.")
        return

    for round_number in range(args.start_round, args.end_round + 1):
        run_round(args, campaign_root, dataset_root, baseline_by_subset, round_number)
    final_winners = rebuild_best(dataset_root)
    write_json(campaign_root / "summary.json", {
        "campaign_id": args.campaign_id,
        "status": "complete",
        "completed_through_round": args.end_round,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "round_results": project_relative_path(results_csv_path(campaign_root)),
        "winners": final_winners,
    })
    print(f"Campaign {args.campaign_id} completed through Round {args.end_round}: {campaign_root}")


if __name__ == "__main__":
    main()
