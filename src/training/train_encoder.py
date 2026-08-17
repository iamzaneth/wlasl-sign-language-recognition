"""Train one or more encoder-only landmark classifiers from a shared feature cache.

Feature-set examples: ``hand``; ``pose,hand``; ``pose,hand,mouth,eye,eyebrow``.
Trials are deliberately executed sequentially, which makes multi-configuration
runs safe on one GPU and produces an experiment-level summary CSV.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import platform
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError as error:  # pragma: no cover - depends on the local installation
    raise SystemExit("PyTorch is required. Install the appropriate torch build for your CPU/CUDA environment, then rerun.") from error

from src.config.paths import DATA_DIR, PROJECT_ROOT, normalize_project_paths, project_relative_path
from src.training.encoder_data import VALID_LANDMARK_SUBSETS, canonical_modalities, feature_columns, landmark_subset_components, landmark_subset_modalities, load_cache, parse_feature_set
from src.training.encoder_model import KeypointTransformerEncoderOnly


DEFAULTS: dict[str, Any] = {
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.25,
    "batch_size": 32,
    "epochs": 90,
    "learning_rate": 2e-4,
    "weight_decay": 1e-3,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "early_stopping_patience": 24,
    "use_class_weight": True,
    "balanced_sampler": True,
    "augment_train": True,
    "augment_noise_std": 0.006,
    "standardize": True,
    "label_smoothing": 0.0,
    "seed": 42,
    "num_workers": "auto",
    "cpu_threads": "auto",
    "prefetch_factor": 4,
    "pin_memory": True,
    "use_amp": True,
    "compile_model": False,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LandmarkDataset(Dataset):
    def __init__(self, indices: np.ndarray, features: np.ndarray, frame_mask: np.ndarray, labels: np.ndarray, columns: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None, augment: bool, noise_std: float) -> None:
        self.indices, self.features, self.frame_mask, self.labels = indices, features, frame_mask, labels
        self.columns, self.mean, self.std = columns, mean, std
        self.augment, self.noise_std = augment, noise_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        x = self.features[index][:, self.columns].astype(np.float32, copy=True)
        mask = self.frame_mask[index].astype(bool, copy=True)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        if self.augment:
            x += np.random.normal(0.0, self.noise_std, x.shape).astype(np.float32)
        x[~mask] = 0.0  # padded tokens must stay neutral after standardization/augmentation
        return torch.from_numpy(x), torch.tensor(int(self.labels[index]), dtype=torch.long), torch.from_numpy(mask), torch.tensor(index, dtype=torch.long)


def resolve_auto(value: int | str, *, reserve_one: bool) -> int:
    if value != "auto":
        return max(0, int(value))
    available = os.cpu_count() or 1
    if reserve_one:
        # Windows uses spawn: every DataLoader worker imports the CUDA-enabled
        # interpreter, which is costly in RAM. Two mmap workers consistently
        # keep this lightweight cache fed without exhausting host memory.
        return min(2, max(1, available - 1)) if os.name == "nt" else min(8, max(1, available - 1))
    return available


def configure_runtime(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Apply reproducible, high-throughput CPU/CUDA settings and record them."""
    workers = resolve_auto(config["num_workers"], reserve_one=True)
    cpu_threads = resolve_auto(config["cpu_threads"], reserve_one=False)
    torch.set_num_threads(cpu_threads)
    # Inter-op work is small for this model; keeping it bounded avoids CPU
    # oversubscription while DataLoader workers are decoding memory maps.
    try:
        torch.set_num_interop_threads(min(4, cpu_threads))
    except RuntimeError:
        # PyTorch only permits this before the first parallel operation. Later
        # trials share the already configured setting.
        pass
    config["num_workers"] = workers
    config["cpu_threads"] = cpu_threads
    runtime: dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_logical_cores": os.cpu_count(), "torch_version": torch.__version__,
        "device": str(device), "num_workers": workers, "cpu_threads": cpu_threads,
        "pin_memory": bool(config["pin_memory"]), "prefetch_factor": int(config["prefetch_factor"]),
        "amp_enabled": bool(config["use_amp"] and device.type == "cuda"),
    }
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        properties = torch.cuda.get_device_properties(device)
        runtime.update({
            "gpu_name": properties.name, "gpu_memory_gb": round(properties.total_memory / 1024**3, 2),
            "cudnn_benchmark": True, "tf32_matmul": True, "tf32_cudnn": True,
            "float32_matmul_precision": "high",
        })
    return runtime


def split_indices(samples: list[dict[str, str]]) -> dict[str, np.ndarray]:
    result = {name: np.asarray([i for i, row in enumerate(samples) if row["split"] == name], dtype=np.int64) for name in ("train", "val", "test")}
    missing = [name for name, indices in result.items() if not len(indices)]
    if missing:
        raise ValueError(f"Cache has empty required split(s): {', '.join(missing)}")
    return result


def training_stats(features: np.ndarray, masks: np.ndarray, indices: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-only, per-coordinate statistics without loading all data at once."""
    total = np.zeros(len(columns), dtype=np.float64)
    total_sq = np.zeros(len(columns), dtype=np.float64)
    count = 0
    for index in indices:
        valid = masks[int(index)]
        values = np.asarray(features[int(index)][valid][:, columns], dtype=np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += len(values)
    if not count:
        raise ValueError("Training set has no valid frames")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def class_weights(labels: np.ndarray, indices: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(np.asarray(labels[indices], dtype=np.int64), minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = len(indices) / (present.sum() * counts[present])
    sample_weights = 1.0 / counts[np.asarray(labels[indices], dtype=np.int64)]
    return weights, sample_weights


def make_loader(config: dict[str, Any], cache: dict[str, Any], indices: np.ndarray, columns: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None, *, train: bool, sample_weights: np.ndarray | None = None) -> DataLoader:
    dataset = LandmarkDataset(indices, cache["features"], cache["frame_mask"], cache["labels"], columns, mean, std, train and config["augment_train"], config["augment_noise_std"])
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True) if train and config["balanced_sampler"] and sample_weights is not None else None
    kwargs: dict[str, Any] = {
        "batch_size": config["batch_size"], "shuffle": train and sampler is None, "sampler": sampler,
        "num_workers": config["num_workers"], "pin_memory": bool(config["pin_memory"]),
        "persistent_workers": config["num_workers"] > 0,
    }
    if config["num_workers"] > 0:
        kwargs["prefetch_factor"] = int(config["prefetch_factor"])
    return DataLoader(dataset, **kwargs)


def metrics(y_true: list[int], y_pred: list[int], prefix: str) -> dict[str, float]:
    truth, predicted = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(truth)
    per_class_f1, recalls, supports = [], [], []
    for label in labels:
        tp = int(((truth == label) & (predicted == label)).sum())
        fp = int(((truth != label) & (predicted == label)).sum())
        fn = int(((truth == label) & (predicted != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        recalls.append(recall)
        supports.append(int((truth == label).sum()))
    f1_values, support_values = np.asarray(per_class_f1), np.asarray(supports)
    accuracy = float((truth == predicted).mean())
    return {f"{prefix}acc": accuracy, f"{prefix}macro_f1": float(f1_values.mean()), f"{prefix}weighted_f1": float(np.average(f1_values, weights=support_values)), f"{prefix}balanced_acc": float(np.mean(recalls))}


def amp_context(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager[Any]:
    return autocast("cuda", enabled=True) if enabled and device.type == "cuda" else contextlib.nullcontext()


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, *, optimizer: torch.optim.Optimizer | None, scaler: GradScaler, use_amp: bool, prefix: str, return_predictions: bool = False) -> dict[str, float] | tuple[dict[str, float], list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total, top5_total = 0, 0
    y_true: list[int] = []
    y_pred: list[int] = []
    predictions: list[dict[str, Any]] = []
    context = contextlib.nullcontext() if training else torch.no_grad()
    with context:
        for x, y, mask, sample_indices in loader:
            x, y, mask = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with amp_context(device, use_amp):
                logits = model(x, mask)
                loss = criterion(logits, y)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            batch = x.shape[0]
            total_loss += float(loss.detach()) * batch
            total += batch
            top5_total += int(logits.topk(min(5, logits.shape[1]), dim=1).indices.eq(y[:, None]).any(dim=1).sum())
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
            if return_predictions:
                probabilities = torch.softmax(logits, dim=1)
                top_probabilities, top_labels = probabilities.topk(min(5, logits.shape[1]), dim=1)
                for index, true_label, predicted_label, labels, probs in zip(sample_indices.tolist(), y.cpu().tolist(), logits.argmax(dim=1).detach().cpu().tolist(), top_labels.cpu().tolist(), top_probabilities.cpu().tolist()):
                    predictions.append({"sample_idx": index, "true_label": true_label, "pred_label": predicted_label, "top_labels": labels, "top_probabilities": probs})
    prefix = f"{prefix}_"
    result = metrics(y_true, y_pred, prefix)
    result.update({f"{prefix}loss": total_loss / max(total, 1), f"{prefix}top1_acc": result[f"{prefix}acc"], f"{prefix}top5_acc": top5_total / max(total, 1), f"{prefix}num_samples": float(total)})
    return (result, predictions) if return_predictions else result


def build_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    loaded: dict[str, Any] = {}
    if args.config:
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
    defaults = {**DEFAULTS, **loaded.get("defaults", {})}
    source_trials = loaded.get("trials", [])
    if args.landmark_subsets:
        source_trials = [{"name": key, "landmark_subset": key} for key in args.landmark_subsets]
    elif args.feature_sets:
        source_trials = [{"name": value.replace(",", "_"), "modalities": list(parse_feature_set(value))} for value in args.feature_sets]
    if not source_trials:
        raise ValueError("Provide --feature-sets or a config JSON containing trials")
    trials = []
    for position, trial in enumerate(source_trials, start=1):
        config = {**defaults, **trial}
        if args.epochs is not None:
            config["epochs"] = args.epochs
        subset_key = config.get("landmark_subset")
        raw_modalities = landmark_subset_modalities(str(subset_key)) if subset_key else config.pop("modalities", config.pop("feature_set", None))
        if isinstance(raw_modalities, str):
            raw_modalities = parse_feature_set(raw_modalities)
        if not raw_modalities:
            raise ValueError(f"Trial {position} has no modalities")
        config["modalities"] = canonical_modalities(raw_modalities)
        config["name"] = str(config.get("name", "_".join(config["modalities"])))
        config["landmark_subset"] = str(subset_key or config["name"])
        config["landmark_components"] = landmark_subset_components(str(subset_key)) if subset_key else list(config["modalities"])
        trials.append(config)
    return trials


def write_json(path: Path, payload: dict[str, Any]) -> None:
    portable = normalize_project_paths(payload)
    path.write_text(json.dumps(portable, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def completed_trial_names(output_root: Path, subset: str) -> set[str]:
    """Return trial names with a complete summary for resumable sweeps."""
    names: set[str] = set()
    for summary_path in (output_root / subset).glob("*/runs/*/summary.json"):
        config_path = summary_path.parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("name"):
                names.add(str(config["name"]))
    return names


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_state_for_save(model: nn.Module) -> dict[str, torch.Tensor]:
    """Strip the torch.compile wrapper so checkpoints load into the base model."""
    return getattr(model, "_orig_mod", model).state_dict()


def cloned_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model_state_for_save(model).items()}


def per_class_rows(predictions: list[dict[str, Any]], label_to_gloss: dict[int, str]) -> list[dict[str, Any]]:
    true = np.asarray([row["true_label"] for row in predictions])
    predicted = np.asarray([row["pred_label"] for row in predictions])
    rows: list[dict[str, Any]] = []
    for label in np.unique(true):
        label = int(label)
        tp = int(((true == label) & (predicted == label)).sum())
        fp = int(((true != label) & (predicted == label)).sum())
        fn = int(((true == label) & (predicted != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": label, "gloss": label_to_gloss.get(label, str(label)), "support": int((true == label).sum()), "predicted_count": int((predicted == label).sum()), "precision": precision, "recall": recall, "f1": f1})
    return rows


def annotate_predictions(predictions: list[dict[str, Any]], samples: list[dict[str, str]], label_to_gloss: dict[int, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        sample = samples[prediction.pop("sample_idx")]
        labels = [int(label) for label in prediction.pop("top_labels")]
        probabilities = [float(probability) for probability in prediction.pop("top_probabilities")]
        rows.append({
            "video_id": sample["video_id"], "split": sample["split"], "true_id": prediction["true_label"], "true_gloss": label_to_gloss.get(prediction["true_label"], str(prediction["true_label"])),
            "pred_id": prediction["pred_label"], "pred_gloss": label_to_gloss.get(prediction["pred_label"], str(prediction["pred_label"])),
            "correct": prediction["true_label"] == prediction["pred_label"],
            "top5_ids": json.dumps(labels), "top5_glosses": json.dumps([label_to_gloss.get(label, str(label)) for label in labels], ensure_ascii=False),
            "top5_probabilities": json.dumps(probabilities),
        })
    return rows


def train_trial(trial: dict[str, Any], cache: dict[str, Any], indices: dict[str, np.ndarray], device: torch.device, output_root: Path) -> dict[str, Any]:
    config = dict(trial)
    set_seed(int(config["seed"]))
    runtime = configure_runtime(config, device)
    columns = feature_columns(config["modalities"])
    mean, std = training_stats(cache["features"], cache["frame_mask"], indices["train"], columns) if config["standardize"] else (None, None)
    max_label = int(np.asarray(cache["labels"]).max())
    num_classes = max_label + 1
    weights, sample_weights = class_weights(cache["labels"], indices["train"], num_classes)
    seq_len = int(cache["manifest"]["seq_len"])
    config.update({"input_dim": int(len(columns)), "feature_columns": columns.tolist(), "seq_len": seq_len, "num_classes": num_classes, "subset": cache["manifest"]["subset"], "model_type": "transformer_encoder_only"})
    run_id = f"{config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / config["subset"] / config["landmark_subset"] / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config.update({"run_id": run_id, "started_at": datetime.now().isoformat(timespec="seconds"), "device": str(device)})

    train_loader = make_loader(config, cache, indices["train"], columns, mean, std, train=True, sample_weights=sample_weights)
    val_loader = make_loader(config, cache, indices["val"], columns, mean, std, train=False)
    test_loader = make_loader(config, cache, indices["test"], columns, mean, std, train=False)
    model = KeypointTransformerEncoderOnly(config["input_dim"], num_classes, seq_len, config["d_model"], config["num_heads"], config["num_layers"], config["dim_feedforward"], config["dropout"]).to(device)
    if config["compile_model"]:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_model=True requires PyTorch 2.0 or newer")
        model = torch.compile(model, mode="reduce-overhead")
        runtime["torch_compile"] = "reduce-overhead"
    else:
        runtime["torch_compile"] = "disabled"
    config["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    write_json(run_dir / "config.json", config)
    checkpoint_config = normalize_project_paths(config)
    write_json(run_dir / "runtime.json", runtime)
    label_to_gloss = {int(row["label"]): row["gloss"] for row in cache["samples"]}
    write_json(run_dir / "data_context.json", {
        "cache_manifest": cache["manifest"], "split_sample_counts": {name: int(len(values)) for name, values in indices.items()},
        "class_counts": {name: int(len(set(int(cache["labels"][index]) for index in values))) for name, values in indices.items()},
        "label_to_gloss": label_to_gloss,
    })
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device) if config["use_class_weight"] else None, label_smoothing=float(config["label_smoothing"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler_name = str(config["scheduler"]).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
    elif scheduler_name == "cosine_warmup":
        warmup_epochs = max(1, min(int(config["warmup_epochs"]), int(config["epochs"]) - 1))
        total_epochs = int(config["epochs"])

        def lr_factor(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    elif scheduler_name in {"none", "off", ""}:
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {config['scheduler']}. Choose cosine, cosine_warmup, or none.")
    scaler = GradScaler("cuda", enabled=bool(config["use_amp"]) and device.type == "cuda")

    best_score, best_epoch, waiting, best_state, last_state = -math.inf, 0, 0, None, None
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, use_amp=config["use_amp"], prefix="train")
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None, scaler=scaler, use_amp=config["use_amp"], prefix="val")
        if scheduler:
            scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **train_metrics, **val_metrics, "epoch_seconds": time.time() - epoch_started, "elapsed_seconds": time.time() - started}
        history.append(row)
        score = row["val_macro_f1"]
        improved = score > best_score or (math.isclose(score, best_score) and row["val_acc"] > (history[best_epoch - 1]["val_acc"] if best_epoch else -1))
        if improved:
            best_score, best_epoch, waiting = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"model_state_dict": cloned_model_state(model), "config": checkpoint_config, "feature_mean": mean, "feature_std": std, "best_epoch": best_epoch, "best_val_macro_f1": best_score}, run_dir / "best.pt")
        else:
            waiting += 1
        last_state = cloned_model_state(model)
        print(f"[{config['name']}] epoch {epoch:03d}/{config['epochs']} val_f1={score:.4f} val_acc={row['val_acc']:.4f} best={best_score:.4f}")
        if waiting >= int(config["early_stopping_patience"]):
            break
    write_csv(run_dir / "history.csv", history)
    assert best_state is not None and last_state is not None
    torch.save({"model_state_dict": last_state, "config": checkpoint_config, "feature_mean": mean, "feature_std": std, "last_epoch": len(history), "best_epoch": best_epoch, "best_val_macro_f1": best_score}, run_dir / "last.pt")
    model.load_state_dict(best_state)
    evaluated = run_epoch(model, test_loader, criterion, device, optimizer=None, scaler=scaler, use_amp=config["use_amp"], prefix="test", return_predictions=True)
    test_metrics, test_predictions = evaluated
    prediction_rows = annotate_predictions(test_predictions, cache["samples"], label_to_gloss)
    write_csv(run_dir / "test_predictions.csv", prediction_rows)
    write_csv(run_dir / "test_per_class_metrics.csv", per_class_rows(test_predictions, label_to_gloss))
    metric_rows = [{"split": "validation_best_epoch", "epoch": best_epoch, **history[best_epoch - 1]}, {"split": "test_best_validation_model", "epoch": best_epoch, **test_metrics}]
    write_csv(run_dir / "metrics.csv", metric_rows)
    summary = {"run_id": run_id, "name": config["name"], "modalities": ",".join(config["modalities"]), "input_dim": config["input_dim"], "best_epoch": best_epoch, "best_val_macro_f1": best_score, **test_metrics, "seconds": time.time() - started, "run_dir": project_relative_path(run_dir), "artifacts": ["config.json", "runtime.json", "data_context.json", "history.csv", "metrics.csv", "test_predictions.csv", "test_per_class_metrics.csv", "best.pt", "last.pt"]}
    write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", choices=("wlasl100", "wlasl300", "wlasl1000", "wlasl2000"), required=True, help="WLASL class subset to train.")
    parser.add_argument("--seq-len", type=int, default=60, help="Must match the sequence length used when creating the cache.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional cache override. Defaults to data/processed/encoder_features/<subset>/seq<seq-len>.")
    parser.add_argument("--landmark-subsets", nargs="+", choices=VALID_LANDMARK_SUBSETS, help="One or more named subsets: " + ", ".join(VALID_LANDMARK_SUBSETS))
    parser.add_argument("--feature-sets", nargs="+", help="Legacy custom modality sets, e.g. hand pose,hand. Prefer --landmark-subsets for predefined experiments.")
    parser.add_argument("--config", type=Path, help="JSON with optional defaults and a trials list; ignored when --feature-sets is supplied.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch count for every selected trial (useful for smoke tests).")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "encoder_only")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string such as cuda:0")
    parser.add_argument("--skip-completed", action="store_true", help="Skip trial names that already have a completed summary in this dataset output.")
    args = parser.parse_args()
    cache_dir = args.cache_dir or DATA_DIR / "processed" / "encoder_features" / args.subset / f"seq{args.seq_len}"
    cache = load_cache(cache_dir)
    if cache["manifest"].get("subset") != args.subset:
        raise ValueError(f"Cache belongs to {cache['manifest'].get('subset')}, not requested subset {args.subset}: {cache_dir}")
    if int(cache["manifest"].get("seq_len", -1)) != args.seq_len:
        raise ValueError(f"Cache sequence length is {cache['manifest'].get('seq_len')}, not requested {args.seq_len}: {cache_dir}")
    trials = build_trials(args)
    if args.skip_completed:
        completed = completed_trial_names(args.output_dir, args.subset)
        requested_count = len(trials)
        trials = [trial for trial in trials if trial["name"] not in completed]
        print(f"Resume check: skipped {requested_count - len(trials)} completed trial(s); {len(trials)} remaining.")
        if not trials:
            print("All requested trials are already complete.")
            return
    requested_device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA-enabled PyTorch installation is available")
    indices = split_indices(cache["samples"])
    print(f"Device: {device}; samples: " + ", ".join(f"{name}={len(value)}" for name, value in indices.items()))
    summaries = [train_trial(trial, cache, indices, device, args.output_dir) for trial in trials]
    reports_dir = args.output_dir / cache["manifest"]["subset"] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    print(f"Completed {len(summaries)} trial(s). Summary: {summary_path}")


if __name__ == "__main__":
    main()
