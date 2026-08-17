# Encoder-Only Landmark Training

This guide covers the fixed-length feature cache, Transformer encoder classifier,
explicit landmark experiments, and the adaptive WLASL100 tuning campaign.

The training workflow has three entrypoints:

- `scripts/prepare_encoder_features.py` builds the reusable feature cache.
- `scripts/train_encoder_only.py` trains one or more explicit configurations.
- `scripts/tune_wlasl100.py` runs the multi-round adaptive WLASL100 campaign.

Run all commands from the repository root.

## Prerequisites

Before training a subset, the following must exist:

1. Usable official split CSVs under `data/splits/<subset>`.
2. One processed NPZ for every split video under
   `data/processed/landmarks/shoulder_norm_v1`.
3. A compatible PyTorch installation.

The recommended preparation path is:

```powershell
python -m src.pipeline.run_data_pipeline --subset wlasl100
python scripts/prepare_encoder_features.py --subset wlasl100 --seq-len 60
```

See the [data pipeline guide](DATA_PIPELINE.md) for raw-data setup and preprocessing
details.

## Build the feature cache

```powershell
python scripts/prepare_encoder_features.py --subset wlasl100 --seq-len 60
```

The default cache directory is:

```text
data/processed/encoder_features/<subset>/seq<seq-len>/
```

Cache files:

| File | Purpose |
| --- | --- |
| `features.npy` | Memory-mapped `float32` tensor with shape `[samples, seq_len, 429]`. |
| `frame_mask.npy` | Boolean mask with shape `[samples, seq_len]`; false marks padding. |
| `labels.npy` | Official integer action labels. |
| `samples.csv` | Sample index, video ID, gloss, label, and split. |
| `manifest.json` | Subset, sequence length, sample count, feature layout, and source path. |

The cache always stores all supported modalities. Individual experiments select column
ranges without rebuilding the cache.

### Sequence conversion

- Long clips are uniformly sampled from the first through last decoded frame using
  `seq_len` rounded `linspace` positions.
- Short clips are right-padded with zero frames.
- The frame mask distinguishes real frames from padding.
- `NaN` and infinite landmark coordinates are converted to zero before resizing.
- No separate per-modality observation mask is passed to the model.

Use `--overwrite` to replace an existing cache. Without it, a matching subset and
sequence length are reused; a conflicting manifest raises an error. The trainer also
rejects a cache whose subset or sequence length differs from the requested values.

## Landmark subsets and input dimensions

Every selected point contributes XYZ, so each point contributes three input features.
Canonical definitions live in `src/training/encoder_data.py` and are mirrored in
`configs/landmark_subsets.yaml`.

| Subset | Components | Points | Input dimension |
| --- | --- | ---: | ---: |
| `pose` | Pose | 9 | 27 |
| `hands` | Left and right hands | 42 | 126 |
| `pose_hands` | Pose and both hands | 51 | 153 |
| `pose_hands_mouth` | Pose, hands, and mouth | 91 | 273 |
| `pose_hands_eyes` | Pose, hands, and both eyes | 83 | 249 |
| `pose_hands_eyebrows` | Pose, hands, and both eyebrows | 71 | 213 |
| `pose_hands_face` | Pose, hands, mouth, eyes, and eyebrows | 143 | 429 |

For ad hoc experiments, custom modality blocks are `pose`, `hand`, `mouth`, `eye`, and
`eyebrow`. Named subsets are preferred because their component contract is explicit.

## Model architecture

`KeypointTransformerEncoderOnly` is a sequence classifier with:

1. a linear projection from selected landmark features to `d_model`;
2. a learned CLS token;
3. learned positional embeddings for `seq_len + 1` tokens;
4. pre-normalized Transformer encoder layers with GELU activation;
5. layer normalization, dropout, and a linear class head.

The padding mask is extended with an unmasked CLS position and passed to the Transformer.
The classifier uses the final CLS representation. `d_model` must be divisible by
`num_heads`.

## Default training behavior

Defaults are defined in `src/training/train_encoder.py`:

| Setting | Default |
| --- | ---: |
| `d_model` | 256 |
| `num_heads` | 8 |
| `num_layers` | 3 |
| `dim_feedforward` | 512 |
| `dropout` | 0.25 |
| `batch_size` | 32 |
| `epochs` | 90 |
| `learning_rate` | 0.0002 |
| `weight_decay` | 0.001 |
| `scheduler` | `cosine` |
| `early_stopping_patience` | 24 |
| `use_class_weight` | true |
| `balanced_sampler` | true |
| `augment_train` | true |
| `augment_noise_std` | 0.006 |
| `standardize` | true |
| `label_smoothing` | 0.0 |
| `seed` | 42 |
| `use_amp` | true |
| `compile_model` | false |

Standardization statistics are computed per selected coordinate from training frames
only. Training augmentation adds independent Gaussian coordinate noise. With the default
configuration, class-frequency weights are used in cross-entropy and a weighted random
sampler also balances training samples.

The optimizer is AdamW. Supported scheduler values in JSON configurations are `cosine`,
`cosine_warmup`, and `none`. Gradients are clipped to a norm of `1.0`.

CUDA runs enable AMP, cuDNN benchmarking, TF32, pinned memory, and non-blocking transfers
when configured. `num_workers` and `cpu_threads` default to `auto`; on Windows, automatic
DataLoader workers are capped at two to limit the memory cost of spawned CUDA-enabled
interpreters.

## Train explicit configurations

Train one subset:

```powershell
python scripts/train_encoder_only.py --subset wlasl100 --landmark-subsets hands
```

Train all canonical ablations sequentially:

```powershell
python scripts/train_encoder_only.py --subset wlasl100 --landmark-subsets pose hands pose_hands pose_hands_mouth pose_hands_eyes pose_hands_eyebrows pose_hands_face
```

Trials run sequentially, which allows a multi-configuration experiment to share one GPU.
For a quick integration check:

```powershell
python scripts/train_encoder_only.py --subset wlasl100 --landmark-subsets hands --epochs 2 --device auto
```

Device values include `auto`, `cpu`, `cuda`, and explicit PyTorch device strings such as
`cuda:0`. `auto` selects CUDA when available and CPU otherwise.

### Train from JSON

```powershell
python scripts/train_encoder_only.py --subset wlasl100 --config configs/encoder_only_example.json
```

The JSON structure contains shared defaults and a trial list:

```json
{
  "defaults": {
    "d_model": 256,
    "num_heads": 8,
    "epochs": 90
  },
  "trials": [
    {
      "name": "hands_baseline",
      "landmark_subset": "hands"
    }
  ]
}
```

Each trial overrides shared defaults. `--epochs` overrides the epoch count for every
selected trial. The supplied example config intentionally sets `compile_model` to true,
while the code default is false; disable compilation in the config if the local PyTorch
backend does not support it reliably.

Use stable, unique trial names when resuming a list:

```powershell
python scripts/train_encoder_only.py --subset wlasl100 --config configs/encoder_only_example.json --skip-completed
```

A trial is considered complete when a run with the same `name` has both `config.json`
and `summary.json` under the selected dataset output. Changing hyperparameters without
changing the name will still cause `--skip-completed` to skip that trial.

### Custom modality combinations

```powershell
python scripts/train_encoder_only.py --subset wlasl100 --feature-sets "hand" "pose,hand,mouth"
```

Do not combine `--feature-sets`, `--landmark-subsets`, and config trials in one command;
use one trial-selection mechanism at a time.

## Selection and evaluation protocol

- Validation macro-F1 controls checkpoint selection and early stopping.
- A validation-accuracy tie-break is applied when macro-F1 is equal.
- The test set is evaluated once with the best validation checkpoint.
- Test metrics are reporting-only and are not used to select adaptive profiles.
- Reported aggregate metrics include loss, top-1 accuracy, top-5 accuracy, macro-F1,
  weighted-F1, and balanced accuracy.

Macro-F1 and balanced accuracy are calculated over labels present in the evaluated
split. On an incomplete local corpus, a validation or test split may not contain every
class; always inspect `data_context.json` before comparing metrics.

## Per-run artifacts

The default run path is:

```text
output/encoder_only/<subset>/<landmark-subset>/runs/<run-id>/
```

Every completed run writes:

| Artifact | Contents |
| --- | --- |
| `config.json` | Resolved model, data, optimization, and run configuration. |
| `runtime.json` | Python, PyTorch, platform, CPU, CUDA, and throughput settings. |
| `data_context.json` | Cache manifest, sample/class counts, and label-to-gloss mapping. |
| `history.csv` | Epoch-level train and validation metrics. |
| `metrics.csv` | Best validation row and test metrics for that checkpoint. |
| `test_predictions.csv` | Per-video top-1 and top-5 predictions/probabilities. |
| `test_per_class_metrics.csv` | Test precision, recall, F1, and support per present class. |
| `summary.json` | Compact best-epoch, test-metric, timing, and artifact summary. |
| `best.pt` | Best validation checkpoint with feature standardization statistics. |
| `last.pt` | Final-epoch checkpoint. |

After a direct multi-trial command, an experiment summary CSV is written under
`output/encoder_only/<subset>/reports`. Direct training does not update
`best_models/`; publication of global winners is performed by the adaptive tuning
runner.

Metadata paths owned by the repository are serialized in project-relative POSIX form so
the JSON and CSV files remain portable across clones.

## Adaptive WLASL100 tuning

The adaptive runner is specialized to WLASL100 and all seven canonical landmark subsets:

```powershell
python scripts/tune_wlasl100.py --campaign-id adaptive_large_model_tuning --device cuda
```

Each round evaluates five profiles over five matched seeds for every subset:

```text
5 profiles x 5 seeds x 7 landmark subsets = 175 trials per round
```

The default five-round campaign therefore plans 875 trials. The search stages are:

1. expand model width and include a deeper architecture;
2. refine depth and feed-forward capacity around the previous winner;
3. search local learning-rate multipliers;
4. search dropout and weight decay; and
5. re-evaluate the five strongest distinct prior profiles on fresh seeds.

Rounds after five continue the capacity, learning-rate, regularization, and portfolio
cycle. Mean validation macro-F1 over the five seeds determines the next search center.
Test means are retained for reporting only.

Validate the next round plan without starting training:

```powershell
python scripts/tune_wlasl100.py --campaign-id adaptive_large_model_tuning --start-round 1 --end-round 5 --dry-run
```

Resume a specific range with the same campaign ID:

```powershell
python scripts/tune_wlasl100.py --campaign-id adaptive_large_model_tuning --start-round 5 --end-round 5 --device cuda
```

Completed rounds are recognized from `results.csv`. Completed trial names are skipped by
the child training process. `active_trials.json` is retained while a round may need to
resume and removed after successful aggregation.

The first round requires a baseline configuration for every canonical subset. The
default tracked campaign includes `baseline_best.json`; a new campaign ID instead reads
local `best_models/<subset>/selection.json` and its referenced source run. Because run
directories and checkpoints are ignored by Git, starting a new campaign or rebuilding
winners from a fresh clone requires recreating the local run history first.

## Campaign and published outputs

```text
output/encoder_only/wlasl100/
|-- <landmark-subset>/runs/<run-id>/
|-- best_models/<landmark-subset>/
|   |-- selection.json
|   |-- config.json
|   |-- runtime.json
|   |-- data_context.json
|   |-- metrics.csv
|   |-- summary.json
|   `-- best.pt                 local only; ignored by Git
|-- best_by_landmark_subset.csv
|-- best_by_landmark_subset.json
|-- logs/
`-- experiments/<campaign-id>/
    |-- baseline_best.json
    |-- manifest.json
    |-- results.csv
    |-- status.csv
    |-- summary.json
    `-- active_trials.json      present only for an incomplete round
```

`results.csv` contains one aggregate row per profile, round, and landmark subset.
`status.csv` contains one lifecycle row per round. After aggregation, the runner scans
all local completed runs and publishes the highest individual validation macro-F1 run
for each landmark subset to `best_models` and the two `best_by_landmark_subset` files.
That global single-run publication rule is distinct from the five-seed mean used to move
the adaptive search center.
