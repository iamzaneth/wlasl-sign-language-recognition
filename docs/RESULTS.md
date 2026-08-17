# Results and Limitations

This page documents the experiment state currently tracked in the repository. It is a
snapshot of one local WLASL corpus and should not be read as a full-dataset benchmark.

## Local dataset coverage

The audit found 11,980 readable videos out of 21,083 annotation instances. Generated
splits preserve official assignments but omit unavailable videos.

| Subset | Official manifest entries | Usable local videos | Coverage | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WLASL100 | 2,038 | 1,013 | 49.71% | 748 | 165 | 100 |
| WLASL300 | 5,117 | 2,660 | 51.98% | 1,897 | 446 | 317 |
| WLASL1000 | 13,168 | 7,232 | 54.92% | 5,001 | 1,290 | 941 |
| WLASL2000 | 21,083 | 11,980 | 56.82% | 8,313 | 2,253 | 1,414 |

Source: `data/metadata/dataset_summary.json` and the tracked split CSV files.

The current feature cache and training results cover WLASL100 at sequence length 60.
Its class coverage is:

| Split | Samples | Classes represented |
| --- | ---: | ---: |
| Train | 748 | 100 |
| Validation | 165 | 89 |
| Test | 100 | 72 |

The test set therefore has fewer than 1.4 samples per represented class on average and
no sample for 28 classes. Macro-F1 and balanced accuracy in this implementation are
computed over the labels present in the evaluated split.

## Published WLASL100 single-run winners

The following values come from
`output/encoder_only/wlasl100/best_by_landmark_subset.csv` and each published
`summary.json`. They were current when the five-round adaptive campaign completed on
2026-08-17.

| Landmark subset | Best validation macro-F1 | Test macro-F1 | Test top-1 | Test top-5 | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pose | 0.2933 | 0.1703 | 0.1900 | 0.4500 | 64 |
| Hands | **0.7158** | **0.7019** | **0.6900** | 0.8300 | 77 |
| Pose + hands | 0.7021 | 0.6292 | 0.6400 | **0.8800** | 85 |
| Pose + hands + mouth | 0.7039 | 0.6032 | 0.6100 | 0.8400 | 96 |
| Pose + hands + eyes | 0.6863 | 0.5657 | 0.5800 | 0.8400 | 62 |
| Pose + hands + eyebrows | 0.6960 | 0.6626 | 0.6800 | 0.8500 | 86 |
| Pose + hands + full selected face | 0.6638 | 0.6083 | 0.6200 | 0.8300 | 68 |

These are the highest **individual-run** validation macro-F1 values found across all
local completed runs for each subset. Test values are from the checkpoint selected on
validation performance and were not used for checkpoint selection.

The hands-only result is strongest on validation macro-F1, test macro-F1, and test top-1
accuracy in this snapshot. Adding pose or selected face landmarks did not improve those
metrics. This is evidence about this local preprocessing and incomplete split, not proof
that non-hand landmarks are generally unhelpful for WLASL.

## Adaptive campaign record

The tracked `adaptive_large_model_tuning` campaign completed five rounds. Each round
evaluated:

```text
5 profiles x 5 matched seeds x 7 landmark subsets = 175 trials
```

The full campaign therefore covered 875 planned trials. Search decisions used mean
validation macro-F1 over five seeds. Aggregate profile means and standard deviations are
stored in:

- `output/encoder_only/wlasl100/experiments/adaptive_large_model_tuning/results.csv`
- `output/encoder_only/wlasl100/experiments/adaptive_large_model_tuning/status.csv`
- `output/encoder_only/wlasl100/experiments/adaptive_large_model_tuning/summary.json`

The adaptive search-center rule and the published winner rule are intentionally
different:

- the next search center is chosen by five-seed **mean** validation macro-F1;
- `best_models` publishes the highest **single-run** validation macro-F1 found in the
  complete local run history.

Use `results.csv` rather than the single-run winner table when comparing profile
stability across seeds.

## What is available in Git

The repository tracks:

- the current split and cache manifests;
- aggregate campaign CSV/JSON records;
- published winner configurations, runtimes, metrics, data context, and summaries; and
- `best_by_landmark_subset.csv` and `.json`.

It does not track nested run directories, logs, reports, MediaPipe model bundles,
processed per-video landmarks, or PyTorch `.pt` checkpoints. A fresh clone can inspect
the reported metrics and use the tracked WLASL100 feature cache for new explicit
training, but it cannot load the published trained model weights.

## Comparison limitations

Keep the following constraints with any citation or comparison:

1. **Incomplete corpus.** WLASL100 coverage is 49.71%, and missing videos are not evenly
   guaranteed across classes, sources, signers, or splits.
2. **Incomplete evaluation classes.** Only 72 classes occur in the 100-sample local test
   split. Reported macro metrics do not assign a zero to the 28 absent classes.
3. **Small test support.** Per-class estimates are unstable because most represented
   test classes have very few examples.
4. **Multiple comparisons.** Published winners maximize individual validation macro-F1
   over a large local search, so their validation values are more optimistic than a
   pre-registered single configuration.
5. **Single preprocessing path.** Results use MediaPipe XYZ, shoulder-normalized XY,
   preserved MediaPipe Z, short-gap hand interpolation, and fixed sequence length 60.
6. **No RGB baseline.** The repository currently evaluates landmark encoders only and
   does not establish parity with an RGB or two-stream WLASL baseline.
7. **Environment dependence.** Recorded winners used Python 3.12, CUDA-enabled PyTorch
   2.11, and an NVIDIA RTX 4050 Laptop GPU. Exact timing and floating-point results can
   differ on another stack.

For these reasons, label the table as a result on the repository's local usable
WLASL100 subset. Do not present it as state of the art or compare it directly with papers
evaluated on the complete official WLASL100 test protocol.
