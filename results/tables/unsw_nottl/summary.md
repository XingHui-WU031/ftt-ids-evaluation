# Results on unsw_nottl


## Test metrics — threshold rule: default (mean ± std over seeds, %)

| Model | n_seeds | Acc | Prec | Rec | F1 | FPR | MCC | ROC-AUC | AP |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 5 | 87.05 ± 0.08 | 81.96 ± 0.08 | 98.06 ± 0.07 | 89.29 ± 0.07 | 26.45 ± 0.13 | 75.13 ± 0.17 | 0.9821 ± 0.0003 | 0.9869 ± 0.0002 |
| Proposed (per-feature + categorical emb.) | 5 | 87.70 ± 0.47 | 82.79 ± 0.69 | 98.04 ± 0.33 | 89.77 ± 0.34 | 24.98 ± 1.28 | 76.31 ± 0.80 | 0.9823 ± 0.0006 | 0.9868 ± 0.0004 |

## Test metrics — threshold rule: val_f1 (mean ± std over seeds, %)

| Model | n_seeds | Acc | Prec | Rec | F1 | FPR | MCC | ROC-AUC | AP | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 5 | 86.95 ± 0.34 | 81.83 ± 0.47 | 98.09 ± 0.12 | 89.22 ± 0.24 | 26.70 ± 0.88 | 74.98 ± 0.57 | 0.9821 ± 0.0003 | 0.9869 ± 0.0002 | 0.4943 ± 0.0168 |
| Proposed (per-feature + categorical emb.) | 5 | 86.13 ± 0.38 | 80.53 ± 0.46 | 98.66 ± 0.23 | 88.68 ± 0.28 | 29.23 ± 0.87 | 73.71 ± 0.69 | 0.9823 ± 0.0006 | 0.9868 ± 0.0004 | 0.4240 ± 0.0322 |

## Test metrics — threshold rule: val_youden (mean ± std over seeds, %)

| Model | n_seeds | Acc | Prec | Rec | F1 | FPR | MCC | ROC-AUC | AP | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 5 | 87.32 ± 0.44 | 82.39 ± 0.69 | 97.90 ± 0.26 | 89.48 ± 0.30 | 25.64 ± 1.27 | 75.58 ± 0.72 | 0.9821 ± 0.0003 | 0.9869 ± 0.0002 | 0.5168 ± 0.0278 |
| Proposed (per-feature + categorical emb.) | 5 | 87.91 ± 0.54 | 83.12 ± 0.84 | 97.94 ± 0.40 | 89.92 ± 0.38 | 24.39 ± 1.56 | 76.67 ± 0.90 | 0.9823 ± 0.0006 | 0.9868 ± 0.0004 | 0.5101 ± 0.0369 |

## Test metrics — threshold rule: val_fpr01 (mean ± std over seeds, %)

| Model | n_seeds | Acc | Prec | Rec | F1 | FPR | MCC | ROC-AUC | AP | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 5 | 92.66 ± 0.08 | 95.03 ± 0.17 | 91.45 ± 0.18 | 93.21 ± 0.08 | 5.86 ± 0.22 | 85.31 ± 0.16 | 0.9821 ± 0.0003 | 0.9869 ± 0.0002 | 0.8946 ± 0.0059 |
| Proposed (per-feature + categorical emb.) | 5 | 92.68 ± 0.26 | 95.81 ± 0.17 | 90.67 ± 0.32 | 93.17 ± 0.25 | 4.85 ± 0.19 | 85.45 ± 0.52 | 0.9823 ± 0.0006 | 0.9868 ± 0.0004 | 0.9004 ± 0.0091 |

## Effect of validation-selected threshold vs 0.5

| Model | Rule | ΔF1 (pp) | ΔFPR (pp) | ΔRecall (pp) | FP removed (mean) |
|---|---|---|---|---|---|
| XGBoost | val_f1 | -0.07 | +0.25 | +0.03 | -94 |
| XGBoost | val_youden | +0.19 | -0.80 | -0.16 | 298 |
| XGBoost | val_fpr01 | +3.92 | -20.59 | -6.61 | 7618 |
| Proposed (per-feature + categorical emb.) | val_f1 | -1.09 | +4.25 | +0.61 | -1571 |
| Proposed (per-feature + categorical emb.) | val_youden | +0.15 | -0.60 | -0.10 | 221 |
| Proposed (per-feature + categorical emb.) | val_fpr01 | +3.40 | -20.13 | -7.38 | 7448 |

## Detection rate per attack category (Normal = false-positive rate), rule default

| Model | Analysis | Backdoor | DoS | Exploits | Fuzzers | Generic | Normal | Reconnaissance | Shellcode | Worms |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 97.16 ± 0.59 | 100.00 ± 0.00 | 99.79 ± 0.03 | 99.17 ± 0.03 | 87.72 ± 0.55 | 99.98 ± 0.01 | 26.45 ± 0.13 | 99.76 ± 0.08 | 99.42 ± 0.29 | 100.00 ± 0.00 |
| Proposed (per-feature + categorical emb.) | 99.59 ± 0.51 | 100.00 ± 0.00 | 99.47 ± 0.18 | 98.98 ± 0.14 | 88.04 ± 2.05 | 99.96 ± 0.02 | 24.98 ± 1.28 | 99.65 ± 0.13 | 98.89 ± 0.29 | 100.00 ± 0.00 |

## Detection rate per attack category (Normal = false-positive rate), rule val_f1

| Model | Analysis | Backdoor | DoS | Exploits | Fuzzers | Generic | Normal | Reconnaissance | Shellcode | Worms |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 97.22 ± 0.69 | 100.00 ± 0.00 | 99.79 ± 0.03 | 99.18 ± 0.07 | 87.94 ± 0.74 | 99.98 ± 0.01 | 26.70 ± 0.88 | 99.77 ± 0.08 | 99.42 ± 0.29 | 100.00 ± 0.00 |
| Proposed (per-feature + categorical emb.) | 99.82 ± 0.19 | 100.00 ± 0.00 | 99.60 ± 0.15 | 99.21 ± 0.10 | 91.98 ± 1.46 | 99.97 ± 0.02 | 29.23 ± 0.87 | 99.79 ± 0.10 | 99.21 ± 0.19 | 100.00 ± 0.00 |

## Significance (validation-selected threshold; paired bootstrap over test samples, seed-averaged)

| Comparison | seeds | McNemar p (median) | seeds with p<0.05 | ΔF1 [95% CI] (pp) | ΔAUC [95% CI] |
|---|---|---|---|---|---|
| Proposed (per-feature + categorical emb.) vs XGBoost | 5 | 1.87e-31 | 5/5 | -0.55 [-0.63, -0.45] | +0.0002 [-0.0001, +0.0005] |