# Results on unsw_noshortcut


## Test metrics — threshold rule: default (mean ± std over seeds, %)

| Model | n_seeds | Acc | Prec | Rec | F1 | FPR | MCC | ROC-AUC | AP |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 5 | 87.15 ± 0.07 | 81.97 ± 0.06 | 98.29 ± 0.09 | 89.39 ± 0.06 | 26.49 ± 0.10 | 75.40 ± 0.14 | 0.9831 ± 0.0005 | 0.9876 ± 0.0003 |
| Proposed (per-feature + categorical emb.) | 5 | 87.42 ± 0.61 | 82.33 ± 0.82 | 98.25 ± 0.39 | 89.59 ± 0.45 | 25.84 ± 1.52 | 75.88 ± 1.09 | 0.9824 ± 0.0013 | 0.9869 ± 0.0009 |

## Test metrics — threshold rule: val_f1 (mean ± std over seeds, %)

| Model | n_seeds | Acc | Prec | Rec | F1 | FPR | MCC | ROC-AUC | AP | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 5 | 87.27 ± 0.33 | 82.15 ± 0.47 | 98.21 ± 0.12 | 89.47 ± 0.24 | 26.14 ± 0.86 | 75.59 ± 0.56 | 0.9831 ± 0.0005 | 0.9876 ± 0.0003 | 0.5078 ± 0.0168 |
| Proposed (per-feature + categorical emb.) | 5 | 86.44 ± 0.65 | 80.90 ± 0.81 | 98.67 ± 0.26 | 88.90 ± 0.48 | 28.55 ± 1.50 | 74.27 ± 1.16 | 0.9824 ± 0.0013 | 0.9869 ± 0.0009 | 0.4555 ± 0.0288 |

## Test metrics — threshold rule: val_youden (mean ± std over seeds, %)

| Model | n_seeds | Acc | Prec | Rec | F1 | FPR | MCC | ROC-AUC | AP | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 5 | 87.71 ± 0.41 | 82.82 ± 0.57 | 98.03 ± 0.10 | 89.78 ± 0.30 | 24.92 ± 1.01 | 76.33 ± 0.72 | 0.9831 ± 0.0005 | 0.9876 ± 0.0003 | 0.5322 ± 0.0192 |
| Proposed (per-feature + categorical emb.) | 5 | 87.58 ± 1.03 | 82.61 ± 1.32 | 98.14 ± 0.23 | 89.70 ± 0.76 | 25.35 ± 2.40 | 76.14 ± 1.84 | 0.9824 ± 0.0013 | 0.9869 ± 0.0009 | 0.5091 ± 0.0527 |

## Test metrics — threshold rule: val_fpr01 (mean ± std over seeds, %)

| Model | n_seeds | Acc | Prec | Rec | F1 | FPR | MCC | ROC-AUC | AP | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 5 | 92.83 ± 0.08 | 94.99 ± 0.25 | 91.82 ± 0.30 | 93.38 ± 0.08 | 5.93 ± 0.34 | 85.63 ± 0.17 | 0.9831 ± 0.0005 | 0.9876 ± 0.0003 | 0.8907 ± 0.0085 |
| Proposed (per-feature + categorical emb.) | 5 | 92.76 ± 0.18 | 95.55 ± 0.12 | 91.10 ± 0.34 | 93.27 ± 0.18 | 5.19 ± 0.16 | 85.57 ± 0.34 | 0.9824 ± 0.0013 | 0.9869 ± 0.0009 | 0.8878 ± 0.0171 |

## Effect of validation-selected threshold vs 0.5

| Model | Rule | ΔF1 (pp) | ΔFPR (pp) | ΔRecall (pp) | FP removed (mean) |
|---|---|---|---|---|---|
| XGBoost | val_f1 | +0.08 | -0.35 | -0.07 | 130 |
| XGBoost | val_youden | +0.40 | -1.57 | -0.26 | 582 |
| XGBoost | val_fpr01 | +3.99 | -20.56 | -6.47 | 7608 |
| Proposed (per-feature + categorical emb.) | val_f1 | -0.68 | +2.72 | +0.43 | -1005 |
| Proposed (per-feature + categorical emb.) | val_youden | +0.11 | -0.49 | -0.11 | 181 |
| Proposed (per-feature + categorical emb.) | val_fpr01 | +3.69 | -20.65 | -7.15 | 7639 |

## Detection rate per attack category (Normal = false-positive rate), rule default

| Model | Analysis | Backdoor | DoS | Exploits | Fuzzers | Generic | Normal | Reconnaissance | Shellcode | Worms |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 97.37 ± 0.35 | 100.00 ± 0.00 | 99.94 ± 0.04 | 99.55 ± 0.02 | 88.50 ± 0.63 | 99.99 ± 0.00 | 26.49 ± 0.10 | 99.84 ± 0.02 | 99.37 ± 0.14 | 100.00 ± 0.00 |
| Proposed (per-feature + categorical emb.) | 99.73 ± 0.35 | 99.97 ± 0.08 | 99.72 ± 0.09 | 99.30 ± 0.14 | 88.64 ± 2.62 | 99.98 ± 0.01 | 25.84 ± 1.52 | 99.76 ± 0.11 | 98.99 ± 0.39 | 100.00 ± 0.00 |

## Detection rate per attack category (Normal = false-positive rate), rule val_f1

| Model | Analysis | Backdoor | DoS | Exploits | Fuzzers | Generic | Normal | Reconnaissance | Shellcode | Worms |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 97.34 ± 0.43 | 100.00 ± 0.00 | 99.94 ± 0.03 | 99.53 ± 0.03 | 88.00 ± 0.82 | 99.99 ± 0.00 | 26.14 ± 0.86 | 99.83 ± 0.02 | 99.37 ± 0.14 | 100.00 ± 0.00 |
| Proposed (per-feature + categorical emb.) | 99.85 ± 0.10 | 100.00 ± 0.00 | 99.79 ± 0.07 | 99.47 ± 0.05 | 91.39 ± 1.87 | 99.99 ± 0.00 | 28.55 ± 1.50 | 99.81 ± 0.11 | 99.10 ± 0.30 | 100.00 ± 0.00 |

## Significance (validation-selected threshold; paired bootstrap over test samples, seed-averaged)

| Comparison | seeds | McNemar p (median) | seeds with p<0.05 | ΔF1 [95% CI] (pp) | ΔAUC [95% CI] |
|---|---|---|---|---|---|
| Proposed (per-feature + categorical emb.) vs XGBoost | 5 | 4.64e-16 | 4/5 | -0.56 [-0.65, -0.47] | -0.0007 [-0.0010, -0.0003] |