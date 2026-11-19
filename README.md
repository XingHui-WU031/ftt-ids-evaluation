# False-alarm control and explanation reliability: manuscript code

This snapshot accompanies the manuscript and Supplementary analyses v3.8. It applies existing explanation diagnostics to a feature-token detector and evaluates threshold transfer under source-host separation. It does not introduce a new explanation-audit framework. See the manuscript for attribution to Maseno et al., DOI 10.1007/s11416-026-00664-7, and the original sources for SHAP, Integrated Gradients and removal benchmarks.

## Results and provenance

The accompanying Reproducibility_package_v3_8.zip contains 160 validation-only tuning fits and 150 final fits, predictions, available weights, frozen source-host plans, logs and independent numerical audits. No models were retrained for the additional v3.8 analyses. Raw datasets are not included: obtain UNSW-NB15 and TON_IoT from their UNSW providers. Do not interpret absence of raw CSVs from this archive as absence of local raw-data training.

The `strict_host_completed/` directory in that archive preserves the actual Windows execution snapshot and its provenance. Absolute paths inside the archived orchestration scripts document that execution; edit them for another machine. The portable top-level `run.py` entry point is supplied separately. `benchmark_outputs/` retains original fixed-split results. The original explanation checkpoints are unavailable; reproducing those attributions requires retraining. Neural and boosting weights are included where saved; no missing random-forest checkpoints are implied to exist.

## Environment

`requirements.txt` records the declared local training versions. Use a CUDA-enabled PyTorch build compatible with the hardware/driver for GPU training. `requirements-reanalysis.txt` records the CPU document/numerical audit environment; it is not a replacement training environment. Do not install dependencies into an already working environment simply to inspect results.

## Full host study (requires raw data and substantial compute)

From this directory in PowerShell, using your chosen Python interpreter:

```powershell
$env:STRICT_HOST_PLANS = Join-Path (Get-Location) 'fixed_host_plans'
$py = 'PATH_TO_YOUR_PYTHON'
$data = 'PATH_TO_TON_IOT_DIRECTORY'
& $py -X utf8 -B run.py host_strict --data-dir $data --results full_results_within_host --models rf xgboost mlp ft_transformer proposed --seeds 42 43 44
& $py -X utf8 -B run.py host_strict --data-dir $data --dedup-policy none --results full_results_nodedup --models rf xgboost mlp ft_transformer proposed --seeds 42 43 44
```

Both policies must use the same STRICT_HOST_PLANS directory. Plans are based on the raw input before deduplication. Each outer fold uses a single internal host validation split; this is not full inner five-fold nested CV. Tuning uses seed 42 and validation scores only. Existing completed outputs need not be retrained to inspect or recalculate the tables.

## Tests and additional saved-prediction analyses

```powershell
& $py -m unittest discover -s tests -p test_threshold_protocol.py
& $py analysis/additional_host_analyses.py --formal-outputs PATH_TO_COMPLETED_FORMAL_OUTPUTS --out numerical_supplement_recomputed
```

The latter reads the unmodified `formal_outputs` layout in the full archive and uses saved predictions only. The separate-host calibration analysis is post hoc: a common target source host is selected by benign record availability and lexical order, with no score-based selection. All records from that host are excluded from evaluation. Fold 4 has no benign evaluation records and is not estimable; four-fold means must not be compared directly with the five-fold baseline as a controlled effect. No chronological deployment claim is made.

## Updating an existing repository

Merge these tracked code files into the existing repository after reviewing the diff. Preserve `.git`, existing history, and any other required files. Do not delete the repository or commit the full result archive, virtual environments, raw CSVs or model checkpoints. Publish a tagged release or separately hosted research archive for large results, then update the manuscript availability statement to the actual public URL. This file preparation has not published or altered the live repository.

Version 3.8 expands the readable supplementary tables and discussion. Model source files, fitting protocol, final predictions and weights are unchanged from v3.7. The new numerical_supplement/paired_model_fold_values.csv lists the ten paired fold differences used in Supplementary Table S5.
