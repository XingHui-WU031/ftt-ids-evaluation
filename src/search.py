"""Round-3 larger hyperparameter search (validation data only, training seed 42), followed by five-seed runs of the
selected configuration. Checks whether the small grids of round 1 disadvantaged XGBoost or FTT-IDS."""
import os

import numpy as np

from .runner import run_one
from .utils import save_json, load_json

SPACE = {
    "xgboost": dict(max_depth=[4, 6, 8, 10, 12], min_child_weight=[1, 2, 5, 10], subsample=[0.6, 0.8, 1.0],
                    colsample_bytree=[0.5, 0.7, 0.9, 1.0], reg_lambda=[0.1, 1.0, 5.0, 10.0], learning_rate=[0.03, 0.05, 0.1],
                    gamma=[0.0, 0.5, 1.0]),
    "proposed": dict(lr=[1e-3, 5e-4, 2e-4, 1e-4], dropout=[0.0, 0.1, 0.2, 0.3], d_model=[32, 64, 128],
                     num_layers=[1, 2, 3], dim_ff=[128, 256, 512, 2048], weight_decay=[0.0, 1e-5, 1e-4]),
}


def run_search(bundle, models, device, results_dir, log=print, n_trials=20, seeds=(42, 43, 44, 45, 46)):
    best_path = os.path.join(results_dir, "search", bundle.name, "best.json")
    best = load_json(best_path) if os.path.exists(best_path) else {}
    for name in models:
        if name not in best:
            rng = np.random.default_rng(2026)
            trials, seen = [], set()
            while len(trials) < n_trials:
                ov = {k: v[int(rng.integers(len(v)))] for k, v in SPACE[name].items()}
                ov = {k: (v.item() if hasattr(v, "item") else v) for k, v in ov.items()}
                key = tuple(sorted(ov.items()))
                if key in seen:
                    continue
                seen.add(key); i = len(trials)
                rep = run_one(name, bundle, 42, device, results_dir, overrides=ov,
                              tag=os.path.join("search", bundle.name, f"{name}_t{i:02d}"), log=log, save_model=False)
                trials.append(dict(trial=i, overrides=ov, val_auc=rep["val"]["roc_auc"], val_f1=rep["val"]["f1"],
                                   test_auc=rep["test_ranking"]["roc_auc"]))
                save_json(trials, os.path.join(results_dir, "search", bundle.name, f"{name}_trials.json"))
            trials.sort(key=lambda t: (t["val_auc"], t["val_f1"]), reverse=True)
            best[name] = trials[0]["overrides"]
            log(f"SEARCH {name}: best {best[name]} (val AUC {trials[0]['val_auc']:.5f})")
            save_json(best, best_path)
        for s in seeds:
            run_one(name, bundle, s, device, results_dir, overrides=best[name], tag=bundle.name + "_search", log=log)
