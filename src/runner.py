import os
import pickle

import numpy as np

from .configs import MODELS, ABLATION_LR_FROM
from .metrics import full_report
from .trainers import fit_torch, fit_tree, predict_torch, predict_tree, _arrays
from .utils import save_json, load_json


def tuned_overrides(results_dir, dataset, name):
    path = os.path.join(results_dir, "tuning", dataset, "best.json")
    if not os.path.exists(path):
        return {}
    best = load_json(path)
    src = ABLATION_LR_FROM.get(name, name)
    ov = dict(best.get(src, {}))
    if name in ABLATION_LR_FROM:                       # ablations borrow only the learning rate
        ov = {k: v for k, v in ov.items() if k == "lr"}
    return ov


def run_one(name, bundle, seed, device, results_dir, overrides=None, tag=None, log=print, save_model=True):
    out = os.path.join(results_dir, "runs", tag or bundle.name, name, f"seed{seed}")
    if os.path.exists(os.path.join(out, "metrics.json")):
        log(f"  skip {name} seed {seed} (done)")
        return load_json(os.path.join(out, "metrics.json"))
    os.makedirs(out, exist_ok=True)
    cfg = MODELS[name]
    log(f"  run {name} seed {seed} overrides={overrides}")
    if cfg["kind"] == "torch":
        model, hist, info = fit_torch(name, bundle, seed, device, overrides, log)
        xva, cva = _arrays(bundle, "val", cfg["input"]); xte, cte = _arrays(bundle, "test", cfg["input"])
        p_val = predict_torch(model, xva, cva, device); p_test = predict_torch(model, xte, cte, device)
        if save_model:
            import torch
            torch.save({"state_dict": model.state_dict(), "overrides": overrides or {}}, os.path.join(out, "model.pt"))
        if hist:
            import csv
            with open(os.path.join(out, "history.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    else:
        model, hist, info = fit_tree(name, bundle, seed, device, overrides, log)
        p_val = predict_tree(model, bundle.splits["val"]["X_tree"])
        p_test = predict_tree(model, bundle.splits["test"]["X_tree"])
        if save_model and cfg["kind"] != "rf":          # RF pickles are large; predictions are kept
            with open(os.path.join(out, "model.pkl"), "wb") as f:
                pickle.dump(model, f)
    y_val = bundle.splits["val"]["y"]; y_test = bundle.splits["test"]["y"]
    np.savez_compressed(os.path.join(out, "preds.npz"), p_val=p_val, p_test=p_test)
    ytest_path = os.path.join(results_dir, "runs", tag or bundle.name, "y_test.npy")
    if not os.path.exists(ytest_path):
        np.save(ytest_path, y_test)
    meta_path = os.path.join(results_dir, "runs", tag or bundle.name, "meta.npz")   # labels/groups for re-analysis
    if not os.path.exists(meta_path):
        extra = {}
        if bundle.splits["test"]["group"] is not None:
            extra["g_test"] = bundle.splits["test"]["group"].astype(str)
        if getattr(bundle, "host_test", None) is not None:
            extra["host_test"] = bundle.host_test.astype(str)
        np.savez_compressed(meta_path, y_val=y_val, y_test=y_test, **extra)
    rep = full_report(y_val, p_val, y_test, p_test, bundle.splits["test"]["group"])
    rep.update(model=name, seed=seed, dataset=tag or bundle.name, overrides=overrides or {}, info=info)
    save_json(rep, os.path.join(out, "metrics.json"))
    t = rep["test"]
    log(f"  -> {name} s{seed}: AUC {rep['test_ranking']['roc_auc']:.4f} | F1@0.5 {t['default']['f1']:.4f} "
        f"FPR@0.5 {t['default']['fpr']:.4f} | F1@valthr {t['val_f1']['f1']:.4f} FPR {t['val_f1']['fpr']:.4f}")
    return rep
