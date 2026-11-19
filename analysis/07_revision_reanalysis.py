"""Recompute corrected thresholds from deposited predictions, without raw data/GPU.

Run from any directory: python analysis/07_revision_reanalysis.py
Historical predictions are immutable. Host outputs remain EXPLORATORY because
their models used globally selected random-split hyperparameters.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.thresholds import labelled_budget_threshold, benign_budget_threshold, THRESHOLD_VERSION
from src.metrics import at_threshold

OUT = ROOT / "results_revision"
MODELS = ["rf", "xgboost", "mlp", "cnn1d", "ft_transformer", "proposed"]
HOST_MODELS = ["rf", "xgboost", "mlp", "ft_transformer", "proposed"]


def summary(rows):
    a = np.asarray(rows)
    return {"mean": a.mean(0).tolist(), "sd": a.std(0, ddof=1).tolist(),
            "p025": np.quantile(a, .025, axis=0).tolist(),
            "p975": np.quantile(a, .975, axis=0).tolist()}


def run():
    OUT.mkdir(exist_ok=True)
    meta = np.load(ROOT / "results_r3/runs/unsw_search/meta.npz", allow_pickle=False)
    yv, yt = meta["y_val"], meta["y_test"]
    assert np.array_equal(yt, np.load(ROOT / "results/runs/unsw/y_test.npy"))
    out = {"threshold_version": THRESHOLD_VERSION,
           "status": "Saved-score reanalysis completed; strict host retraining NOT completed",
           "unsw": {}, "host_exploratory": {}, "host_composition": {}}
    for mod in MODELS:
        rows = []; matched = []
        for seed in range(42, 47):
            p = np.load(ROOT / f"results/runs/unsw/{mod}/seed{seed}/preds.npz")
            pv, pt = p["p_val"], p["p_test"]
            old = json.loads((ROOT / f"results/runs/unsw/{mod}/seed{seed}/metrics.json").read_text())
            # Check that the recovered labels align with the original saved metrics.
            assert abs(roc_auc_score(yv, pv) - old["val"]["roc_auc"]) < 1e-9
            t = labelled_budget_threshold(yv, pv)
            assert (pv[yv == 0] >= t).mean() <= .01
            m = at_threshold(yt, pt, t)
            rows.append([m[k] for k in ("accuracy", "precision", "recall", "f1", "fpr", "mcc")])
            fpr, tpr, _ = roc_curve(yt, pt)
            matched.append([roc_auc_score(yt, pt), np.interp(.01, fpr, tpr)])
        out["unsw"][mod] = {"budget": summary(rows), "ranking": summary(matched), "recal": {}}
    # Preserve the original UNSW analysis random-draw sequence for comparability.
    rng = np.random.default_rng(0)
    for mod in MODELS:
        for n in (200, 500, 1000, 2000, 5000):
            for alpha in (.01, .05):
                rows = []
                for seed in range(42, 47):
                    pt = np.load(ROOT / f"results/runs/unsw/{mod}/seed{seed}/preds.npz")["p_test"]
                    neg = np.flatnonzero(yt == 0)
                    for rep in range(20):
                        ci = rng.choice(neg, n, replace=False)
                        mask = np.ones(len(yt), bool); mask[ci] = False
                        t = benign_budget_threshold(pt[ci], alpha)
                        cal = float((pt[ci] >= t).mean())
                        assert cal <= alpha + 1e-12
                        m = at_threshold(yt[mask], pt[mask], t)
                        rows.append([m["fpr"], m["recall"], m["f1"], m["accuracy"], cal, float(t > 1)])
                out["unsw"][mod]["recal"][f"{n}_{alpha}"] = summary(rows)
    rng = np.random.default_rng(0)
    detailed = []
    for fold in range(5):
        meta = np.load(ROOT / f"results_r3/runs/toniot_host{fold}/meta.npz", allow_pickle=False)
        yv, yt, group = meta["y_val"], meta["y_test"], meta["g_test"]
        neg = np.flatnonzero(yt == 0)
        out["host_composition"][str(fold+1)] = {str(g): int((group == g).sum()) for g in np.unique(group)}
        for mod in HOST_MODELS:
            rows = []; per_type = {}
            for seed in (42, 43, 44):
                p = np.load(ROOT / f"results_r3/runs/toniot_host{fold}/{mod}/seed{seed}/preds.npz")
                pv, pt = p["p_val"], p["p_test"]
                vt = labelled_budget_threshold(yv, pv)
                for rep in range(20):
                    ci = rng.choice(neg, 1000, replace=False)
                    mask = np.ones(len(yt), bool); mask[ci] = False
                    t = benign_budget_threshold(pt[ci])
                    cal = float((pt[ci] >= t).mean())
                    assert cal <= .01 + 1e-12
                    m = at_threshold(yt[mask], pt[mask], t)
                    rows.append([m["fpr"], m["recall"], m["f1"], cal, float(t > 1)])
                    detailed.append([fold+1, mod, seed, rep, t, cal, m["fpr"], m["recall"]])
                    for g in np.unique(group[yt == 1]):
                        per_type.setdefault(str(g), []).append(float((pt[(group == g) & (yt == 1)] >= t).mean()))
            out["host_exploratory"][f"{fold+1}:{mod}"] = {"recal": summary(rows),
                                  "recall_by_attack_type": {g: float(np.mean(v)) for g,v in per_type.items()}}
    (OUT / "corrected_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (OUT / "host_recalibration_draws.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["host_group", "model", "seed", "draw", "threshold", "cal_fpr", "eval_fpr", "recall"]); w.writerows(detailed)
    print("UNSW (corrected mean percentages, n=1000 alpha=1%)")
    for mod in MODELS:
        print(mod, np.round(100*np.array(out["unsw"][mod]["recal"]["1000_0.01"]["mean"]), 3))
    print("Exploratory host recalibration: fold, min/max FPR, min/max recall (%)")
    for fold in range(1,6):
        vals = np.array([out["host_exploratory"][f"{fold}:{mod}"]["recal"]["mean"] for mod in HOST_MODELS])
        print(fold, np.round(100*vals[:,:2].min(0), 3), np.round(100*vals[:,:2].max(0), 3))
    print("Saved", OUT)


if __name__ == "__main__":
    run()
