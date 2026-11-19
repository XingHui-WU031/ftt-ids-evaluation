"""Summarise NEW strict-host runs. Never substitute historical host predictions.

python analysis/08_strict_host_summary.py --results results_strict_host
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.thresholds import benign_budget_threshold, labelled_budget_threshold, THRESHOLD_VERSION


def rates(y, p, t):
    attack, benign = y == 1, y == 0
    return float((p[benign] >= t).mean()), float((p[attack] >= t).mean()) if attack.any() else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True)
    ap.add_argument("--suffix", default=""); args = ap.parse_args()
    root = Path(args.results); records = []; type_rows = []
    for fold in range(5):
        name = f"toniot_strict_host{fold}{args.suffix}"
        audit = json.loads((root / "protocol" / f"{name}_audit.json").read_text())
        if audit.get("protocol") != "nested-source-host-v1":
            raise ValueError("Strict host audit absent")
        meta = np.load(root / "runs" / name / "meta.npz", allow_pickle=False)
        yv, yt, gt, ht = (meta[k] for k in ("y_val", "y_test", "g_test", "host_test"))
        neg = np.flatnonzero(yt == 0)
        if len(neg) <= 1000:
            raise ValueError(f"{name}: insufficient benign rows for n=1000 plus an evaluation set")
        for mod in ("rf", "xgboost", "mlp", "ft_transformer", "proposed"):
            for seed in (42,43,44):
                path = root / "runs" / name / mod / f"seed{seed}"
                metrics = json.loads((path / "metrics.json").read_text())
                if metrics.get("threshold_version") != THRESHOLD_VERSION:
                    raise ValueError("Stale threshold implementation in cached run")
                p = np.load(path / "preds.npz"); pv, pt = p["p_val"], p["p_test"]
                vt = labelled_budget_threshold(yv, pv)
                for kind,t in (("default", .5), ("validation_budget", vt)):
                    fp, rec = rates(yt, pt, t)
                    records.append([fold+1, mod, seed, kind, -1, t, fp, rec, roc_auc_score(yt, pt)])
                # Same calibration indices for each model and seed within a fold.
                rng = np.random.default_rng(9000 + fold)
                for draw in range(20):
                    ci = rng.choice(neg,1000,replace=False); mask = np.ones(len(yt),bool); mask[ci]=False
                    t = benign_budget_threshold(pt[ci]); assert (pt[ci]>=t).mean() <= .01
                    fp, rec = rates(yt[mask], pt[mask], t)
                    records.append([fold+1,mod,seed,"target_benign",draw,t,fp,rec,roc_auc_score(yt,pt)])
                    for g in sorted(set(gt[yt==1])):
                        idx = (gt==g)&(yt==1)
                        seen = g in audit["attack_counts"]["train"]
                        type_rows.append([fold+1,mod,seed,draw,g,seen,int(idx.sum()),float((pt[idx]>=t).mean())])
                    for host in sorted(set(ht[yt==0])):
                        idx = (ht==host)&(yt==0)&mask
                        if idx.any():
                            type_rows.append([fold+1,mod,seed,draw,"benign_host:"+host,"",int(idx.sum()),float((pt[idx]>=t).mean())])
    out = root / "summary"; out.mkdir(exist_ok=True)
    for filename, headers, rows in (("strict_host_metrics.csv", ["fold","model","seed","rule","draw","threshold","fpr","recall","auc"],records),
                                    ("attack_coverage_and_host_fpr.csv", ["fold","model","seed","draw","group","attack_type_seen_in_training","n","rate"],type_rows)):
        with (out/filename).open("w",newline="") as f:
            w=csv.writer(f);w.writerow(headers);w.writerows(rows)
    print("Strict results saved to",out)


if __name__ == "__main__":
    main()
