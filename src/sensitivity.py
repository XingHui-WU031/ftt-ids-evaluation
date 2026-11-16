"""Test-set sensitivity analysis: re-score saved predictions on (a) the de-duplicated official test set and
(b) the de-duplicated test set without records whose features also occur in the training file. No retraining."""
import glob
import os

import numpy as np
import pandas as pd

from .metrics import at_threshold, ranking
from .utils import load_json


def unsw_test_masks(data_dir):
    tr = pd.read_csv(os.path.join(data_dir, "UNSW_NB15_training-set.csv")).drop(columns=["id"])
    te = pd.read_csv(os.path.join(data_dir, "UNSW_NB15_testing-set.csv")).drop(columns=["id"])
    feats = [c for c in te.columns if c not in ("label", "attack_cat")]
    dedup = ~te.duplicated().to_numpy()
    key = lambda d: pd.util.hash_pandas_object(d[feats].astype(str), index=False).to_numpy()
    overlap = np.isin(key(te), key(tr))
    return {"official": np.ones(len(te), bool), "dedup": dedup, "dedup_no_overlap": dedup & ~overlap}


def sensitivity(results_dir, dataset, data_dir, log=print):
    masks = unsw_test_masks(data_dir)
    y = np.load(os.path.join(results_dir, "runs", dataset, "y_test.npy"))
    rows = []
    for p in glob.glob(os.path.join(results_dir, "runs", dataset, "*", "seed*", "metrics.json")):
        m = load_json(p)
        pt = np.load(os.path.join(os.path.dirname(p), "preds.npz"))["p_test"]
        for mname, mk in masks.items():
            for rule in ("default", "val_fpr01"):
                r = at_threshold(y[mk], pt[mk], m["test"][rule]["threshold"])
                rows.append(dict(model=m["model"], seed=m["seed"], test_set=mname, rule=rule, n=int(mk.sum()),
                                 pos_rate=float(y[mk].mean()), **ranking(y[mk], pt[mk]),
                                 **{k: r[k] for k in ("accuracy", "precision", "recall", "f1", "fpr", "mcc")}))
    df = pd.DataFrame(rows)
    out = os.path.join(results_dir, "tables", dataset); os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "sensitivity_long.csv"), index=False)
    g = df.groupby(["test_set", "rule", "model"])[["n", "pos_rate", "accuracy", "f1", "fpr", "roc_auc", "avg_precision"]]
    s = g.mean().round(4).join(g.std().round(4)[["f1", "fpr", "roc_auc"]], rsuffix="_std").reset_index()
    s.to_csv(os.path.join(out, "table_sensitivity.csv"), index=False)
    log(f"sensitivity table written ({ {k: int(v.sum()) for k, v in masks.items()} })")
    return s
