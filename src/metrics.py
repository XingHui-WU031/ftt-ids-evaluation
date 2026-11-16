import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             roc_curve, matthews_corrcoef)


def at_threshold(y, p, thr):
    pred = (p >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(threshold=float(thr), accuracy=(tp + tn) / len(y), precision=prec, recall=rec, f1=f1,
                fpr=fp / (fp + tn) if fp + tn else 0.0, mcc=float(matthews_corrcoef(y, pred)),
                tp=tp, tn=tn, fp=fp, fn=fn)


def ranking(y, p):
    return dict(roc_auc=float(roc_auc_score(y, p)), avg_precision=float(average_precision_score(y, p)))


def select_threshold(y_val, p_val, method="f1"):
    """Decision threshold chosen on the VALIDATION set only."""
    if method == "f1":
        prec, rec, thr = precision_recall_curve(y_val, p_val)
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        return float(thr[int(np.argmax(f1))])
    if method == "youden":
        fpr, tpr, thr = roc_curve(y_val, p_val)
        i = int(np.argmax(tpr - fpr))
        return float(min(thr[i], 1.0))
    if method.startswith("fpr"):
        # smallest threshold whose validation false-positive rate is <= target (e.g. fpr01 -> 1%)
        target = float(method[3:]) / 100
        neg = np.sort(p_val[y_val == 0])
        k = int(np.floor(target * len(neg)))
        return float(np.nextafter(neg[len(neg) - k - 1], 1.0)) if k < len(neg) else 0.0
    raise ValueError(method)


def per_group_recall(groups, y, p, thr):
    """Detection rate per attack category (and FPR on the normal class)."""
    pred = p >= thr
    out = {}
    for g in np.unique(groups):
        m = groups == g
        if y[m].max() == 0:
            out[g] = dict(n=int(m.sum()), false_positive_rate=float(pred[m].mean()))
        else:
            out[g] = dict(n=int(m.sum()), detection_rate=float(pred[m].mean()))
    return out


def full_report(y_val, p_val, y_test, p_test, groups_test=None):
    rep = {"val": {**ranking(y_val, p_val), **at_threshold(y_val, p_val, 0.5)},
           "test_ranking": ranking(y_test, p_test), "test": {}}
    for name in ("default", "val_f1", "val_youden", "val_fpr01"):
        thr = 0.5 if name == "default" else select_threshold(y_val, p_val, name.split("_")[1])
        rep["test"][name] = at_threshold(y_test, p_test, thr)
        if groups_test is not None:
            rep["test"][name]["per_group"] = per_group_recall(groups_test, y_test, p_test, thr)
    return rep
