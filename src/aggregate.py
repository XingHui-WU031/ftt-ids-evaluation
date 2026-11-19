"""E3/E4/E6: summary tables (mean ± std over seeds), significance tests, threshold analysis,
per-attack-category detection rates. Outputs CSV + Markdown into results/tables/<dataset>/."""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .data import load_dataset  # noqa: F401  (kept for callers)
from .utils import load_json
from .thresholds import THRESHOLD_VERSION

ORDER = ["rf", "xgboost", "lightgbm", "mlp", "cnn1d", "ft_transformer", "single_token", "thesis_ftt",
         "abl_shared_noid", "abl_perfeat_ordinal", "abl_shared_catemb", "proposed"]
LABELS = {"rf": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP",
          "cnn1d": "1D-CNN", "ft_transformer": "FT-Transformer", "single_token": "Single-token Transformer (thesis Exp01)",
          "thesis_ftt": "Feature-token Transformer, shared emb. (thesis)", "abl_shared_noid": "  - w/o identity embedding",
          "abl_perfeat_ordinal": "  per-feature emb., ordinal categoricals", "abl_shared_catemb": "  shared emb. + categorical emb.",
          "proposed": "Proposed (per-feature + categorical emb.)"}


def _collect(results_dir, dataset):
    rows = []
    for p in glob.glob(os.path.join(results_dir, "runs", dataset, "*", "seed*", "metrics.json")):
        m = load_json(p)
        if m.get("threshold_version") != THRESHOLD_VERSION:
            raise ValueError("Historical metrics use the old threshold rule. Run analysis/07_revision_reanalysis.py; do not relabel archived metrics as corrected.")
        base = dict(model=m["model"], seed=m["seed"], dir=os.path.dirname(p),
                    roc_auc=m["test_ranking"]["roc_auc"], avg_precision=m["test_ranking"]["avg_precision"],
                    val_acc=m["val"]["accuracy"], val_auc=m["val"]["roc_auc"],
                    best_epoch=m["info"].get("best_epoch"), train_seconds=m["info"].get("train_seconds"),
                    n_params=m["info"].get("n_params"))
        for th, t in m["test"].items():
            r = dict(base, threshold_rule=th, threshold=t["threshold"])
            for k in ("accuracy", "precision", "recall", "f1", "fpr", "mcc", "tp", "tn", "fp", "fn"):
                r[k] = t[k]
            for g, gv in t.get("per_group", {}).items():
                r["grp_" + g] = gv.get("detection_rate", gv.get("false_positive_rate"))
            rows.append(r)
    return pd.DataFrame(rows)


def _md(t):
    cols = list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def _fmt(mean, std, pct=True):
    return f"{100 * mean:.2f} ± {100 * std:.2f}" if pct else f"{mean:.4f} ± {std:.4f}"


def mcnemar(y, pa, pb):
    """Exact-continuity McNemar test on paired predictions (a vs b)."""
    ca, cb = pa == y, pb == y
    b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    if b + c == 0:
        return 1.0, b, c
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(chi2.sf(stat, 1)), b, c


def paired_bootstrap(y, probs_a, probs_b, thr_a, thr_b, n_boot=1000, seed=0):
    """Bootstrap over test samples; each replicate averages the metric over seeds for both models.
    Returns mean difference (a-b) and 95% CI for F1 and ROC-AUC."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y)

    def f1(yy, pred):
        tp = (pred & (yy == 1)).sum(); fp = (pred & (yy == 0)).sum(); fn = (~pred & (yy == 1)).sum()
        return 2 * tp / max(2 * tp + fp + fn, 1)

    pred_a = [pa >= t for pa, t in zip(probs_a, thr_a)]
    pred_b = [pb >= t for pb, t in zip(probs_b, thr_b)]
    d_f1, d_auc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        d_f1.append(np.mean([f1(yy, p[idx]) for p in pred_a]) - np.mean([f1(yy, p[idx]) for p in pred_b]))
        if len(d_auc) < 200:      # AUC is slower; 200 replicates are enough for its CI
            d_auc.append(np.mean([roc_auc_score(yy, p[idx]) for p in probs_a]) -
                         np.mean([roc_auc_score(yy, p[idx]) for p in probs_b]))
    q = lambda a: (float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(d_f1), q(d_auc)


def aggregate(results_dir, dataset, log=print, reference="proposed"):
    df = _collect(results_dir, dataset)
    if df.empty:
        log(f"no runs for {dataset}"); return
    out = os.path.join(results_dir, "tables", dataset)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "all_runs_long.csv"), index=False)
    models = [m for m in ORDER if m in df.model.unique()] + sorted(set(df.model) - set(ORDER))
    md = [f"# Results on {dataset}\n"]

    # ---------- Table: main metrics, for the default 0.5 rule and the validation-selected rule ----------
    for rule in ("default", "val_f1", "val_youden", "val_fpr01"):
        sub = df[df.threshold_rule == rule]
        rows = []
        for m in models:
            s = sub[sub.model == m]
            if s.empty:
                continue
            r = {"Model": LABELS.get(m, m), "n_seeds": len(s)}
            for k, lab in [("accuracy", "Acc"), ("precision", "Prec"), ("recall", "Rec"), ("f1", "F1"),
                           ("fpr", "FPR"), ("mcc", "MCC")]:
                r[lab] = _fmt(s[k].mean(), s[k].std(ddof=1) if len(s) > 1 else 0)
            for k, lab in [("roc_auc", "ROC-AUC"), ("avg_precision", "AP")]:
                r[lab] = _fmt(s[k].mean(), s[k].std(ddof=1) if len(s) > 1 else 0, pct=False)
            if rule != "default":
                r["Threshold"] = _fmt(s["threshold"].mean(), s["threshold"].std(ddof=1) if len(s) > 1 else 0, pct=False)
            rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(out, f"table_metrics_{rule}.csv"), index=False)
        md += [f"\n## Test metrics — threshold rule: {rule} (mean ± std over seeds, %)\n", _md(t)]

    # ---------- Threshold effect (E4): change in FPR / F1 from default to validation-selected ----------
    rows = []
    for m in models:
        a = df[(df.model == m) & (df.threshold_rule == "default")].set_index("seed")
        for rule in ("val_f1", "val_youden", "val_fpr01"):
            b = df[(df.model == m) & (df.threshold_rule == rule)].set_index("seed")
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            rows.append({"Model": LABELS.get(m, m), "Rule": rule,
                         "ΔF1 (pp)": f"{100 * (b.f1 - a.f1)[common].mean():+.2f}",
                         "ΔFPR (pp)": f"{100 * (b.fpr - a.fpr)[common].mean():+.2f}",
                         "ΔRecall (pp)": f"{100 * (b.recall - a.recall)[common].mean():+.2f}",
                         "FP removed (mean)": f"{(a.fp - b.fp)[common].mean():.0f}"})
    if rows:
        t = pd.DataFrame(rows); t.to_csv(os.path.join(out, "table_threshold_effect.csv"), index=False)
        md += ["\n## Effect of validation-selected threshold vs 0.5\n", _md(t)]

    # ---------- Per attack category (E6) ----------
    grp_cols = [c for c in df.columns if c.startswith("grp_")]
    if grp_cols:
        for rule in ("default", "val_f1"):
            sub = df[df.threshold_rule == rule]
            rows = []
            for m in models:
                s = sub[sub.model == m]
                if s.empty:
                    continue
                r = {"Model": LABELS.get(m, m)}
                for c in grp_cols:
                    r[c[4:]] = _fmt(s[c].mean(), s[c].std(ddof=1) if len(s) > 1 else 0)
                rows.append(r)
            t = pd.DataFrame(rows); t.to_csv(os.path.join(out, f"table_per_category_{rule}.csv"), index=False)
            md += [f"\n## Detection rate per attack category (Normal = false-positive rate), rule {rule}\n",
                   _md(t)]

    # ---------- Significance vs reference (E3) ----------
    if reference in df.model.unique():
        y = None
        rows = []
        ref = df[(df.model == reference) & (df.threshold_rule == "val_f1")].set_index("seed")
        ref_preds = {s: np.load(os.path.join(ref.loc[s, "dir"], "preds.npz"))["p_test"] for s in ref.index}
        y = _load_test_labels(ref.iloc[0]["dir"], results_dir, dataset)
        for m in models:
            if m == reference:
                continue
            oth = df[(df.model == m) & (df.threshold_rule == "val_f1")].set_index("seed")
            seeds = sorted(set(ref.index) & set(oth.index))
            if not seeds or y is None:
                continue
            o_preds = {s: np.load(os.path.join(oth.loc[s, "dir"], "preds.npz"))["p_test"] for s in seeds}
            pvals = []
            for s in seeds:
                p, _, _ = mcnemar(y, ref_preds[s] >= ref.loc[s, "threshold"], o_preds[s] >= oth.loc[s, "threshold"])
                pvals.append(p)
            (df1, lo1, hi1), (dauc, loa, hia) = paired_bootstrap(
                y, [ref_preds[s] for s in seeds], [o_preds[s] for s in seeds],
                [ref.loc[s, "threshold"] for s in seeds], [oth.loc[s, "threshold"] for s in seeds])
            rows.append({"Comparison": f"{LABELS.get(reference, reference)} vs {LABELS.get(m, m)}",
                         "seeds": len(seeds), "McNemar p (median)": f"{np.median(pvals):.2e}",
                         "seeds with p<0.05": f"{sum(p < 0.05 for p in pvals)}/{len(seeds)}",
                         "ΔF1 [95% CI] (pp)": f"{100 * df1:+.2f} [{100 * lo1:+.2f}, {100 * hi1:+.2f}]",
                         "ΔAUC [95% CI]": f"{dauc:+.4f} [{loa:+.4f}, {hia:+.4f}]"})
            log(f"  significance {m}: done")
        if rows:
            t = pd.DataFrame(rows); t.to_csv(os.path.join(out, "table_significance.csv"), index=False)
            md += ["\n## Significance (validation-selected threshold; paired bootstrap over test samples, seed-averaged)\n",
                   _md(t)]

    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    log(f"tables written to {out}")
    return df


def _load_test_labels(run_dir, results_dir, dataset):
    path = os.path.join(results_dir, "runs", dataset, "y_test.npy")
    return np.load(path) if os.path.exists(path) else None
