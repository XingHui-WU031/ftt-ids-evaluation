"""E7-E9, E11 + error analysis.

E7  SHAP (GradientExplainer / expected gradients) on the attack margin  f(x) = logit_attack - logit_normal,
    larger background and explanation sets, completeness (additivity) error reported.
E8  Stability: rank agreement of global importance across model seeds and across background draws.
E9  Faithfulness: mask global / per-sample top-k features with reference values vs random-k.
E11 Permutation importance (model-agnostic) and its agreement with SHAP.
ERR SHAP by prediction outcome (TP/TN/FP/FN): what drives false alarms.
Categorical embeddings are explained in embedding space and summed per feature.
"""
import os

import numpy as np
import pandas as pd

from .configs import MODELS
from .trainers import build_torch_model, feature_names_for, _arrays, predict_torch
from .utils import save_json, load_json


# ------------------------------------------------------------------ helpers
def _stratified(y, n, rng):
    idx = []
    for c in np.unique(y):
        pool = np.where(y == c)[0]
        k = int(round(n * len(pool) / len(y)))
        idx.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return np.sort(np.concatenate(idx))


def _load_model(name, bundle, seed, device, results_dir):
    import torch
    run = os.path.join(results_dir, "runs", bundle.name, name, f"seed{seed}")
    ck = torch.load(os.path.join(run, "model.pt"), map_location=device, weights_only=False)
    model = build_torch_model(name, bundle, ck.get("overrides"))
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    thr = load_json(os.path.join(run, "metrics.json"))["test"]["val_f1"]["threshold"]
    return model, thr


class _MarginWrapper:
    """Builds a torch module mapping z = [x_num, flattened categorical embeddings] -> attack margin."""
    def __new__(cls, model, n_num, n_cat):
        import torch.nn as nn

        class W(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, z):
                xn = z[:, :n_num]
                ce = z[:, n_num:].reshape(z.size(0), n_cat, -1) if n_cat else None
                lg = self.m.forward_embedded(xn, ce)
                return (lg[:, 1] - lg[:, 0]).unsqueeze(1)
        return W()


def _to_z(model, x_num, x_cat, device):
    import torch
    xn = torch.as_tensor(x_num, device=device)
    if getattr(model, "n_cat", 0):
        with torch.no_grad():
            ce = model.cat_embed(torch.as_tensor(x_cat, device=device))
        return torch.cat([xn, ce.flatten(1)], dim=1)
    return xn


def _aggregate_to_features(sv, n_num, n_cat):
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 0]
    if not n_cat:
        return sv
    e = (sv.shape[1] - n_num) // n_cat
    cat = sv[:, n_num:].reshape(len(sv), n_cat, e).sum(-1)
    return np.concatenate([sv[:, :n_num], cat], axis=1)


def shap_values(model, n_num, n_cat, bg, ex, device, nsamples, rseed, batch_size=256):
    import shap
    import torch
    w = _MarginWrapper(model, n_num, n_cat).to(device).eval()
    zb, ze = _to_z(model, *bg, device), _to_z(model, *ex, device)
    expl = shap.GradientExplainer(w, zb, batch_size=batch_size)
    sv = expl.shap_values(ze, nsamples=nsamples, rseed=rseed)
    phi = _aggregate_to_features(sv[0] if isinstance(sv, list) else sv, n_num, n_cat)
    with torch.no_grad():
        base = float(w(zb).mean()); margin = w(ze).squeeze(1).cpu().numpy()
    err = np.abs(base + phi.sum(1) - margin)
    return phi, base, margin, err


def _masked_predict(model, x_num, x_cat, mask_cols, ref_num, ref_cat, n_num, device, per_sample=None):
    """Replace selected features with reference values. mask_cols indexes the explained feature list
    (numeric first, then categorical). per_sample: (n, k) array of columns to mask per row."""
    xn, xc = x_num.copy(), (x_cat.copy() if x_cat is not None else None)
    if per_sample is None:
        per_sample = np.tile(np.asarray(mask_cols, dtype=int), (len(xn), 1))
    rows = np.arange(len(xn))[:, None]
    for j in range(per_sample.shape[1]):
        cols = per_sample[:, j]
        isn = cols < n_num
        xn[rows[isn, 0], cols[isn]] = ref_num[cols[isn]]
        if xc is not None and (~isn).any():
            cc = cols[~isn] - n_num
            xc[rows[~isn, 0], cc] = ref_cat[cc]
    return predict_torch(model, xn, xc, device)


# ------------------------------------------------------------------ main
def run_explain(bundle, name, seeds, device, results_dir, log=print, n_background=500, n_explain=2000,
                nsamples=500, perm_n=20000, perm_repeats=3, ks=(1, 2, 3, 5, 10, 15, 20), n_outcome=400):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, f1_score
    from scipy.stats import spearmanr, kendalltau

    out = os.path.join(results_dir, "explain", bundle.name, name)
    os.makedirs(out, exist_ok=True)
    inp = MODELS[name]["input"]
    feats = feature_names_for(bundle, inp)
    n_num = len(bundle.feature_names) if inp == "ordinal" else len(bundle.num_cols)
    n_cat = 0 if inp == "ordinal" else len(bundle.cat_cols)
    ref_num = bundle.ref_ord if inp == "ordinal" else bundle.ref_num
    ref_cat = bundle.ref_cat
    rng = np.random.default_rng(2026)
    xtr, ctr = _arrays(bundle, "train", inp); xte, cte = _arrays(bundle, "test", inp)
    ytr, yte = bundle.splits["train"]["y"], bundle.splits["test"]["y"]
    bg_idx = _stratified(ytr, n_background, rng)            # identical samples for every seed
    ex_idx = _stratified(yte, n_explain, rng)
    sub = lambda x, i: x[i] if x is not None else None
    np.savez(os.path.join(out, "indices.npz"), bg=bg_idx, ex=ex_idx)

    glob_imp, summary = {}, {"model": name, "features": feats, "seeds": {}}
    first = None
    for s in seeds:
        model, thr = _load_model(name, bundle, s, device, results_dir)
        phi, base, margin, err = shap_values(model, n_num, n_cat, (sub(xtr, bg_idx), sub(ctr, bg_idx)),
                                             (sub(xte, ex_idx), sub(cte, ex_idx)), device, nsamples, s)
        np.savez_compressed(os.path.join(out, f"shap_seed{s}.npz"), phi=phi, base=base, margin=margin)
        glob_imp[s] = np.abs(phi).mean(0)
        summary["seeds"][s] = dict(base_value=base, additivity_mae=float(err.mean()),
                                   additivity_rel=float(err.mean() / (np.abs(margin).mean() + 1e-12)))
        log(f"  SHAP seed {s}: additivity MAE {err.mean():.4f} (rel {summary['seeds'][s]['additivity_rel']:.3%})")
        if first is None:
            first = (s, model, thr, phi, base, margin)

    # ---------------- E7 global importance table + plots ----------------
    G = np.vstack([glob_imp[s] for s in seeds])
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": G.mean(0), "std_over_seeds": G.std(0, ddof=1) if len(seeds) > 1 else 0})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.insert(0, "rank", np.arange(1, len(imp) + 1))
    imp.to_csv(os.path.join(out, "global_importance.csv"), index=False)
    top = imp.head(15)[::-1]
    plt.figure(figsize=(6, 5))
    plt.barh(top.feature, top.mean_abs_shap, xerr=top.std_over_seeds, color="#4C72B0", ecolor="#333333", capsize=2)
    plt.xlabel("mean |SHAP| on attack margin (mean ± std over seeds)"); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_global_importance.png"), dpi=300); plt.close()

    import shap
    s0, model0, thr0, phi0, base0, margin0 = first
    xe_show = np.concatenate([sub(xte, ex_idx)] + ([sub(cte, ex_idx).astype(np.float32)] if n_cat else []), axis=1)
    shap.summary_plot(phi0, xe_show, feature_names=feats, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_beeswarm.png"), dpi=300); plt.close()
    p0 = 1 / (1 + np.exp(-margin0))
    tp = np.where((yte[ex_idx] == 1) & (p0 >= thr0))[0]
    if len(tp):
        i = tp[0]
        shap.plots.waterfall(shap.Explanation(phi0[i], base0, xe_show[i], feature_names=feats), max_display=15, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(out, "fig_waterfall_tp.png"), dpi=300); plt.close()

    # ---------------- E8 stability ----------------
    stab = []
    for a in range(len(seeds)):
        for b in range(a + 1, len(seeds)):
            ga, gb = glob_imp[seeds[a]], glob_imp[seeds[b]]
            ta, tb = set(np.argsort(-ga)[:10]), set(np.argsort(-gb)[:10])
            stab.append(dict(kind="across_seeds", a=seeds[a], b=seeds[b], spearman=spearmanr(ga, gb)[0],
                             kendall=kendalltau(ga, gb)[0], top10_jaccard=len(ta & tb) / len(ta | tb)))
    bgs = [glob_imp[s0]]
    for r in (1, 2):
        bi = _stratified(ytr, n_background, np.random.default_rng(100 + r))
        phi_r, *_ = shap_values(model0, n_num, n_cat, (sub(xtr, bi), sub(ctr, bi)),
                                (sub(xte, ex_idx), sub(cte, ex_idx)), device, nsamples, s0)
        bgs.append(np.abs(phi_r).mean(0))
    for r in (1, 2):
        ta, tb = set(np.argsort(-bgs[0])[:10]), set(np.argsort(-bgs[r])[:10])
        stab.append(dict(kind="across_backgrounds", a=0, b=r, spearman=spearmanr(bgs[0], bgs[r])[0],
                         kendall=kendalltau(bgs[0], bgs[r])[0], top10_jaccard=len(ta & tb) / len(ta | tb)))
    stab = pd.DataFrame(stab); stab.to_csv(os.path.join(out, "stability.csv"), index=False)
    agg = stab.groupby("kind")[["spearman", "kendall", "top10_jaccard"]].agg(["mean", "std"]).round(4)
    summary["stability"] = {str(kind): {f"{c}_{st}": float(agg.loc[kind, (c, st)]) for c, st in agg.columns}
                            for kind in agg.index}
    log(f"  stability: {stab.groupby('kind')[['spearman', 'top10_jaccard']].mean().round(3).to_dict()}")

    # ---------------- E9 faithfulness ----------------
    order = [feats.index(f) for f in imp.feature]
    faith = []
    for s in seeds:
        model, thr = _load_model(name, bundle, s, device, results_dir)
        p_full = predict_torch(model, xte, cte, device)
        faith.append(dict(seed=s, k=0, kind="none", auc=roc_auc_score(yte, p_full), f1=f1_score(yte, p_full >= thr)))
        for k in ks:
            p = _masked_predict(model, xte, cte, order[:k], ref_num, ref_cat, n_num, device)
            faith.append(dict(seed=s, k=k, kind="shap_topk", auc=roc_auc_score(yte, p), f1=f1_score(yte, p >= thr)))
            for r in range(5):
                cols = np.random.default_rng(1000 * k + r).choice(len(feats), size=k, replace=False)
                p = _masked_predict(model, xte, cte, cols, ref_num, ref_cat, n_num, device)
                faith.append(dict(seed=s, k=k, kind="random_k", rep=r, auc=roc_auc_score(yte, p), f1=f1_score(yte, p >= thr)))
    faith = pd.DataFrame(faith); faith.to_csv(os.path.join(out, "faithfulness_global.csv"), index=False)
    fsum = faith.groupby(["kind", "k"])[["auc", "f1"]].agg(["mean", "std"]).reset_index()
    fsum.columns = ["_".join(c).strip("_") for c in fsum.columns]
    fsum.to_csv(os.path.join(out, "faithfulness_global_summary.csv"), index=False)
    plt.figure(figsize=(5, 4))
    for kind, lab in (("shap_topk", "mask SHAP top-k"), ("random_k", "mask random k")):
        d = fsum[(fsum.kind == kind)]
        d = pd.concat([fsum[fsum.kind == "none"], d])
        plt.errorbar(d.k, d.auc_mean, yerr=d.auc_std, marker="o", capsize=2, label=lab)
    plt.xlabel("k features masked"); plt.ylabel("test ROC-AUC"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_faithfulness.png"), dpi=300); plt.close()

    # local (per-sample) deletion on the explanation set, first seed: drop in attack margin
    xe, ce = sub(xte, ex_idx), sub(cte, ex_idx)
    p_base = predict_torch(model0, xe, ce, device)
    loc = []
    rank_local = np.argsort(-np.abs(phi0), axis=1)
    pos = p_base >= thr0
    for k in ks:
        p_top = _masked_predict(model0, xe, ce, None, ref_num, ref_cat, n_num, device, per_sample=rank_local[:, :k])
        rnd = np.vstack([np.random.default_rng(i).choice(len(feats), size=k, replace=False) for i in range(len(xe))])
        p_rnd = _masked_predict(model0, xe, ce, None, ref_num, ref_cat, n_num, device, per_sample=rnd)
        loc.append(dict(k=k, flip_rate_shap=float((p_top[pos] < thr0).mean()), flip_rate_random=float((p_rnd[pos] < thr0).mean()),
                        prob_drop_shap=float((p_base[pos] - p_top[pos]).mean()), prob_drop_random=float((p_base[pos] - p_rnd[pos]).mean())))
    pd.DataFrame(loc).to_csv(os.path.join(out, "faithfulness_local.csv"), index=False)

    # ---------------- E11 permutation importance ----------------
    pi_idx = _stratified(yte, min(perm_n, len(yte)), np.random.default_rng(7))
    xp, cp, yp = sub(xte, pi_idx), sub(cte, pi_idx), yte[pi_idx]
    auc0 = roc_auc_score(yp, predict_torch(model0, xp, cp, device))
    perm = []
    prng = np.random.default_rng(11)
    for j, f in enumerate(feats):
        drops = []
        for _ in range(perm_repeats):
            xq, cq = xp.copy(), (cp.copy() if cp is not None else None)
            if j < n_num:
                xq[:, j] = xq[prng.permutation(len(xq)), j]
            else:
                cq[:, j - n_num] = cq[prng.permutation(len(cq)), j - n_num]
            drops.append(auc0 - roc_auc_score(yp, predict_torch(model0, xq, cq, device)))
        perm.append(dict(feature=f, auc_drop_mean=float(np.mean(drops)), auc_drop_std=float(np.std(drops))))
    perm = pd.DataFrame(perm).sort_values("auc_drop_mean", ascending=False)
    perm.to_csv(os.path.join(out, "permutation_importance.csv"), index=False)
    m = imp.set_index("feature").join(perm.set_index("feature"))
    summary["shap_vs_permutation"] = dict(spearman=float(spearmanr(m.mean_abs_shap, m.auc_drop_mean)[0]),
                                          top10_overlap=len(set(imp.feature[:10]) & set(perm.feature[:10])))
    log(f"  SHAP vs permutation: {summary['shap_vs_permutation']}")

    # ---------------- error analysis: SHAP by outcome ----------------
    p_all = predict_torch(model0, xte, cte, device)
    pred = p_all >= thr0
    groups = {"TP": (yte == 1) & pred, "TN": (yte == 0) & ~pred, "FP": (yte == 0) & pred, "FN": (yte == 1) & ~pred}
    oc_rows, oc_counts = [], {}
    orng = np.random.default_rng(5)
    for g, mask in groups.items():
        pool = np.where(mask)[0]; oc_counts[g] = int(len(pool))
        if len(pool) == 0:
            continue
        gi = orng.choice(pool, size=min(n_outcome, len(pool)), replace=False)
        phi_g, *_ = shap_values(model0, n_num, n_cat, (sub(xtr, bg_idx), sub(ctr, bg_idx)),
                                (sub(xte, gi), sub(cte, gi)), device, nsamples, s0)
        top1 = np.argmax(np.abs(phi_g), axis=1)
        row = {"group": g, "n_total": len(pool), "n_explained": len(gi)}
        for f in imp.feature[:10]:
            row[f"meanSHAP_{f}"] = float(phi_g[:, feats.index(f)].mean())
        row["top1_feature_most_common"] = feats[np.bincount(top1, minlength=len(feats)).argmax()]
        row["top1_share"] = float(np.bincount(top1, minlength=len(feats)).max() / len(gi))
        oc_rows.append(row)
    oc = pd.DataFrame(oc_rows); oc.to_csv(os.path.join(out, "shap_by_outcome.csv"), index=False)
    summary["outcome_counts_seed_first"] = oc_counts
    hm = oc.set_index("group")[[c for c in oc.columns if c.startswith("meanSHAP_")]]
    hm.columns = [c[9:] for c in hm.columns]
    plt.figure(figsize=(8, 2.8))
    v = np.abs(hm.values).max()
    plt.imshow(hm.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    plt.xticks(range(hm.shape[1]), hm.columns, rotation=45, ha="right"); plt.yticks(range(hm.shape[0]), hm.index)
    plt.colorbar(label="mean SHAP (attack margin)"); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_shap_by_outcome.png"), dpi=300); plt.close()
    log(f"  outcome analysis: {oc[['group', 'n_total', 'top1_feature_most_common', 'top1_share']].to_dict('records')}")

    summary["seeds"] = {str(k): v for k, v in summary["seeds"].items()}
    save_json(summary, os.path.join(out, "summary.json"))
    log(f"explanations written to {out}")
