"""Round-3 explanation checks for the trained models of round 1 (no retraining).

R3-a  Expected gradients (GradientExplainer) with more samples per explanation: does the additivity error shrink,
      and does the global ranking change?
R3-b  Integrated Gradients (own implementation, training-mean baseline in the embedded input space) as a second
      attribution method: completeness error, agreement with the expected-gradients ranking, stability across
      seeds, masking faithfulness, and the attribution profile of false positives vs. true positives.
Uses the same background / explanation records as run_explain (indices.npz).
"""
import os

import numpy as np

from .configs import MODELS
from .explain import (_stratified, _load_model, _MarginWrapper, _to_z, _aggregate_to_features, shap_values,
                      _masked_predict)
from .trainers import feature_names_for, _arrays, predict_torch
from .utils import save_json


def integrated_gradients(model, n_num, n_cat, bg, ex, device, steps=64, batch=8):
    """IG of the attack margin w.r.t. z = [numeric features, categorical embeddings]; baseline = mean background z."""
    import torch
    w = _MarginWrapper(model, n_num, n_cat).to(device).eval()
    zb, ze = _to_z(model, *bg, device), _to_z(model, *ex, device)
    base = zb.mean(0, keepdim=True)
    alphas = ((torch.arange(steps, device=device, dtype=ze.dtype) + 0.5) / steps).view(steps, 1, 1)   # midpoint rule
    out = []
    for i in range(0, len(ze), batch):
        z = ze[i:i + batch]
        path = (base.unsqueeze(0) + alphas * (z - base).unsqueeze(0)).reshape(-1, z.size(1)).detach().requires_grad_(True)
        g = torch.autograd.grad(w(path).sum(), path)[0].reshape(steps, len(z), -1).mean(0)
        out.append(((z - base) * g).detach().cpu().numpy())
    attr = _aggregate_to_features(np.concatenate(out), n_num, n_cat)
    with torch.no_grad():
        f_base = float(w(base)); margin = w(ze).squeeze(1).cpu().numpy()
    err = np.abs(f_base + attr.sum(1) - margin)
    return attr, f_base, margin, err


def _rank_stats(a, b):
    from scipy.stats import spearmanr, kendalltau
    ta, tb = set(np.argsort(-a)[:10]), set(np.argsort(-b)[:10])
    return dict(spearman=float(spearmanr(a, b)[0]), kendall=float(kendalltau(a, b)[0]),
                top10_jaccard=len(ta & tb) / len(ta | tb))


def run_explain_r3(bundle, name, seeds, device, model_results, out_results, log=print, n_background=500,
                   n_explain=2000, eg_nsamples=2000, eg_seeds=(42, 43), ks=(1, 2, 3, 5, 10)):
    from sklearn.metrics import roc_auc_score
    out = os.path.join(out_results, "explain_r3", bundle.name, name)
    os.makedirs(out, exist_ok=True)
    inp = MODELS[name]["input"]
    feats = feature_names_for(bundle, inp)
    n_num = len(bundle.feature_names) if inp == "ordinal" else len(bundle.num_cols)
    n_cat = 0 if inp == "ordinal" else len(bundle.cat_cols)
    ref_num = bundle.ref_ord if inp == "ordinal" else bundle.ref_num
    xtr, ctr = _arrays(bundle, "train", inp); xte, cte = _arrays(bundle, "test", inp)
    ytr, yte = bundle.splits["train"]["y"], bundle.splits["test"]["y"]
    old = os.path.join(model_results, "explain", bundle.name, name)
    if os.path.exists(os.path.join(old, "indices.npz")):
        ix = np.load(os.path.join(old, "indices.npz")); bg_idx, ex_idx = ix["bg"], ix["ex"]
    else:
        rng = np.random.default_rng(2026)
        bg_idx = _stratified(ytr, n_background, rng); ex_idx = _stratified(yte, n_explain, rng)
    sub = lambda x, i: x[i] if x is not None else None
    bg = (sub(xtr, bg_idx), sub(ctr, bg_idx)); ex = (sub(xte, ex_idx), sub(cte, ex_idx))
    res = {"model": name, "features": feats, "ig": {}, "eg": {}}
    rng = np.random.default_rng(7)

    # ---------------- R3-b integrated gradients, all seeds
    ig_global = {}
    for s in seeds:
        model, _ = _load_model(name, bundle, s, device, model_results)
        attr, f_base, margin, err = integrated_gradients(model, n_num, n_cat, bg, ex, device)
        np.savez_compressed(os.path.join(out, f"ig_seed{s}.npz"), phi=attr, base=f_base, margin=margin)
        g = np.abs(attr).mean(0); ig_global[s] = g
        r = dict(additivity_mae=float(err.mean()), additivity_rel=float(err.mean() / np.abs(margin - f_base).mean()),
                 top10=[feats[i] for i in np.argsort(-g)[:10]])
        p_old = os.path.join(old, f"shap_seed{s}.npz")
        if os.path.exists(p_old):
            eg = np.load(p_old)["phi"]
            r["vs_eg500"] = _rank_stats(g, np.abs(eg).mean(0))
            r["signed_corr_per_record"] = float(np.mean([np.corrcoef(attr[i], eg[i])[0, 1] for i in range(len(attr))]))
        # masking faithfulness with the IG ranking (same protocol as round 1)
        p0 = predict_torch(model, xte, cte, device); order = np.argsort(-g)
        fa = {"none": float(roc_auc_score(yte, p0))}
        for k in ks:
            fa[f"ig_top{k}"] = float(roc_auc_score(yte, _masked_predict(model, xte, cte, order[:k], ref_num, bundle.ref_cat, n_num, device)))
            rr = [roc_auc_score(yte, _masked_predict(model, xte, cte, rng.choice(len(feats), k, replace=False), ref_num,
                                                     bundle.ref_cat, n_num, device)) for _ in range(5)]
            fa[f"random{k}"] = float(np.mean(rr))
        r["mask_auc"] = fa
        # attribution profile by outcome at threshold 0.5 (explained records only)
        pe = p0[ex_idx]; ye = yte[ex_idx]; top = np.argsort(-g)[:10]
        prof = {}
        for gname, mk in (("TP", (pe >= .5) & (ye == 1)), ("FP", (pe >= .5) & (ye == 0)),
                          ("TN", (pe < .5) & (ye == 0)), ("FN", (pe < .5) & (ye == 1))):
            if mk.sum():
                prof[gname] = dict(n=int(mk.sum()), mean={feats[i]: float(attr[mk, i].mean()) for i in top},
                                   top1={feats[i]: float((np.argmax(attr[mk], 1) == i).mean()) for i in top})
        r["by_outcome"] = prof
        res["ig"][str(s)] = r
        log(f"  IG seed {s}: additivity rel {100 * r['additivity_rel']:.2f}% | top5 {r['top10'][:5]} | "
            f"vs EG500 {r.get('vs_eg500')}")
        save_json(res, os.path.join(out, "summary_r3.json"))
    ss = list(ig_global)
    pairs = [_rank_stats(ig_global[a], ig_global[b]) for i, a in enumerate(ss) for b in ss[i + 1:]]
    if pairs:
        res["ig_stability_across_seeds"] = {k: [float(np.mean([p[k] for p in pairs])), float(np.std([p[k] for p in pairs]))]
                                            for k in pairs[0]}
    res["ig_global_mean"] = {feats[i]: float(np.mean([ig_global[s][i] for s in ss])) for i in range(len(feats))}
    save_json(res, os.path.join(out, "summary_r3.json"))

    # ---------------- R3-a expected gradients with more samples
    for s in [x for x in eg_seeds if x in seeds]:
        model, _ = _load_model(name, bundle, s, device, model_results)
        phi, base, margin, err = shap_values(model, n_num, n_cat, bg, ex, device, eg_nsamples, s)
        np.savez_compressed(os.path.join(out, f"eg{eg_nsamples}_seed{s}.npz"), phi=phi, base=base, margin=margin)
        g = np.abs(phi).mean(0)
        r = dict(nsamples=eg_nsamples, additivity_mae=float(err.mean()),
                 additivity_rel=float(err.mean() / np.abs(margin - base).mean()),
                 vs_ig=_rank_stats(g, ig_global[s]))
        p_old = os.path.join(old, f"shap_seed{s}.npz")
        if os.path.exists(p_old):
            eg = np.load(p_old)["phi"]
            r["vs_eg500"] = _rank_stats(g, np.abs(eg).mean(0))
        res["eg"][str(s)] = r
        log(f"  EG nsamples={eg_nsamples} seed {s}: additivity rel {100 * r['additivity_rel']:.2f}% | {r}")
        save_json(res, os.path.join(out, "summary_r3.json"))
    log(f"round-3 explanation checks written to {out}")
