"""Host-disjoint development and evaluation without global random-split tuning.

Historical toniot_host* results are NOT overwritten. Strict runs use a separate
toniot_strict_host* namespace. Raw rows are retained across hosts; duplicates
may be removed within each source host, or retained for sensitivity analysis.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import _prep_toniot, build_bundle, SPLIT_SEED

PROTOCOL = "nested-source-host-v1"
SPLIT_REVISION = "fixed-raw-host-v2"


def load_strict_host(data_dir, fold, dedup_policy="within_host", development_only=False,
                     log1p=True, manifest_dir=None):
    if dedup_policy not in ("within_host", "none") or fold not in range(5):
        raise ValueError("Expected fold 0..4 and within_host/none deduplication")
    df, cat_cols, host = _prep_toniot(data_dir, "Train_Test_Network.csv")
    source = Path(data_dir) / "Train_Test_Network.csv"
    raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    plans_dir = Path(os.environ.get("STRICT_HOST_PLANS", Path(__file__).resolve().parents[1] / "fixed_host_plans"))
    plans_dir = plans_dir / raw_hash
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plans_dir / f"fold{fold}.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["source_sha256"] != raw_hash or plan["revision"] != SPLIT_REVISION:
            raise RuntimeError("Frozen host plan provenance mismatch")
    else:
        if host.nunique() < 10:
            raise ValueError("At least ten source hosts required")
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=SPLIT_SEED)
        dev_raw, te_raw = list(outer.split(df, df.label, host))[fold]
        inner = StratifiedGroupKFold(5, shuffle=True, random_state=SPLIT_SEED + fold + 1)
        a, b = next(inner.split(df.iloc[dev_raw], df.label.iloc[dev_raw], host.iloc[dev_raw]))
        plan = dict(revision=SPLIT_REVISION, source_sha256=raw_hash, fold=fold,
                    hosts={key: sorted(host.iloc[idx].unique().tolist()) for key, idx in
                           (("train", dev_raw[a]), ("val", dev_raw[b]), ("test", te_raw))})
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    row_id = np.arange(len(df))
    if dedup_policy == "within_host":
        keep = ~df.assign(__source_host=host).duplicated()
        row_id = row_id[keep]; df = df.loc[keep].reset_index(drop=True)
        host = host.loc[keep].reset_index(drop=True)
    tr_idx, va_idx, te_idx = [np.flatnonzero(host.isin(plan["hosts"][key])) for key in ("train", "val", "test")]
    dev_idx = np.concatenate([tr_idx, va_idx])
    if len(dev_idx) + len(te_idx) != len(df):
        raise RuntimeError("Frozen plan does not cover all rows")
    sets = [set(host.iloc[i]) for i in (tr_idx, va_idx, te_idx)]
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Host overlap across training, validation and test")
    for idx in (tr_idx, va_idx):
        if df.label.iloc[idx].nunique() != 2:
            raise ValueError("A development partition has only one class; revise the predeclared split")
    name = f"toniot_strict_host{fold}" + ("_nodedup" if dedup_policy == "none" else "")
    # Development-only bundles contain NO outer-test rows or labels in their arrays.
    test_idx = va_idx if development_only else te_idx
    bundle = build_bundle(name, df.iloc[tr_idx].reset_index(drop=True),
                          df.iloc[va_idx].reset_index(drop=True),
                          df.iloc[test_idx].reset_index(drop=True), cat_cols,
                          label_col="label", group_col="type", log1p=log1p)
    bundle.host_test = host.iloc[test_idx].to_numpy()
    manifest = pd.DataFrame({"raw_row": row_id, "source_host": host, "split": "test"})
    manifest.loc[tr_idx, "split"] = "train"; manifest.loc[va_idx, "split"] = "val"
    source = Path(data_dir) / "Train_Test_Network.csv"
    raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    split_hash = hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest()
    bundle.info.update(protocol=PROTOCOL, dedup_policy=dedup_policy,
                       source_sha256=raw_hash, split_sha256=split_hash,
                       n_hosts={s: len(v) for s, v in zip(("train", "val", "test"), sets)},
                       development_only=development_only, split_revision=SPLIT_REVISION, host_plan=plan,
                       class_counts={s: df.iloc[idx].label.value_counts().to_dict() for s, idx in (("train", tr_idx), ("val", va_idx), ("test", te_idx))})
    # Test composition/overlap is audited ONLY after model selection, never used to select a model.
    if not development_only:
        feat = [c for c in df if c not in ("label", "type")]
        keys = pd.util.hash_pandas_object(df[feat], index=False).to_numpy()
        bundle.info["feature_overlap_test_vs_development"] = int(np.isin(keys[te_idx], keys[dev_idx]).sum())
        bundle.info["attack_counts"] = {s: df.iloc[idx].type.value_counts().to_dict()
                                       for s, idx in (("train", tr_idx), ("val", va_idx), ("test", te_idx))}
        if manifest_dir:
            out = Path(manifest_dir); out.mkdir(parents=True, exist_ok=True)
            manifest.to_csv(out / f"{name}_split.csv", index=False)
    return bundle


def run_strict(args, log):
    import itertools
    from .configs import MODELS
    from .metrics import ranking, at_threshold
    from .trainers import fit_torch, fit_tree, predict_torch, predict_tree, _arrays
    from .runner import run_one
    from .utils import save_json, load_json
    from run import get_device, data_dir_for

    data_dir = args.data_dir or data_dir_for("toniot")
    models = args.models or ["rf", "xgboost", "mlp", "ft_transformer", "proposed"]
    device = "cpu" if args.cpu or all(MODELS[m]["kind"] != "torch" for m in models) else get_device(False)
    for fold in args.host_folds:
        dev = load_strict_host(data_dir, fold, args.dedup_policy, development_only=True)
        path = Path(args.results) / "tuning" / dev.name / "best.json"
        provenance = {k: dev.info[k] for k in ("protocol", "source_sha256", "split_sha256", "dedup_policy")}
        saved = load_json(path) if path.exists() else {"provenance": provenance, "models": {}}
        if saved.get("provenance") != provenance:
            raise RuntimeError("Tuning provenance changed; use a fresh output directory")
        for name in models:
            if name in saved["models"]:
                continue
            grid = MODELS[name].get("tune")
            if not grid:
                raise ValueError(f"No declared tuning grid for {name}")
            trials = []
            for vals in itertools.product(*grid.values()):
                ov = dict(zip(grid, vals)); cfg = MODELS[name]
                # Fit and predict validation only. No run_one/test evaluation in tuning.
                if cfg["kind"] == "torch":
                    model, _, _ = fit_torch(name, dev, 42, device, ov, log)
                    xv, cv = _arrays(dev, "val", cfg["input"])
                    pv = predict_torch(model, xv, cv, device)
                else:
                    model, _, _ = fit_tree(name, dev, 42, device, ov, log)
                    pv = predict_tree(model, dev.splits["val"]["X_tree"])
                y = dev.splits["val"]["y"]
                trials.append({"overrides": ov, **ranking(y, pv), "f1": at_threshold(y, pv, .5)["f1"]})
                del model
            trials.sort(key=lambda t: (t["roc_auc"], t["f1"]), reverse=True)
            saved["models"][name] = trials[0]["overrides"]
            save_json(trials, str(path.parent / f"{name}_trials.json"))
            save_json(saved, str(path))
        full = load_strict_host(data_dir, fold, args.dedup_policy,
                                manifest_dir=Path(args.results) / "protocol")
        save_json(full.info, str(Path(args.results) / "protocol" / f"{full.name}_audit.json"))
        for name in models:
            for seed in args.seeds or [42, 43, 44]:
                run_one(name, full, seed, device, args.results, overrides=saved["models"][name], log=log)
