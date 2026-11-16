"""Unified data pipeline shared by every model (E0).

All models use exactly the same split and the same preprocessing statistics,
which are fitted on the training partition only.
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42  # the data split is fixed; model seeds vary independently

UNSW_CAT_COLS = ["proto", "service", "state"]


@dataclass
class Bundle:
    name: str
    feature_names: list            # original feature order (used by ordinal-input models)
    num_cols: list
    cat_cols: list
    cat_cards: list                # number of known categories per categorical column (index 0 = unknown)
    # per split: dict with keys X_ord, X_num, X_cat, X_tree, y, group
    splits: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    host_test: np.ndarray = None   # source host of each test record (host-disjoint TON_IoT only)
    ref_ord: np.ndarray = None     # per-feature reference values (train mean / mode) in X_ord space
    ref_num: np.ndarray = None
    ref_cat: np.ndarray = None


def _fit_category_maps(train_df, cat_cols):
    maps = {}
    for c in cat_cols:
        cats = sorted(train_df[c].astype(str).unique())      # alphabetical, like OrdinalEncoder
        maps[c] = {v: i + 1 for i, v in enumerate(cats)}       # 0 is reserved for unknown
    return maps


def _encode_cats(df, cat_cols, maps):
    out = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        out[:, j] = df[c].astype(str).map(maps[c]).fillna(0).astype(np.int64).to_numpy()
    return out


def build_bundle(name, train_df, val_df, test_df, cat_cols, label_col="label",
                 group_col=None, drop_cols=(), log1p=False):
    drop = set(drop_cols) | {label_col}
    if group_col:
        drop.add(group_col)
    feature_names = [c for c in train_df.columns if c not in drop]
    cat_cols = [c for c in cat_cols if c in feature_names]
    num_cols = [c for c in feature_names if c not in cat_cols]

    maps = _fit_category_maps(train_df, cat_cols)
    cat_cards = [len(maps[c]) + 1 for c in cat_cols]

    def numeric(df):
        x = df[num_cols].to_numpy(dtype=np.float64)
        if log1p:
            x = np.sign(x) * np.log1p(np.abs(x))
        return x

    tr_num_raw = numeric(train_df)
    mu, sd = tr_num_raw.mean(0), tr_num_raw.std(0)
    sd[sd == 0] = 1.0

    tr_cat = _encode_cats(train_df, cat_cols, maps)
    # ordinal value = alphabetical index (unknown -> -1), identical to the thesis OrdinalEncoder
    ord_mu = (tr_cat - 1).mean(0)
    ord_sd = (tr_cat - 1).std(0)
    ord_sd[ord_sd == 0] = 1.0

    pos_num = [feature_names.index(c) for c in num_cols]
    pos_cat = [feature_names.index(c) for c in cat_cols]

    splits = {}
    for split, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        x_num = ((numeric(df) - mu) / sd).astype(np.float32)
        x_cat = _encode_cats(df, cat_cols, maps)
        x_ord = np.zeros((len(df), len(feature_names)), dtype=np.float32)
        x_ord[:, pos_num] = x_num
        x_ord[:, pos_cat] = ((x_cat - 1 - ord_mu) / ord_sd).astype(np.float32)
        # tree models: raw numeric values + ordinal codes (no scaling needed)
        x_tree = np.zeros((len(df), len(feature_names)), dtype=np.float32)
        x_tree[:, pos_num] = df[num_cols].to_numpy(dtype=np.float32)
        x_tree[:, pos_cat] = (x_cat - 1).astype(np.float32)
        splits[split] = {
            "X_ord": x_ord, "X_num": x_num, "X_cat": x_cat, "X_tree": x_tree,
            "y": df[label_col].to_numpy(dtype=np.int64),
            "group": df[group_col].astype(str).to_numpy() if group_col else None,
        }

    # reference values used for masking in faithfulness tests (train mean for numeric, mode for categorical)
    mode_idx = np.array([np.bincount(tr_cat[:, j]).argmax() for j in range(len(cat_cols))], dtype=np.int64)
    ref_num = np.zeros(len(num_cols), dtype=np.float32)
    ref_ord = np.zeros(len(feature_names), dtype=np.float32)
    ref_ord[pos_cat] = ((mode_idx - 1 - ord_mu) / ord_sd).astype(np.float32)

    info = {
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "pos_rate": {s: float(splits[s]["y"].mean()) for s in splits},
        "unknown_categories_test": {c: int((splits["test"]["X_cat"][:, j] == 0).sum()) for j, c in enumerate(cat_cols)},
        "log1p": log1p,
    }
    return Bundle(name, feature_names, num_cols, cat_cols, cat_cards, splits, info,
                  ref_ord=ref_ord, ref_num=ref_num, ref_cat=mode_idx)


def load_unsw(data_dir, dedup=True, log1p=False, drop_features=(), tag=None):
    tr = pd.read_csv(os.path.join(data_dir, "UNSW_NB15_training-set.csv"))
    te = pd.read_csv(os.path.join(data_dir, "UNSW_NB15_testing-set.csv"))
    tr = tr.drop(columns=["id"])
    te = te.drop(columns=["id"])
    n_before = len(tr)
    if dedup:
        # exact duplicates (all features + label) would otherwise land on both sides of the
        # train/validation split and inflate validation scores
        tr = tr.drop_duplicates().reset_index(drop=True)
    tr_part, val_part = train_test_split(tr, test_size=0.2, random_state=SPLIT_SEED, stratify=tr["label"])
    if tag is None:
        tag = "unsw" if not drop_features else ("unsw_nottl" if "dttl" in drop_features else "unsw_noshortcut")
    b = build_bundle(tag,
                     tr_part.reset_index(drop=True), val_part.reset_index(drop=True), te,
                     UNSW_CAT_COLS, label_col="label", group_col="attack_cat",
                     drop_cols=tuple(drop_features), log1p=log1p)
    b.info["duplicates_removed"] = n_before - len(tr)
    return b


# TON_IoT network dataset (Train_Test_Network.csv). Columns removed: identifiers/time that leak
# the capture setup, and free-text fields. Verified against the file when it is downloaded.
TON_DROP = ["ts", "src_ip", "dst_ip", "src_port", "dst_port",
            "dns_query", "ssl_subject", "ssl_issuer", "http_uri", "http_user_agent"]
TON_CAT = ["proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
           "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established", "http_trans_depth",
           "http_method", "http_version", "http_orig_mime_types", "http_resp_mime_types",
           "weird_name", "weird_addl", "weird_notice"]


def load_toniot(data_dir, dedup=True, log1p=False, filename="Train_Test_Network.csv"):
    df = pd.read_csv(os.path.join(data_dir, filename), low_memory=False, encoding="utf-8-sig")  # file has a BOM
    df = df.drop(columns=[c for c in TON_DROP if c in df.columns])
    cat_cols = [c for c in TON_CAT if c in df.columns]
    # any remaining object column is treated as categorical
    for c in df.columns:
        if c not in cat_cols + ["label", "type"] and not pd.api.types.is_numeric_dtype(df[c]):
            cat_cols.append(c)
    for c in df.columns:
        if c not in cat_cols + ["label", "type"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    n_before = len(df)
    if dedup:
        df = df.drop_duplicates().reset_index(drop=True)
    # no official split: stratified 60/20/20 on the attack type
    tr, rest = train_test_split(df, test_size=0.4, random_state=SPLIT_SEED, stratify=df["type"])
    val, te = train_test_split(rest, test_size=0.5, random_state=SPLIT_SEED, stratify=rest["type"])
    b = build_bundle("toniot", tr.reset_index(drop=True), val.reset_index(drop=True), te.reset_index(drop=True),
                     cat_cols, label_col="label", group_col="type", log1p=log1p)
    b.info["duplicates_removed"] = n_before - len(df)
    return b


def _prep_toniot(data_dir, filename):
    """Read the TON_IoT network file, drop identifiers/free text, coerce types. Returns (df, cat_cols, src_ip)."""
    df = pd.read_csv(os.path.join(data_dir, filename), low_memory=False, encoding="utf-8-sig")
    host = df["src_ip"].astype(str)
    df = df.drop(columns=[c for c in TON_DROP if c in df.columns])
    cat_cols = [c for c in TON_CAT if c in df.columns]
    for c in df.columns:
        if c not in cat_cols + ["label", "type"] and not pd.api.types.is_numeric_dtype(df[c]):
            cat_cols.append(c)
    for c in df.columns:
        if c not in cat_cols + ["label", "type"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df, cat_cols, host


def load_toniot_host(data_dir, fold, n_folds=5, dedup=True, log1p=False, filename="Train_Test_Network.csv"):
    """Host-disjoint evaluation (round 3). The source hosts (src_ip) are partitioned into n_folds groups with
    StratifiedGroupKFold; fold k is the test set, the remaining hosts are split 80/20 into training/validation
    (stratified by attack type). src_ip is used only to form the groups, never as a feature. Exact duplicates are
    removed first (first occurrence kept), so no feature vector occurs on both sides of the host split."""
    from sklearn.model_selection import StratifiedGroupKFold
    df, cat_cols, host = _prep_toniot(data_dir, filename)
    n_before = len(df)
    if dedup:
        keep = ~df.duplicated()
        df = df[keep].reset_index(drop=True); host = host[keep].reset_index(drop=True)
    sgk = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SPLIT_SEED)
    dev_idx, te_idx = list(sgk.split(df, df["label"], host))[fold]
    dev, te = df.iloc[dev_idx], df.iloc[te_idx].reset_index(drop=True)
    tr, val = train_test_split(dev, test_size=0.2, random_state=SPLIT_SEED, stratify=dev["type"])
    b = build_bundle(f"toniot_host{fold}", tr.reset_index(drop=True), val.reset_index(drop=True), te,
                     cat_cols, label_col="label", group_col="type", log1p=log1p)
    b.info["duplicates_removed"] = n_before - len(df)
    b.info["test_hosts"] = int(host.iloc[te_idx].nunique()); b.info["train_val_hosts"] = int(host.iloc[dev_idx].nunique())
    b.host_test = host.iloc[te_idx].to_numpy()
    return b


# Feature sets for the retraining-based faithfulness test (ROAR). "top" follows the five-seed mean |SHAP| ranking of
# FTT-IDS on UNSW-NB15 (results/explain/unsw/proposed/global_importance.csv); "rnd" sets were drawn once with
# numpy.random.default_rng(2026) from the 42 features and are fixed here for transparency.
UNSW_SHAP_RANK = ["ct_state_ttl", "sttl", "dttl", "service", "proto", "ct_srv_dst", "swin", "ct_srv_src",
                  "ct_dst_ltm", "dmean"]
ROAR_SETS = {
    "top5": UNSW_SHAP_RANK[:5], "top10": UNSW_SHAP_RANK[:10],
    "rnd5a": ['proto', 'sbytes', 'sinpkt', 'smean', 'ct_dst_ltm'], "rnd5b": ['sloss', 'trans_depth', 'ct_state_ttl', 'ct_dst_ltm', 'ct_flw_http_mthd'],
    "rnd10a": ['state', 'dpkts', 'dttl', 'sload', 'synack', 'ackdat', 'smean', 'ct_state_ttl', 'ct_dst_sport_ltm', 'ct_ftp_cmd'], "rnd10b": ['state', 'spkts', 'dbytes', 'sttl', 'dttl', 'dinpkt', 'swin', 'dmean', 'ct_srv_src', 'ct_dst_sport_ltm'],
}


def load_dataset(name, data_dir, **kw):
    if name == "unsw_nodedup":       # control: official training file without duplicate removal
        kw = dict(kw); kw["dedup"] = False
        return load_unsw(data_dir, tag="unsw_nodedup", **kw)
    if name.startswith("unsw_roar_"):
        return load_unsw(data_dir, drop_features=tuple(ROAR_SETS[name[len("unsw_roar_"):]]), tag=name, **kw)
    if name.startswith("toniot_host"):
        return load_toniot_host(data_dir, int(name[len("toniot_host"):]), **kw)
    if name == "unsw":
        return load_unsw(data_dir, **kw)
    if name == "unsw_noshortcut":
        return load_unsw(data_dir, drop_features=("sttl", "ct_state_ttl"), **kw)
    if name == "unsw_nottl":   # remove every TTL-derived feature
        return load_unsw(data_dir, drop_features=("sttl", "dttl", "ct_state_ttl"), **kw)
    if name == "toniot":
        return load_toniot(data_dir, **kw)
    raise ValueError(name)
