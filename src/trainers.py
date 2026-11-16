import copy
import time

import numpy as np

from .configs import MODELS, TRAIN_DEFAULTS


def _arrays(bundle, split, inp):
    s = bundle.splits[split]
    if inp == "ordinal":
        return s["X_ord"], None
    if inp == "mixed":
        return s["X_num"], s["X_cat"]
    return s["X_tree"], None


def n_num_for(bundle, inp):
    return len(bundle.feature_names) if inp == "ordinal" else len(bundle.num_cols)


def feature_names_for(bundle, inp):
    return list(bundle.feature_names) if inp in ("ordinal", "tree") else list(bundle.num_cols) + list(bundle.cat_cols)


def build_torch_model(name, bundle, overrides=None):
    from . import models as M
    cfg = MODELS[name]
    params = dict(cfg["params"])
    params.update({k: v for k, v in (overrides or {}).items() if k not in TRAIN_DEFAULTS})
    cls = getattr(M, cfg["cls"])
    cards = bundle.cat_cards if cfg["input"] == "mixed" else ()
    return cls(n_num_for(bundle, cfg["input"]), cards, **params)


def train_config(name, overrides=None):
    tc = dict(TRAIN_DEFAULTS)
    tc.update(MODELS[name].get("train", {}))
    tc.update({k: v for k, v in (overrides or {}).items() if k in TRAIN_DEFAULTS})
    return tc


# ------------------------------------------------------------------ torch
def predict_torch(model, x_num, x_cat, device, batch=8192):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_num), batch):
            xn = torch.as_tensor(x_num[i:i + batch], device=device)
            xc = torch.as_tensor(x_cat[i:i + batch], device=device) if x_cat is not None else None
            out.append(torch.softmax(model(xn, xc), dim=1)[:, 1].float().cpu().numpy())
    return np.concatenate(out)


def fit_torch(name, bundle, seed, device, overrides=None, log=print):
    import torch
    import torch.nn.functional as F
    from . import models as M
    from .utils import set_seed
    set_seed(seed)
    cfg = MODELS[name]
    tc = train_config(name, overrides)
    model = build_torch_model(name, bundle, overrides).to(device)
    xtr, ctr = _arrays(bundle, "train", cfg["input"])
    xva, cva = _arrays(bundle, "val", cfg["input"])
    ytr = bundle.splits["train"]["y"]; yva = bundle.splits["val"]["y"]

    Xtr = torch.as_tensor(xtr, device=device); Ytr = torch.as_tensor(ytr, device=device)
    Ctr = torch.as_tensor(ctr, device=device) if ctr is not None else None
    Xva = torch.as_tensor(xva, device=device); Yva = torch.as_tensor(yva, device=device)
    Cva = torch.as_tensor(cva, device=device) if cva is not None else None

    if tc["optimizer"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bs, n = tc["batch_size"], len(Xtr)
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    hist = []
    t0 = time.time()
    for ep in range(1, tc["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            logits = model(Xtr[idx], Ctr[idx] if Ctr is not None else None)
            loss = F.cross_entropy(logits, Ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl, correct = 0.0, 0
            for i in range(0, len(Xva), 8192):
                lg = model(Xva[i:i + 8192], Cva[i:i + 8192] if Cva is not None else None)
                vl += F.cross_entropy(lg, Yva[i:i + 8192], reduction="sum").item()
                correct += (lg.argmax(1) == Yva[i:i + 8192]).sum().item()
        vl /= len(Xva)
        hist.append(dict(epoch=ep, train_loss=tot / n, val_loss=vl, val_acc=correct / len(Xva)))
        if vl < best - 1e-5:
            best, best_epoch, bad = vl, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            log(f"    [{name} s{seed}] ep {ep:3d} train {tot / n:.4f} val {vl:.4f} acc {correct / len(Xva):.4f}")
        if bad >= tc["patience"]:
            break
    model.load_state_dict(best_state)
    info = dict(best_epoch=best_epoch, best_val_loss=best, epochs_run=len(hist),
                train_seconds=time.time() - t0, n_params=M.count_params(model), train_cfg=tc)
    return model, hist, info


# ------------------------------------------------------------------ trees
def fit_tree(name, bundle, seed, device, overrides=None, log=print):
    cfg = MODELS[name]
    params = dict(cfg["params"])
    params.update({k: v for k, v in (overrides or {}).items() if k not in TRAIN_DEFAULTS})
    xtr = bundle.splits["train"]["X_tree"]; ytr = bundle.splits["train"]["y"]
    xva = bundle.splits["val"]["X_tree"]; yva = bundle.splits["val"]["y"]
    t0 = time.time()
    if cfg["kind"] == "rf":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=seed, n_jobs=-1, **params).fit(xtr, ytr)
        extra = {}
    elif cfg["kind"] == "xgb":
        import xgboost as xgb
        dev = "cuda" if str(device).startswith("cuda") else "cpu"
        model = xgb.XGBClassifier(random_state=seed, tree_method="hist", device=dev,
                                  eval_metric="logloss", **params)
        model.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
        extra = dict(best_iteration=int(model.best_iteration))
        model.set_params(device="cpu")      # predictions are made on numpy arrays
    elif cfg["kind"] == "lgbm":
        import lightgbm as lgb
        es = params.pop("early_stopping_rounds", 100)
        model = lgb.LGBMClassifier(random_state=seed, verbose=-1, **params)
        model.fit(xtr, ytr, eval_set=[(xva, yva)], callbacks=[lgb.early_stopping(es, verbose=False)])
        extra = dict(best_iteration=int(model.best_iteration_ or params["n_estimators"]))
    else:
        raise ValueError(cfg["kind"])
    info = dict(train_seconds=time.time() - t0, params=params, **extra)
    return model, [], info


def predict_tree(model, x):
    return model.predict_proba(x)[:, 1].astype(np.float32)
