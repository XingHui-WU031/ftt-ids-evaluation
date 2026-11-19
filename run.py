"""Single entry point.

  python run.py check                                   # environment + GPU check
  python run.py smoke                                   # 2-minute end-to-end test on a tiny subset
  python run.py tune  --dataset unsw                    # validation grid on seed 42 (E1 fairness)
  python run.py train --dataset unsw --group main       # E1/E2 (5 seeds)
  python run.py train --dataset unsw --group ablation   # E5
  python run.py train --dataset unsw_noshortcut --group shortcut   # E10
  python run.py aggregate --dataset unsw                # tables + significance (E3, E4, E6)
  python run.py explain --dataset unsw --model proposed # SHAP + stability + faithfulness + permutation (E7-E9, E11)
  python run.py profile --dataset unsw                  # params + latency (E13)
"""
import argparse
import itertools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.configs import MODELS, GROUPS, TUNE_ORDER          # noqa: E402
from src.data import load_dataset                           # noqa: E402
from src.utils import save_json, load_json                  # noqa: E402

DEFAULT_DATA = {"unsw": os.path.join(HERE, "..", "数据集"),
                "unsw_noshortcut": os.path.join(HERE, "..", "数据集"),
                "unsw_nottl": os.path.join(HERE, "..", "数据集"),
                "toniot": os.path.join(HERE, "..", "数据集", "TON_IoT")}
SEEDS = [42, 43, 44, 45, 46]


def data_dir_for(name):
    if name in DEFAULT_DATA:
        return DEFAULT_DATA[name]
    return DEFAULT_DATA["toniot"] if name.startswith("toniot") else DEFAULT_DATA["unsw"]


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n"); self.f.flush()


def get_device(force_cpu=False):
    import torch
    return torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")


def load(args, log):
    data_dir = args.data_dir or data_dir_for(args.dataset)
    b = load_dataset(args.dataset, data_dir, dedup=not args.no_dedup, log1p=not args.no_log1p)
    log(f"dataset {b.name}: {b.info}")
    return b


def subsample(b, n=4000):
    import numpy as np
    rng = np.random.default_rng(0)
    for s in b.splits:
        m = len(b.splits[s]["y"]); idx = rng.choice(m, size=min(n, m), replace=False)
        b.splits[s] = {k: (v[idx] if v is not None else None) for k, v in b.splits[s].items()}
    return b


def cmd_check(args, log):
    import platform
    import torch
    log(f"python {platform.python_version()} | torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)} | torch CUDA {torch.version.cuda}")
    for mod in ("sklearn", "pandas", "numpy", "shap", "xgboost", "lightgbm", "scipy", "matplotlib"):
        try:
            m = __import__(mod); log(f"{mod} {getattr(m, '__version__', '?')}")
        except ImportError:
            log(f"{mod} MISSING")


def cmd_smoke(args, log):
    from src.runner import run_one
    from src.explain import run_explain
    args.dataset = "unsw"
    b = subsample(load(args, log))
    dev = get_device(args.cpu)
    res = os.path.join(HERE, "results_smoke")
    names = [n for n in MODELS if MODELS[n]["kind"] == "torch"] + ["rf", "xgboost"]
    for name in names:
        try:
            run_one(name, b, 42, dev, res, overrides={"max_epochs": 2} if MODELS[name]["kind"] == "torch" else {"n_estimators": 20}, log=log)
        except ImportError as e:
            log(f"  {name}: skipped ({e})")
    run_explain(b, "proposed", [42], dev, res, log, n_background=50, n_explain=100, nsamples=50, perm_n=500)
    log("SMOKE TEST PASSED")


def cmd_tune(args, log):
    from src.runner import run_one
    b = load(args, log); dev = get_device(args.cpu)
    best_path = os.path.join(args.results, "tuning", b.name, "best.json")
    best = load_json(best_path) if os.path.exists(best_path) else {}
    for name in (args.models or TUNE_ORDER):
        grid = MODELS[name].get("tune")
        if not grid or name in best:
            continue
        keys = list(grid)
        scores = []
        for vals in itertools.product(*[grid[k] for k in keys]):
            ov = dict(zip(keys, vals))
            tag = os.path.join("tuning", b.name, name + "_" + "_".join(f"{k}{v}" for k, v in ov.items()))
            rep = run_one(name, b, 42, dev, args.results, overrides=ov, tag=tag, log=log, save_model=False)
            # selection criterion uses validation data only (log-loss proxy: val ROC-AUC, tie-break val F1)
            scores.append((rep["val"]["roc_auc"], rep["val"]["f1"], ov))
        scores.sort(key=lambda t: (t[0], t[1]), reverse=True)
        best[name] = scores[0][2]
        log(f"TUNED {name}: {best[name]} (val AUC {scores[0][0]:.5f})")
        save_json(best, best_path)


def cmd_train(args, log):
    from src.runner import run_one, tuned_overrides
    b = load(args, log); dev = get_device(args.cpu)
    tune_ds = "unsw" if b.name.startswith("unsw") else ("toniot" if b.name.startswith("toniot") else b.name)
    names = args.models or GROUPS[args.group]
    for seed in (args.seeds or SEEDS):
        for name in names:
            # hyperparameters always come from the original validation grids (results/tuning/<dataset>/best.json)
            ov = tuned_overrides(args.tune_results, tune_ds, name)
            try:
                run_one(name, b, seed, dev, args.results, overrides=ov, log=log)
            except ImportError as e:
                log(f"  {name}: skipped ({e})")


def cmd_host_strict(args, log):
    from src.host_protocol import run_strict
    run_strict(args, log)


def cmd_aggregate(args, log):
    from src.aggregate import aggregate
    aggregate(args.results, args.dataset, log, reference=args.model)


def cmd_sensitivity(args, log):
    from src.sensitivity import sensitivity
    sensitivity(args.results, args.dataset, args.data_dir or data_dir_for(args.dataset), log)


def cmd_explain(args, log):
    from src.explain import run_explain
    b = load(args, log)
    run_explain(b, args.model, args.seeds or SEEDS, get_device(args.cpu), args.results, log)


def cmd_explain_r3(args, log):
    from src.explain_r3 import run_explain_r3
    b = load(args, log)
    run_explain_r3(b, args.model, args.seeds or SEEDS, get_device(args.cpu), args.tune_results, args.results, log)


def cmd_search(args, log):
    from src.search import run_search
    b = load(args, log)
    run_search(b, args.models or ["xgboost", "proposed"], get_device(args.cpu), args.results, log,
               n_trials=args.trials, seeds=args.seeds or SEEDS)


def cmd_profile(args, log):
    from src.profile import run_profile
    run_profile(load(args, log), args.results, get_device(args.cpu), log, args.models)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "smoke", "tune", "train", "aggregate", "explain", "profile", "sensitivity",
                                    "explain_r3", "search", "host_strict"])
    ap.add_argument("--host-folds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--dedup-policy", choices=["within_host", "none"], default="within_host")
    ap.add_argument("--dataset", default="unsw")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--group", default="main", choices=list(GROUPS))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--model", default="proposed")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--tune-results", default=os.path.join(HERE, "results"),
                    help="results directory that holds tuning/<dataset>/best.json and the trained round-1/2 models")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--no-log1p", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    lg = Logger(os.path.join(a.results if a.cmd != "smoke" else os.path.join(HERE, "results_smoke"), "logs", f"{a.cmd}.log"))
    lg(f"=== {' '.join(sys.argv)}")
    globals()[f"cmd_{a.cmd}"](a, lg)
