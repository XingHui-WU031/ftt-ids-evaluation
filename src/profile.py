"""E13: parameter count and inference cost (no 'lightweight' claim; numbers only)."""
import os
import time

import numpy as np
import pandas as pd

from .configs import MODELS, GROUPS
from .runner import tuned_overrides
from .trainers import build_torch_model, _arrays, n_num_for


def _time(fn, reps):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def run_profile(bundle, results_dir, device, log=print, models=None):
    import torch
    rows = []
    for name in models or GROUPS["main"]:
        cfg = MODELS[name]
        if cfg["kind"] != "torch":
            continue
        model = build_torch_model(name, bundle, tuned_overrides(results_dir, bundle.name, name)).eval()
        x, c = _arrays(bundle, "test", cfg["input"])
        row = {"model": name, "params": sum(p.numel() for p in model.parameters())}
        for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            m = model.to(dev)
            for bs, reps in ((1, 300), (1024, 30)):
                xn = torch.as_tensor(x[:bs], device=dev); xc = torch.as_tensor(c[:bs], device=dev) if c is not None else None

                def f():
                    with torch.no_grad():
                        m(xn, xc)
                    if dev == "cuda":
                        torch.cuda.synchronize()
                t = _time(f, reps)
                row[f"{dev}_bs{bs}_ms"] = 1000 * t
                row[f"{dev}_bs{bs}_samples_per_s"] = bs / t
        rows.append(row); log(f"  profile {row}")
    torch.set_num_threads(torch.get_num_threads())
    out = os.path.join(results_dir, "tables", bundle.name); os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_profile.csv"), index=False)
