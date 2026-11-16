import json, os, random
import numpy as np


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _clean_keys(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _clean_keys(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_keys(v) for v in o]
    return o


def save_json(obj, path):
    obj = _clean_keys(obj)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
