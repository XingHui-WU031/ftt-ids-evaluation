import tempfile
import unittest
import argparse
import json
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from src.thresholds import benign_budget_threshold
from src.host_protocol import load_strict_host, run_strict


class BudgetTests(unittest.TestCase):
    def test_saturation_rejects_all_when_needed(self):
        for dtype in (np.float32, np.float64):
            scores = np.ones(1000, dtype=dtype)
            t = benign_budget_threshold(scores)
            self.assertGreater(t, 1)
            self.assertEqual(int((scores >= t).sum()), 0)

    def test_ties_cannot_exceed_empirical_budget(self):
        rng = np.random.default_rng(4)
        for dtype in (np.float32, np.float64):
            for alpha in (0, .001, .01, .1, 1):
                for n in (1, 20, 1000):
                    scores = (rng.integers(0, 101, n)/100).astype(dtype)
                    t = benign_budget_threshold(scores, alpha)
                    self.assertLessEqual(int((scores >= t).sum()), int(np.floor(n*alpha)))

    def test_bad_inputs_fail(self):
        for scores, alpha in (([], .01), ([np.nan], .01), ([1.1], .01), ([.2], -.1)):
            with self.assertRaises(ValueError):
                benign_budget_threshold(scores, alpha)


class HostTests(unittest.TestCase):
    def test_strict_rf_smoke_ignores_global_tuning(self):
        from src.configs import MODELS
        with tempfile.TemporaryDirectory() as td:
            rows=[]
            for h in range(20):
                for j in range(10):
                    y=j%2
                    rows.append(dict(src_ip=f"host{h}",label=y,type="normal" if y==0 else "attack",
                                     proto="tcp",duration=float(y),src_bytes=j))
            pd.DataFrame(rows).to_csv(Path(td)/"Train_Test_Network.csv",index=False)
            root=Path(td)/"output";glob=root/"tuning/toniot/best.json";glob.parent.mkdir(parents=True)
            glob.write_text(json.dumps({"rf":{"min_samples_leaf":999}}))
            args=argparse.Namespace(data_dir=td,cpu=True,models=["rf"],host_folds=[0],
                                    dedup_policy="within_host",results=str(root),seeds=[42])
            small={**MODELS["rf"],"params":{**MODELS["rf"]["params"],"n_estimators":5}}
            with patch.dict(MODELS,{"rf":small}):
                run_strict(args,lambda s:None)
            best=json.loads((root/"tuning/toniot_strict_host0/best.json").read_text())
            self.assertIn(best["models"]["rf"]["min_samples_leaf"],(1,5))
            trials=json.loads((root/"tuning/toniot_strict_host0/rf_trials.json").read_text())
            self.assertTrue(all("test_auc" not in t for t in trials))
            metrics=json.loads((root/"runs/toniot_strict_host0/rf/seed42/metrics.json").read_text())
            self.assertEqual(metrics["data_protocol"]["protocol"],"nested-source-host-v1")

    def test_nested_disjoint_and_cross_host_duplicates_retained(self):
        with tempfile.TemporaryDirectory() as td:
            rows = []
            for h in range(20):
                for y in (0, 1):
                    rows.append(dict(src_ip=f"host{h}", label=y, type="normal" if y==0 else "attack",
                                     proto="tcp", duration=float(y), src_bytes=2*y))
            # Same feature vector across all hosts; only within-host exact copies removed.
            pd.DataFrame(rows + rows).to_csv(Path(td)/"Train_Test_Network.csv", index=False)
            for fold in range(5):
                b = load_strict_host(td, fold, manifest_dir=td)
                m = pd.read_csv(Path(td)/f"toniot_strict_host{fold}_split.csv")
                hosts = [set(m.loc[m.split==s, "source_host"]) for s in ("train", "val", "test")]
                self.assertFalse(hosts[0] & hosts[1] or hosts[0] & hosts[2] or hosts[1] & hosts[2])
                self.assertEqual(len(m), 40)
                self.assertGreater(b.info["feature_overlap_test_vs_development"], 0)
                dev = load_strict_host(td, fold, development_only=True)
                self.assertTrue(dev.info["development_only"])
                self.assertFalse(set(dev.host_test) & hosts[2])
                self.assertTrue(np.array_equal(dev.splits["val"]["X_ord"], dev.splits["test"]["X_ord"]))


if __name__ == "__main__":
    unittest.main()
