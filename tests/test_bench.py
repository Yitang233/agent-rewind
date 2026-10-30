import json
import os
import tempfile
import unittest

from agent_rewind import bench

CFG = {
    "name": "t", "experiments": [
        {"exp": "E1", "workloads": ["w1", "w3"], "faults": ["clean", "*"], "methods": ["none"], "models": ["scripted"], "repeats": 2},
        {"exp": "E2", "workloads": [{"name": "w2", "params": {"n": 12, "layers": 3}, "grid": {"position": [0.2, 0.8]}}],
         "faults": ["auto"], "methods": ["rollback", "selective"], "models": ["scripted"], "repeats": 2},
        {"exp": "E8", "workloads": [{"name": "w3", "faults": ["wrong_issue"]}],
         "variants": [{"label": "a", "method": "restart", "flags": {"style": "cot"}},
                      {"label": "b", "method": "selective", "flags": {"effects": "diff", "audit": "m3+m4"}}],
         "models": ["1.5b", "7b"], "repeats": 1},
    ]}


class BenchTest(unittest.TestCase):
    def test_expand_counts_and_keys(self):
        cells = bench.expand(CFG)
        by = {}
        for c in cells:
            by[c["exp"]] = by.get(c["exp"], 0) + 1
        self.assertEqual(by["E1"], ((1 + 3) + (1 + 3)) * 2)          # (clean + 3 faults) per workload, 2 repeats
        self.assertEqual(by["E2"], 2 * 2 * 2)
        self.assertEqual(by["E8"], 4)
        self.assertEqual(len({bench.cell_key(c) for c in cells}), len(cells))
        models = [c["model"] for c in cells]
        self.assertEqual(models, sorted(models, key=["scripted", "1.5b", "7b"].index))     # one pass per model
        self.assertTrue(all(c["method"] == "none" for c in cells if c["fault"] == "clean"))

    def test_dry_run_collapses_models_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "c.json")
            json.dump(CFG, open(cfg, "w"))
            out = os.path.join(d, "out")
            self.assertEqual(bench.main(["--config", cfg, "--dry-run", "--out", out]), 0)
            rows = [json.loads(x) for x in open(os.path.join(out, "cells.jsonl"))]
            self.assertTrue(all(r["spec"]["model"] == "scripted" and "error" not in r for r in rows))
            n = len(rows)
            prog = json.load(open(os.path.join(out, "progress.json")))
            self.assertEqual((prog["done"], prog["total"]), (n, n))
            bench.main(["--config", cfg, "--dry-run", "--out", out])                   # nothing left to do
            self.assertEqual(len(open(os.path.join(out, "cells.jsonl")).read().splitlines()), n)
            rec = next(r for r in rows if r["spec"]["exp"] == "E2")
            self.assertTrue(rec["recovery"]["values_match"])
            e8 = [r for r in rows if r["spec"]["exp"] == "E8"]
            self.assertEqual({r["spec"]["label"] for r in e8}, {"a", "b"})
            self.assertTrue(all(r["recovery"]["ok_recovered"] for r in e8))

    def test_estimate_uses_measured_rows(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(CFG, experiments=[CFG["experiments"][2]])
            cells = bench.expand(cfg)
            path = os.path.join(d, "m.jsonl")
            with open(path, "w") as f:
                for c in cells:
                    f.write(json.dumps({"spec": c, "key": bench.cell_key(c), "wall_s": 3600.0}) + "\n")
            est = bench.estimate(cfg, path)
            self.assertEqual(est["hours"], 4.0)
            self.assertEqual(est["unestimated_cells"], 0)

    def test_max_hours_stops(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "c.json")
            json.dump(CFG, open(cfg, "w"))
            out = os.path.join(d, "out")
            bench.main(["--config", cfg, "--dry-run", "--out", out, "--max-hours", "0.0000001"])
            self.assertLess(len(open(os.path.join(out, "cells.jsonl")).read().splitlines()), 5)


if __name__ == "__main__":
    unittest.main()
