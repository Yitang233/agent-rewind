"""Measurement hygiene: v1 keys unchanged, live-cost flags, call hashes, reuse check, validator."""
import copy
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest

from agent_rewind import bench
from agent_rewind.engine import execute
from agent_rewind.llm import CachedLLM
from agent_rewind.workloads import tamas
from agent_rewind.world import World

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load_check_cells():
    spec = importlib.util.spec_from_file_location("check_cells", os.path.join(ROOT, "scripts", "check_cells.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CC = _load_check_cells()

# sha256 over the sorted cell keys of each v1 config, taken before the hygiene flags changed bench.py
# phase1 and phase1_baselines grew 1563 -> 1808 and 112 -> 152 when origin/main added the timeout and planner
# faults (PR #5); the config files are unchanged. full and s6_measure still expand to their v1 keys.
V1_KEYS = {"full": (540, "f5b0296f6b0dcb7a"), "phase1": (1808, "8d60f3c0ff48c1b7"),
           "s6_measure": (47, "b615fe56be599bf8"), "phase1_baselines": (152, "def5eaf37fe789a6")}

CFG = {"name": "t", "experiments": [
    {"exp": "E5v2", "workloads": ["w1"], "faults": ["wrong_capacity"], "methods": ["selective"],
     "flags": {"replay": ["off", "resync"], "cache": ["off"], "reuse_check": ["on"]}, "models": ["scripted"], "repeats": 1},
    {"exp": "E5v2", "workloads": ["w3"], "faults": ["wrong_issue"], "methods": ["selective"],
     "flags": {"replay": ["off", "resync"], "cache": ["off"], "reuse_check": ["on"]}, "models": ["scripted"], "repeats": 1},
    {"exp": "E2v2", "workloads": [{"name": "w2", "params": {"n": 24, "layers": 4}}], "faults": ["auto"],
     "methods": ["restart", "selective"], "models": ["scripted"], "repeats": 1}]}


def run_dry(cfg, d):
    p = os.path.join(d, "c.json")
    json.dump(cfg, open(p, "w"))
    out = os.path.join(d, "out")
    bench.main(["--config", p, "--dry-run", "--out", out])
    return [json.loads(x) for x in open(os.path.join(out, "cells.jsonl"))]


class KeysTest(unittest.TestCase):
    @unittest.skipUnless(tamas.available(), "TAMAS not cloned at ~/src/TAMAS (the W4 cells need its scenarios)")
    def test_v1_configs_expand_to_the_same_keys(self):
        for name, (n, h) in V1_KEYS.items():
            cfg = json.load(open(os.path.join(ROOT, "bench", "configs", f"{name}.json")))
            keys = sorted(bench.cell_key(c) for c in bench.expand(cfg))
            self.assertEqual(len(keys), n, name)
            self.assertEqual(hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16], h, name)

    def test_new_flags_do_not_appear_unless_configured(self):
        cells = bench.expand(CFG)
        w1 = [c for c in cells if c["workload"] == "w1"][0]
        self.assertEqual(w1["flags"]["cache"], "off")
        old = bench.expand({"name": "o", "experiments": [CFG["experiments"][2]]})
        self.assertTrue(all(not ({"cache", "reuse_check", "audit_p"} & set(c["flags"])) for c in old))

    def test_bad_flag_values_are_rejected(self):
        bad = copy.deepcopy(CFG)
        bad["experiments"][0]["flags"]["cache"] = ["maybe"]
        with self.assertRaises(ValueError):
            bench.expand(bad)


class PiecesTest(unittest.TestCase):
    def test_uniform_audit_gets_p_and_others_do_not(self):
        self.assertEqual(bench._audit_args("uniform", True, p=0.25)["p"], 0.25)
        self.assertEqual(bench._audit_args("uniform", True)["p"], 1.0)
        self.assertNotIn("p", bench._audit_args("m3", True, p=0.25))

    def test_uniform_p_changes_audits(self):
        from agent_rewind.audit import Auditor, RuleChecker
        case = bench.build_case("w1", 0, {})
        try:
            ref = execute(case.workflow, case.world, CachedLLM(case.scripted()))
        finally:
            case.close()
        n = {}
        for p in (1.0, 0.25):
            c = bench.build_case("w1", 0, {})
            run = execute(c.workflow, c.world, CachedLLM(c.scripted()),
                          audit=Auditor(RuleChecker(), "uniform", **bench._audit_args("uniform", True, None, 0, p)))
            n[p] = Auditor.summary(run)["audits_run"]
            c.close()
        self.assertGreater(n[1.0], n[0.25])
        self.assertTrue(ref.records)

    def test_duplicates_count_exact_key_repeats_of_irreversible_calls_only(self):
        w = World({"team.members": []})
        for _ in range(3):
            w.call("email.send", "k1", to=["a"], body="x")
        w.call("email.send", "k2", to=["a"], body="x")
        w.call("email.send", None, to=["a"], body="x")        # no key: not countable
        w.call("doc.set", "d", doc="d", value=1)
        w.call("doc.set", "d", doc="d", value=1)               # idempotent: not irreversible
        self.assertEqual(w.duplicates_irreversible(), 2)

    def test_recording_llm_dedupe_and_hash(self):
        class Echo:
            def __init__(self):
                self.n = 0

            def complete(self, role, task, inputs):
                self.n += 1
                return {"v": inputs["x"]}

        e = Echo()
        r = bench.RecordingLLM(e, dedupe=True)
        r.cur_step = "s"
        r.complete("a", "t", {"x": 1})
        r.complete("a", "t", {"x": 1})
        self.assertEqual((e.n, r.hits), (1, 1))
        e2 = Echo()
        live = bench.RecordingLLM(e2, dedupe=False)
        live.cur_step = "s"
        live.complete("a", "t", {"x": 1})
        live.complete("a", "t", {"x": 1})
        self.assertEqual((e2.n, live.hits), (2, 0))
        self.assertEqual([c["call_index"] for c in live.log], [0, 1])
        self.assertEqual(live.log[0]["sha256"], live.log[1]["sha256"])
        self.assertEqual(len(live.log[0]["sha256"]), 64)

    def test_metered_cutoff_counts_hashes(self):
        c = bench.build_case("w1", 0, {})
        try:
            mc, m = bench.MeteredCutoff(None), bench.Meter()
            execute(c.workflow, c.world, CachedLLM(c.scripted()), cutoff=mc, meter=m)
        finally:
            c.close()
        self.assertGreater(mc.n, 0)
        self.assertGreater(m.reads, 0)
        bk = bench._bookkeeping(m, mc)
        self.assertGreater(bk["journal_bytes"], 0)
        self.assertLessEqual(bk["ns_hash_outside"], bk["ns_hash"])


class DryRowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.rows = run_dry(CFG, cls.dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_rows_carry_hygiene_fields(self):
        self.assertTrue(all("error" not in r for r in self.rows))
        for r in self.rows:
            for ph in ("faulty", "recovery"):
                self.assertIn("cache_hits", r[ph])
                self.assertIn("bookkeeping", r[ph])
                self.assertGreater(r[ph]["bookkeeping"]["n_read"], 0)
            self.assertIn("duplicates", r)
        w1 = [r for r in self.rows if r["spec"]["workload"] == "w1"]
        for r in w1:
            rec = r["recovery"]
            self.assertTrue(rec["values_match"])
            self.assertEqual(r["clean_ref"]["cache_hits"], 0)
            self.assertTrue(rec["calls"] and set(rec["calls"][0]) == {"step", "call_index", "sha256", "n_tokens"})
            rc = rec["reuse_check"]
            self.assertEqual(rc["sampled"], min(2, rc["candidates"]))
            self.assertEqual(rc["equal"], rc["sampled"])

    def test_w2_unauthorized_is_null_and_values_match_kept(self):
        w2 = [r for r in self.rows if r["spec"]["workload"] == "w2"]
        self.assertTrue(w2)
        self.assertTrue(all(r["unauthorized"] is None for r in w2))
        self.assertTrue(all(r["recovery"]["values_match"] is True for r in w2))

    def test_replay_pair_has_identical_call_hashes_in_scripted_runs(self):
        by = {r["spec"]["flags"]["replay"]: r for r in self.rows if r["spec"]["workload"] == "w1"}
        self.assertEqual([c["sha256"] for c in by["off"]["recovery"]["calls"]],
                         [c["sha256"] for c in by["resync"]["recovery"]["calls"]])

    def test_validator_accepts_and_rejects(self):
        e5 = [r for r in self.rows if r["spec"]["exp"] == "E5v2"]
        self.assertEqual(CC.check(e5, "E5v2"), [])
        self.assertEqual(CC.check([r for r in self.rows if r["spec"]["exp"] == "E2v2"], "E2v2"), [])
        # an error row
        self.assertTrue(CC.check(e5 + [{"spec": {}, "key": "k", "error": "boom"}], "E5v2"))
        # a missing pair
        half = [r for r in e5 if r["spec"]["flags"]["replay"] == "off"]
        self.assertTrue(any("pair" in b for b in CC.check(half, "E5v2")))
        # a cache hit in a measured phase of a model cell with cache=off
        r = copy.deepcopy(e5[0])
        r["spec"]["model"] = "7b"
        r["recovery"]["cache_hits"] = 2
        self.assertTrue(any("cache_hits=2" in b for b in CC.check([r], "E5v2")))
        # a model cell without cache=off
        r = copy.deepcopy(e5[0])
        r["spec"]["model"] = "7b"
        r["spec"]["flags"]["cache"] = "on"
        self.assertTrue(any("without flag cache=off" in b for b in CC.check([r], "E5v2")))


if __name__ == "__main__":
    unittest.main()
