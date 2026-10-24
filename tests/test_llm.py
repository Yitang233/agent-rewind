import json
import os
import tempfile
import unittest

from mas_recovery import atoms as A

ENV = os.path.join(os.path.dirname(__file__), "..", "runs", "env.json")


def small_model():
    try:
        m = json.load(open(ENV))["models"]["1.5b"]
    except (OSError, KeyError):
        return None
    return m if os.path.exists(m["path"]) else None


@unittest.skipUnless(small_model(), "1.5b model not present")
class LlamaCppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mas_recovery.llm import LlamaCppLLM
        m = small_model()
        cls.dir = tempfile.TemporaryDirectory()
        cls.llm = LlamaCppLLM(m["path"], model_sha=m["sha256"], cache_path=os.path.join(cls.dir.name, "c.sqlite"),
                              n_ctx=2048)

    @classmethod
    def tearDownClass(cls):
        del cls.llm
        cls.dir.cleanup()

    INP = {"groups": A.groups_to_json([
        A.Group("Rooms", {"/m/capacity.A": 8, "/m/capacity.B": 16, "/m/capacity.C": 30}),
        A.Group("Team", {"/kv/team/size": 12}),
        A.Group("Meeting 0", {"/m/guests": 0, "/m/topic": "Q4 roadmap"})])}

    def test_grammar_output_parses_with_stats_and_cache(self):
        llm = self.llm
        out = llm.complete("planner", "plan_meeting", self.INP)
        self.assertIn(out["room"], "ABC")
        self.assertEqual(len(llm.last_atoms), 3)
        st = llm.last_stats
        self.assertFalse(st["cached_call"])
        self.assertEqual(st["prompt"], st["cached_prefix"] + st["prefill"])
        self.assertGreater(st["decoded"], 10)
        self.assertEqual(len(llm.last_conf["atoms"]), 3)
        self.assertLessEqual(llm.last_conf["atoms"][0]["min"], llm.last_conf["atoms"][0]["mean"] + 1e-9)
        self.assertLess(llm.last_conf["atoms"][0]["min"], -0.001)         # real logprobs, not a constant
        first = (out, llm.last_text)
        again = llm.complete("planner", "plan_meeting", self.INP)       # served from SQLite
        self.assertTrue(llm.last_stats["cached_call"])
        self.assertEqual(llm.last_stats["decoded"], 0)
        self.assertEqual((again, llm.last_text), first)

    def test_shared_prefix_is_reused(self):
        llm = self.llm
        llm.complete("planner", "plan_meeting", self.INP)
        other = json.loads(json.dumps(self.INP))
        other["groups"][2]["paths"]["/m/topic"] = "Budget review"
        llm.complete("planner", "plan_meeting", other)
        self.assertGreater(llm.last_stats["cached_prefix"], 100)

    def test_no_cache_switch(self):
        from mas_recovery.llm import LlamaCppLLM
        self.assertIsNone(LlamaCppLLM.__init__.__defaults__[0])           # cache_path defaults to off


if __name__ == "__main__":
    unittest.main()
