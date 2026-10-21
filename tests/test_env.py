import json
import os
import unittest

ENV = os.path.join(os.path.dirname(__file__), "..", "runs", "env.json")


class TestEnv(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(ENV):
            self.skipTest("runs/env.json missing")
        self.env = json.load(open(ENV))

    def test_keys(self):
        need = {"path", "sha256", "prefill_tps", "decode_tps", "verify_ms_64",
                "equiv_rate", "kv_crop_ok", "peak_rss_mb"}
        for name, m in self.env["models"].items():
            self.assertTrue(need <= set(m), name)

    def test_small_model_primitives(self):
        m = self.env["models"]["1.5b"]
        if not os.path.exists(m["path"]):
            self.skipTest("model not present")
        self.assertGreaterEqual(m["equiv_rate"], 0.95)
        self.assertTrue(m["kv_crop_ok"])


if __name__ == "__main__":
    unittest.main()
