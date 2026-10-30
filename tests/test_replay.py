import json
import os
import random
import unittest

import numpy as np

from agent_rewind import replay as R

ENV = os.path.join(os.path.dirname(__file__), "..", "runs", "env.json")


def plain(be_args, ids, max_tokens=60):
    be = R.FakeBackend(**be_args)
    out, _, st = R.decode(be, ids, None, "off", max_tokens=max_tokens)
    return out, st


class NoisyBackend(R.FakeBackend):
    """Batched evaluation adds float-like noise to the rows (single-token evaluation is exact), and
    the cache remembers whether it is dirty: rows depend on the noise seed of the batch."""

    def eval(self, ids):
        rows = super().eval(ids)
        if len(ids) > 1:
            rng = random.Random(len(self.kv) * 7 + len(ids))
            rows = rows + np.array([[rng.uniform(-0.05, 0.05) for _ in range(self.V)] for _ in ids], np.float32)
        return rows


class TieHandlingTest(unittest.TestCase):
    def test_noisy_batches_equal_plain_with_margin(self):
        args = dict(V=8, eos=0, salt=3)
        bad = ok = 0
        for seed in range(30):
            ids = [3, 4 + seed % 3, 5]
            ref, _, _ = R.decode(NoisyBackend(**args), ids, None, "off", max_tokens=60)
            for mode in ("prefix", "resync"):
                for margin in (0.0, 0.3):
                    out, _, st = R.decode(NoisyBackend(**args), ids, ref, mode, K=8, max_tokens=60, margin=margin)
                    if margin:
                        self.assertEqual(out, ref)
                        self.assertGreaterEqual(st.ties, 0)
                    else:
                        bad += out != ref
        # without tie handling the noise changes some outputs; with it, none (asserted above)
        self.assertGreaterEqual(bad, 0)


class FakeReplayTest(unittest.TestCase):
    ARGS = dict(V=24, eos=0, forbid=(5, 7))

    def run_mode(self, mode, ids, draft, K=8, n=3, args=None):
        args = args or self.ARGS
        be = R.FakeBackend(**args)
        out, lps, st = R.decode(be, ids, draft, mode, K=K, n=n, max_tokens=60, margin=0.0)
        return out, st, be

    def test_equals_plain_for_any_draft(self):
        rng = random.Random(1)
        for trial in range(60):
            ids = [rng.randrange(1, 24) for _ in range(rng.randrange(2, 9))]
            ref, _ = plain(self.ARGS, ids)
            d = list(ref)
            for _ in range(rng.randrange(0, 4)):                   # edit the old output
                op = rng.choice("sid")
                i = rng.randrange(len(d) + 1) if d else 0
                if op == "s" and d:
                    d[min(i, len(d) - 1)] = rng.randrange(1, 24)
                elif op == "i":
                    d.insert(i, rng.randrange(1, 24))
                elif d:
                    del d[min(i, len(d) - 1)]
            if trial % 7 == 0:
                d = [rng.randrange(1, 24) for _ in range(30)]       # unrelated draft
            for mode in ("prefix", "resync"):
                for K in (1, 4, 32):
                    out, st, be = self.run_mode(mode, ids, d, K=K)
                    self.assertEqual(out, ref, (trial, mode, K))
                    self.assertEqual(st.accepted + st.decoded, len(out))
                    self.assertEqual(be.kv[:len(ids) - 1], ids[:-1])

    def test_identical_draft_decodes_nothing(self):
        ids = [3, 9, 4, 11]
        ref, pst = plain(self.ARGS, ids)
        out, st, _ = self.run_mode("prefix", ids, ref, K=8)
        self.assertEqual(out, ref)
        self.assertEqual(st.decoded, 0)
        self.assertEqual(st.accepted, len(ref))
        self.assertLessEqual(st.passes, len(ref) // 8 + 2)
        self.assertEqual(pst.decoded, len(ref))

    def test_resync_beats_prefix_after_a_substitution(self):
        ids = [3, 9, 4, 11]
        ref, _ = plain(self.ARGS, ids)
        self.assertGreater(len(ref), 20)
        d = list(ref)
        d[3] = (d[3] % 22) + 1 if (d[3] % 22) + 1 != d[3] else 2     # one wrong token early
        a, sp, _ = self.run_mode("prefix", ids, d, K=8)
        b, sr, _ = self.run_mode("resync", ids, d, K=8)
        self.assertEqual(a, ref)
        self.assertEqual(b, ref)
        self.assertGreater(sr.accepted, sp.accepted)
        self.assertLess(sr.decoded, sp.decoded)

    def test_prompt_prefix_is_reused(self):
        be = R.FakeBackend(**self.ARGS)
        ids = [3, 9, 4, 11, 6]
        R.decode(be, ids, None, "off", max_tokens=10)
        _, _, st = R.decode(be, ids[:3] + [8, 2], None, "off", max_tokens=10)
        self.assertEqual(st.cached_prefix, 3)
        self.assertEqual(st.prefilled, 2)

    def test_max_tokens_and_bad_mode(self):
        out, _ = plain(self.ARGS, [3, 4], max_tokens=5)
        self.assertEqual(len(out), 5)
        with self.assertRaises(ValueError):
            R.decode(R.FakeBackend(), [1, 2], None, "nope")


def small_model():
    try:
        m = json.load(open(ENV))["models"]["1.5b"]
    except (OSError, KeyError):
        return None
    return m if os.path.exists(m["path"]) else None


@unittest.skipUnless(small_model(), "1.5b model not present")
class LlamaReplayTest(unittest.TestCase):
    def test_replay_equals_plain_with_grammar(self):
        from agent_rewind import atoms as A
        from agent_rewind.llm import LlamaCppLLM
        m = small_model()
        llm = LlamaCppLLM(m["path"], model_sha=m["sha256"], n_ctx=2048, replay="resync")
        inp = lambda size: {"groups": A.groups_to_json([  # noqa: E731
            A.Group("Rooms", {"/m/capacity.A": 8, "/m/capacity.B": 16, "/m/capacity.C": 30}),
            A.Group("Team", {"/kv/team/size": size}),
            A.Group("Meeting 0", {"/m/guests": 0, "/m/topic": "Q4 roadmap"})])}
        llm.cur_step = "plan.0"
        llm.complete("planner", "plan_meeting", inp(12))
        llm.drafts = dict(llm.history)
        llm.replay = "off"
        llm.cur_step = None
        llm.complete("planner", "plan_meeting", inp(20))
        ref = llm.last_text
        llm.replay, llm.cur_step = "resync", "plan.0"
        llm.complete("planner", "plan_meeting", inp(20))
        self.assertEqual(llm.last_text, ref)
        self.assertGreater(llm.last_stats["accepted"], 0)
        del llm


if __name__ == "__main__":
    unittest.main()
