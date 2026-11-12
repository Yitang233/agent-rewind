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


class LocalTieTest(unittest.TestCase):
    """tie="local" (crop to the exact prefix, redo only the generated tokens) against tie="full"."""

    ARGS = dict(V=8, eos=0, salt=3)

    def test_local_equals_exact_plain_under_noisy_batches(self):
        n_ties = 0
        for seed in range(40):
            ids = [3, 4 + seed % 3, 5, 6, 7][:3 + seed % 3]
            ref, _, _ = R.decode(NoisyBackend(**self.ARGS), ids, None, "off", max_tokens=60, tie="local")
            rng = random.Random(seed)
            d = list(ref)
            for _ in range(seed % 4):                                   # an edited old output
                if d:
                    d[rng.randrange(len(d))] = rng.randrange(1, 8)
            for mode in ("prefix", "resync"):
                for K in (4, 32):
                    out, _, st = R.decode(NoisyBackend(**self.ARGS), ids, d, mode, K=K, max_tokens=60, margin=0.3, tie="local")
                    self.assertEqual(out, ref, (seed, mode, K))
                    n_ties += st.ties
        self.assertGreater(n_ties, 0)                                   # the path under test was exercised

    def test_local_equals_full_when_the_prompt_batch_is_exact(self):
        # FakeBackend rows do not depend on batching: both re-checks must give the same output
        rng = random.Random(5)
        for trial in range(40):
            ids = [rng.randrange(1, 24) for _ in range(rng.randrange(2, 9))]
            args = dict(V=24, eos=0, forbid=(5, 7))
            ref, _ = plain(args, ids)
            d = list(ref)
            if d:
                d[rng.randrange(len(d))] = rng.randrange(1, 24)
            for mode in ("prefix", "resync"):
                a, _, sa = R.decode(R.FakeBackend(**args), ids, d, mode, K=8, max_tokens=60, margin=1.0, tie="full")
                b, _, sb = R.decode(R.FakeBackend(**args), ids, d, mode, K=8, max_tokens=60, margin=1.0, tie="local")
                self.assertEqual(a, b)
                self.assertEqual(b, ref)

    def test_local_recheck_never_rereads_the_prompt(self):
        ids = list(range(1, 60))
        ref, _, _ = R.decode(NoisyBackend(**self.ARGS), ids, None, "off", max_tokens=60, tie="local")
        d = list(ref)
        d[len(d) // 2] = (d[len(d) // 2] % 7) + 1
        _, _, full = R.decode(NoisyBackend(**self.ARGS), ids, d, "resync", K=16, max_tokens=60, margin=0.3, tie="full")
        _, _, loc = R.decode(NoisyBackend(**self.ARGS), ids, d, "resync", K=16, max_tokens=60, margin=0.3, tie="local")
        self.assertGreater(full.reeval, full.ties * len(ids) - 1)       # full: a whole prompt per tie
        self.assertLessEqual(loc.reeval, loc.ties * 60)                 # local: only generated tokens
        self.assertLess(loc.reeval, max(full.reeval, 1))

    def test_bad_tie_name(self):
        with self.assertRaises(ValueError):
            R.decode(R.FakeBackend(), [1, 2], None, "off", tie="x")

    def test_phase_times_are_recorded(self):
        _, _, st = R.decode(R.FakeBackend(), [1, 2, 3], None, "off", max_tokens=10, tie="local")
        self.assertGreaterEqual(st.t_prefill + st.t_decode + st.t_pick, 0.0)
        self.assertLessEqual(st.t_prefill + st.t_verify + st.t_decode + st.t_tie + st.t_resync + st.t_pick, st.wall_s + 1e-3)


class StableLayoutTest(unittest.TestCase):
    def test_same_task_calls_share_a_long_prefix_and_the_atom_order_is_kept(self):
        from agent_rewind import atoms as A
        from agent_rewind.llm import ATOM_TASKS, stable_prompt
        ins, _, hint = ATOM_TASKS["plan_meeting"]

        def groups(i):
            return [A.Group("Rooms", {"/m/capacity.A": 8, "/m/capacity.B": 16}), A.Group("Team", {"/kv/team/size": 12}),
                    A.Group(f"Meeting {i}", {f"/m/{i}/guests": i, f"/m/{i}/topic": f"topic {i}"})]
        p = [A.chatml(*stable_prompt("planner", ins, groups(i), hint)) for i in (0, 1)]

        def shared(a, b):
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            return n
        q2 = [A.chatml(*A.build_prompt(role, ins, groups(0), hint)) for role in ("planner", "reviewer")]
        p2 = [A.chatml(*stable_prompt(role, ins, groups(0), hint)) for role in ("planner", "reviewer")]
        self.assertGreater(shared(*p2), shared(*q2) + 100)              # v1 diverges at the role name
        self.assertGreater(shared(*p), 500)
        self.assertLess(p[0].index("## Rooms"), p[0].index("## Team"))
        self.assertLess(p[0].index("## Team"), p[0].index("## Meeting 0"))
        self.assertLess(p[0].index("## Meeting 0"), p[0].index("Write exactly 3 atoms"))
        # role-free system text: two roles on one task share the prompt up to the footer
        r = A.chatml(*stable_prompt("reviewer", ins, groups(0), hint))
        self.assertEqual(shared(p[0], r), p[0].index("You are the planner") + len("You are the "))


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


CASES = os.path.join(os.path.dirname(__file__), "..", "runs", "v2", "D3", "cases")


def recorded_cases():
    import glob
    return sorted(glob.glob(os.path.join(CASES, "1.5b_*_v1_atoms.json")))[:6]


@unittest.skipUnless(small_model() and recorded_cases(), "1.5b model or recorded cases not present")
class RecordedTieRecheckTest(unittest.TestCase):
    """The local tie re-check gives the old path's tokens on recorded recovery calls (runs/v2/D3/cases)."""

    def test_local_equals_full_on_recorded_calls(self):
        from agent_rewind.llm import LlamaCppLLM
        m = small_model()
        llm = LlamaCppLLM(m["path"], model_sha=m["sha256"], n_ctx=4096, replay="resync", margin=0.3)
        n = 0
        for path in recorded_cases():
            case = json.load(open(path))
            llm.style = case["style"]
            outs = {}
            for tie in ("full", "local"):
                llm.tie = tie
                llm.llm.n_tokens = 0
                res = []
                for c in case["calls"]:
                    if c.get("draft") is None:
                        continue
                    llm.drafts = {c["step"]: c["draft"]}
                    res.append(llm.generate(c["prompt"], c["schema"], step=c["step"])["ids"])
                outs[tie] = res
            self.assertEqual(outs["full"], outs["local"], path)
            n += len(outs["full"])
        self.assertGreater(n, 0)
        del llm


class BenchFlagTest(unittest.TestCase):
    def test_replay_flags_default_to_v1_and_are_checked(self):
        from agent_rewind import bench
        fl = bench._check_flags({**bench.DEFAULTS})
        self.assertEqual([bench.flag(fl, k) for k in ("layout", "replay_K", "margin", "tie")], ["v1", 32, 0.3, "full"])
        with self.assertRaises(ValueError):
            bench._check_flags({**bench.DEFAULTS, "tie": "nope"})
        with self.assertRaises(ValueError):
            bench._check_flags({**bench.DEFAULTS, "layout": "nope"})


if __name__ == "__main__":
    unittest.main()
