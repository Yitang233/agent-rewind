import copy
import unittest

from mas_recovery import audit as AU
from mas_recovery.engine import execute
from mas_recovery.llm import CachedLLM, ScriptedLLM
from mas_recovery.recovery import recover
from mas_recovery.store import canon
from mas_recovery.workloads import booking
from mas_recovery.world import World

K = 3


def clean(policy="m3+cache", **kw):
    world = World(booking.meetings_kb(K))
    wf = booking.meetings_workflow(K)
    aud = AU.Auditor(AU.RuleChecker(), policy, threshold=-1.0, **kw)
    run = execute(wf, world, CachedLLM(ScriptedLLM()), audit=aud)
    return wf, world, run, aud


def faulty_recover(policy, prior_audit=True, **kw):
    wf, world, run, aud = clean(policy, **kw)
    fault = {("research.capacity", wf.path("fact.capacity.A")): 12}
    world2 = World(booking.meetings_kb(K))
    faulty = execute(wf, world2, CachedLLM(ScriptedLLM()), overrides=fault, audit=AU.Auditor(AU.RuleChecker(), policy, **kw))
    return wf, world2, faulty


def atom_keys(run):
    return {n: v["key"] for n, v in run.audits.items()}


class AuditTest(unittest.TestCase):
    def test_clean_run_audits_every_atom_and_passes(self):
        _, _, run, _ = clean()
        rep = run.report["audit"]
        self.assertEqual(rep["audits_run"], 3 * 3 + 3 * 2)            # 3 atoms per plan, 2 per email
        self.assertEqual(rep["flagged"], [])
        self.assertIn("/audit/plan.0/_atoms/0/ok", run.store.leaves(run.branch))

    def test_reruns_iff_cited_source_hash_changes(self):
        wf, world, faulty = faulty_recover("m3+cache")
        # clean reference for the same workflow: which atoms differ (text or cited values)?
        _, _, ref, _ = clean()

        def ident(run):
            out = {}
            for r in run.records:
                for c in r.calls:
                    vals = {p: v for g in c["groups"] for p, v in g["paths"].items()}
                    for k, a in enumerate(c["atoms"]):
                        out[(r.step, a["text"], canon({p: vals[p] for p in a["cites"]}))] = 1
            return out
        # recover to the clean state; fixing the fault changes exactly the atoms whose identity differs
        w3 = copy.deepcopy(world)
        aud = AU.Auditor(AU.RuleChecker(), "m3+cache")
        new = execute(wf, w3, CachedLLM(ScriptedLLM()), prior=faulty, dirty={"research.capacity"}, audit=aud)
        before, after = ident(faulty), ident(new)
        expected = {k for k in after if k not in before}
        reran = {n for n, v in new.audits.items() if v["audited"] and not v["reused"]}
        self.assertEqual(len(reran), len(expected))
        for n, v in new.audits.items():
            old = faulty.audits[n]
            self.assertEqual(v["reused"], v["key"] == old["key"], n)
            if v["reused"]:
                self.assertEqual(v["tokens"], 0)
        self.assertTrue(reran)                                     # the fix did change something
        self.assertTrue(any(v["reused"] for v in new.audits.values()))   # and most stayed

    def test_nothing_changed_nothing_reaudited(self):
        wf, world, run, _ = clean()
        aud = AU.Auditor(AU.RuleChecker(), "m3+cache")
        new = execute(wf, World(booking.meetings_kb(K)), CachedLLM(ScriptedLLM()), prior=run, dirty=(), audit=aud)
        self.assertEqual(new.report["audit"]["audits_run"], 0)
        self.assertEqual(aud.checker.calls, 0)

    def test_no_cache_policy_reaudits_everything(self):
        wf, world, run, _ = clean("m3")
        aud = AU.Auditor(AU.RuleChecker(), "m3")
        new = execute(wf, World(booking.meetings_kb(K)), CachedLLM(ScriptedLLM()), prior=run, dirty=(), audit=aud)
        self.assertEqual(new.report["audit"]["audits_run"], 15)

    def test_atom_text_change_alone_changes_the_key(self):
        wf, world, run, _ = clean()
        edits = {"plan.0": lambda atoms, ans: ([dict(atoms[0], text=atoms[0]["text"] + " (checked)")] + atoms[1:], ans)}
        aud = AU.Auditor(AU.RuleChecker(), "m3+cache")
        new = execute(wf, World(booking.meetings_kb(K)), AU.FaultyLLM(CachedLLM(ScriptedLLM()), edits), prior=run,
                      dirty={"plan.0"}, audit=aud)
        self.assertEqual({n for n, v in new.audits.items() if not v["reused"] and v["audited"]}, {"plan.0/0"})

    def test_source_change_reaudits_only_atoms_citing_it(self):
        wf, world, faulty = faulty_recover("m3+cache")
        new = execute(wf, World(booking.meetings_kb(K)), CachedLLM(ScriptedLLM()), prior=faulty,
                      dirty={"research.capacity"}, audit=AU.Auditor(AU.RuleChecker(), "m3+cache"))
        again = {n for n, v in new.audits.items() if v["audited"] and not v["reused"]}
        # team size and meeting atoms of the planner cite nothing that changed
        self.assertFalse(any(n.endswith(("/1", "/2")) and n.startswith("plan.") for n in again))
        self.assertTrue(all(n.startswith(("plan.", "execute.email.")) for n in again))

    def test_hallucinated_atom_flagged_and_injection_flagged(self):
        wf = booking.meetings_workflow(K)
        edits = {"plan.0": lambda a, ans: ([dict(a[0], text="Rooms: capacity.A=20, B=16, C=30")] + a[1:], ans),
                 "execute.email.1": lambda a, ans: ([dict(a[0], text=a[0]["text"] + " and eve@evil.example")] + a[1:], ans)}
        run = execute(wf, World(booking.meetings_kb(K)), AU.FaultyLLM(CachedLLM(ScriptedLLM()), edits),
                      audit=AU.Auditor(AU.RuleChecker(), "m3"))
        self.assertEqual(run.report["audit"]["flagged"], ["execute.email.1/0", "plan.0/0"])

    def test_always_verify_checks_whole_output_every_execution(self):
        wf, world, run, _ = clean("always")
        self.assertEqual(run.report["audit"]["audits_run"], 6)
        new = execute(wf, World(booking.meetings_kb(K)), CachedLLM(ScriptedLLM()), prior=run, dirty=(),
                      audit=AU.Auditor(AU.RuleChecker(), "always"))
        self.assertEqual(new.report["audit"]["audits_run"], 6)      # reused steps are re-checked too

    def test_none_and_default_engine_unchanged(self):
        wf = booking.meetings_workflow(K)
        a = execute(wf, World(booking.meetings_kb(K)), CachedLLM(ScriptedLLM()))
        b = execute(wf, World(booking.meetings_kb(K)), CachedLLM(ScriptedLLM()), audit=AU.Auditor(AU.RuleChecker(), "none"))
        self.assertEqual(a.values(), b.values())
        self.assertNotIn("audit", a.report)

    def test_uniform_budget(self):
        wf = booking.meetings_workflow(K)
        run = execute(wf, World(booking.meetings_kb(K)), CachedLLM(ScriptedLLM()),
                      audit=AU.Auditor(AU.RuleChecker(), "uniform", p=0.0))
        self.assertEqual(run.report["audit"]["audits_run"], 0)


class PlacementTest(unittest.TestCase):
    def test_rule(self):
        self.assertTrue(AU.should_audit(0.5, 1000, False, 300, 0))
        self.assertFalse(AU.should_audit(0.01, 1000, False, 300, 0))
        self.assertTrue(AU.should_audit(0.01, 1000, True, 300, 300))         # irreversible gate
        self.assertFalse(AU.should_audit(0.01, 1000, False, 300, 300))

    def test_risk_from_confidence(self):
        self.assertAlmostEqual(AU.risk_of({"atoms": [{"mean": 0.0}]}, 0), 0.0)
        self.assertEqual(AU.risk_of({}, 0), 1.0)

    def test_cone_cost_and_irreversible_gate_from_trace(self):
        wf, world, run, _ = clean()
        cones = AU.cone_costs(run)
        self.assertTrue(cones["plan.0"][1])                         # plan -> book / email (irreversible)
        self.assertGreater(cones["plan.0"][0], 0)
        self.assertGreater(cones["plan.0"][0], cones["execute.email.0"][0] - 1)
        self.assertFalse(cones["research.team"][0] == 0)

    def test_m4_skips_confident_low_cone_atoms(self):
        wf, world, run, _ = clean()
        aud = AU.Auditor(AU.RuleChecker(), "m3+m4", trace=run, w_irr=0.0)
        new = execute(wf, World(booking.meetings_kb(K)), CachedLLM(ScriptedLLM()), audit=aud)
        # ScriptedLLM has no confidence: risk 1.0, so every atom with a nonzero cone is audited
        self.assertEqual(new.report["audit"]["skipped"], 0)

        class Conf(CachedLLM):
            def complete(s, *a):
                r = super().complete(*a)
                s.last_conf = {"atoms": [{"mean": -0.001}] * 3}
                return r
        new = execute(wf, World(booking.meetings_kb(K)), Conf(ScriptedLLM()), audit=aud)
        self.assertGreater(new.report["audit"]["skipped"], 0)
        gated = AU.Auditor(AU.RuleChecker(), "m3+m4", trace=run, w_irr=10 ** 6)
        new = execute(wf, World(booking.meetings_kb(K)), Conf(ScriptedLLM()), audit=gated)
        self.assertEqual(new.report["audit"]["skipped"], 0)         # all cones reach an email

    def test_key_changed_forces_reaudit_under_m4(self):
        wf, world, run, _ = clean()

        class Conf(CachedLLM):
            def complete(s, *a):
                r = super().complete(*a)
                s.last_conf = {"atoms": [{"mean": -0.001}] * 3}
                return r
        # first run audits everything (trace-less cone fallback gives risk*nominal > cost? force with w_irr)
        first = execute(wf, World(booking.meetings_kb(K)), Conf(ScriptedLLM()),
                        audit=AU.Auditor(AU.RuleChecker(), "m3+m4", trace=run, w_irr=10 ** 6))
        wf2, w2, faulty = faulty_recover("m3")
        aud = AU.Auditor(AU.RuleChecker(), "m3+m4", trace=run, w_irr=0.0)
        new = execute(wf, World(booking.meetings_kb(K)), Conf(ScriptedLLM()), prior=first, dirty={"research.capacity"},
                      audit=aud)
        for n, v in new.audits.items():
            if not v["reused"] and first.audits[n]["audited"] is True and v["key"] != first.audits[n]["key"]:
                self.assertTrue(v["audited"], n)


class CalibrationTest(unittest.TestCase):
    def test_threshold_fpr(self):
        s = [-0.1 * i for i in range(100)]
        tau = AU.calibrate(s, 0.05)
        self.assertLessEqual(sum(x < tau for x in s) / len(s), 0.05)


if __name__ == "__main__":
    unittest.main()
