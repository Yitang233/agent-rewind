"""The four cases of method-design 4.3, on small hand-built workflows.

Each case builds a clean reference, a faulty run (a fault replaces what one step writes),
and recovers with `selective` (verifying trace) and `selective-d2` (forward propagation).
"""
import unittest
from unittest import mock

from mas_recovery import engine
from mas_recovery.engine import Step, Workflow, execute, producers
from mas_recovery.experiment import clean_reference, compare
from mas_recovery.faults import FAULTS
from mas_recovery.llm import CachedLLM, ScriptedLLM
from mas_recovery.store import Store
from mas_recovery.world import World


def setup(steps, fault=None):
    wf = Workflow(steps)
    llm = lambda: CachedLLM(ScriptedLLM({}))
    clean = execute(wf, World({}), llm())
    world = World({})
    faulty = execute(wf, world, llm(), overrides=fault or {})
    return wf, llm, clean, faulty, world


def recover(wf, llm, faulty, world, step, policy):
    return execute(wf, world, llm(), prior=faulty, dirty=producers(faulty, {step}) if step.startswith("/") else {step},
                   policy=policy)


class EngineBasics(unittest.TestCase):
    def test_clean_rerun_reuses_everything(self):
        wf, llm, clean, _, _ = setup([
            Step("a", "x", lambda c: c.write("/kv/a", 1)),
            Step("b", "x", lambda c: c.write("/kv/b", c.read("/kv/a") + 1)),
        ])
        again = execute(wf, World({}), llm(), prior=clean)
        self.assertEqual(again.report["reused"], ["a", "b"])
        self.assertEqual(again.values(), clean.values())

    def test_changed_code_reruns_the_step(self):
        wf, llm, clean, _, _ = setup([Step("a", "x", lambda c: c.write("/kv/a", 1))])
        wf2 = Workflow([Step("a", "x", lambda c: c.write("/kv/a", 2))])
        self.assertEqual(execute(wf2, World({}), llm(), prior=clean).report["rerun"], ["a"])

    def test_restart_and_rollback(self):
        steps = [Step(n, "x", (lambda n: lambda c: c.write(f"/kv/{n}", n))(n)) for n in "abc"]
        wf, llm, clean, _, w = setup(steps)
        self.assertEqual(execute(wf, w, llm(), prior=clean, policy="restart").report["rerun"], ["a", "b", "c"])
        r = execute(wf, w, llm(), prior=clean, dirty={"b"}, policy="rollback")
        self.assertEqual((r.report["reused"], r.report["rerun"]), (["a"], ["b", "c"]))


class Case1WriteSetChange(unittest.TestCase):
    """plan stops writing /kv/hold after recovery; book read it."""

    def steps(self):
        def prep(c): c.write("/kv/flag", "ok")
        def plan(c):
            if c.read("/kv/flag") == "bad":
                c.write("/kv/hold", "h")
            c.write("/msg/planner/plan/room", "B")
        def book(c): c.write("/kv/booking", {"hold": c.read("/kv/hold")})
        return [Step("prep", "x", prep), Step("plan", "x", plan), Step("book", "x", book)]

    def test_d2_keeps_the_stale_hold_selective_does_not(self):
        wf, llm, clean, faulty, world = setup(self.steps(), {("prep", "/kv/flag"): "bad"})
        self.assertEqual(faulty.value("/kv/booking"), {"hold": "h"})
        d2 = recover(wf, llm, faulty, world, "prep", "selective-d2")
        sel = recover(wf, llm, faulty, setup(self.steps(), {("prep", "/kv/flag"): "bad"})[4], "prep", "selective")
        self.assertEqual(sel.values(), clean.values())
        self.assertEqual(sel.report["rerun"], ["prep", "plan", "book"])
        self.assertNotEqual(d2.values(), clean.values())
        self.assertEqual(d2.value("/kv/booking"), {"hold": "h"})
        self.assertEqual(d2.report["reused"], ["book"])


class Case2ShadowedWrite(unittest.TestCase):
    """s1 writes /kv/x, s2 overwrites it, s3 reads s2's version."""

    def steps(self):
        return [Step("s1", "x", lambda c: c.write("/kv/x", "good")),
                Step("s2", "x", lambda c: c.write("/kv/x", "final")),
                Step("s3", "x", lambda c: c.write("/kv/out", c.read("/kv/x") + "!"))]

    def test_selective_reuses_s3_d2_invalidates_it(self):
        fault = {("s1", "/kv/x"): "bad"}
        wf, llm, clean, faulty, world = setup(self.steps(), fault)
        sel = recover(wf, llm, faulty, world, "s1", "selective")
        wf, llm, clean, faulty, world = setup(self.steps(), fault)
        d2 = recover(wf, llm, faulty, world, "s1", "selective-d2")
        self.assertEqual(sel.values(), clean.values())
        self.assertEqual(d2.values(), clean.values())
        self.assertEqual(sel.report["rerun"], ["s1"])
        self.assertEqual(d2.report["rerun"], ["s1", "s3"])


class Case3BranchFlip(unittest.TestCase):
    """The `when` predicate selects `email` instead of `escalate` after recovery."""

    def steps(self):
        def book(room):
            def fn(c):
                c.tool("calendar.book", room=room)
                c.write("/kv/action", room)
            return fn
        return [Step("decide", "x", lambda c: c.write("/kv/severity", "low")),
                Step("escalate", "x", book("A"), when=lambda c: c.read("/kv/severity") == "high"),
                Step("email", "x", book("B"), when=lambda c: c.read("/kv/severity") != "high")]

    def test_orphan_reconciled_and_new_branch_runs(self):
        fault = {("decide", "/kv/severity"): "high"}
        wf, llm, clean, _, _ = setup(self.steps())
        out = lambda w: w.outcome()["rooms"]
        w1 = World({})
        f1 = execute(wf, w1, llm(), overrides=fault)
        self.assertEqual(out(w1), ["A"])
        sel = execute(wf, w1, llm(), prior=f1, dirty={"decide"})
        self.assertEqual(sel.orphans, ["escalate"])
        self.assertEqual(sel.report["rerun"], ["decide", "email"])
        self.assertEqual(sel.values(), clean.values())
        self.assertEqual(out(w1), ["B"])                    # escalate's booking compensated
        self.assertEqual(sel.report["compensated"], 1)

        w2 = World({})
        f2 = execute(wf, w2, llm(), overrides=fault)
        d2 = execute(wf, w2, llm(), prior=f2, dirty={"decide"}, policy="selective-d2")
        self.assertNotEqual(d2.values(), clean.values())    # escalate reran, email never ran
        self.assertEqual(d2.value("/kv/action"), "A")
        self.assertEqual(out(w2), ["A"])


class Case4ReadOrder(unittest.TestCase):
    """use reads /kv/mode, then /kv/a or /kv/b. Verification stops at the first mismatch."""

    def steps(self):
        def prep(c):
            c.write("/kv/mode", "b")
            c.write("/kv/a", "A1")
            c.write("/kv/b", "B1")
        def use(c): c.write("/kv/out", c.read("/kv/a" if c.read("/kv/mode") == "a" else "/kv/b"))
        return [Step("prep", "x", prep), Step("use", "x", use)]

    def test_never_checks_a_path_the_new_run_would_not_read(self):
        class Spy(Store):
            asked: list = []
            on = False
            def hash(self, branch, path, at=None):
                if Spy.on:
                    Spy.asked.append(path)
                return super().hash(branch, path, at)

        real = engine.verified
        def watched(old, store, branch):
            Spy.on = True
            try:
                return real(old, store, branch)
            finally:
                Spy.on = False
        wf = Workflow(self.steps())
        mk = lambda: CachedLLM(ScriptedLLM({}))
        clean = execute(wf, World({}), mk())
        faulty = execute(wf, World({}), mk(), overrides={("prep", "/kv/mode"): "a"}, store=Spy())
        self.assertEqual(faulty.value("/kv/out"), "A1")
        with mock.patch.object(engine, "verified", watched):
            rec = execute(wf, World({}), mk(), prior=faulty, dirty={"prep"})
        self.assertEqual(rec.values(), clean.values())
        # verifying `use`: /kv/mode differs, so /kv/a (read by the old run) is never hashed
        self.assertEqual(Spy.asked, ["/kv/mode"])
        self.assertEqual(rec.report["rerun"], ["prep", "use"])
        self.assertEqual([p for p, _ in rec.records[-1].reads], ["/kv/mode", "/kv/b"])


class BookingRegression(unittest.TestCase):
    def test_selective_matches_old_fact_unit(self):
        rows = {r["method"]: r for r in compare(FAULTS["wrong_capacity"], clean_reference())}
        s = rows["selective/fact"]
        self.assertEqual((s["rerun"], s["reused"], s["llm_calls"], s["tokens"], s["irreversible_redone"]),
                         (3, 3, 4, 73, 0))
        self.assertTrue(s["correct"] and s["values_match"])
        self.assertEqual(rows["selective-d2/fact"]["rerun"], 3)


if __name__ == "__main__":
    unittest.main()
