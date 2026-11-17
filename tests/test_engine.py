"""The four cases of method-design 4.3, on small hand-built workflows.

Each case builds a clean reference, a faulty run (a fault replaces what one step writes),
and recovers with `selective` (verifying trace) and `selective-d2` (forward propagation).
"""
import unittest
from unittest import mock

from agent_rewind import engine
from agent_rewind.engine import Step, Workflow, execute, producers
from agent_rewind.experiment import clean_reference, compare
from agent_rewind.faults import FAULTS
from agent_rewind.llm import CachedLLM, ScriptedLLM
from agent_rewind.store import Store
from agent_rewind.world import World


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

    def test_reading_an_ancestor_of_an_own_write_is_a_dependency(self):
        def s(c):
            c.write("/kv/cfg/count", 0)
            c.write("/kv/out", c.read("/kv/cfg")["x"])        # the parent also holds another step's x
        wf, llm, _, faulty, w = setup([Step("up", "x", lambda c: c.write("/kv/cfg/x", 1)), Step("s", "x", s)],
                                      {("up", "/kv/cfg/x"): 99})
        self.assertEqual(faulty.value("/kv/out"), 99)
        rec = recover(wf, llm, faulty, w, "up", "selective")
        self.assertEqual(rec.value("/kv/out"), 1)
        self.assertIn("s", rec.report["rerun"])

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


class OutOfConeReruns(unittest.TestCase):
    """Every rerun outside the path-overlap cone has a cause the dynamic cone explains."""

    def _cone(self, faulty, path):
        from agent_rewind.bench import affected_steps
        return affected_steps(faulty, path)

    def test_control_flow_change_reruns_a_reader_outside_the_static_cone(self):
        steps = [Step("a", "x", lambda c: c.write("/kv/a", 1)),
                 Step("g", "x", lambda c: c.write("/kv/g", 7), when=lambda c: c.read("/kv/a") == 1),
                 Step("r", "x", lambda c: c.write("/kv/r", c.read("/kv/g", 0) + 1))]
        wf, llm, clean, faulty, world = setup(steps, {("a", "/kv/a"): 0})
        self.assertEqual(self._cone(faulty, "/kv/a"), set())                  # static cone is empty ...
        rec = execute(wf, world, llm(), prior=faulty, dirty={"a"})
        self.assertEqual(rec.report["rerun"], ["a", "g", "r"])                # ... yet r must rerun
        d = engine.dynamic_cone(faulty, rec)
        self.assertEqual((d["cone"], d["new"]), ({"r"}, {"g"}))
        cls = engine.explain_reruns(faulty, rec, {"a"}, set())
        self.assertEqual(cls, {"a": "origin", "g": "new_step", "r": "control_flow"})
        self.assertEqual(rec.values(), clean.values())

    def test_write_set_change_is_in_the_dynamic_cone_not_the_static_one(self):
        steps = [Step("a", "x", lambda c: c.write("/kv/a", 1)),
                 Step("w", "x", lambda c: (c.write("/kv/o", 1), c.read("/kv/a") % 2 and c.write("/kv/alt", 5))),
                 Step("r", "x", lambda c: c.write("/kv/r", c.read("/kv/alt", 0)))]
        wf, llm, clean, faulty, world = setup(steps, {("a", "/kv/a"): 2})
        rec = execute(wf, world, llm(), prior=faulty, dirty={"a"})
        self.assertEqual(rec.report["rerun"], ["a", "w", "r"])
        self.assertEqual(self._cone(faulty, "/kv/a"), {"w"})
        self.assertEqual(engine.dynamic_cone(faulty, rec)["cone"], {"w", "r"})
        self.assertEqual(engine.dynamic_cone(faulty, clean)["cone"], {"w", "r"})
        self.assertEqual(engine.explain_reruns(faulty, rec, {"a"}, {"w"})["r"], "shadow")

    def test_dirty_seed_on_a_shared_key_is_the_callers_over_invalidation(self):
        steps = [Step("s1", "x", lambda c: c.write("/kv/k", 1)),
                 Step("s2", "x", lambda c: c.write("/kv/k", 2)),
                 Step("s3", "x", lambda c: c.write("/kv/o", c.read("/kv/k")))]
        wf, llm, clean, faulty, world = setup(steps, {("s1", "/kv/k"): 9})
        over = execute(wf, world, llm(), prior=faulty, dirty=producers(faulty, {"/kv/k"}))
        self.assertEqual(over.report["rerun"], ["s1", "s2"])                  # s2 rerun with nothing changed
        self.assertEqual(engine.explain_reruns(faulty, over, {"s1"}, set())["s2"], "dirty_seed")
        exact = execute(wf, world, llm(), prior=faulty, dirty={"s1"})
        self.assertEqual(exact.report["rerun"], ["s1"])                       # the engine reuses s2 and s3
        self.assertEqual(exact.values(), clean.values())


if __name__ == "__main__":
    unittest.main()


class RerunReadsCleanWorld(unittest.TestCase):
    """W3: a rerun step that reads the world must not see the faulty run's compensable effects."""

    def faulty(self):
        from agent_rewind.workloads import codefix as cf
        case = cf.make(0)
        self.addCleanup(case.close)
        f = case.faults["stale_code"]
        ov = {(f.step, case.workflow.path(f.name)): f.value,
              ("plan", cf.PLAN + "/file"): "cart.py",
              ("plan", cf.PLAN + "/old"): "return 0.0 if total >= 50 else 5.0",
              ("plan", cf.PLAN + "/new"): "return 0.0 if total > 50 else 5.0"}
        run = execute(case.workflow, case.world, CachedLLM(case.scripted()), effects="diff", overrides=ov)
        return cf, case, f, run

    def test_selective_rerun_reads_clean_world(self):
        cf, case, f, faulty = self.faulty()
        rec = execute(case.workflow, case.world, CachedLLM(case.scripted()), prior=faulty,
                      dirty={"research.code"}, policy="selective", revert_early=True)
        self.assertIn("    return 0.0 if total >= 50 else 5.0", rec.store.get(rec.branch, cf.CODE_PATH))
        self.assertTrue(cf.check(case.world))
        self.assertEqual(rec.report["effects"]["compensated"], 1)

    def test_off_by_default(self):
        cf, case, f, faulty = self.faulty()
        rec = execute(case.workflow, case.world, CachedLLM(case.scripted()), prior=faulty,
                      dirty={"research.code"}, policy="selective")
        self.assertIn("    return 0.0 if total > 50 else 5.0", rec.store.get(rec.branch, cf.CODE_PATH))

    def test_reverted_effect_is_reapplied_when_its_step_is_reused(self):
        cf, case, f, faulty = self.faulty()
        rec = execute(case.workflow, case.world, CachedLLM(case.scripted()), prior=faulty, dirty={"research.code"},
                      policy="selective", revert_early=True,
                      overrides={(f.step, case.workflow.path(f.name)): f.value})
        self.assertIn("execute.apply", rec.report["reused"])
        with open(case.world._path("cart.py")) as fh:
            self.assertIn("total > 50", fh.read())
        self.assertEqual(rec.report["effects"]["compensated"], 0)
