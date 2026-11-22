"""v3: the gate release rule (release_gate / flag release=on), on scripted W1.

Why it exists. In E8v2 (W4.ipi, 7B checker) a post stayed held after a correct recovery in 3 of 5 cells: a step the
recovery reused (its read hashes unchanged, so the fault never reached it) kept a flag from the checker, and the gate
holds an intent while any step of its backward cone is flagged. `StaleFlag` below is that situation with the scripted
model: a false positive on the atoms of meeting 1, a step pair that the injection in research.meeting.0 does not touch."""
import unittest

from agent_rewind import audit as AU
from agent_rewind.engine import execute
from agent_rewind.llm import CachedLLM, ScriptedLLM
from agent_rewind.workloads import booking
from agent_rewind.world import World

from tests.test_trust import K, UNTRUSTED, clean_trace, fault, wf


class StaleFlag(AU.RuleChecker):
    """Rule checker plus a false positive on every atom that cites a source under `bad_prefix`."""

    def __init__(self, bad_prefix) -> None:
        super().__init__()
        self.bad = bad_prefix

    def check_atom(self, sources, atom):
        s, t = super().check_atom(sources, atom)
        return (-8.0 if any(p.startswith(self.bad) for p in sources) else s), t


def auditor(trace, bad=None):
    chk = StaleFlag(bad) if bad else AU.RuleChecker()
    return AU.Auditor(chk, "m3t+m4+gate", threshold=-1.0, tau_inj=-1.0, trace=trace, w_irr=1e6, untrusted=UNTRUSTED)


def faulty_run(bad=None):
    w, world = wf(), World(booking.meetings_kb(K))
    run = execute(w, world, CachedLLM(ScriptedLLM()), overrides=fault(w), effects="diff+outbox",
                  untrusted=UNTRUSTED, audit=auditor(clean_trace(), bad))
    return w, world, run


def recover(w, world, faulty, bad=None, release=False, dirty=("research.meeting.0",)):
    return execute(w, world, CachedLLM(ScriptedLLM()), prior=faulty, dirty=set(dirty), policy="selective",
                   untrusted=UNTRUSTED, audit=auditor(faulty, bad), release_gate=release)


BAD = booking.m_path(1, "topic")      # meeting 1's agenda: read by research.meeting.1 and plan.1, not by the injected meeting 0


class Release(unittest.TestCase):
    def test_reproduction_a_flag_on_a_reused_step_keeps_the_intent_held(self):
        w, world, faulty = faulty_run(BAD)
        self.assertTrue(any(n.startswith("plan.1/") for n in faulty.report["audit"]["flagged"]))
        rec = recover(w, world, faulty, BAD)
        self.assertNotIn("plan.1", rec.report["rerun"])                   # reused: its reads did not change
        self.assertIn("execute.email.1", rec.report["held_steps"])        # held for a flag the fault did not cause
        self.assertEqual(len(world.outbox), K - 1)                        # the other two emails left at once
        self.assertFalse(booking.meetings_ok(world, K))

    def test_release_rule_lets_it_go_and_nothing_unauthorized_leaves(self):
        w, world, faulty = faulty_run(BAD)
        rec = recover(w, world, faulty, BAD, release=True)
        self.assertEqual(rec.report["held"], [])
        self.assertEqual(len(world.outbox), K)
        self.assertEqual(world.unauthorized(), 0)
        self.assertEqual(rec.report["effects"]["leaked"], 0)
        self.assertTrue(booking.meetings_ok(world, K))
        self.assertTrue(any(n.startswith("plan.1/") for n in rec.report["audit"]["flagged"]))   # still reported

    def test_clean_recovery_is_released_with_or_without_the_rule(self):
        for release in (False, True):
            w, world, faulty = faulty_run()
            rec = recover(w, world, faulty, release=release)
            self.assertEqual(rec.report["held"], [])
            self.assertEqual(len(world.outbox), K)
            self.assertEqual(world.unauthorized(), 0)

    def test_a_flag_on_a_rerun_step_still_holds(self):
        w, world, faulty = faulty_run()
        rec = recover(w, world, faulty, release=True, dirty=("execute.email.0",))   # wrong origin: the injected plan stays
        self.assertEqual(rec.report["held"], ["email.send"])
        self.assertEqual(world.unauthorized(), 0)

    def test_a_flag_on_a_rerun_step_that_is_still_unsupported_holds_too(self):
        w, world, faulty = faulty_run(booking.m_room(0))                     # a false positive on a step that is rerun
        rec = recover(w, world, faulty, booking.m_room(0), release=True)
        self.assertIn("plan.0", rec.report["rerun"])
        self.assertIn("execute.email.0", rec.report["held_steps"])
        self.assertEqual(world.unauthorized(), 0)

    def test_default_off_is_identical(self):
        reports = []
        for kw in ({}, {"release_gate": False}):
            w, world, faulty = faulty_run(BAD)
            rec = execute(w, world, CachedLLM(ScriptedLLM()), prior=faulty, dirty={"research.meeting.0"},
                          policy="selective", untrusted=UNTRUSTED, audit=auditor(faulty, BAD), **kw)
            r = dict(rec.report)
            r.pop("latency_s"), r.pop("world_ms")
            reports.append((r, world.outbox))
        self.assertEqual(reports[0], reports[1])

    def test_first_run_ignores_the_flag(self):
        w, world = wf(), World(booking.meetings_kb(K))
        run = execute(w, world, CachedLLM(ScriptedLLM()), overrides=fault(w), effects="diff+outbox",
                      untrusted=UNTRUSTED, audit=auditor(clean_trace()), release_gate=True)
        self.assertEqual(run.report["held"], ["email.send"])
        self.assertEqual(world.unauthorized(), 0)

if __name__ == "__main__":
    unittest.main()
