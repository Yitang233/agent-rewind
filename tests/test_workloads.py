import tempfile
import unittest

from agent_rewind.engine import execute, producers
from agent_rewind.llm import ATOM_TASKS, CachedLLM
from agent_rewind.workloads import codefix, tamas


def faulty_then_recover(case, fault, effects="diff"):
    wf = case.workflow
    path = wf.path(fault.name)
    faulty = execute(wf, case.world, CachedLLM(case.scripted()), overrides={(fault.step, path): fault.value}, effects=effects)
    ok_faulty = case.check(case.world)
    mark = case.mark(case.world)
    rec = execute(wf, case.world, CachedLLM(case.scripted()), prior=faulty, dirty=producers(faulty, {path}))
    return faulty, ok_faulty, rec, mark


class CodefixTest(unittest.TestCase):
    def test_clean_run_passes_for_every_variant(self):
        for v in range(5):
            c = codefix.make(v)
            try:
                self.assertFalse(c.world.run_tests()["passed"])             # the seeded bug is real
                execute(c.workflow, c.world, CachedLLM(c.scripted()))
                self.assertTrue(c.check(c.world), v)
                self.assertEqual(c.world.touched(), [codefix.CODE_FILE])
            finally:
                c.close()

    def test_faults_break_then_recovery_restores(self):
        for v in (0, 3):
            for label in ("wrong_issue", "stale_code", "inject_issue"):
                c = codefix.make(v)
                try:
                    faulty, ok_faulty, rec, _ = faulty_then_recover(c, c.faults[label])
                    self.assertFalse(ok_faulty, label)
                    self.assertTrue(c.check(c.world), (v, label))
                    self.assertIn("plan", rec.report["reused"] + rec.report["rerun"])
                finally:
                    c.close()

    def test_compensation_checks_out_and_keeps_other_edits(self):
        c = codefix.make(0)
        w = c.world
        try:
            a = w.call("fs.write", file="cart.py", old="return price * qty", new="return price * qty * 1")
            b = w.call("fs.write", file="cart.py", old="if qty > 10:", new="if qty >= 10:")
            self.assertTrue(a and b)
            w.compensate("fs.write", a)
            text = open(w.dir + "/cart.py").read()
            self.assertIn("if qty >= 10:", text)
            self.assertNotIn("* 1", text)
            w.compensate("fs.write", b)
            self.assertEqual(w.touched(), [])
        finally:
            c.close()

    def test_no_match_writes_nothing(self):
        c = codefix.make(1)
        try:
            self.assertIsNone(c.world.call("fs.write", file="cart.py", old="no such line", new="x"))
            self.assertEqual(c.world.touched(), [])
            with self.assertRaises(ValueError):
                c.world._path("../outside")
        finally:
            c.close()

    def test_task_registered_for_models(self):
        self.assertIn("plan_patch", ATOM_TASKS)


@unittest.skipUnless(tamas.available(), "TAMAS not cloned at ~/src/TAMAS")
class TamasTest(unittest.TestCase):
    def test_tools_and_scenarios_come_from_tamas(self):
        tools = tamas.load_tools()
        self.assertEqual(tools["post_on_x"]["returns"], "Posted to X successfully!")
        self.assertIn("publish_to_unverified_article", tools)
        self.assertIn("RSS", tamas.load_scenario("ipi")["user query"])
        self.assertEqual(tamas.EFFECT_CLASS["post_on_x"], "irreversible")
        self.assertEqual(tamas.EFFECT_CLASS["fetch_latest_rss_feeds"], "read")

    def test_clean_runs_pass(self):
        for sc in ("ipi", "byz"):
            for v in range(5):
                c = tamas.make(sc, v)
                execute(c.workflow, c.world, CachedLLM(c.scripted()), effects="diff")
                self.assertTrue(c.check(c.world), (sc, v))
                self.assertEqual(len(c.world.posts), len(tamas.SCENARIOS[sc]["channels"]))

    def test_faults_reach_the_outcome_and_irreversible_posts_stay(self):
        for sc, label in [("ipi", "inject_item"), ("byz", "byz_factcheck"), ("byz", "byz_distribute")]:
            c = tamas.make(sc, 1)
            faulty, ok_faulty, rec, mark = faulty_then_recover(c, c.faults[label])
            self.assertFalse(ok_faulty, label)
            self.assertTrue(c.check(c.world, mark), label)          # what recovery posted is right
            self.assertFalse(c.check(c.world), label)                # but the bad posts cannot be unsent
            self.assertGreaterEqual(rec.report["effects"]["leaked"], 1)

    def test_outbox_withholds_untrusted_posts(self):
        from agent_rewind.world import OPEN_WORLD
        c = tamas.make("ipi", 0)
        OPEN_WORLD.add("fetch_latest_rss_feeds")          # opt in: the feed is outside data, so its readers are untrusted
        try:
            run = execute(c.workflow, c.world, CachedLLM(c.scripted()), effects="diff+outbox", commit=True)
        finally:
            OPEN_WORLD.discard("fetch_latest_rss_feeds")
        self.assertEqual(c.world.posts, [])
        self.assertTrue(run.report["held"])


if __name__ == "__main__":
    unittest.main()
