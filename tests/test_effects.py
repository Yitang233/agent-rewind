"""Effect layer (intents, reconcile table, outbox, trust labels) and equality tiers."""
import copy
import unittest

from agent_rewind.cutoff import Canonical, get as get_cutoff
from agent_rewind.effects import Intent, action, intent_key
from agent_rewind.engine import Step, Workflow, execute
from agent_rewind.world import LATENCY_MS, World

KB = {"team.members": [], "http://x/0": "page"}


def flow(tool_args=None):
    """prep -> mail (reads /kv/n, sends mail and books room, args from n) -> tail (reads mail's output)."""
    def prep(c): c.write("/kv/n", 1)

    def mail(c):
        n = c.read("/kv/n")
        c.tool("email.send", to=["a@x.org"], body=f"hello {n}")
        c.tool("calendar.book", room="AB"[n % 2])
        c.write("/kv/mail", n)

    def tail(c): c.write("/kv/tail", c.read("/kv/mail") * 10)

    return Workflow([Step("prep", "x", prep), Step("mail", "x", mail), Step("tail", "x", tail)])


def faulty_and_world(wf, effects, commit=True, value=2):
    w = World(KB)
    faulty = execute(wf, w, None, overrides={("prep", "/kv/n"): value}, effects=effects, commit=commit)
    return w, faulty


class Keys(unittest.TestCase):
    def test_key_depends_on_step_tool_args_only(self):
        k = intent_key("s", "email.send", {"to": ["a"], "body": "x"})
        self.assertEqual(k, intent_key("s", "email.send", {"body": "x", "to": ["a"]}))
        self.assertNotEqual(k, intent_key("t", "email.send", {"to": ["a"], "body": "x"}))
        self.assertNotEqual(k, intent_key("s", "email.send", {"to": ["a"], "body": "y"}))

    def test_same_call_after_different_inputs_is_the_same_effect(self):
        def make(v):
            return Workflow([Step("a", "x", lambda c: c.write("/kv/a", v)),
                             Step("b", "x", lambda c: (c.read("/kv/a"), c.tool("email.send", to=["z"], body="same"),
                                                       c.write("/kv/b", 1)))])
        w = World(KB)
        r1 = execute(make(1), w, None, effects="diff")
        r2 = execute(make(2), w, None, prior=r1, policy="restart")
        self.assertEqual(len(w.outbox), 1)
        self.assertEqual(r2.report["effects"]["retained"], 1)


class Table(unittest.TestCase):
    def test_action_table(self):
        mk = lambda cls, released=False: Intent("s", "t", "h", "k", cls, released=released)
        self.assertEqual(action(mk("irreversible"), True), "reuse")
        self.assertEqual(action(mk("compensable"), False), "compensate")
        self.assertEqual(action(mk("idempotent"), False), "compensate")
        self.assertEqual(action(mk("irreversible", True), False), "leak")
        self.assertEqual(action(mk("irreversible", False), False), "withdraw")
        self.assertEqual(action(mk("read"), False), "none")
        self.assertEqual(action(None, False), "apply")

    def test_naive_restart_duplicates_diff_does_not(self):
        counts = {}
        for eff in ("naive", "diff", "diff+outbox"):
            wf = flow()
            w = World(KB)
            clean = execute(wf, w, None, effects=eff)
            execute(wf, w, None, prior=clean, policy="restart")
            counts[eff] = (len(w.outbox), sum(b["active"] for b in w.bookings))
        self.assertEqual(counts["naive"], (2, 1))          # sent twice; booking compensated and redone
        self.assertEqual(counts["diff"], (1, 1))
        self.assertEqual(counts["diff+outbox"], (1, 1))

    def test_changed_args_compensate_and_leak(self):
        for eff in ("diff", "diff+outbox"):
            wf = flow()
            w, faulty = faulty_and_world(wf, eff)
            new = execute(wf, w, None, prior=faulty, dirty={"prep"}, policy="selective")
            rep = new.report["effects"]
            self.assertEqual((rep["compensated"], rep["leaked"], rep["applied"]), (1, 1, 2), eff)
            self.assertEqual(new.report["irreversible_redone"], ["email.send"])
            self.assertEqual(w.outcome()["rooms"], ["B"])            # n=1 -> room "B"; the faulty "A" is undone
            self.assertEqual([m["body"] for m in w.outbox], ["hello 2", "hello 1"])
            self.assertEqual(w.duplicates(), 0)


class Outbox(unittest.TestCase):
    def test_irreversible_waits_for_commit(self):
        seen = []
        def mail(c):
            c.tool("email.send", to=["a"], body="x")
            seen.append(len(c.world.outbox))
        wf = Workflow([Step("m", "x", mail)])
        w = World(KB)
        run = execute(wf, w, None, effects="diff+outbox")
        self.assertEqual((seen, len(w.outbox)), ([0], 1))
        self.assertEqual(run.report["effects"]["released"], 1)
        w2 = World(KB)
        execute(wf, w2, None, effects="diff")
        self.assertEqual(seen[-1], 1)                              # without the outbox it is sent at once

    def test_uncommitted_run_is_withdrawn_not_leaked(self):
        wf = flow()
        w, faulty = faulty_and_world(wf, "diff+outbox", commit=False)
        self.assertEqual(w.outbox, [])
        new = execute(wf, w, None, prior=faulty, dirty={"prep"})
        self.assertEqual(new.report["effects"]["leaked"], 0)
        self.assertEqual(new.report["effects"]["withdrawn"], 1)
        self.assertEqual([m["body"] for m in w.outbox], ["hello 1"])

    def test_released_key_is_not_released_again(self):
        wf = flow()
        w = World(KB)
        a = execute(wf, w, None, effects="diff+outbox")
        b = execute(wf, w, None, prior=a, dirty={"mail"}, policy="selective")   # mail rerun, same key
        c = execute(wf, w, None, prior=b, policy="restart")
        self.assertEqual((len(w.outbox), w.duplicates()), (1, 0))
        self.assertEqual(c.report["effects"]["released"], 0)


class Trust(unittest.TestCase):
    def wf(self):
        def fetch(c): c.write("/msg/web/page", c.tool("web.fetch", url="http://x/0"))
        def safe(c): c.write("/kv/safe", 1)
        def send_tainted(c): c.tool("email.send", to=["a"], body=c.read("/msg/web/page"))
        def send_safe(c): c.tool("email.send", to=["a"], body=f"s{c.read('/kv/safe')}")
        return Workflow([Step("fetch", "w", fetch), Step("safe", "s", safe),
                         Step("tainted", "x", send_tainted), Step("clean", "x", send_safe)])

    def test_backward_cone_with_open_world_source_is_held(self):
        w = World(KB)
        run = execute(self.wf(), w, None, effects="diff+outbox")
        self.assertEqual([m["body"] for m in w.outbox], ["s1"])
        self.assertEqual(run.report["held"], ["email.send"])
        self.assertTrue(run.last()["tainted"].untrusted)
        self.assertFalse(run.last()["clean"].untrusted)

    def test_verified_node_releases(self):
        w = World(KB)
        execute(self.wf(), w, None, effects="diff+outbox", verified_paths={"/msg/web"})
        self.assertEqual(sorted(m["body"] for m in w.outbox), ["page", "s1"])

    def test_seeded_untrusted_path_and_overwrite(self):
        def inbox(c): c.write("/msg/other/in", "hi")
        def send(c): c.tool("email.send", to=["a"], body=c.read("/msg/other/in"))
        wf = Workflow([Step("send", "x", send)])
        w = World(KB)
        w.kb["x"] = 1
        w1 = World(KB)
        r = execute(Workflow([Step("inbox", "o", inbox), Step("send", "x", send)]), w1, None,
                    effects="diff+outbox", untrusted={"/msg/other/in"})
        self.assertEqual(len(w1.outbox), 1)                         # the step rewrites the path: trusted again
        self.assertEqual(r.report["held"], [])

    def test_no_hold_without_outbox(self):
        w = World(KB)
        execute(self.wf(), w, None, effects="diff")
        self.assertEqual(len(w.outbox), 2)


class Classes(unittest.TestCase):
    def test_idempotent_compensated_and_reads_always_called(self):
        def a(c): c.write("/kv/a", 1)
        def b(c):
            n = c.read("/kv/a")
            c.tool("doc.set", doc="d", value=n)
            c.tool("kb.lookup", key="team.members")
        wf = Workflow([Step("a", "x", a), Step("b", "x", b)])
        w = World(KB)
        f = execute(wf, w, None, overrides={("a", "/kv/a"): 5}, effects="diff")
        self.assertEqual(w.docs, {"d": 5})
        r = execute(wf, w, None, prior=f, dirty={"a"})
        self.assertEqual(w.docs, {"d": 1})                          # overwritten, not deleted afterwards
        self.assertEqual(r.report["effects"]["compensated"], 0)
        self.assertEqual(execute(Workflow([wf.steps[0]]), w, None, prior=r).orphans, ["b"])
        self.assertEqual(w.docs, {})                                # an orphan's document is removed
        self.assertEqual(sum(t == "kb.lookup" for t, _ in w.calls), 2)   # reads are never reused

    def test_replayed_intents_are_copies(self):
        wf = flow()
        w = World(KB)
        a = execute(wf, w, None, effects="diff")
        b = execute(wf, w, None, prior=a)
        self.assertIsNot(a.last()["mail"].intents[0], b.last()["mail"].intents[0])
        self.assertTrue(b.last()["mail"].reused)


class Tiers(unittest.TestCase):
    def wf(self):
        def src(c): c.write("/msg/a/src/out", {"room": "B", "note": "x"})
        def use(c): c.write("/kv/use", c.read("/msg/a/src/out"))
        return Workflow([Step("src", "a", src), Step("use", "b", use)])

    def recover(self, cutoff, wrong):
        wf = self.wf()
        w = World(KB)
        f = execute(wf, w, None, overrides={("src", "/msg/a/src/out"): wrong}, cutoff=cutoff)
        return execute(wf, w, None, prior=f, dirty={"src"})

    def test_exact_reruns_canonical_cuts_off(self):
        wrong = {"room": " b ", "note": "x"}
        self.assertEqual(self.recover("exact", wrong).report["rerun"], ["src", "use"])
        r = self.recover("canonical", wrong)
        self.assertEqual((r.report["rerun"], r.report["reused"]), (["src"], ["use"]))

    def test_canonical_still_sees_real_changes(self):
        self.assertEqual(self.recover("canonical", {"room": "C", "note": "x"}).report["rerun"], ["src", "use"])

    def test_free_text_fields_are_excluded_by_schema(self):
        wrong = {"room": "B", "note": "something else entirely"}
        self.assertEqual(self.recover("canonical", wrong).report["rerun"], ["src", "use"])
        r = self.recover(Canonical(free_text=("note",)), wrong)
        self.assertEqual(r.report["reused"], ["use"])

    def test_normalisation(self):
        c = Canonical()
        self.assertEqual(c.norm({"b": 3.0, "a": " Hi  There "}), c.norm({"a": "hi there", "b": 3}))
        self.assertEqual(c.norm(0.1 + 0.2), c.norm(0.3))
        self.assertNotEqual(c.norm(1), c.norm(2))
        self.assertIs(get_cutoff(None), get_cutoff("exact"))

    def test_run_inherits_the_tier(self):
        wf = self.wf()
        f = execute(wf, World(KB), None, cutoff="canonical")
        self.assertEqual(execute(wf, World(KB), None, prior=f).cutoff.name, "canonical")


class WorldLatency(unittest.TestCase):
    def test_off_by_default_and_virtual_clock(self):
        w = World(KB)
        w.call("kb.lookup", key="x")
        self.assertEqual(w.clock_ms, 0)
        w = World(KB, latency=True)
        for tool, args in [("kb.lookup", {"key": "x"}), ("calendar.book", {"room": "A"}),
                           ("email.send", {"to": [], "body": ""})]:
            w.call(tool, **args)
        self.assertEqual(w.clock_ms, 50 + 300 + 200)
        self.assertEqual(LATENCY_MS["web.fetch"], 50)

    def test_jitter_is_seeded(self):
        a, b = World(KB, latency=True, jitter=0.2, seed=3), World(KB, latency=True, jitter=0.2, seed=3)
        for w in (a, b):
            for _ in range(5):
                w.call("kb.lookup", key="x")
        self.assertEqual(a.clock_ms, b.clock_ms)
        self.assertNotEqual(a.clock_ms, 250)

    def test_report_has_world_ms(self):
        w = World(KB, latency=True)
        r = execute(flow(), w, None)
        self.assertEqual(r.report["world_ms"], 500)


if __name__ == "__main__":
    unittest.main()
