import os
import tempfile
import unittest

from mas_recovery.store import BOT, Store, cut, overlaps


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.s = Store()

    def test_absent_hashes_to_bottom_and_leaf_is_content_hash(self):
        self.assertEqual(self.s.hash("main", "/kv/x"), BOT)
        self.s.write("main", "/kv/x", {"b": 1, "a": 2})
        self.s.write("main", "/kv/y", {"a": 2, "b": 1})
        self.assertEqual(self.s.hash("main", "/kv/x"), self.s.hash("main", "/kv/y"))   # canonical JSON
        self.assertNotEqual(self.s.hash("main", "/kv/x"), BOT)

    def test_json_values_are_subtrees(self):
        self.s.write("main", "/msg/p/plan/agenda", {"items": ["a", "b", "c"], "topic": "t"})
        self.assertEqual(self.s.get("main", "/msg/p/plan/agenda/items/2"), "c")
        self.assertEqual(self.s.get("main", "/msg/p/plan/agenda"), {"items": ["a", "b", "c"], "topic": "t"})
        self.s.write("main", "/msg/p/plan/agenda/items/2", "z")
        self.assertEqual(self.s.get("main", "/msg/p/plan/agenda/items"), ["a", "b", "z"])

    def test_merkle_ancestors_change_siblings_do_not(self):
        self.s.write("main", "/msg/p/plan", {"room": "B", "agenda": "x"})
        root, plan, agenda = (self.s.hash("main", p) for p in ("/", "/msg/p/plan", "/msg/p/plan/agenda"))
        self.s.write("main", "/msg/p/plan/room", "C")
        self.assertNotEqual(self.s.hash("main", "/"), root)
        self.assertNotEqual(self.s.hash("main", "/msg/p/plan"), plan)
        self.assertEqual(self.s.hash("main", "/msg/p/plan/agenda"), agenda)

    def test_delete_returns_to_bottom_and_prunes(self):
        h = self.s.hash("main", "/")
        self.s.write("main", "/kv/a/b", 1)
        self.s.delete("main", "/kv/a/b")
        self.assertEqual(self.s.hash("main", "/kv/a"), BOT)
        self.assertEqual(self.s.hash("main", "/"), h)

    def test_mvcc_reads_past_versions(self):
        t1 = self.s.write("main", "/kv/x", 1)
        t2 = self.s.write("main", "/kv/x", 2)
        self.assertEqual(self.s.get("main", "/kv/x", at=t1), 1)
        self.assertEqual(self.s.get("main", "/kv/x", at=t2), 2)
        self.assertEqual(self.s.get("main", "/kv/x", at=0), None)
        self.assertEqual(self.s.hash("main", "/kv/x", at=0), BOT)

    def test_branches_share_history_and_diverge(self):
        self.s.write("main", "/kv/x", 1)
        self.s.write("main", "/kv/y", 1)
        b = self.s.new_branch("b", parent="main")
        self.s.write(b, "/kv/x", 2)
        self.s.write("main", "/kv/y", 9)               # after the fork: invisible to b
        self.assertEqual(self.s.get(b, "/kv/y"), 1)
        self.assertEqual(self.s.get("main", "/kv/x"), 1)
        self.assertEqual(self.s.diff("main", b), ["/kv/x", "/kv/y"])
        again = self.s.snapshot(b, self.s.seq)          # rebuilt from the log, not the live view
        self.assertEqual(again.get("/kv/x"), 2)
        self.assertEqual(again.get("/kv/y"), 1)

    def test_empty_branch_starts_empty(self):
        self.s.write("main", "/kv/x", 1)
        b = self.s.new_branch()
        self.assertEqual(self.s.hash(b, "/kv/x"), BOT)

    def test_depth_cuts(self):
        p = "/msg/planner/plan/agenda/items/2"
        self.assertEqual(cut(p, 2), "/msg/planner")
        self.assertEqual(cut(p, 3), "/msg/planner/plan")
        self.assertEqual(cut(p, 4), "/msg/planner/plan/agenda")
        self.assertEqual(cut(p), p)
        self.s.write("main", p, "v")
        c, h = self.s.cut("main", p, 3)
        self.assertEqual((c, h), ("/msg/planner/plan", self.s.hash("main", "/msg/planner/plan")))
        self.assertTrue(overlaps("/a/b", "/a/b/c") and not overlaps("/a/b", "/a/c"))

    def test_coarse_cut_never_misses_a_change(self):
        self.s.write("main", "/msg/p/plan/a/b", 1)
        before = {d: self.s.cut("main", "/msg/p/plan/a/b", d)[1] for d in (1, 2, 3, 4, None)}
        self.s.write("main", "/msg/p/plan/a/b", 2)
        for d, h in before.items():
            self.assertNotEqual(self.s.cut("main", "/msg/p/plan/a/b", d)[1], h, d)

    def test_cones(self):
        s = self.s
        s.note("main", "s1", [], ["/kv/a"])
        s.note("main", "s2", ["/kv/a"], ["/kv/b"])
        s.note("main", "s3", ["/kv/b"], ["/kv/c"])
        s.note("main", "s4", ["/kv/z"], ["/kv/d"])
        self.assertEqual(s.forward_cone("main", {"/kv/a"}), ["s2", "s3"])
        steps, paths = s.backward_cone("main", "/kv/c")
        self.assertEqual(steps, ["s1", "s2", "s3"])
        self.assertIn("/kv/a", paths)
        self.assertNotIn("/kv/z", paths)

    def test_sqlite_roundtrip(self):
        self.s.write("main", "/kv/x", {"a": [1, 2]})
        b = self.s.new_branch("b", parent="main")
        self.s.write(b, "/kv/x/a/0", 5)
        self.s.note(b, "s", ["/kv/x"], ["/kv/x/a/0"])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.db")
            self.s.save(path)
            t = Store.load(path)
        self.assertEqual(t.get("main", "/kv/x"), {"a": [1, 2]})
        self.assertEqual(t.get("b", "/kv/x"), {"a": [5, 2]})
        self.assertEqual(t.hash("b", "/"), self.s.hash("b", "/"))
        self.assertEqual(t.forward_cone("b", {"/kv/x"}), ["s"])


if __name__ == "__main__":
    unittest.main()
