import json
import unittest

from agent_rewind import atoms as A
from agent_rewind.llm import CachedLLM, ScriptedLLM
from agent_rewind.workloads import booking

G = [A.Group("Rooms", {"/a/A": 8, "/a/B": 16}), A.Group("Team", {"/kv/team/size": 12})]
TEXT = ('{"atoms":[{"cites":["/a/A","/a/B"],"text":"A 8, B 16"},{"cites":["/kv/team/size"],"text":"12"}],'
        '"answer":{"room":"B"}}')


class ParserTest(unittest.TestCase):
    def test_parse_and_spans(self):
        p = A.parse(TEXT)
        self.assertTrue(p.ok)
        self.assertEqual(p.answer, {"room": "B"})
        self.assertEqual([a["cites"] for a in p.atoms], [["/a/A", "/a/B"], ["/kv/team/size"]])
        for (s, e), a in zip(p.spans, p.atoms):
            self.assertEqual(json.loads(TEXT[s:e]), a)
        self.assertEqual(TEXT[slice(*p.answer_span)], '{"room":"B"}')

    def test_truncated_output_is_not_ok_but_keeps_atoms(self):
        p = A.parse(TEXT[:TEXT.index('"answer"') + 12])
        self.assertFalse(p.ok)
        self.assertEqual(len(p.atoms), 2)

    def test_garbage_never_raises(self):
        for t in ["", "nope", "[]", '{"atoms": 3}']:
            self.assertFalse(A.parse(t).ok)

    def test_schema_fixes_atom_count_and_cites(self):
        s = A.atom_schema(G, {"type": "object"})
        atoms = s["properties"]["atoms"]
        self.assertEqual((atoms["minItems"], atoms["maxItems"]), (2, 2))
        self.assertEqual(atoms["items"]["properties"]["cites"]["items"]["enum"], ["/a/A", "/a/B", "/kv/team/size"])

    def test_prompt_keeps_group_order_and_paths(self):
        _, user = A.build_prompt("planner", "do it", G, "hint")
        self.assertLess(user.index("## Rooms"), user.index("## Team"))
        self.assertIn('[/kv/team/size] = 12', user)

    def test_confidence_maps_tokens_to_atoms(self):
        toks = [TEXT[i:i + 5] for i in range(0, len(TEXT), 5)]
        lps = [-0.1] * len(toks)
        lps[3] = -2.0                                   # inside the first atom
        c = A.confidence(A.parse(TEXT), toks, lps)
        self.assertEqual(len(c["atoms"]), 2)
        self.assertAlmostEqual(c["atoms"][0]["min"], -2.0)
        self.assertAlmostEqual(c["atoms"][1]["min"], -0.1)
        self.assertGreater(c["answer"]["n"], 0)

    def test_cited_cutoff(self):
        atoms = A.parse(TEXT).atoms
        self.assertTrue(A.can_skip(atoms, ["/kv/other"]))
        self.assertFalse(A.can_skip(atoms, ["/kv/team"]))          # ancestor of a cited path
        self.assertFalse(A.can_skip(atoms, ["/a/B"]))

    def test_check_citations(self):
        chk = A.check_citations(A.parse(TEXT).atoms, G)
        self.assertTrue(chk["count_ok"])
        self.assertEqual((chk["unknown_cites"], chk["own_group"], chk["uncited"]), (0, 2, []))


class ScriptedAtomsTest(unittest.TestCase):
    def test_scripted_emits_one_atom_per_group_in_order(self):
        llm = CachedLLM(ScriptedLLM())
        run, world = booking.run_meetings(llm, 3)
        self.assertTrue(booking.meetings_ok(world, 3))
        out, _ = llm.complete("planner", "plan_meeting", {"groups": A.groups_to_json([
            A.Group("Rooms", {"/x/capacity.A": 8, "/x/capacity.B": 16}), A.Group("Team", {"/kv/team/size": 12}),
            A.Group("M", {"/m/guests": 0, "/m/topic": "T"})])})
        self.assertEqual(out["room"], "B")
        self.assertEqual([a["cites"][0] for a in llm.last_atoms], ["/x/capacity.A", "/kv/team/size", "/m/guests"])

    def test_widened_w1_shape(self):
        for k in (1, 3, 5):
            wf = booking.meetings_workflow(k)
            self.assertEqual(len(wf.steps), 2 + 4 * k)
            run, world = booking.run_meetings(CachedLLM(ScriptedLLM()), k)
            self.assertTrue(booking.meetings_ok(world, k))
        # all meetings read the shared team facts
        run, _ = booking.run_meetings(CachedLLM(ScriptedLLM()), 3)
        readers = [r.step for r in run.records if any(p.startswith("/kv/team/size") for p, _ in r.reads)]
        self.assertEqual(readers, ["plan.0", "plan.1", "plan.2"])

    def test_variants_are_solvable(self):
        for v in range(5):
            _, world = booking.run_meetings(CachedLLM(ScriptedLLM()), 3, variant=v)
            self.assertTrue(booking.meetings_ok(world, 3), v)

    def test_prompt_groups_put_shared_facts_before_meeting_facts(self):
        seen = []

        class Spy(ScriptedLLM):
            def complete(self, role, task, inputs):
                seen.append((task, [g["title"] for g in inputs["groups"]]))
                return super().complete(role, task, inputs)
        booking.run_meetings(CachedLLM(Spy()), 2)
        self.assertEqual(seen[0], ("plan_meeting", ["Rooms", "Team", "Meeting 0"]))


if __name__ == "__main__":
    unittest.main()
