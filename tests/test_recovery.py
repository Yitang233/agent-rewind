import copy
import unittest

from mas_recovery.experiment import clean_reference, compare, make_llm, propagation
from mas_recovery.faults import FAULTS
from mas_recovery.recovery import recover


def rows_by_method(label):
    clean = clean_reference()
    return {r["method"]: r for r in compare(FAULTS[label], clean)}


class PropagationTest(unittest.TestCase):
    def test_clean_run_books_smallest_fitting_room(self):
        self.assertEqual(clean_reference()["outcome"]["rooms"], ["B"])

    def test_wrong_fact_reaches_the_booking(self):
        prop = propagation(FAULTS["wrong_capacity"], clean_reference())
        self.assertTrue(prop["outcome_changed"])
        self.assertIn("execute.book", prop["affected_steps"])

    def test_masked_fault_is_read_but_changes_nothing(self):
        prop = propagation(FAULTS["masked_capacity"], clean_reference())
        self.assertTrue(prop["read_downstream"])
        self.assertFalse(prop["outcome_changed"])

    def test_injection_causes_unauthorized_email(self):
        self.assertEqual(propagation(FAULTS["inject_agenda"], clean_reference())["unauthorized"], 1)


class RecoveryTest(unittest.TestCase):
    def test_every_method_restores_the_clean_result(self):
        for label in FAULTS:
            for method, row in rows_by_method(label).items():
                with self.subTest(fault=label, method=method):
                    self.assertTrue(row["correct"])
                    self.assertTrue(row["values_match"])

    def test_selective_reruns_less_and_keeps_the_email(self):
        rows = rows_by_method("wrong_capacity")
        self.assertLess(rows["selective/fact"]["rerun"], rows["restart"]["rerun"])
        self.assertEqual(rows["selective/fact"]["irreversible_redone"], 0)
        self.assertEqual(rows["restart"]["irreversible_redone"], 1)

    def test_coarser_units_invalidate_more(self):
        rows = rows_by_method("wrong_capacity")
        self.assertGreater(rows["selective/message"]["rerun"], rows["selective/fact"]["rerun"])
        self.assertGreater(rows["selective/fact/no-cutoff"]["rerun"], rows["selective/fact"]["rerun"])

    def test_shared_state_fault_recovers_through_store(self):
        rows = rows_by_method("timeout_team")
        self.assertEqual(rows["selective/fact"]["rerun"], 3)

    def test_sent_email_is_reported_not_undone(self):
        prop = propagation(FAULTS["inject_agenda"], clean_reference())
        world, trace = copy.deepcopy((prop["world"], prop["trace"]))
        _, rep = recover(trace, world, make_llm(), {"fact.agenda.topic"})
        self.assertEqual(rep["irreversible_redone"], ["email.send"])
        self.assertEqual(world.unauthorized(), 1)


if __name__ == "__main__":
    unittest.main()
