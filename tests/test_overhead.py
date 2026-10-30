"""P2 counters: ns per read/write and journal bytes per step, by depth tier."""
import unittest

from agent_rewind.deps import TIERS, Meter, depth_of, measure
from agent_rewind.engine import execute
from agent_rewind.workloads.dag import generate


class Overhead(unittest.TestCase):
    def setUp(self):
        self.dag = generate(n=40, layers=5, fan_in=3, shared_keys=3, seed=11)

    def test_tiers(self):
        self.assertEqual([depth_of(t) for t in TIERS], [2, 3, 4, None])
        self.assertEqual(depth_of(3), 3)

    def test_meter_counts_reads_writes_and_bytes(self):
        m = Meter()
        run = execute(self.dag.workflow, self.dag.world(), None, meter=m)
        s = m.summary()
        self.assertEqual(s["steps"], len(run.records))
        self.assertGreater(s["reads"], 0)
        self.assertGreater(s["writes"], 0)
        self.assertGreater(s["ns_per_read"], 0)
        self.assertGreater(s["ns_per_write"], 0)
        self.assertGreater(s["bytes_per_step"], 0)
        self.assertGreater(s["value_bytes_per_step"], 0)

    def test_metering_does_not_change_the_run(self):
        a = execute(self.dag.workflow, self.dag.world(), None)
        b = execute(self.dag.workflow, self.dag.world(), None, meter=Meter())
        self.assertEqual(a.values(), b.values())
        self.assertEqual([r.reads for r in a.records], [r.reads for r in b.records])

    def test_coarser_tier_records_fewer_reads_and_fewer_bytes(self):
        res = measure(self.dag.workflow, self.dag.world, lambda: None)
        order = ["agent", "step", "artifact", "field"]
        for lo, hi in zip(order, order[1:]):
            self.assertLessEqual(res[lo]["reads_per_step"], res[hi]["reads_per_step"] + 1e-9, (lo, hi))
            self.assertLessEqual(res[lo]["bytes_per_step"], res[hi]["bytes_per_step"] + 1e-9, (lo, hi))
        self.assertLess(res["agent"]["bytes_per_step"], res["field"]["bytes_per_step"])

    def test_canonical_tier_is_measurable_too(self):
        res = measure(self.dag.workflow, self.dag.world, lambda: None, tiers=("field",), cutoff="canonical")
        self.assertGreater(res["field"]["ns_per_read"], 0)


if __name__ == "__main__":
    unittest.main()
