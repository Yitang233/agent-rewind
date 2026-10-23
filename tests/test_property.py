"""P1 property test on seeded W2 DAGs with an injected fault.

For every case, recovering from the faulty run must satisfy
  final(selective) == final(rollback) == final(restart) == the clean run's final state,
  reruns(selective) <= reruns(rollback) <= reruns(restart),
  no irreversible effect delivered twice under `diff+outbox`, also over a chain of recoveries
  (a second, itself faulty recovery and a third that fixes it) and when the faulty run never
  reached its commit point (then nothing is leaked and the world matches the clean run).
"""
import copy
import random
import time
import unittest

from mas_recovery.engine import execute
from mas_recovery.workloads.dag import generate

N_CASES = 520
EFF = "diff+outbox"


def params(rng: random.Random) -> dict:
    mix = {k: rng.choice([0, 1, 1, 2, 3]) for k in ("none", "read", "idempotent", "compensable", "irreversible")}
    mix["irreversible"] = max(mix["irreversible"], 1)
    return dict(n=rng.randint(6, 28), layers=rng.randint(2, 5), fan_in=rng.randint(1, 3),
                shared_keys=rng.randint(0, 4), effect_mix=mix, seed=rng.randrange(10**6),
                branch_p=rng.choice([0.0, 0.2, 0.4]), min_mod=rng.choice([2, 2, 4]))


def check_case(seed: int) -> tuple[list[str], dict]:
    rng = random.Random(seed)
    dag = generate(**params(rng))
    wf, bad = dag.workflow, []
    tier = rng.choice([None, None, 4, 3, 2])
    clean_world = dag.world()
    clean = execute(wf, clean_world, None, depth=tier, effects=EFF)
    fault = dag.fault_for(clean, rng)
    step, path, wrong = fault
    stats = {"restart_dups_naive": 0}

    committed = rng.random() < 0.6
    world = dag.world()
    faulty = execute(wf, world, None, overrides={(step, path): wrong}, depth=tier, effects=EFF, commit=committed)

    runs = {}
    for policy in ("restart", "rollback", "selective"):
        w = copy.deepcopy(world)
        r = execute(wf, w, None, prior=faulty, dirty={step}, policy=policy)
        runs[policy] = (r, w)
        if r.values() != clean.values():
            bad.append(f"{policy}: final differs from clean")
        if w.duplicates():
            bad.append(f"{policy}: duplicated irreversible effect")
        if not committed:
            if r.report["effects"]["leaked"]:
                bad.append(f"{policy}: leak from an uncommitted run")
            if {o["key"] for o in w.outbox} != {o["key"] for o in clean_world.outbox}:
                bad.append(f"{policy}: delivered set differs from clean")
    if runs["selective"][0].values() != runs["restart"][0].values():
        bad.append("selective != restart")
    sel, rb, rs = (set(runs[p][0].report["rerun"]) for p in ("selective", "rollback", "restart"))
    if not (sel <= rb <= rs):
        bad.append(f"rerun chain broken: {len(sel)} {len(rb)} {len(rs)}")
    if step not in sel:
        bad.append("faulty step not rerun by selective")

    if rng.random() < 0.35:                      # chain: faulty recovery, then a fixing one
        step2, path2, wrong2 = dag.fault_for(clean, rng)
        w = copy.deepcopy(runs["selective"][1])
        mid = execute(wf, w, None, prior=runs["selective"][0], dirty={step2}, overrides={(step2, path2): wrong2})
        end = execute(wf, w, None, prior=mid, dirty={step2}, policy=rng.choice(["selective", "rollback", "restart"]))
        if end.values() != clean.values():
            bad.append("chain: final differs from clean")
        if w.duplicates():
            bad.append("chain: duplicated irreversible effect")
    return bad, stats


class PropertyTest(unittest.TestCase):
    def test_seeded_dag_fault_cases(self):
        t0, failures = time.perf_counter(), []
        for seed in range(N_CASES):
            bad, _ = check_case(seed)
            failures += [f"seed {seed}: {b}" for b in bad]
        elapsed = time.perf_counter() - t0
        self.assertEqual(failures[:5], [], f"{len(failures)} violations")
        self.assertLess(elapsed, 60, f"{N_CASES} cases took {elapsed:.1f}s")

    def test_naive_policy_does_duplicate(self):
        """The property is not vacuous: without the intent diff a restart sends mails twice."""
        dups = 0
        for seed in range(40):
            rng = random.Random(seed)
            dag = generate(**params(rng))
            clean = execute(dag.workflow, dag.world(), None)
            step, path, wrong = dag.fault_for(clean, rng)
            world = dag.world()
            faulty = execute(dag.workflow, world, None, overrides={(step, path): wrong})
            execute(dag.workflow, world, None, prior=faulty, dirty={step}, policy="restart")
            dups += world.duplicates()
        self.assertGreater(dups, 0)

    def test_selective_reruns_less_than_restart(self):
        saved = 0
        for seed in range(40):
            rng = random.Random(seed)
            dag = generate(**params(rng))
            clean = execute(dag.workflow, dag.world(), None)
            step, path, wrong = dag.fault_for(clean, rng)
            faulty = execute(dag.workflow, dag.world(), None, overrides={(step, path): wrong})
            sel = execute(dag.workflow, dag.world(), None, prior=faulty, dirty={step})
            rs = execute(dag.workflow, dag.world(), None, prior=faulty, policy="restart")
            saved += len(rs.report["rerun"]) - len(sel.report["rerun"])
        self.assertGreater(saved, 0)


if __name__ == "__main__":
    unittest.main()
