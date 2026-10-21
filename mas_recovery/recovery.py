"""Recovery methods, given the identity of the faulty artifact(s).

restart    -- undo what can be undone, rerun every step
rollback   -- time order: keep steps before the first faulty one, rerun everything after
selective  -- dependency-aware: rerun the faulty producer, then only steps that read a
              changed unit; with early cutoff a rerun whose output hash is unchanged
              stops the invalidation there

`granularity` sets what counts as one dependency unit: an artifact name ("fact"),
a whole step's output ("message") or everything one agent produced ("agent").
Coarser units need less bookkeeping but invalidate more work.
"""
from __future__ import annotations

import time

from .trace import Trace
from .workflow import STEPS, run_step
from .world import World


def unit(trace: Trace, art_id: str, granularity: str) -> str:
    art = trace.artifacts[art_id]
    return {"fact": art.name, "message": art.step, "agent": art.agent}[granularity]


def recover(old: Trace, world: World, llm, faulty: set[str], method: str = "selective",
            granularity: str = "fact", early_cutoff: bool = True) -> tuple[Trace, dict]:
    start = time.perf_counter()
    new = Trace.continue_from(old)
    changed: set[str] = set()
    first_faulty = next(i for i, c in enumerate(old.commits)
                        if any(old.artifacts[w].name in faulty for w in c.writes))
    report = {"rerun": [], "reused": [], "irreversible_redone": [], "compensated": 0}

    for i, c in enumerate(old.commits):
        produces_faulty = any(old.artifacts[w].name in faulty for w in c.writes)
        if method == "restart":
            rerun = True
        elif method == "rollback":
            rerun = i >= first_faulty
        else:
            rerun = produces_faulty or any(unit(old, r, granularity) in changed for r in c.reads)

        if not rerun:
            new.commits.append(c)
            for w in c.writes:
                new.latest[old.artifacts[w].name] = w
            report["reused"].append(c.step)
            continue

        for e in c.effects:
            if e.effect_class == "compensable" and not e.undone:
                e.undone = world.compensate(e.tool, e.result)
                report["compensated"] += 1
            elif e.effect_class == "irreversible":
                report["irreversible_redone"].append(e.tool)
        old_hash = {old.artifacts[w].name: old.artifacts[w].hash for w in c.writes}
        nc = run_step(STEPS[c.step], new, world, llm, faults=[], rerun_of=c.id)
        for w in nc.writes:
            art = new.artifacts[w]
            if not early_cutoff or old_hash.get(art.name) != art.hash:
                changed.add(unit(new, w, granularity))
        report["rerun"].append(c.step)

    rerun_commits = [c for c in new.commits if c.rerun_of]
    report.update(
        llm_calls=sum(c.llm_calls for c in rerun_commits),
        tokens=sum(c.tokens for c in rerun_commits),
        latency_s=time.perf_counter() - start,
    )
    return new, report
