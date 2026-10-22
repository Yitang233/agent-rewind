"""Compatibility: `recover()` is now a thin call into `engine.execute`.

restart       -- rerun every step
rollback      -- rerun the first faulty step and everything after it
selective     -- rerun the faulty producer; every other step is reused when its recorded
                 reads still hash the same (early cutoff comes from the hashes)
selective-d2  -- the earlier forward-propagation algorithm (baseline)

`early_cutoff=False` with `selective` selects the D2 algorithm without cutoff, as the old
code did. `granularity` is a property of the prior run (its depth tier), so it is accepted
for compatibility and not used here.
"""
from __future__ import annotations

from .engine import Run, execute, producers
from .world import World

GRANULARITY_DEPTH = {"fact": None, "message": 3, "agent": 2}


def recover(old: Run, world: World, llm, faulty: set[str], method: str = "selective",
            granularity: str = "fact", early_cutoff: bool = True) -> tuple[Run, dict]:
    wf = old.workflow
    paths = {wf.path(n) for n in faulty}
    policy = method
    if method == "selective" and not early_cutoff:
        policy = "selective-d2"
    new = execute(wf, world, llm, prior=old, dirty=producers(old, paths), policy=policy,
                  early_cutoff=early_cutoff)
    return new, new.report
