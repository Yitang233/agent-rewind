"""Dependency depth tiers and overhead counters.

A read is recorded as (path cut to d components, hash of the cut). Paths look like
`/msg/<agent>/<step>/<artifact>/<field>`, so the tiers are:

    agent -> 2   step -> 3   artifact -> 4   field -> None (full path)

Paths shorter than d (`/kv/<key>`) are kept whole, so a coarse tier never merges them.
`Meter` wraps `read` and `write` in `perf_counter_ns` and counts journal bytes per step.
"""
from __future__ import annotations

from .store import canon

TIERS = {"agent": 2, "step": 3, "artifact": 4, "field": None}


def depth_of(tier) -> int | None:
    """Tier name (or an int / None) -> depth."""
    return TIERS[tier] if isinstance(tier, str) else tier


class Meter:
    """Timing around ctx.read / ctx.write and the size of what a step leaves in the journal."""

    def __init__(self) -> None:
        self.read_ns = self.write_ns = 0
        self.reads = self.writes = 0
        self.steps = 0
        self.meta_bytes = 0       # reads + writes (path, hash) + intent headers
        self.value_bytes = 0      # written values, kept for replay

    def add_record(self, rec) -> None:
        self.steps += 1
        self.meta_bytes += len(canon([rec.reads, rec.writes, [(i.tool, i.key, i.cls) for i in rec.intents]]))
        self.value_bytes += len(canon(rec.values))

    def summary(self) -> dict:
        return {
            "ns_per_read": self.read_ns / self.reads if self.reads else 0.0,
            "ns_per_write": self.write_ns / self.writes if self.writes else 0.0,
            "reads": self.reads, "writes": self.writes, "steps": self.steps,
            "reads_per_step": self.reads / self.steps if self.steps else 0.0,
            "bytes_per_step": self.meta_bytes / self.steps if self.steps else 0.0,
            "value_bytes_per_step": self.value_bytes / self.steps if self.steps else 0.0,
        }


def measure(workflow, make_world, make_llm, tiers=("agent", "step", "artifact", "field"), cutoff=None) -> dict:
    """Run `workflow` once per tier with a Meter and return {tier: summary}."""
    from .engine import execute
    out = {}
    for tier in tiers:
        meter = Meter()
        execute(workflow, make_world(), make_llm(), depth=depth_of(tier), meter=meter, cutoff=cutoff)
        out[tier] = meter.summary()
    return out
