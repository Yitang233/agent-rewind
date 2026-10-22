"""Phase 1 propagation measurement and recovery comparison under identical fault schedules."""
from __future__ import annotations

import copy

from .faults import Fault
from .llm import CachedLLM, ScriptedLLM
from .engine import Run
from .recovery import GRANULARITY_DEPTH, recover
from .workloads.booking import KB, NAMES, run
from .world import World

RECOVERY_CONFIGS = [
    ("restart", "fact", True),
    ("rollback", "fact", True),
    ("selective", "agent", True),
    ("selective", "message", True),
    ("selective", "fact", False),
    ("selective", "fact", True),
    ("selective-d2", "fact", True),
]


def make_llm(backend=None, **kw) -> CachedLLM:
    """A fresh cache per run, so every method pays for its own model calls."""
    return CachedLLM(backend() if backend else ScriptedLLM(KB, **kw))


def final_values(trace: Run) -> dict:
    return trace.values()


def clean_reference(backend=None) -> dict:
    world = World(KB)
    trace, stats = run(world, make_llm(backend))
    return {"trace": trace, "world": world, "outcome": world.outcome(), "values": final_values(trace),
            "tokens": sum(r.cost.tokens for r in trace.records), "latency_s": stats["latency_s"]}


def propagation(fault: Fault, clean: dict, backend=None) -> dict:
    """Did the fault actually flow anywhere, and what did it reach?"""
    world = World(KB)
    trace, _ = run(world, make_llm(backend), [fault])
    cone = set(trace.store.forward_cone(trace.branch, {NAMES[fault.name]}))
    affected = [r for r in trace.records if r.step in cone]
    return {
        "fault": fault.label, "kind": fault.kind, "adversarial": fault.adversarial,
        "read_downstream": bool(affected),
        "outcome_changed": world.outcome() != clean["outcome"],
        "affected_steps": [c.step for c in affected],
        "effects_reached": [e.tool for c in affected for e in c.effects if e.effect_class != "read"],
        "unauthorized": world.unauthorized(),
        "wasted_tokens": sum(c.cost.tokens for c in affected),
        "trace": trace, "world": world,
    }


def compare(fault: Fault, clean: dict, backend=None, verify_miss_rate: float = 0.0) -> list[dict]:
    rows, faulty = [], {}
    for method, gran, cutoff in RECOVERY_CONFIGS:
        # The depth tier is chosen when the faulty run records its reads.
        depth = GRANULARITY_DEPTH[gran]
        if depth not in faulty:
            w = World(KB)
            faulty[depth] = (w, run(w, make_llm(backend), [fault], depth=depth)[0])
        world, trace = copy.deepcopy(faulty[depth])
        sent_before = len(world.outbox)
        new, rep = recover(trace, world, make_llm(backend), {fault.name}, method, gran, cutoff)
        name = {"selective": f"selective/{gran}{'' if cutoff else '/no-cutoff'}",
                "selective-d2": f"selective-d2/{gran}"}.get(method, method)
        rows.append(_row(fault, name, world, new, clean, rep, sent_before))

    # Always-verify is prevention, not recovery: it pays a check on every run.
    world = World(KB)
    trace, stats = run(world, make_llm(backend, verifier_miss_rate=verify_miss_rate), [fault], verify=True)
    rows.append({
        "fault": fault.label, "method": "always-verify",
        "correct": world.outcome() == clean["outcome"], "values_match": final_values(trace) == clean["values"],
        "rerun": stats["verify_flags"], "reused": "-", "llm_calls": stats["verify_calls"],
        "tokens": sum(r.cost.tokens for r in trace.records), "irreversible_redone": 0,
        "unauthorized": world.unauthorized(), "latency_ms": round(stats["latency_s"] * 1e3, 2),
    })
    return rows


def _row(fault, name, world, new, clean, rep, sent_before) -> dict:
    return {
        "fault": fault.label, "method": name,
        "correct": world.outcome() == clean["outcome"],
        # Reuse check: every final artifact must equal the clean run's, not just the outcome.
        "values_match": final_values(new) == clean["values"],
        "rerun": len(rep["rerun"]), "reused": len(rep["reused"]),
        "llm_calls": rep["llm_calls"], "tokens": rep["tokens"],
        "irreversible_redone": len(rep["irreversible_redone"]),
        # Unauthorized emails stay sent: recovery cannot take them back.
        "unauthorized": world.unauthorized(),
        "latency_ms": round(rep["latency_s"] * 1e3, 2),
    }
