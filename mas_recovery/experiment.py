"""Phase 1 propagation measurement and recovery comparison under identical fault schedules."""
from __future__ import annotations

import copy

from .faults import Fault
from .llm import CachedLLM, ScriptedLLM
from .recovery import recover
from .trace import Trace
from .workflow import KB, run
from .world import World

RECOVERY_CONFIGS = [
    ("restart", "fact", True),
    ("rollback", "fact", True),
    ("selective", "agent", True),
    ("selective", "message", True),
    ("selective", "fact", False),
    ("selective", "fact", True),
]


def make_llm(backend=None, **kw) -> CachedLLM:
    """A fresh cache per run, so every method pays for its own model calls."""
    return CachedLLM(backend() if backend else ScriptedLLM(KB, **kw))


def final_values(trace: Trace) -> dict:
    return {n: trace.value(n) for n in sorted(trace.latest)}


def clean_reference(backend=None) -> dict:
    world = World(KB)
    trace, stats = run(world, make_llm(backend))
    return {"trace": trace, "world": world, "outcome": world.outcome(), "values": final_values(trace),
            "tokens": sum(c.tokens for c in trace.commits), "latency_s": stats["latency_s"]}


def propagation(fault: Fault, clean: dict, backend=None) -> dict:
    """Did the fault actually flow anywhere, and what did it reach?"""
    world = World(KB)
    trace, _ = run(world, make_llm(backend), [fault])
    affected = trace.downstream({fault.name})
    return {
        "fault": fault.label, "kind": fault.kind, "adversarial": fault.adversarial,
        "read_downstream": bool(affected),
        "outcome_changed": world.outcome() != clean["outcome"],
        "affected_steps": [c.step for c in affected],
        "effects_reached": [e.tool for c in affected for e in c.effects if e.effect_class != "read"],
        "unauthorized": world.unauthorized(),
        "wasted_tokens": sum(c.tokens for c in affected),
        "trace": trace, "world": world,
    }


def compare(fault: Fault, clean: dict, backend=None, verify_miss_rate: float = 0.0) -> list[dict]:
    faulty = propagation(fault, clean, backend)
    rows = []
    for method, gran, cutoff in RECOVERY_CONFIGS:
        world, trace = copy.deepcopy((faulty["world"], faulty["trace"]))
        sent_before = len(world.outbox)
        new, rep = recover(trace, world, make_llm(backend), {fault.name}, method, gran, cutoff)
        name = method if method != "selective" else f"selective/{gran}{'' if cutoff else '/no-cutoff'}"
        rows.append(_row(fault, name, world, new, clean, rep, sent_before))

    # Always-verify is prevention, not recovery: it pays a check on every run.
    world = World(KB)
    trace, stats = run(world, make_llm(backend, verifier_miss_rate=verify_miss_rate), [fault], verify=True)
    rows.append({
        "fault": fault.label, "method": "always-verify",
        "correct": world.outcome() == clean["outcome"], "values_match": final_values(trace) == clean["values"],
        "rerun": stats["verify_flags"], "reused": "-", "llm_calls": stats["verify_calls"],
        "tokens": sum(c.tokens for c in trace.commits), "irreversible_redone": 0,
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
