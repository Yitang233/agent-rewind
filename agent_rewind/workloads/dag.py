"""W2: seeded synthetic DAG workflows with scripted, deterministic steps.

    dag = generate(n=24, layers=4, fan_in=2, shared_keys=3, effect_mix={"irreversible": 1, ...}, seed=7)
    run = execute(dag.workflow, dag.world(), None)

Steps sit in layers; a step reads the outputs of earlier layers and some shared keys, mixes
them into a small integer (the modulus controls how often a changed input is absorbed, which
is what early cutoff feeds on) and writes `/msg/<agent>/<step>/out = {v, w, note}`. Features
that stress recovery:
  * `when` branches: a step runs only if an earlier value passes a test (dynamic step set);
  * dynamic reads: which extra path is read depends on a value already read;
  * shared keys `/kv/k<j>/val` written by several steps (shadowed writes), sometimes after a
    read-modify-write, and sometimes written twice inside one step;
  * write-set changes: some steps write `/kv/alt/<step>` only for some inputs;
  * one tool call per step, drawn from `effect_mix`, with arguments that depend on the inputs.
Step functions carry only plain data in their closure, so `Step.code` is stable per plan.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

from ..engine import Step, StepContext, Workflow
from ..world import World

AGENTS = 4
KB = {f"k{i}": i * 10 for i in range(3)} | {f"http://x/{i}": f"page{i}" for i in range(3)} | {"team.members": []}
ROOMS = ["A", "B", "C"]
DEFAULT_MIX = {"none": 2, "read": 2, "idempotent": 1, "compensable": 1, "irreversible": 1}


def _mix(values) -> int:
    h = hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).digest()
    return int.from_bytes(h[:6], "big")


def _num(x) -> int:
    """The part of a read value that steps compute with (a whole artifact or one field)."""
    if isinstance(x, dict):
        return int(x.get("v", 0))
    return int(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else 0


def _body(plan: dict):
    def fn(ctx: StepContext) -> None:
        vals = [ctx.read(p, 0) for p in plan["reads"]]
        if plan["probe"]:                                   # dynamic read: which path depends on a value
            probe, a, b = plan["probe"]
            vals.append(ctx.read(a if _num(ctx.read(probe, 0)) % 2 == 0 else b, 0))
        if plan["rmw"]:
            vals.append(ctx.read(plan["rmw"], 0))
        mixed = _mix(vals)
        v = mixed % plan["mod"]
        w = (mixed // 7) % 5
        extra = None
        tool = plan["tool"]
        if tool == "kb.lookup":
            extra = ctx.tool("kb.lookup", key=f"k{mixed % 3}")
        elif tool == "web.fetch":
            extra = ctx.tool("web.fetch", url=f"http://x/{mixed % 3}")
        elif tool == "doc.set":
            ctx.tool("doc.set", doc=ctx.step, value=v % 4)
        elif tool == "calendar.book":
            ctx.tool("calendar.book", room=ROOMS[mixed % 3])
        elif tool == "email.send":
            ctx.tool("email.send", to=[f"u{mixed % 3}@x.org"], body=f"m{v}")
        if extra is not None:
            v = (v + (len(str(extra)) % 2)) % plan["mod"]
        if plan["shared"]:
            ctx.write(plan["shared"], v)                    # provisional ...
            if plan["twice"]:
                ctx.write(plan["shared"], (v + 1) % plan["mod"])   # ... shadowed inside the step
        if plan["alt"] and mixed % 2:
            ctx.write(plan["alt"], v)
        ctx.write(plan["out"], {"v": v, "w": w, "note": f"{ctx.step} saw {mixed % 1000}"})
    return fn


def _guard(plan: dict):
    def fn(ctx: StepContext) -> bool:
        return _num(ctx.read(plan["when"], 0)) % 3 != 0
    return fn


@dataclass
class Dag:
    workflow: Workflow
    plans: dict[str, dict]
    spec: dict
    kb: dict = field(default_factory=lambda: dict(KB))

    def world(self, **kw) -> World:
        return World(dict(self.kb), **kw)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.workflow.steps]

    def fault_for(self, run, rng: random.Random) -> tuple[str, str, object] | None:
        """Pick a step that ran in `run` and a path it wrote; return (step, path, wrong value)."""
        recs = [r for r in run.records if r.writes]
        if not recs:
            return None
        rec = rng.choice(recs)
        path = rng.choice([p for p, _ in rec.writes])
        return rec.step, path, perturb(rec.values[path], rng)


def perturb(value, rng: random.Random):
    if isinstance(value, dict):
        out = dict(value)
        out["v"] = int(out.get("v", 0)) + rng.randint(1, 3)
        return out
    return (value if isinstance(value, int) else 0) + rng.randint(1, 3)


def generate(n: int = 24, layers: int = 4, fan_in: int = 2, shared_keys: int = 3, effect_mix: dict | None = None,
             seed: int = 0, branch_p: float = 0.2, shadow_p: float = 0.35, alt_p: float = 0.2,
             probe_p: float = 0.2, min_mod: int = 2, max_mod: int = 9) -> Dag:
    rng = random.Random(seed)
    mix = {k: w for k, w in (effect_mix or DEFAULT_MIX).items() if w > 0}
    kinds, weights = list(mix), list(mix.values())
    layer_of = sorted(rng.randrange(layers) for _ in range(n))
    names = [f"s{i:03d}" for i in range(n)]
    out = {nm: f"/msg/a{i % AGENTS}/{nm}/out" for i, nm in enumerate(names)}
    skeys = [f"/kv/k{j}/val" for j in range(shared_keys)]
    plans, steps = {}, []
    for i, nm in enumerate(names):
        earlier = [names[j] for j in range(i) if layer_of[j] < layer_of[i]]
        prev = [x for x in earlier if layer_of[names.index(x)] == layer_of[i] - 1] or earlier
        picks = (rng.sample(prev, min(fan_in, len(prev))) if prev else [])
        reads = [out[x] + rng.choice(["", "/v"]) for x in picks]
        if skeys and rng.random() < 0.5:
            reads.append(rng.choice(skeys))
        probe = None
        if len(earlier) >= 3 and rng.random() < probe_p:
            probe = [out[rng.choice(earlier)] + "/v", out[rng.choice(earlier)], rng.choice(skeys + [out[rng.choice(earlier)] + "/w"])]
        kind = rng.choices(kinds, weights)[0]
        tool = {"none": None, "idempotent": "doc.set", "compensable": "calendar.book",
                "irreversible": "email.send"}.get(kind)
        if kind == "read":
            tool = rng.choice(["kb.lookup", "kb.lookup", "web.fetch"])
        shared = rng.choice(skeys) if skeys and rng.random() < shadow_p else None
        plan = {
            "reads": reads, "probe": probe, "out": out[nm], "mod": rng.randint(min_mod, max_mod), "tool": tool,
            "shared": shared, "twice": bool(shared) and rng.random() < 0.3,
            "rmw": shared if shared and rng.random() < 0.5 else None,
            "alt": f"/kv/alt/{nm}" if rng.random() < alt_p else None,
            "when": (out[rng.choice(earlier)] + "/v") if earlier and rng.random() < branch_p else None,
        }
        plans[nm] = plan
        steps.append(Step(nm, f"a{i % AGENTS}", _body(plan), when=_guard(plan) if plan["when"] else None))
    spec = dict(n=n, layers=layers, fan_in=fan_in, shared_keys=shared_keys, effect_mix=mix, seed=seed)
    return Dag(Workflow(steps), plans, spec)
