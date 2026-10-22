"""One engine for clean runs and every recovery method: re-execution against a verifying trace.

    run = execute(workflow, world, llm)                                   # clean run
    new = execute(workflow, world, llm, prior=run, dirty={"plan"},        # recovery
                  policy="selective")

A step of the prior run is reused when it is not dirty, its code is unchanged and every
recorded read still hashes the same in the store being rebuilt; otherwise it is rerun.
Policies only decide which steps start dirty (`restart`, `rollback`, `selective`).
`selective-d2` is the earlier forward-propagation algorithm, kept as a baseline: it walks
the prior steps, follows recorded read -> write edges and never looks at hashes of
anything that was not written in this recovery.

Depth tiers: `depth=d` records every read as (path cut to d components, hash of the cut).
It is a property of the run that records, so the prior run and the recovery should use the
same tier; `execute` inherits it from `prior` when `depth` is not given.
"""
from __future__ import annotations

import inspect
import json
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Callable

from .store import BOT, Store, canon, cut, digest, overlaps
from .trace import Effect
from .world import EFFECT_CLASS, World

POLICIES = ("restart", "rollback", "selective", "selective-d2")


@dataclass
class Cost:
    llm_calls: int = 0
    tokens: int = 0
    wall_s: float = 0.0


@dataclass
class StepRecord:
    step: str
    agent: str
    code: str
    reads: list[tuple[str, str]] = field(default_factory=list)    # (cut path, hash), in read order
    writes: list[tuple[str, str]] = field(default_factory=list)   # (path, hash)
    values: dict = field(default_factory=dict)                    # path -> written value (for replay)
    effects: list[Effect] = field(default_factory=list)           # S2 replaces this by intents
    cost: Cost = field(default_factory=Cost)
    reused: bool = False
    rerun_of: str | None = None


def _stable(v) -> str:
    return getattr(v, "__qualname__", None) or canon(v)


@dataclass
class Step:
    name: str
    agent: str
    fn: Callable[["StepContext"], None]
    when: Callable[["StepContext"], bool] | None = None   # reads the store; recorded as step reads

    @property
    def code(self) -> str:
        try:
            src = inspect.getsource(self.fn)
        except (OSError, TypeError):
            src = self.fn.__code__.co_code.hex()
        cells = [_stable(c.cell_contents) for c in (self.fn.__closure__ or ())]
        return digest(canon([self.name, self.agent, src, cells]))


@dataclass
class Workflow:
    steps: list[Step]
    names: dict[str, str] = field(default_factory=dict)   # legacy artifact name -> path

    @classmethod
    def of(cls, wf) -> "Workflow":
        return wf if isinstance(wf, Workflow) else cls(list(wf))

    def path(self, name: str) -> str:
        return name if name.startswith("/") else self.names[name]


class StepContext:
    """What a step sees: `read`, `write`, `tool`, `ask`. Everything is recorded."""

    def __init__(self, store: Store, branch: str, world: World, llm, step: Step,
                 overrides: dict | None = None, depth: int | None = None) -> None:
        self.store, self.branch, self.world, self.llm = store, branch, world, llm
        self.step, self.agent, self.depth = step.name, step.agent, depth
        self.overrides = overrides or {}
        self.reads: list[tuple[str, str]] = []
        self.writes: dict[str, str] = {}
        self.values: dict = {}
        self.write_order: list[tuple[str, str]] = []
        self.effects: list[Effect] = []
        self.cost = Cost()

    def read(self, path: str, default=None):
        if not any(overlaps(path, w) for w in self.writes):       # own output is not a dependency
            c, h = self.store.cut(self.branch, path, self.depth)
            if all(c != rc for rc, _ in self.reads):
                self.reads.append((c, h))
        return self.store.get(self.branch, path, default)

    def write(self, path: str, value) -> None:
        value = self.overrides.get((self.step, path), value)
        self.store.write(self.branch, path, value)
        self.writes[path] = self.store.hash(self.branch, path)
        self.values[path] = value
        self.write_order.append((path, self.writes[path]))

    def tool(self, tool: str, **args):
        result = self.world.call(tool, **args)
        self.effects.append(Effect(tool, EFFECT_CLASS[tool], args, result))
        return result

    def ask(self, task: str, inputs: dict) -> dict:
        out, used = self.llm.complete(self.agent, task, inputs)
        self.cost.llm_calls += 1
        self.cost.tokens += used
        return out


@dataclass
class Run:
    workflow: Workflow
    store: Store
    branch: str
    depth: int | None = None
    policy: str = "clean"
    records: list[StepRecord] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    report: dict = field(default_factory=dict)

    def value(self, path: str, default=None):
        return self.store.get(self.branch, path, default)

    def values(self) -> dict:
        return self.store.leaves(self.branch)

    def last(self) -> dict[str, StepRecord]:
        return {r.step: r for r in self.records}

    def dump(self, path) -> None:
        with open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r), default=str) + "\n")


def run_step(step: Step, store: Store, branch: str, world: World, llm, overrides=None,
             depth: int | None = None, ctx: StepContext | None = None, rerun_of: str | None = None) -> StepRecord:
    ctx = ctx or StepContext(store, branch, world, llm, step, overrides, depth)
    t0 = time.perf_counter()
    step.fn(ctx)
    ctx.cost.wall_s = time.perf_counter() - t0
    rec = StepRecord(step.name, step.agent, step.code, list(ctx.reads), list(ctx.write_order), dict(ctx.values),
                     ctx.effects, ctx.cost, rerun_of=rerun_of)
    store.note(branch, step.name, [p for p, _ in rec.reads], [p for p, _ in rec.writes])
    return rec


def producers(prior: Run, paths) -> set[str]:
    """Steps of `prior` that wrote any of `paths` (or something below or above them)."""
    return {r.step for r in prior.records if any(overlaps(w, p) for w, _ in r.writes for p in paths)}


def verified(old: StepRecord, store: Store, branch: str) -> bool:
    """Every recorded read still hashes the same. `all` stops at the first mismatch, so a path
    the new execution would not read is never looked at."""
    return all(store.hash(branch, p) == h for p, h in old.reads)


def _replay(old: StepRecord, store: Store, branch: str) -> StepRecord:
    for path, _ in old.writes:
        store.write(branch, path, old.values[path])
    store.note(branch, old.step, [p for p, _ in old.reads], [p for p, _ in old.writes])
    return replace(old, reused=True)


def _reconcile(old: StepRecord | None, world: World, report: dict) -> None:
    """Current effect policy: compensate what can be, only report the irreversible."""
    for e in (old.effects if old else ()):
        if e.effect_class == "compensable" and not e.undone:
            e.undone = world.compensate(e.tool, e.result)
            report["compensated"] += 1
        elif e.effect_class == "irreversible":
            report["irreversible_redone"].append(e.tool)


def execute(workflow, world: World, llm, prior: Run | None = None, dirty=(), policy: str = "selective",
            depth: int | None = None, overrides: dict | None = None, early_cutoff: bool = True,
            store: Store | None = None, after_step: Callable | None = None) -> Run:
    """Run `workflow` on a new branch of the store. See the module docstring.

    overrides   {(step, path): value} replaces what a step writes (fault injection).
    early_cutoff only matters for `selective-d2`: False marks every rerun write as changed.
    after_step  after_step(step, record, store, branch) -> True reruns the step once without
                overrides (the always-verify baseline).
    """
    if policy not in POLICIES:
        raise ValueError(f"policy {policy!r}")
    wf = Workflow.of(workflow)
    if depth is None and prior is not None:
        depth = prior.depth
    store = prior.store if prior is not None else (store or Store())
    branch = store.new_branch()
    run = Run(wf, store, branch, depth, policy if prior is not None else "clean")
    report = {"rerun": [], "reused": [], "irreversible_redone": [], "compensated": 0}
    start = time.perf_counter()
    prior_recs = prior.last() if prior is not None else {}
    names = [s.name for s in wf.steps]
    dirty = set(dirty)
    if policy == "restart":
        dirty = set(names)
    elif policy == "rollback" and dirty:
        first = min(i for i, n in enumerate(names) if n in dirty)
        dirty = set(names[first:])

    def rerun(step: Step, old: StepRecord | None, ctx: StepContext | None) -> StepRecord:
        _reconcile(old, world, report)
        rec = run_step(step, store, branch, world, llm, overrides, depth, ctx, rerun_of=old.step if old else None)
        report["rerun"].append(step.name)
        if after_step and after_step(step, rec, store, branch):
            run.records.append(rec)
            _reconcile(rec, world, report)
            rec = run_step(step, store, branch, world, llm, None, depth, rerun_of=step.name)
        return rec

    if policy == "selective-d2":
        if prior is None:
            raise ValueError("selective-d2 needs a prior run")
        changed: set[str] = set()
        for old in prior.records:
            hit = old.step in dirty or any(overlaps(cut(p, depth), c) for p, _ in old.reads for c in changed)
            if not hit:
                run.records.append(_replay(old, store, branch))
                report["reused"].append(old.step)
                continue
            step = next(s for s in wf.steps if s.name == old.step)
            rec = rerun(step, old, None)
            old_h = dict(old.writes)
            changed |= {cut(p, depth) for p, h in rec.writes if not early_cutoff or old_h.get(p) != h}
            run.records.append(rec)
    else:
        seen = set()
        for step in wf.steps:
            ctx = StepContext(store, branch, world, llm, step, overrides, depth)
            if step.when is not None and not step.when(ctx):
                continue                                   # guard reads stay in ctx.reads for the record
            seen.add(step.name)
            old = prior_recs.get(step.name)
            if (old is not None and step.name not in dirty and old.code == step.code
                    and verified(old, store, branch)):
                run.records.append(_replay(old, store, branch))
                report["reused"].append(step.name)
                continue
            run.records.append(rerun(step, old, ctx))
        for name, old in prior_recs.items():               # orphans: steps that no longer run
            if name not in seen:
                _reconcile(old, world, report)
                run.orphans.append(name)

    fresh = [r for r in run.records if not r.reused]
    report.update(orphans=list(run.orphans), llm_calls=sum(r.cost.llm_calls for r in fresh),
                  tokens=sum(r.cost.tokens for r in fresh), latency_s=time.perf_counter() - start)
    run.report = report
    return run
