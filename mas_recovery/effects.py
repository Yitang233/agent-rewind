"""Effect layer: every tool call is an intent with an idempotency key.

key = h(step, tool, canonical args). It deliberately leaves out the step's read hashes:
if the inputs changed but the call is identical, it is the same effect and is reused.

Classes: read | idempotent | compensable | irreversible. Reconcile table, per old intent K:

    old K, new K                         reuse the result, no tool call     (retained)
    old K compensable/idempotent, gone   compensate                         (compensated)
    old K irreversible and released      cannot undo                        (leaked)
    old K irreversible, not released     drop it from the outbox            (withdrawn)
    no old, new K'                       call it (irreversible: stage it)   (applied)

Policies (`execute(effects=...)`):
    naive         every rerun calls every tool again; old compensables are undone first and
                  every old irreversible effect counts as redone (the earlier behaviour)
    diff          the table above; irreversible calls are made at once
    diff+outbox   as diff, but irreversible calls wait in an outbox and leave at the commit
                  point (end of `execute`). An intent whose backward cone holds an untrusted,
                  unverified node is held, not executed.

Trust labels: tool results of open-world tools and paths listed in `untrusted` start
untrusted. A step's label is the join of what it read (computed in `ctx.read`); its writes and
its intents carry that label. `verified` paths count as trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .store import canon, digest, split
from .world import EFFECT_CLASS, World

CLASSES = ("read", "idempotent", "compensable", "irreversible")
POLICIES = ("naive", "diff", "diff+outbox")


def args_hash(args: dict) -> str:
    return digest(canon(args))


def intent_key(step: str, tool: str, args: dict) -> str:
    return digest(canon([step, tool, args_hash(args)]))


@dataclass
class Intent:
    step: str
    tool: str
    args_hash: str
    key: str
    cls: str
    result: object = None
    released: bool = False       # irreversible: has it left the outbox (or been made at once)?
    args: dict = field(default_factory=dict)
    undone: bool = False
    held: bool = False           # blocked at commit by the trust rule
    withdrawn: bool = False      # dropped before release
    untrusted: bool = False      # label of its backward cone

    @property
    def effect_class(self) -> str:   # name used by the earlier Effect objects
        return self.cls


def action(old: Intent | None, still_wanted: bool) -> str:
    """The reconcile table as a function. `still_wanted`: the new execution has the same key."""
    if old is None:
        return "apply"
    if old.cls == "read":
        return "none"
    if still_wanted:
        return "reuse"
    if old.cls in ("compensable", "idempotent"):
        return "compensate"
    return "leak" if old.released else "withdraw"


def _under(path: str, root: str) -> bool:
    p, r = split(path), split(root)
    return p[:len(r)] == r


class Ledger:
    """Effect bookkeeping for one `execute` call."""

    def __init__(self, world: World, policy: str = "naive", untrusted=(), verified=()) -> None:
        if policy not in POLICIES:
            raise ValueError(f"effects policy {policy!r}")
        self.world, self.policy = world, policy
        self.untrusted, self.verified = set(untrusted), set(verified)
        self.stats = dict(retained=0, applied=0, compensated=0, leaked=0, released=0, held=0,
                          withdrawn=0, deduped=0)
        self.held: list[str] = []
        self.redone: list[str] = []          # tools of irreversible effects that were made again

    # ---- trust labels ---------------------------------------------------------------
    def is_untrusted(self, path: str) -> bool:
        if not self.untrusted:
            return False
        return any((_under(u, path) or _under(path, u)) and not any(_under(u, v) for v in self.verified)
                   for u in self.untrusted)

    def set_label(self, paths, tainted: bool) -> None:
        for p in paths:
            self.untrusted = {u for u in self.untrusted if not _under(u, p)}
            if tainted:
                self.untrusted.add(p)

    # ---- one tool call ----------------------------------------------------------------
    def call(self, ctx, tool: str, args: dict):
        cls = EFFECT_CLASS[tool]
        key = intent_key(ctx.step, tool, args)
        if cls == "read":
            result = self.world.call(tool, _key=key, **args)
            if tool in _open_world():
                ctx.tainted = True
            ctx.intents.append(Intent(ctx.step, tool, args_hash(args), key, cls, result, args=dict(args)))
            return result
        if self.policy != "naive":
            seen = ctx.local.get(key)
            if seen is not None:                         # the same call twice in one step is one intent
                return seen.result
            old = ctx.prior.get(key)
            if old is not None:
                ctx.prior_used.add(key)
                new = replace(old, held=False)
                ctx.intents.append(new)
                ctx.local[key] = new
                self.stats["retained"] += 1
                return new.result
        it = Intent(ctx.step, tool, args_hash(args), key, cls, args=dict(args), untrusted=ctx.tainted)
        if cls == "irreversible" and self.policy == "diff+outbox":
            it.result = key                              # handle; the real call happens at commit
        else:
            it.result = self.world.call(tool, _key=key, **args)
            it.released = cls == "irreversible"
        self.stats["applied"] += 1
        ctx.intents.append(it)
        ctx.local[key] = it
        return it.result

    # ---- around one step run ----------------------------------------------------------
    def begin(self, ctx, old) -> None:
        ctx.prior = {i.key: i for i in (old.intents if old else ()) if i.cls != "read" and not i.withdrawn}
        if self.policy == "naive":
            self.settle(old.intents if old else (), ())

    def end(self, ctx, old) -> None:
        if self.policy != "naive":
            self.settle(old.intents if old else (), ctx.prior_used, ctx.intents)

    def settle(self, old_intents, used, new=()) -> None:
        """Apply the table to old intents that the new execution does not have. An idempotent
        effect whose target the new execution sets again is overwritten, not compensated."""
        targets = {(i.tool, i.result) for i in new if i.cls == "idempotent"}
        for it in old_intents:
            if it.cls == "read" or it.withdrawn:
                continue
            act = action(it, it.key in used)
            if self.policy == "naive":
                if it.cls == "compensable" and not it.undone:
                    it.undone = self.world.compensate(it.tool, it.result)
                    self.stats["compensated"] += 1
                elif it.cls == "irreversible":
                    self.redone.append(it.tool)
                    self.stats["leaked"] += 1
                continue
            if act == "compensate" and (it.tool, it.result) in targets:
                continue
            if act == "compensate" and not it.undone:
                self.world.compensate(it.tool, it.result)
                self.stats["compensated"] += 1
            elif act == "leak":
                self.stats["leaked"] += 1
                self.redone.append(it.tool)
            elif act == "withdraw":
                self.stats["withdrawn"] += 1

    def replay(self, old) -> list[Intent]:
        self.set_label([p for p, _ in old.writes], old.untrusted)
        out = [replace(i) for i in old.intents]
        self.stats["retained"] += sum(i.cls != "read" for i in out)
        return out

    def commit(self, records) -> None:
        """Commit point: release staged irreversible intents unless the trust rule holds them."""
        if self.policy != "diff+outbox":
            return
        for rec in records:
            for it in rec.intents:
                if it.cls != "irreversible" or it.released or it.withdrawn:
                    continue
                if it.untrusted:
                    it.held = True
                    self.held.append(it.tool)
                    self.stats["held"] += 1
                elif it.key in self.world.delivered:
                    it.released = True
                    self.stats["deduped"] += 1
                else:
                    self.world.call(it.tool, _key=it.key, **it.args)
                    it.released = True
                    self.stats["released"] += 1


def _open_world():
    from .world import OPEN_WORLD
    return OPEN_WORLD
