"""Researcher -> planner -> executor workflow with a checkable outcome.

Task: book the smallest room that fits the team, and email the team the agenda.
Facts live under `/msg/<agent>/<step>/...`, shared team state under `/kv/...`.
`NAMES` maps the artifact names used by `faults.py` to those paths.
"""
from __future__ import annotations

import time

from ..engine import Run, Step, StepContext, Workflow, execute
from ..faults import Fault
from ..world import World

ROOMS = ["A", "B", "C"]

KB = {
    "capacity.A": 8,
    "capacity.B": 16,
    "capacity.C": 30,
    "team.size": 12,
    "team.members": ["ana@lab.edu", "bo@lab.edu", "cy@lab.edu"],
    "agenda.topic": "Q4 roadmap",
}

CAP = "/msg/researcher/research.capacity"
AGENDA = "/msg/researcher/research.agenda/topic"
SIZE, MEMBERS = "/kv/team/size", "/kv/team/members"
ROOM, PLAN_AGENDA = "/msg/planner/plan/room", "/msg/planner/plan/agenda"
BOOKING, EMAIL = "/kv/booking", "/kv/email"

NAMES = {
    "state.team.size": SIZE,
    "state.team.members": MEMBERS,
    "fact.agenda.topic": AGENDA,
    **{f"fact.capacity.{r}": f"{CAP}/{r}" for r in ROOMS},
    "plan.room": ROOM,
    "plan.agenda": PLAN_AGENDA,
    "state.booking": BOOKING,
    "state.email": EMAIL,
}
KEY = {path: name.split(".", 1)[1] for name, path in NAMES.items()}   # path -> kb key ("capacity.A")


def research(keys: list[tuple[str, str]]):
    """Look up each kb key and publish it at a path."""
    def fn(ctx: StepContext) -> None:
        for key, path in keys:
            found = ctx.tool("kb.lookup", key=key)
            ctx.write(path, ctx.ask("answer", {"question": key, "lookup": found})["value"])
    return fn


def plan(ctx: StepContext) -> None:
    out = ctx.ask("plan", {
        "capacities": {r: ctx.read(f"{CAP}/{r}") for r in ROOMS},
        "team_size": ctx.read(SIZE),
        "agenda_topic": ctx.read(AGENDA),
    })
    ctx.write(ROOM, out["room"])
    ctx.write(PLAN_AGENDA, out["agenda"])


def book(ctx: StepContext) -> None:
    room = ctx.read(ROOM)
    ctx.tool("calendar.book", room=room)
    ctx.write(BOOKING, {"room": room})


def email(ctx: StepContext) -> None:
    out = ctx.ask("compose_email", {"agenda": ctx.read(PLAN_AGENDA), "members": ctx.read(MEMBERS)})
    ctx.tool("email.send", to=out["to"], body=out["body"])
    ctx.write(EMAIL, out)


WORKFLOW = [
    # Team facts go to shared state; later steps read them from there, not from a message.
    Step("research.team", "researcher", research([("team.size", SIZE), ("team.members", MEMBERS)])),
    Step("research.agenda", "researcher", research([("agenda.topic", AGENDA)])),
    Step("research.capacity", "researcher", research([(f"capacity.{r}", f"{CAP}/{r}") for r in ROOMS])),
    Step("plan", "planner", plan),
    Step("execute.book", "executor", book),
    Step("execute.email", "executor", email),
]
STEPS = {s.name: s for s in WORKFLOW}
BOOKING_WORKFLOW = Workflow(WORKFLOW, NAMES)


def overrides(faults) -> dict:
    return {(f.step, NAMES[f.name]): f.value for f in faults}


def run(world: World, llm, faults: list[Fault] = (), verify: bool = False,
        depth: int | None = None) -> tuple[Run, dict]:
    """Run the workflow once. With verify=True every researcher output is checked by an
    extra model call and the step is retried without the fault when the check fails
    (the always-verify baseline)."""
    start = time.perf_counter()
    stats = {"verify_calls": 0, "verify_flags": 0}

    def check(step, rec, store, branch) -> bool:
        if step.agent != "researcher":
            return False
        for path, _ in rec.writes:
            ok, used = llm.complete("verifier", "verify", {"key": KEY[path], "value": rec.values[path]})
            stats["verify_calls"] += 1
            rec.cost.tokens += used
            if not ok["ok"]:
                stats["verify_flags"] += 1
                return True
        return False

    result = execute(BOOKING_WORKFLOW, world, llm, depth=depth, overrides=overrides(faults),
                     after_step=check if verify else None)
    stats["latency_s"] = time.perf_counter() - start
    return result, stats
