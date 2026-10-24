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


# ---- W1 widened: k meetings sharing /kv/team/* -------------------------------------------------
# Researcher steps are tool-only (no model call); the planner and executor steps call the model
# with a blackboard prompt (named paths, shared groups first) and cite them in atoms.

TOPICS = ["Q4 roadmap", "Budget review", "Hiring plan", "Offsite logistics", "Release retro"]
GUESTS = [0, 3, 5, 0, 3]          # extra seats per meeting: team 12 -> B(16), B, C(30), B, B


SIZES = [12, 9, 13, 10, 14]       # team size per variant (variant 0 is the default)


def meetings_kb(k: int, variant: int = 0) -> dict:
    """Ground truth for k meetings. `variant` changes team size and topic order, so repeated runs
    of a greedy model are not identical."""
    kb = dict(KB)
    kb["team.size"] = SIZES[variant % len(SIZES)]
    for i in range(k):
        kb[f"meeting.{i}.topic"] = TOPICS[(i + variant) % len(TOPICS)]
        kb[f"meeting.{i}.guests"] = GUESTS[(i + variant) % len(GUESTS)]
    return kb


def expected_rooms(kb: dict, k: int) -> list[str]:
    need = [kb["team.size"] + kb[f"meeting.{i}.guests"] for i in range(k)]
    return sorted(min((r for r in ROOMS if kb[f"capacity.{r}"] >= n), key=lambda r: kb[f"capacity.{r}"])
                  for n in need)


def m_path(i: int, leaf: str) -> str:
    return f"/msg/researcher/research.meeting.{i}/{leaf}"


def m_room(i: int) -> str:
    return f"/msg/planner/plan.{i}/room"


def m_agenda(i: int) -> str:
    return f"/msg/planner/plan.{i}/agenda"


def _group(title: str, paths: list[str], ctx) -> dict:
    return {"title": title, "paths": {p: ctx.read(p) for p in paths}}


def _lookups(keys: list[tuple[str, str]]):
    def fn(ctx: StepContext) -> None:
        for key, path in keys:
            ctx.write(path, ctx.tool("kb.lookup", key=key))
    return fn


def plan_meeting(i: int):
    def fn(ctx: StepContext) -> None:
        groups = [   # shared groups first (stable prefix), meeting-specific last
            _group("Rooms", [f"{CAP}/{r}" for r in ROOMS], ctx),
            _group("Team", [SIZE], ctx),
            _group(f"Meeting {i}", [m_path(i, "guests"), m_path(i, "topic")], ctx),
        ]
        out = ctx.ask("plan_meeting", {"groups": groups})
        ctx.write(m_room(i), out["room"])
        ctx.write(m_agenda(i), out["agenda"])
    return fn


def book_meeting(i: int):
    def fn(ctx: StepContext) -> None:
        room = ctx.read(m_room(i))
        ctx.tool("calendar.book", room=room)
        ctx.write(f"/kv/booking/{i}", {"room": room})
    return fn


def email_meeting(i: int):
    def fn(ctx: StepContext) -> None:
        groups = [_group("Team", [MEMBERS], ctx), _group(f"Meeting {i}", [m_room(i), m_agenda(i)], ctx)]
        out = ctx.ask("email_meeting", {"groups": groups})
        ctx.tool("email.send", to=out["to"], body=out["body"])
        ctx.write(f"/kv/email/{i}", out)
    return fn


def meetings_workflow(k: int = 3) -> Workflow:
    steps = [
        Step("research.team", "researcher", _lookups([("team.size", SIZE), ("team.members", MEMBERS)])),
        Step("research.capacity", "researcher", _lookups([(f"capacity.{r}", f"{CAP}/{r}") for r in ROOMS])),
    ]
    for i in range(k):
        steps.append(Step(f"research.meeting.{i}", "researcher", _lookups(
            [(f"meeting.{i}.topic", m_path(i, "topic")), (f"meeting.{i}.guests", m_path(i, "guests"))])))
    for i in range(k):
        steps += [Step(f"plan.{i}", "planner", plan_meeting(i)),
                  Step(f"execute.book.{i}", "executor", book_meeting(i)),
                  Step(f"execute.email.{i}", "executor", email_meeting(i))]
    names = {k_: v for k_, v in NAMES.items() if k_.startswith(("state.team", "fact.capacity"))}
    for i in range(k):
        names[f"fact.agenda.topic.{i}"] = m_path(i, "topic")
        names[f"fact.agenda.guests.{i}"] = m_path(i, "guests")
        names[f"plan.room.{i}"] = m_room(i)
    return Workflow(steps, names)


def meetings_ok(world: World, k: int) -> bool:
    """Task checker: the expected active bookings, one email per meeting, each to exactly the team
    members and carrying that meeting's topic."""
    kb, sent = world.kb, world.outbox
    members = set(kb["team.members"])
    topics = [kb[f"meeting.{i}.topic"].casefold() for i in range(k)]
    return (world.outcome()["rooms"] == expected_rooms(kb, k) and len(sent) == k
            and all(set(m["to"]) == members and len(m["to"]) == len(members) for m in sent)
            and sorted(t for t in topics if any(t in m["body"].casefold() for m in sent)) == sorted(topics)
            and all(sum(t in m["body"].casefold() for t in topics) == 1 for m in sent))


def run_meetings(llm, k: int = 3, world: World | None = None, **kw) -> tuple[Run, World]:
    world = world or World(meetings_kb(k, kw.pop("variant", 0)))
    return execute(meetings_workflow(k), world, llm, **kw), world
