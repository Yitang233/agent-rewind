"""Researcher -> planner -> executor workflow with a checkable outcome.

Task: book the smallest room that fits the team, and email the team the agenda.
Steps read and write named artifacts through a StepContext, which records every
read, write, tool effect and model call into the current commit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .faults import Fault
from .trace import Commit, Effect, Trace
from .world import EFFECT_CLASS, World

ROOMS = ["A", "B", "C"]

KB = {
    "capacity.A": 8,
    "capacity.B": 16,
    "capacity.C": 30,
    "team.size": 12,
    "team.members": ["ana@lab.edu", "bo@lab.edu", "cy@lab.edu"],
    "agenda.topic": "Q4 roadmap",
}


class StepContext:
    def __init__(self, trace: Trace, commit: Commit, world: World, llm, faults: list[Fault]) -> None:
        self.trace, self.commit, self.world, self.llm, self.faults = trace, commit, world, llm, faults

    def read(self, name: str):
        art_id = self.trace.latest[name]
        if art_id not in self.commit.reads:
            self.commit.reads.append(art_id)
        return self.trace.artifacts[art_id].value

    def write(self, name: str, value) -> None:
        for f in self.faults:
            if f.step == self.commit.step and f.name == name:
                value = f.value
        art = self.trace.add_artifact(name, value, self.commit)
        self.commit.writes.append(art.id)

    def tool(self, tool: str, **args):
        result = self.world.call(tool, **args)
        self.commit.effects.append(Effect(tool, EFFECT_CLASS[tool], args, result))
        return result

    def ask(self, task: str, inputs: dict) -> dict:
        out, used = self.llm.complete(self.commit.agent, task, inputs)
        self.commit.llm_calls += 1
        self.commit.tokens += used
        return out


@dataclass
class Step:
    name: str
    agent: str
    fn: Callable[[StepContext], None]


def research(keys: list[tuple[str, str]]):
    """Look up each kb key and publish it under an artifact name."""
    def fn(ctx: StepContext) -> None:
        for key, name in keys:
            found = ctx.tool("kb.lookup", key=key)
            ctx.write(name, ctx.ask("answer", {"question": key, "lookup": found})["value"])
    return fn


def plan(ctx: StepContext) -> None:
    out = ctx.ask("plan", {
        "capacities": {r: ctx.read(f"fact.capacity.{r}") for r in ROOMS},
        "team_size": ctx.read("state.team.size"),
        "agenda_topic": ctx.read("fact.agenda.topic"),
    })
    ctx.write("plan.room", out["room"])
    ctx.write("plan.agenda", out["agenda"])


def book(ctx: StepContext) -> None:
    room = ctx.read("plan.room")
    ctx.tool("calendar.book", room=room)
    ctx.write("state.booking", {"room": room})


def email(ctx: StepContext) -> None:
    out = ctx.ask("compose_email", {"agenda": ctx.read("plan.agenda"), "members": ctx.read("state.team.members")})
    ctx.tool("email.send", to=out["to"], body=out["body"])
    ctx.write("state.email", out)


WORKFLOW = [
    # Team facts go to shared state; later steps read them from there, not from a message.
    Step("research.team", "researcher", research([("team.size", "state.team.size"),
                                                  ("team.members", "state.team.members")])),
    Step("research.agenda", "researcher", research([("agenda.topic", "fact.agenda.topic")])),
    Step("research.capacity", "researcher", research([(f"capacity.{r}", f"fact.capacity.{r}") for r in ROOMS])),
    Step("plan", "planner", plan),
    Step("execute.book", "executor", book),
    Step("execute.email", "executor", email),
]
STEPS = {s.name: s for s in WORKFLOW}


def run_step(step: Step, trace: Trace, world: World, llm, faults: list[Fault], rerun_of: str | None = None) -> Commit:
    commit = trace.new_commit(step.name, step.agent, rerun_of)
    step.fn(StepContext(trace, commit, world, llm, faults))
    return commit


def run(world: World, llm, faults: list[Fault] = (), verify: bool = False) -> tuple[Trace, dict]:
    """Run the workflow once. With verify=True every researcher output is checked by an
    extra model call and the step is retried without the fault when the check fails
    (the always-verify baseline)."""
    trace, start = Trace(), time.perf_counter()
    stats = {"verify_calls": 0, "verify_flags": 0}
    for step in WORKFLOW:
        commit = run_step(step, trace, world, llm, list(faults))
        if verify and step.agent == "researcher":
            for art_id in commit.writes:
                art = trace.artifacts[art_id]
                key = art.name.split(".", 1)[1]
                ok, used = llm.complete("verifier", "verify", {"key": key, "value": art.value})
                stats["verify_calls"] += 1
                commit.tokens += used
                if not ok["ok"]:
                    stats["verify_flags"] += 1
                    run_step(step, trace, world, llm, [], rerun_of=commit.id)
                    break
    stats["latency_s"] = time.perf_counter() - start
    return trace, stats
