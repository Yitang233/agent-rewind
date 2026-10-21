"""Commit log: every agent step is a commit; everything it passes on is a versioned artifact.

Messages and shared-state keys are both artifacts (`fact.*`, `plan.*`, `state.*`).
Writing a name again creates a new version, so the trace doubles as a versioned
key-value store, and every read records the exact version it saw.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class Artifact:
    id: str          # "<name>@<version>"
    name: str        # logical name, e.g. "fact.capacity.A"
    value: object
    producer: str    # commit id
    step: str        # producer's step and agent, kept for coarse dependency units
    agent: str
    hash: str


@dataclass
class Effect:
    tool: str
    effect_class: str   # read | compensable | irreversible
    args: dict
    result: object = None
    undone: bool = False


@dataclass
class Commit:
    id: str
    step: str
    agent: str
    reads: list[str] = field(default_factory=list)    # artifact ids
    writes: list[str] = field(default_factory=list)   # artifact ids
    effects: list[Effect] = field(default_factory=list)
    llm_calls: int = 0
    tokens: int = 0
    rerun_of: str | None = None


class Trace:
    def __init__(self) -> None:
        self.artifacts: dict[str, Artifact] = {}
        self.commits: list[Commit] = []
        self.latest: dict[str, str] = {}      # name -> artifact id visible to the next step
        self._versions: dict[str, int] = {}
        self._next = 0

    @classmethod
    def continue_from(cls, old: "Trace") -> "Trace":
        """Empty trace for a recovery run: shares old artifacts, continues versions and ids."""
        new = cls()
        new.artifacts = dict(old.artifacts)
        new._versions = dict(old._versions)
        new._next = old._next
        return new

    def new_commit(self, step: str, agent: str, rerun_of: str | None = None) -> Commit:
        commit = Commit(id=f"c{self._next}", step=step, agent=agent, rerun_of=rerun_of)
        self._next += 1
        self.commits.append(commit)
        return commit

    def add_artifact(self, name: str, value, producer: Commit) -> Artifact:
        version = self._versions.get(name, 0) + 1
        self._versions[name] = version
        art = Artifact(id=f"{name}@{version}", name=name, value=value, producer=producer.id,
                       step=producer.step, agent=producer.agent, hash=digest(value))
        self.artifacts[art.id] = art
        self.latest[name] = art.id
        return art

    def value(self, name: str):
        return self.artifacts[self.latest[name]].value

    def downstream(self, names: set[str]) -> list[Commit]:
        """Commits that transitively read any version of `names` in this trace."""
        tainted_ids = {a.id for a in self.artifacts.values() if a.name in names}
        hit = []
        for c in self.commits:
            if tainted_ids & set(c.reads):
                hit.append(c)
                tainted_ids |= set(c.writes)
        return hit

    def dump(self, path) -> None:
        with open(path, "w") as f:
            for c in self.commits:
                row = asdict(c)
                row["outputs"] = {a: self.artifacts[a].value for a in c.writes}
                f.write(json.dumps(row, default=str) + "\n")
