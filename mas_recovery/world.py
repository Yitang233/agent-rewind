"""Simulated environment and tools, each tagged with an effect class.

read          -- no side effect
idempotent    -- repeating the call with the same key changes nothing more (a document set)
compensable   -- has a compensating action (a booking can be cancelled)
irreversible  -- cannot be undone (an email stays sent); recovery can only report it

`OPEN_WORLD` tools return data from outside the system; their results start untrusted.
Latency is off by default. `latency=True` (or a {tool: ms} dict) charges each call to a
virtual clock (`clock_ms`); `sleep=True` also sleeps for real.
"""
from __future__ import annotations

import random
import time

EFFECT_CLASS = {
    "kb.lookup": "read",
    "web.fetch": "read",
    "doc.set": "idempotent",
    "calendar.book": "compensable",
    "email.send": "irreversible",
}
OPEN_WORLD = {"web.fetch"}
LATENCY_MS = {"kb.lookup": 50, "web.fetch": 50, "doc.set": 50, "calendar.book": 300, "email.send": 200}


class World:
    def __init__(self, kb: dict, latency=None, seed: int = 0, jitter: float = 0.0, sleep: bool = False) -> None:
        self.kb = kb                 # ground truth the researcher looks up
        self.bookings: list[dict] = []
        self.outbox: list[dict] = []
        self.docs: dict[str, object] = {}
        self.calls: list[tuple[str, str | None]] = []      # (tool, idempotency key) of every call
        self.delivered: dict[str, object] = {}             # key -> result of released irreversible calls
        self.latency = dict(LATENCY_MS) if latency is True else (dict(latency) if latency else None)
        self.jitter, self.sleep, self.clock_ms = jitter, sleep, 0.0
        self.rng = random.Random(seed)

    def _charge(self, tool: str) -> None:
        if not self.latency:
            return
        ms = self.latency.get(tool, 0.0) * (1 + self.jitter * (2 * self.rng.random() - 1))
        self.clock_ms += ms
        if self.sleep:
            time.sleep(ms / 1000)

    def call(self, tool: str, _key: str | None = None, **args):
        self._charge(tool)
        self.calls.append((tool, _key))
        if tool == "kb.lookup":
            return self.kb.get(args["key"])
        if tool == "web.fetch":
            return self.kb.get(args["url"])
        if tool == "doc.set":
            self.docs[args["doc"]] = args["value"]
            return args["doc"]
        if tool == "calendar.book":
            booking = {"id": f"b{len(self.bookings)}", "room": args["room"], "active": True}
            self.bookings.append(booking)
            return booking["id"]
        if tool == "email.send":
            self.outbox.append({"to": list(args["to"]), "body": args["body"], "key": _key})
            result = len(self.outbox) - 1
            if _key is not None:
                self.delivered[_key] = result
            return result
        raise KeyError(tool)

    def compensate(self, tool: str, result) -> bool:
        """Undo one effect if its tool allows it. Returns False for irreversible effects."""
        if tool == "calendar.book":
            for b in self.bookings:
                if b["id"] == result:
                    b["active"] = False
            return True
        if tool == "doc.set":
            self.docs.pop(result, None)
            return True
        return EFFECT_CLASS[tool] == "read"

    def duplicates(self, tool: str = "email.send") -> int:
        """Irreversible calls beyond the first for the same idempotency key."""
        keys = [k for t, k in self.calls if t == tool and k is not None]
        return len(keys) - len(set(keys))

    def outcome(self) -> dict:
        """What the task checker looks at: active bookings and the last email sent."""
        return {
            "rooms": sorted(b["room"] for b in self.bookings if b["active"]),
            "last_email": ({k: v for k, v in self.outbox[-1].items() if k != "key"} if self.outbox else None),
        }

    def unauthorized(self) -> int:
        """Emails ever sent to anyone outside the team (the policy is checked, not enforced)."""
        members = set(self.kb["team.members"])
        return sum(1 for m in self.outbox for to in m["to"] if to not in members)
