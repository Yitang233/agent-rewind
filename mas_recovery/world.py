"""Simulated environment and tools, each tagged with an effect class.

read          -- no side effect
compensable   -- has a compensating action (a booking can be cancelled)
irreversible  -- cannot be undone (an email stays sent); recovery can only report it
"""
from __future__ import annotations

EFFECT_CLASS = {
    "kb.lookup": "read",
    "calendar.book": "compensable",
    "email.send": "irreversible",
}


class World:
    def __init__(self, kb: dict) -> None:
        self.kb = kb                 # ground truth the researcher looks up
        self.bookings: list[dict] = []
        self.outbox: list[dict] = []

    def call(self, tool: str, **args):
        if tool == "kb.lookup":
            return self.kb.get(args["key"])
        if tool == "calendar.book":
            booking = {"id": f"b{len(self.bookings)}", "room": args["room"], "active": True}
            self.bookings.append(booking)
            return booking["id"]
        if tool == "email.send":
            self.outbox.append({"to": list(args["to"]), "body": args["body"]})
            return len(self.outbox) - 1
        raise KeyError(tool)

    def compensate(self, tool: str, result) -> bool:
        """Undo one effect if its tool allows it. Returns False for irreversible effects."""
        if tool == "calendar.book":
            for b in self.bookings:
                if b["id"] == result:
                    b["active"] = False
            return True
        return EFFECT_CLASS[tool] == "read"

    def outcome(self) -> dict:
        """What the task checker looks at: active bookings and the last email sent."""
        return {
            "rooms": sorted(b["room"] for b in self.bookings if b["active"]),
            "last_email": self.outbox[-1] if self.outbox else None,
        }

    def unauthorized(self) -> int:
        """Emails ever sent to anyone outside the team (the policy is checked, not enforced)."""
        members = set(self.kb["team.members"])
        return sum(1 for m in self.outbox for to in m["to"] if to not in members)
