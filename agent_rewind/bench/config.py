"""Flag defaults and choices, the repository ROOT, and flag access."""
from __future__ import annotations

import os


ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DEFAULTS = {"style": "atoms", "replay": "off", "audit": "none", "cutoff": "exact", "effects": "naive", "depth": "field"}
# Kept out of DEFAULTS so that v1 configs expand to the same cell keys; read with flag().
NEW_DEFAULTS = {"cache": "on", "reuse_check": "off", "audit_p": 1.0, "dirty": "producers",
                # replay settings; the defaults are the v1 behaviour of LlamaCppLLM
                "layout": "v1", "replay_K": 32, "margin": 0.3, "tie": "full",
                # audit checker: llama (the workflow model) or minicheck (lytang/MiniCheck-Flan-T5-Large)
                "checker": "llama",
                # v3: compensate a later step's faulty-run effects before a rerun step reads the world (engine.execute)
                "revert": "off",
                # v3: after a recovery a flag on a reused step no longer holds an irreversible intent (audit.Auditor.gate)
                "release": "off"}
CHOICES = {"revert": ("off", "on"), "release": ("off", "on"), "cache": ("on", "off"), "reuse_check": ("on", "off"), "dirty": ("producers", "origin"),
           "layout": ("v1", "stable"), "tie": ("full", "local"), "checker": ("llama", "minicheck")}


def flag(fl: dict, name: str):
    return fl.get(name, NEW_DEFAULTS[name]) if name in NEW_DEFAULTS else fl[name]


def _check_flags(fl: dict) -> dict:
    for k, ok in CHOICES.items():
        if k in fl and fl[k] not in ok:
            raise ValueError(f"flag {k}={fl[k]!r}: expected one of {ok}")
    if "audit_p" in fl and not 0.0 <= float(fl["audit_p"]) <= 1.0:
        raise ValueError(f"flag audit_p={fl['audit_p']!r}: expected a probability")
    return fl


METHODS = ("none", "restart", "rollback", "selective", "selective-d2")
MODEL_KEYS = {"1.5b", "7b"}
