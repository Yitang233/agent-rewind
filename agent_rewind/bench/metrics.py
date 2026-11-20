"""Per-cell measurement helpers: call recording, hash timing, bookkeeping, outcome hashes."""
from __future__ import annotations

import hashlib
import resource
import sys
import time

from .. import cutoff as _cutoff
from ..cutoff import Canonical
from ..deps import Meter
from ..llm import CachedLLM
from ..store import canon


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def _totals(inner) -> dict:
    t = getattr(inner, "totals", None)
    return dict(t) if t else {}


def _delta(a: dict, b: dict) -> dict:
    return {k: round(b.get(k, 0) - a.get(k, 0), 4) for k in b}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class RecordingLLM(CachedLLM):
    """CachedLLM that also logs, per model call, (step, call_index, sha256 of the output text, n_tokens).
    `dedupe=False` sends every call to the backend (live-cost runs). For a backend without text (scripted) the
    hash is over the canonical JSON of the answer and n_tokens is 0."""

    def __init__(self, inner, dedupe: bool = True) -> None:
        super().__init__(inner, dedupe)
        self.log: list[dict] = []
        self._n: dict[str, int] = {}

    def complete(self, role: str, task: str, inputs: dict):
        t = getattr(self.inner, "totals", None)
        before = t["calls"] if t else 0
        out, used = super().complete(role, task, inputs)
        step = getattr(self, "cur_step", None)
        i = self._n.get(step, 0)
        self._n[step] = i + 1
        if t is not None and t["calls"] > before:
            text, n = self.inner.last_text, int(getattr(self.inner, "last_out_tokens", 0))
        else:
            text, n = canon(out), 0
        self.log.append({"step": step, "call_index": i, "sha256": _sha(text), "n_tokens": n})
        return out, used


class MeteredCutoff:
    """Equality-tier proxy that times every `hash` call. `ns_inside` is the part spent under StepContext.read/write,
    which deps.Meter already counts in ns_read / ns_write; the rest (`verified`, audit keys) is outside them."""

    def __init__(self, base) -> None:
        self.base = _cutoff.get(base)
        self.ns = self.ns_inside = self.n = 0

    def hash(self, store, branch, path):
        inside = sys._getframe(1).f_code.co_name in ("read", "write")
        t0 = time.perf_counter_ns()
        h = self.base.hash(store, branch, path)
        dt = time.perf_counter_ns() - t0
        self.ns += dt
        self.n += 1
        if inside:
            self.ns_inside += dt
        return h

    def value(self, path, value):
        return self.base.value(path, value)

    def __getattr__(self, name):
        return getattr(self.base, name)


def _bookkeeping(meter: Meter, mc: MeteredCutoff) -> dict:
    return {"ns_read": meter.read_ns, "ns_write": meter.write_ns, "ns_hash": mc.ns, "ns_hash_outside": mc.ns - mc.ns_inside,
            "n_read": meter.reads, "n_write": meter.writes, "n_hash": mc.n,
            "journal_bytes": meter.meta_bytes + meter.value_bytes, "journal_meta_bytes": meter.meta_bytes,
            "steps": meter.steps}


def _outcome(leaves: dict) -> dict:
    """Canonical hash per outcome path: every leaf outside /msg (agent messages) and /audit."""
    c = Canonical()
    return {p: c.value(p, v) for p, v in leaves.items() if not p.startswith(("/msg/", "/audit/"))}


def _checker_counts(models, real: bool) -> tuple[int, int]:
    c = models.checker if real else None
    return (getattr(c, "calls", 0), getattr(c, "memo_hits", 0)) if c is not None else (0, 0)
