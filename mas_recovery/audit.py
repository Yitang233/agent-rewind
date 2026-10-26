"""Scoped, cached audits (M3) and cone-cost placement (M4).

An audit checks one atom (unit) against the values of only the paths it cites (context; with
`scope='step'` the step's whole read set, the conservative choice). The score is the logprob of
"yes" from a checker; below the calibrated threshold the verdict is "unsupported".

Verdicts are graph nodes `/audit/<step>/_atoms/<k>` keyed by hash(atom text, source hashes).
`Auditor.after_step` is called by `engine.execute(audit=...)` after every step, rerun or reused.
A verdict of `prior.audits` is reused while its key is unchanged, so after a recovery an audit
reruns iff the atom text or a cited source hash changed.

Policies
    none        no audits
    always      Inspector-style: the whole output of every model call, checked on every
                execution (rerun or reused steps alike); no cache
    uniform     each atom audited with probability `p` (the budget of another policy); no cache
    m3          every atom, scoped, no cache
    m3+cache    every atom, scoped, verdicts cached by key
    m3+m4       cached, and an atom with no valid verdict is audited only when placement says so:
                  risk(a) * cost(cone(a)) + w_irr * [cone reaches an irreversible intent] > cost(audit)
                or it had an audited verdict whose key changed.
"""
from __future__ import annotations

import hashlib
import math
import random
import re

from .store import canon, digest, overlaps

POLICIES = ("none", "always", "uniform", "m3", "m3+cache", "m3+m4")


# ---- prompts and checkers -----------------------------------------------------------------------

def sources_text(sources: dict) -> str:
    return "\n".join(f"[{p}] = {canon(v)}" for p, v in sources.items())


ATOM_SYSTEM = "You are a strict fact checker. Answer with one word: yes or no."


def atom_prompt(sources: dict, atom: str) -> tuple[str, str]:
    return ATOM_SYSTEM, (f"Sources:\n{sources_text(sources)}\n\nClaim: {atom}\n\n"
                         "Is the claim fully supported by the sources above? Answer yes or no.")


def output_prompt(groups, atoms, answer) -> tuple[str, str]:
    """Inspector-style: all inputs of the call and the whole output."""
    inputs = "\n".join(f"## {g['title']}\n{sources_text(g['paths'])}" for g in groups)
    out = "\n".join(f"- {a['text']}" for a in atoms)
    return ATOM_SYSTEM, (f"Inputs:\n{inputs}\n\nAgent output:\n{out}\nAnswer: {canon(answer)}\n\n"
                         "Is every statement of the agent output correct and supported by the inputs, and is the "
                         "answer right? Answer yes or no.")


class LlamaChecker:
    """Checker A: the same local model; score = log P("yes") of the first answer token."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.llm = inner.llm
        self.memo: dict[str, tuple[float, int]] = {}
        self.yes = [self.llm.tokenize(w, add_bos=False)[0] for w in (b"yes", b"Yes")]
        self.calls = 0

    def _ids(self, sp: tuple[str, str]) -> list[int]:
        from .atoms import chatml
        return self.llm.tokenize(chatml(*sp).encode(), add_bos=False, special=True)

    def _score(self, sp: tuple[str, str]) -> tuple[float, int]:
        key = hashlib.sha256(canon(sp).encode()).hexdigest()
        ids = self._ids(sp)
        if key not in self.memo:                       # memo only saves compute; tokens are charged anyway
            import llama_cpp
            import numpy as np
            self.llm.reset()
            self.llm.eval(ids)
            row = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(self.llm._ctx.ctx, -1),
                                        shape=(self.llm.n_vocab(),)).astype(np.float64)
            m = row.max()
            lse = m + np.log(np.exp(row - m).sum())
            self.memo[key] = (float(np.logaddexp.reduce(row[self.yes]) - lse), 0)
            self.calls += 1
        return self.memo[key][0], len(ids) + 1

    def cost(self, sources: dict, atom: str) -> int:
        return len(self._ids(atom_prompt(sources, atom))) + 1

    def check_atom(self, sources: dict, atom: str) -> tuple[float, int]:
        return self._score(atom_prompt(sources, atom))

    def check_output(self, groups, atoms, answer) -> tuple[float, int]:
        return self._score(output_prompt(groups, atoms, answer))


class RuleChecker:
    """Deterministic stand-in for tests: every number and address of the text must occur in the sources."""

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _toks(s: str) -> set[str]:
        return set(re.findall(r"\d+|[\w.]+@[\w.]+\w", s))

    def _supported(self, sources_s: str, text: str) -> float:
        self.calls += 1
        return 0.0 if self._toks(text) <= self._toks(sources_s) else -8.0

    def cost(self, sources: dict, atom: str) -> int:
        return len((sources_text(sources) + atom).split())

    def check_atom(self, sources: dict, atom: str) -> tuple[float, int]:
        return self._supported(sources_text(sources), atom), self.cost(sources, atom)

    def check_output(self, groups, atoms, answer) -> tuple[float, int]:
        src = "\n".join(sources_text(g["paths"]) for g in groups)
        text = " ".join(a["text"] for a in atoms) + " " + canon(answer)
        return self._supported(src, text), len((src + text).split())


# ---- cone cost (M4) -----------------------------------------------------------------------------

def step_tokens(rec) -> int:
    return sum(c.get("nominal", 0) for c in rec.calls) or rec.cost.tokens


def cone_costs(trace) -> dict[str, tuple[int, bool]]:
    """step -> (tokens of the step and everything downstream of it, does the cone hold an
    irreversible intent), read from the recorded trace of a run."""
    recs = trace.records
    down: dict[str, set[str]] = {r.step: set() for r in recs}
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            if any(overlaps(w, p) for w, _ in a.writes for p, _ in b.reads):
                down[a.step].add(b.step)
    by = {r.step: r for r in recs}
    out = {}
    for s in by:
        seen, todo = {s}, [s]
        while todo:
            for n in down[todo.pop()]:
                if n not in seen:
                    seen.add(n)
                    todo.append(n)
        out[s] = (sum(step_tokens(by[n]) for n in seen),
                  any(i.cls == "irreversible" for n in seen for i in by[n].intents))
    return out


def risk_of(conf: dict | None, k: int) -> float:
    """1 - geometric-mean token probability of atom k; 1.0 when no confidence is known."""
    try:
        m = conf["atoms"][k]["mean"]
    except (KeyError, IndexError, TypeError):
        return 1.0
    return 1.0 if m is None else 1.0 - math.exp(m)


def should_audit(risk: float, cone_tokens: float, irreversible: bool, audit_tokens: float, w_irr: float) -> bool:
    return risk * cone_tokens + (w_irr if irreversible else 0.0) > audit_tokens


# ---- the auditor --------------------------------------------------------------------------------

class Auditor:
    def __init__(self, checker, policy: str = "m3+cache", threshold: float = -1.0, scope: str = "cited",
                 trace=None, w_irr: float = 0.0, p: float = 1.0, seed: int = 0,
                 out_threshold: float | None = None) -> None:
        if policy not in POLICIES:
            raise ValueError(f"audit policy {policy!r}")
        self.checker, self.policy, self.threshold, self.scope = checker, policy, threshold, scope
        self.out_threshold = threshold if out_threshold is None else out_threshold
        self.w_irr, self.p, self.rng = w_irr, p, random.Random(seed)
        self.cones = cone_costs(trace) if trace is not None else {}

    # sources of atom k of a call
    def sources(self, call: dict, atom: dict) -> dict:
        everything = {p: v for g in call["groups"] for p, v in g["paths"].items()}
        if self.scope == "step" or not atom.get("cites"):
            return everything
        return {p: everything[p] for p in atom["cites"] if p in everything}

    def key(self, atom: dict, sources: dict, store, branch, cutoff) -> str:
        return digest(canon([atom["text"], [(p, cutoff.hash(store, branch, p)) for p in sources], self.scope]))

    def _decide(self, call, k, atom, sources, pv) -> bool:
        if self.policy in ("m3", "m3+cache"):
            return True
        if self.policy == "uniform":
            return self.rng.random() < self.p
        # m3+m4
        if pv is not None and pv.get("audited"):
            return True                                           # key changed: re-audit
        cone, irr = self.cones.get(call["step"], (call.get("nominal", 0), False))
        return should_audit(risk_of(call.get("conf"), k), cone, irr, self.checker.cost(sources, atom["text"]),
                            self.w_irr)

    def after_step(self, run, rec, prior, store, branch, cutoff) -> None:
        if self.policy == "none":
            return
        cache = self.policy in ("m3+cache", "m3+m4") and prior is not None
        k = 0
        for j, call in enumerate(rec.calls):
            call = dict(call, step=rec.step)
            if self.policy == "always":
                score, tokens = self.checker.check_output(call["groups"], call["atoms"], call["answer"])
                run.audits[f"{rec.step}/_output/{j}"] = {"kind": "output", "step": rec.step, "audited": True,
                    "reused": False, "score": score, "ok": score >= self.out_threshold, "tokens": tokens}
                continue
            for atom in call["atoms"]:
                node = f"{rec.step}/{k}"
                src = self.sources(call, atom)
                key = self.key(atom, src, store, branch, cutoff)
                pv = prior.audits.get(node) if cache else None
                if pv is not None and pv["key"] == key and (pv["audited"] or self.policy == "m3+m4"):
                    v = dict(pv, reused=True, tokens=0)                     # unchanged key: nothing to do
                elif self._decide(call, k, atom, src, pv):
                    score, tokens = self.checker.check_atom(src, atom["text"])
                    v = {"kind": "atom", "step": rec.step, "k": k, "key": key, "audited": True, "reused": False,
                         "score": score, "ok": score >= self.threshold, "tokens": tokens, "atom": atom["text"],
                         "risk": risk_of(call.get("conf"), k)}
                else:
                    v = {"kind": "atom", "step": rec.step, "k": k, "key": key, "audited": False, "reused": False,
                         "score": None, "ok": None, "tokens": 0, "atom": atom["text"], "risk": risk_of(call.get("conf"), k)}
                run.audits[node] = v
                if v["audited"]:
                    store.write(branch, f"/audit/{rec.step}/_atoms/{k}",
                                {"key": v["key"], "ok": v["ok"], "score": v["score"]})
                k += 1

    @staticmethod
    def summary(run) -> dict:
        vs = list(run.audits.values())
        done = [v for v in vs if v["audited"] and not v["reused"]]
        return {"audits_run": len(done), "audit_tokens": sum(v["tokens"] for v in done),
                "reused": sum(1 for v in vs if v["reused"]), "skipped": sum(1 for v in vs if not v["audited"]),
                "flagged": sorted(n for n, v in run.audits.items() if v["audited"] and v["ok"] is False)}


# ---- helpers for experiments --------------------------------------------------------------------

class FaultyLLM:
    """Wraps a CachedLLM and edits the atoms/answer of chosen steps after the model call
    (a hallucinating or injected agent). edits: {step: fn(atoms, answer) -> (atoms, answer)}."""

    def __init__(self, inner, edits: dict) -> None:
        self.inner, self.edits = inner, edits
        self.cur_step = None
        self.last_atoms, self.last_conf, self.last_nominal = [], {}, 0

    def complete(self, role, task, inputs):
        self.inner.cur_step = self.cur_step
        out, used = self.inner.complete(role, task, inputs)
        atoms, answer = [dict(a) for a in self.inner.last_atoms], dict(out)
        if self.cur_step in self.edits:
            atoms, answer = self.edits[self.cur_step](atoms, answer)
        self.last_atoms, self.last_conf, self.last_nominal = atoms, self.inner.last_conf, self.inner.last_nominal
        return answer, used


def calibrate(scores: list[float], fpr: float = 0.05) -> float:
    """Threshold such that at most `fpr` of the clean scores fall strictly below it."""
    s = sorted(scores)
    return s[int(fpr * len(s))] if s else -1.0
