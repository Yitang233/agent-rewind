"""Structured reasoning atoms (M2): schema, prompt layout, parser, confidence, cited cutoff.

An LLM step's output is {"atoms": [{"cites": [paths], "text": str}, ...], "answer": {...}}.
There is one atom per input *group*, in the order the groups appear in the prompt (the
canonical order). Inputs are a blackboard: named paths with values, grouped, shared
(stable) groups first.

    groups = [Group("Rooms", {"/msg/.../A": 8, ...}), Group("Team", {"/kv/team/size": 12})]
    prompt = build_prompt("planner", INSTRUCTIONS, groups)
    schema = atom_schema(groups, answer_schema)

`parse` is tolerant of the model's own formatting; `confidence` maps token logprobs onto
atoms; `can_skip` is the optional (lossy) cited-cutoff check.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .store import overlaps


@dataclass
class Group:
    title: str
    paths: dict                       # path -> value, in prompt order


def groups_to_json(groups) -> list:
    return [{"title": g.title, "paths": g.paths} for g in groups]


def groups_from_json(raw) -> list[Group]:
    return [Group(g["title"], dict(g["paths"])) for g in raw]


def all_paths(groups) -> list[str]:
    return [p for g in groups for p in g.paths]


def build_prompt(role: str, instructions: str, groups, answer_hint: str) -> tuple[str, str]:
    """(system, user) texts. Stable text first; the volatile groups come last."""
    system = f"You are the {role} in a multi-agent workflow. Reply with JSON only."
    lines = [instructions, "",
             f"Write exactly {len(groups)} atoms, one per section below and in the same order. "
             "Each atom has `cites` (the paths of that section you used) and `text` (one short "
             f"sentence). Then give the answer. {answer_hint}", ""]
    for g in groups:
        lines.append(f"## {g.title}")
        lines += [f"[{p}] = {json.dumps(v)}" for p, v in g.paths.items()]
        lines.append("")
    return system, "\n".join(lines).rstrip() + "\n"


def chatml(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")


def atom_schema(groups, answer_schema: dict, max_text: int = 160) -> dict:
    paths = all_paths(groups)
    atom = {
        "type": "object",
        "properties": {
            "cites": {"type": "array", "items": {"enum": paths}, "maxItems": max(len(paths), 1)},
            "text": {"type": "string", "maxLength": max_text},
        },
        "required": ["cites", "text"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "atoms": {"type": "array", "items": atom, "minItems": len(groups), "maxItems": len(groups)},
            "answer": answer_schema,
        },
        "required": ["atoms", "answer"],
        "additionalProperties": False,
    }


@dataclass
class Parsed:
    atoms: list[dict] = field(default_factory=list)
    answer: dict | None = None
    spans: list[tuple[int, int]] = field(default_factory=list)   # char span of each atom in the text
    answer_span: tuple[int, int] | None = None
    ok: bool = False
    error: str = ""


def parse(text: str) -> Parsed:
    """Parse model output into atoms + answer. Never raises; `ok` says whether both are usable.
    Falls back to scanning when the whole text is not valid JSON (truncated output)."""
    out = Parsed()
    dec = json.JSONDecoder()
    try:
        whole = json.loads(text)
        atoms, answer = whole.get("atoms"), whole.get("answer")
        if not isinstance(atoms, list) or not isinstance(answer, dict):
            raise ValueError("missing atoms or answer")
    except (ValueError, AttributeError) as e:
        out.error = str(e)
        atoms, answer = [], None
    m = re.search(r'"atoms"\s*:\s*\[', text)
    if m:
        i = m.end()
        spans, found = [], []
        while True:
            while i < len(text) and text[i] in " \n\t,":
                i += 1
            if i >= len(text) or text[i] != "{":
                break
            try:
                obj, j = dec.raw_decode(text, i)
            except ValueError:
                break
            spans.append((i, j))
            found.append(obj)
            i = j
        out.spans = spans
        if not atoms:
            atoms = found
    m = re.search(r'"answer"\s*:\s*', text)
    if m:
        try:
            obj, j = dec.raw_decode(text, m.end())
            out.answer_span = (m.end(), j)
            if answer is None and isinstance(obj, dict):
                answer = obj
        except ValueError:
            pass
    out.atoms = [{"cites": list(a.get("cites", [])), "text": str(a.get("text", ""))}
                 for a in atoms if isinstance(a, dict)]
    out.answer = answer
    out.ok = bool(out.atoms) and isinstance(answer, dict)
    if out.ok:
        out.error = ""
    return out


def token_char_spans(tokens: list[str]) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for t in tokens:
        spans.append((pos, pos + len(t)))
        pos += len(t)
    return spans


def _stats(lps: list[float]) -> dict:
    if not lps:
        return {"mean": None, "min": None, "n": 0}
    return {"mean": sum(lps) / len(lps), "min": min(lps), "n": len(lps)}


def confidence(parsed: Parsed, tokens: list[str], logprobs: list[float | None]) -> dict:
    """Mean and min token logprob per atom and for the answer. A token belongs to an atom when its
    character span overlaps the atom's span."""
    tspans = token_char_spans(tokens)

    def within(span):
        if span is None:
            return []
        a, b = span
        return [lp for (s, e), lp in zip(tspans, logprobs) if lp is not None and s < b and e > a]

    return {"atoms": [_stats(within(s)) for s in parsed.spans],
            "answer": _stats(within(parsed.answer_span))}


def can_skip(atoms: list[dict], changed: list[str]) -> bool:
    """Cited-cutoff check (lossy, optional tier): True when no atom cites a changed path or one
    below/above it. The model may have used an input it did not cite, so every skip is audited
    by a scratch rerun elsewhere."""
    return not any(overlaps(c, p) for a in atoms for c in a["cites"] for p in changed)


def check_citations(atoms: list[dict], groups) -> dict:
    """Structural quality: atom count matches groups, every cite is a known path and lies in its
    own group, and every group's paths are cited at least once somewhere."""
    known = all_paths(groups)
    n_ok = len(atoms) == len(groups)
    unknown = [c for a in atoms for c in a["cites"] if c not in known]
    own = sum(1 for a, g in zip(atoms, groups) if all(c in g.paths for c in a["cites"]))
    cited = {c for a in atoms for c in a["cites"]}
    return {"count_ok": n_ok, "unknown_cites": len(unknown), "own_group": own,
            "uncited": [p for p in known if p not in cited]}
