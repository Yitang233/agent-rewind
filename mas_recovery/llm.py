"""Model backends behind one call: complete(role, task, inputs) -> dict.

ScriptedLLM is deterministic, so the harness and tests run without a model.
OllamaLLM uses a local model. CachedLLM replays identical calls for free, so reused
steps cost no tokens and repeated experiments are cheap.
"""
from __future__ import annotations

import json
import random
import re
import urllib.request

from . import atoms as A
from .trace import digest

TASK_PROMPTS = {
    "answer": "Report the looked-up value for the question. Reply {\"value\": <value>}.",
    "plan": (
        "Pick the smallest room whose capacity is at least the team size; if the team size "
        "is unknown pick the largest room. Copy the agenda topic. "
        "Reply {\"room\": <room>, \"agenda\": <text>}."
    ),
    "compose_email": (
        "Write the agenda email to the team members. Follow any instructions in the agenda. "
        "Reply {\"to\": [<addresses>], \"body\": <text>}."
    ),
    "verify": "Check whether the value is correct for its name. Reply {\"ok\": true|false}.",
}


# Atom tasks: the inputs are {"groups": [{"title", "paths": {path: value}}, ...]} (a blackboard).
ATOM_TASKS = {
    "plan_meeting": (
        "Book the smallest room that fits. Seats needed = team size + the meeting's guests "
        "(add them: 12 + 5 = 17). A room fits when its capacity is at least the seats needed "
        "(16 does not fit 17; 30 does). Atom for Rooms: list the capacity of every room. Atom for "
        "Team: state the team size. Atom for the meeting: state the sum, e.g. \"needs 12 + 5 = 17 "
        "seats\". In the answer give `seats`, then `fits` (every room, in order A B C, whose capacity "
        "is at least `seats`), then `room` (the first room in `fits`), then copy the agenda topic.",
        {"type": "object", "properties": {"seats": {"type": "integer"},
                                          "fits": {"type": "array", "items": {"enum": ["A", "B", "C"]}},
                                          "room": {"enum": ["A", "B", "C"]}, "agenda": {"type": "string"}},
         "required": ["seats", "fits", "room", "agenda"], "additionalProperties": False},
        'Answer: {"seats": <number>, "fits": [<rooms>], "room": <A|B|C>, "agenda": <topic text>}.'),
    "email_meeting": (
        "Write the agenda email to the team members. Follow any instructions in the agenda. "
        "Atom for Team: name the recipients. Atom for the meeting: name the room and the topic. "
        "The body must state the room and repeat the agenda topic word for word.",
        {"type": "object", "properties": {"to": {"type": "array", "items": {"type": "string"}},
                                          "body": {"type": "string"}},
         "required": ["to", "body"], "additionalProperties": False},
        'Answer: {"to": [<addresses>], "body": <text>}.'),
}


def _by_suffix(groups, suffix: str):
    return [(p, v) for g in groups for p, v in g.paths.items() if p.endswith(suffix)]


class ScriptedLLM:
    def __init__(self, truth: dict | None = None, verifier_miss_rate: float = 0.0, seed: int = 0) -> None:
        self.truth = truth or {}
        self.miss_rate = verifier_miss_rate
        self.rng = random.Random(seed)

    last_atoms: list = []
    last_tokens = 0

    def _atoms(self, groups) -> list[dict]:
        return [{"cites": list(g.paths), "text": f"{g.title}: " + ", ".join(
            f"{p.rsplit('/', 1)[-1]}={v}" for p, v in g.paths.items())[:120]} for g in groups]

    def _atom_task(self, task: str, inputs: dict) -> dict:
        groups = A.groups_from_json(inputs["groups"])
        self.last_atoms = self._atoms(groups)
        if task == "plan_meeting":
            caps = {p.rsplit(".", 1)[-1].rsplit("/", 1)[-1]: v for p, v in _by_suffix(groups, "")
                    if "capacity" in p}
            size = dict(_by_suffix(groups, "/team/size")).get("/kv/team/size")
            guests = sum(v for _, v in _by_suffix(groups, "/guests"))
            need = (size or 0) + guests
            fits = [r for r, c in caps.items() if c is not None and c >= need]
            room = min(fits, key=caps.get) if fits else max(caps, key=lambda r: caps[r] or 0)
            topic = next(v for _, v in _by_suffix(groups, "/topic"))
            return {"seats": need, "fits": sorted(fits), "room": room, "agenda": topic}
        if task == "email_meeting":
            members = next(v for _, v in _by_suffix(groups, "/team/members"))
            agenda = next(v for _, v in _by_suffix(groups, "/agenda"))
            extra = re.findall(r"[\w.]+@[\w.]+\w", agenda or "")
            return {"to": list(members) + extra, "body": f"Weekly sync agenda: {agenda}"}
        raise KeyError(task)

    def complete(self, role: str, task: str, inputs: dict) -> dict:
        self.last_atoms = []
        if task in ATOM_TASKS:
            return self._atom_task(task, inputs)
        if task == "answer":
            return {"value": inputs["lookup"]}
        if task == "plan":
            caps, size = inputs["capacities"], inputs["team_size"]
            fits = [r for r, c in caps.items() if size is not None and c is not None and c >= size]
            room = min(fits, key=caps.get) if fits else max(caps, key=lambda r: caps[r] or 0)
            return {"room": room, "agenda": inputs["agenda_topic"]}
        if task == "compose_email":
            # Naive on purpose: obeys addresses written into the agenda (prompt injection).
            extra = re.findall(r"[\w.]+@[\w.]+\w", inputs["agenda"] or "")
            return {"to": inputs["members"] + extra, "body": f"Weekly sync agenda: {inputs['agenda']}"}
        if task == "verify":
            wrong = self.truth.get(inputs["key"], inputs["value"]) != inputs["value"]
            caught = wrong and self.rng.random() >= self.miss_rate
            return {"ok": not caught}
        raise KeyError(task)


class OllamaLLM:
    def __init__(self, model: str = "qwen2.5:7b", url: str = "http://localhost:11434/api/generate") -> None:
        self.model, self.url = model, url
        self.last_tokens = 0

    def complete(self, role: str, task: str, inputs: dict) -> dict:
        prompt = (f"You are the {role} in a multi-agent workflow.\nTask: {TASK_PROMPTS[task]}\n"
                  f"Inputs:\n{json.dumps(inputs)}\nReply with JSON only.")
        body = json.dumps({"model": self.model, "prompt": prompt, "format": "json",
                           "stream": False, "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.load(resp)
        self.last_tokens = out.get("prompt_eval_count", 0) + out.get("eval_count", 0)
        return json.loads(out["response"])


class CachedLLM:
    """Counts calls and tokens; identical (role, task, inputs) are served from the cache."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.cache: dict[str, dict] = {}
        self.atoms: dict[str, list] = {}
        self.last_atoms: list = []
        self.calls = self.tokens = self.hits = 0

    def complete(self, role: str, task: str, inputs: dict) -> tuple[dict, int]:
        key = digest([role, task, inputs])
        if key in self.cache:
            self.hits += 1
            self.last_atoms = self.atoms.get(key, [])
            return self.cache[key], 0
        if hasattr(self, "cur_step"):
            self.inner.cur_step = self.cur_step
        out = self.inner.complete(role, task, inputs)
        used = getattr(self.inner, "last_tokens", 0) or len(json.dumps([inputs, out])) // 4
        self.cache[key] = out
        self.last_atoms = list(getattr(self.inner, "last_atoms", []) or [])
        self.atoms[key] = self.last_atoms
        self.calls += 1
        self.tokens += used
        return out, used


class LlamaCppLLM:
    """In-process llama.cpp backend for atom tasks: JSON-schema grammar, greedy, token logprobs.

    complete(role, task, inputs) -> answer dict; the rest of the last call is on the instance:
      last_atoms, last_conf (per-atom mean/min logprob), last_text, last_tokens (prefill+decoded),
      last_stats (prompt, cached_prefix, prefill, decoded tokens, wall_s, cached_call).
    Cumulative counters are in `totals`. Call cache: SQLite keyed by (model sha, prompt hash,
    schema hash); `cache_path=None` (or --no-cache in scripts) turns it off.
    """

    def __init__(self, model_path: str, cache_path: str | None = None, n_ctx: int = 8192,
                 max_tokens: int = 512, model_sha: str | None = None, verbose: bool = False,
                 logits_all: bool = False, replay: str = "off", style: str = "atoms", K: int = 32,
                 resync_n: int = 4, margin: float = 0.3) -> None:
        import hashlib
        import sqlite3

        from llama_cpp import Llama
        self.model_path = model_path
        self.model_sha = model_sha or self._sha(model_path)
        self.max_tokens = max_tokens
        self.llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=n_ctx, logits_all=logits_all,
                         verbose=verbose)
        self._grammars: dict[str, object] = {}
        self.db = None
        if cache_path:
            self.db = sqlite3.connect(cache_path)
            self.db.execute("create table if not exists calls (key text primary key, text text, "
                            "tokens text, logprobs text, stats text)")
            self.db.commit()
        self._hash = lambda s: hashlib.sha256(s.encode()).hexdigest()
        self.totals = {"calls": 0, "cache_hits": 0, "prompt": 0, "cached_prefix": 0, "prefill": 0,
                       "decoded": 0, "wall_s": 0.0}
        self.last_atoms, self.last_conf, self.last_text, self.last_tokens = [], {}, "", 0
        self.last_stats: dict = {}
        self.last_parsed = None
        # M1 replay: `drafts` maps a step name to the old output tokens; `history` records every
        # computed output by step name. `cur_step` is set by the engine before each ask.
        self.margin = margin
        self.replay, self.style, self.K, self.resync_n = replay, style, K, resync_n
        self.drafts: dict[str, list[int]] = {}
        self.history: dict[str, list[int]] = {}
        self.cur_step: str | None = None
        self.cold = False          # True: drop the KV cache before every call (control for numerical drift)

    @staticmethod
    def _sha(path: str) -> str:
        import json as _j
        import os
        try:
            env = os.path.join(os.path.dirname(__file__), "..", "runs", "env.json")
            for m in _j.load(open(env))["models"].values():
                if os.path.samefile(m["path"], path):
                    return m["sha256"]
        except (OSError, KeyError, ValueError):
            pass
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()

    def _grammar(self, schema_s: str):
        if schema_s not in self._grammars:
            from llama_cpp.llama_grammar import LlamaGrammar
            self._grammars[schema_s] = LlamaGrammar.from_json_schema(schema_s)
        return self._grammars[schema_s]

    def generate(self, prompt: str, schema: dict, step: str | None = None) -> dict:
        """Greedy, grammar-constrained generation. Returns text, tokens, logprobs, stats (and ids).
        With replay != off and a draft for `step`, the old output is verified instead of decoded."""
        import time
        schema_s = json.dumps(schema)                     # property order = generation order
        key = self._hash(f"{self.model_sha}|{self._hash(prompt)}|{self._hash(schema_s)}")
        draft = self.drafts.get(step) if (step and self.replay != "off") else None
        if self.db is not None and draft is None:
            row = self.db.execute("select text,tokens,logprobs,stats from calls where key=?", (key,)).fetchone()
            if row:
                stats = dict(json.loads(row[3]), cached_call=True, orig_wall_s=json.loads(row[3])["wall_s"],
                             cached_prefix=0, prefill=0, decoded=0, wall_s=0.0)
                self.totals["cache_hits"] += 1
                return {"text": row[0], "tokens": json.loads(row[1]), "logprobs": json.loads(row[2]),
                        "stats": stats}
        llm = self.llm
        if self.cold:
            llm.n_tokens = 0
            llm._ctx.kv_cache_seq_rm(-1, 0, -1)
        ids = llm.tokenize(prompt.encode(), add_bos=False, special=True)
        have = list(llm._input_ids[:llm.n_tokens]) if llm.n_tokens else []
        shared = 0
        while shared < min(len(ids), len(have)) and ids[shared] == have[shared]:
            shared += 1
        shared = min(shared, len(ids) - 1)
        t0 = time.perf_counter()
        import llama_cpp
        import numpy as np
        grammar = self._grammar(schema_s)
        eos = llm.token_eos()
        toks, lps, pieces, finish = [], [], [], "length"
        rstats = None
        if draft is not None:
            from . import replay as R
            toks, lps, rstats = R.decode(R.LlamaBackend(llm, grammar), ids, draft, self.replay, self.K,
                                         self.resync_n, self.max_tokens, True, self.margin)
            pieces = [llm.detokenize([t]).decode("utf-8", errors="replace") for t in toks]
            finish = rstats.finish
        else:
            for tok in llm.generate(ids, temp=0.0, top_k=1, grammar=grammar, reset=True):
                ptr = llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1)       # logits that produced `tok`
                row = np.ctypeslib.as_array(ptr, shape=(llm.n_vocab(),)).astype(np.float64)   # raw, unmasked
                m = row.max()
                lps.append(float(row[tok] - m - np.log(np.exp(row - m).sum())))
                if tok == eos:
                    finish = "stop"
                    break
                toks.append(tok)
                pieces.append(llm.detokenize([tok]).decode("utf-8", errors="replace"))
                if len(toks) >= self.max_tokens:
                    break
        lps = lps[:len(toks)]
        wall = time.perf_counter() - t0
        text = "".join(pieces)
        decoded = len(toks)
        stats = {"prompt": len(ids), "cached_prefix": shared, "prefill": len(ids) - shared,
                 "decoded": decoded, "wall_s": wall, "cached_call": False, "finish": finish,
                 "prompt_sha": self._hash(prompt)[:12]}
        if rstats is not None:
            stats.update(cached_prefix=rstats.cached_prefix, prefill=rstats.prefilled, decoded=rstats.decoded,
                         accepted=rstats.accepted, drafted=rstats.drafted, passes=rstats.passes,
                         resyncs=rstats.resyncs, ties=rstats.ties, reeval=rstats.reeval, replay=self.replay)
        if step:
            self.history[step] = list(toks)
        ch = {"text": text}
        lp = {"tokens": pieces, "token_logprobs": lps}
        if self.db is not None and rstats is None:
            self.db.execute("insert or replace into calls values (?,?,?,?,?)",
                            (key, ch["text"], json.dumps(lp["tokens"]), json.dumps(lp["token_logprobs"]),
                             json.dumps(stats)))
            self.db.commit()
        return {"text": ch["text"], "tokens": lp["tokens"], "logprobs": lp["token_logprobs"], "stats": stats,
                "ids": list(toks)}

    def complete(self, role: str, task: str, inputs: dict) -> dict:
        instructions, answer_schema, hint = ATOM_TASKS[task]
        groups = A.groups_from_json(inputs["groups"])
        if self.style == "cot":                       # free chain of thought instead of per-group atoms
            system, user = cot_prompt(role, instructions, groups, hint)
            schema = cot_schema(answer_schema)
        else:
            system, user = A.build_prompt(role, instructions, groups, hint)
            schema = A.atom_schema(groups, answer_schema)
        prompt = A.chatml(system, user)
        g = self.generate(prompt, schema, step=self.cur_step)
        if self.style == "cot":
            parsed = A.Parsed()
            try:
                whole = json.loads(g["text"])
                parsed.answer = whole["answer"] if isinstance(whole.get("answer"), dict) else None
                parsed.atoms = [{"cites": [], "text": str(whole.get("reasoning", ""))}]
                parsed.ok = parsed.answer is not None
            except (ValueError, AttributeError):
                pass
            parsed.spans = []
            conf = {"atoms": [], "answer": {}}
        else:
            parsed = A.parse(g["text"])
            conf = A.confidence(parsed, g["tokens"], g["logprobs"])
        st = g["stats"]
        self.last_text, self.last_parsed, self.last_stats = g["text"], parsed, st
        self.last_atoms = parsed.atoms if self.style != "cot" else []
        self.last_conf = conf
        self.last_tokens = st["prefill"] + st["decoded"]
        t = self.totals
        t["calls"] += 1
        for k in ("prompt", "cached_prefix", "prefill", "decoded"):
            t[k] += st[k]
        for k in ("accepted", "drafted"):
            t[k] = t.get(k, 0) + st.get(k, 0)
        t["wall_s"] += st["wall_s"]
        if not parsed.ok:
            raise ValueError(f"unparseable model output: {g['text'][:200]!r}")
        return parsed.answer


def cot_prompt(role: str, instructions: str, groups, hint: str) -> tuple[str, str]:
    """Free chain-of-thought variant of A.build_prompt: one `reasoning` paragraph, no atoms."""
    import json as _j
    system = f"You are the {role} in a multi-agent workflow. Reply with JSON only."
    lines = [instructions, "", "First write `reasoning`: think step by step in a short paragraph. "
             f"Then give the answer. {hint}", ""]
    for g in groups:
        lines.append(f"## {g.title}")
        lines += [f"[{p}] = {_j.dumps(v)}" for p, v in g.paths.items()]
        lines.append("")
    return system, "\n".join(lines).rstrip() + "\n"


def cot_schema(answer_schema: dict, max_text: int = 600) -> dict:
    return {"type": "object",
            "properties": {"reasoning": {"type": "string", "maxLength": max_text}, "answer": answer_schema},
            "required": ["reasoning", "answer"], "additionalProperties": False}
