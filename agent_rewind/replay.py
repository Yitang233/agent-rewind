"""M1 speculative replay: rerun an LLM step by verifying its old output as a draft.

The model checks K old tokens in one batched forward pass; the longest prefix on which its own
(grammar-masked) greedy choice agrees is accepted, the cache is cropped at the first
disagreement, and the model's own token replaces the old one. Under greedy decoding the result
equals a plain token-by-token decode (up to floating-point differences between batched and
sequential evaluation, which `scripts/replay_check.py` measures).

    replay=off      plain decode (no draft)
    replay=prefix   verify the old output until the first mismatch, then decode normally
    replay=resync   after a mismatch, find the last n generated tokens inside the old output
                    (n-gram match) and keep verifying from the position after it

`decode` works on any `Backend`; `LlamaBackend` is llama-cpp-python, `FakeBackend` a tiny
deterministic model for tests. A Backend provides
    eos, begin(ids) -> shared, eval(ids) -> rows [len, V], crop(n), constraint() -> Constraint
and a Constraint provides pick(row) -> token (greedy under the grammar), accept(token), close().
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

MODES = ("off", "prefix", "resync")


@dataclass
class ReplayStats:
    mode: str = "off"
    prompt: int = 0           # prompt tokens
    cached_prefix: int = 0    # prompt tokens already in the KV cache
    prefilled: int = 0        # prompt tokens evaluated
    accepted: int = 0         # output tokens taken from the draft
    drafted: int = 0          # draft tokens sent for verification
    decoded: int = 0          # output tokens the model chose itself (one sequential step each in a plain decode)
    passes: int = 0           # forward calls in the decode loop
    resyncs: int = 0          # successful n-gram re-alignments
    ties: int = 0             # near-tie positions re-verified with sequential evaluation
    reeval: int = 0           # single-token evaluations spent on those re-verifications
    wall_s: float = 0.0
    finish: str = "stop"

    def as_dict(self) -> dict:
        return asdict(self)


def _resync(old: list[int], out: list[int], n: int, expect: int) -> int | None:
    """Index in `old` after the last n generated tokens, if they occur there; the occurrence
    nearest to `expect` wins."""
    if n <= 0 or len(out) < n or len(old) < n:
        return None
    tail = out[-n:]
    best = None
    for j in range(len(old) - n + 1):
        if old[j:j + n] == tail and (best is None or abs(j + n - expect) < abs(best - expect)):
            best = j + n
    return best


def _gap(row) -> float:
    a, b = np.partition(row, -2)[-2:]
    return float(b - a)


def decode(be, ids: list[int], draft: list[int] | None = None, mode: str = "resync", K: int = 32,
           n: int = 4, max_tokens: int = 512, want_logprobs: bool = True, margin: float = 0.3):
    """Greedy-decode a continuation of `ids`. Returns (tokens without eos, logprobs, ReplayStats).
    `draft` is the old output (tokens without eos); the end of the old output counts as eos.

    Tie handling: batched and one-token evaluation differ by float noise (up to ~0.1 in logits on
    the 1.5B model). Once any batch > 1 has touched the cache, a position whose top-2 raw logit gap
    is below `margin` is re-decided from an exact state: the cache is cleared, the prompt is re-evaluated as one batch
    and the generated tokens one at a time (as a cold plain decode would), and the pick is
    taken from that row. Rare, so the cost is small; margin=0 disables it."""
    if mode not in MODES:
        raise ValueError(f"replay mode {mode!r}")
    t0 = time.perf_counter()
    st = ReplayStats(mode=mode, prompt=len(ids))
    old = (list(draft) + [be.eos]) if draft and mode != "off" else []
    shared = be.begin(ids)
    st.cached_prefix, st.prefilled = shared, len(ids) - shared
    con = be.constraint()
    out: list[int] = []
    lps: list[float] = []
    pending = [ids[-1]]
    ptr: int | None = 0 if old else None          # index in `old` of the next draft token
    dirty = False                                 # the cache holds entries computed in a batch > 1
    expect = 0
    try:
        while True:
            dr = old[ptr:ptr + K] if ptr is not None else []
            rows = be.eval(pending + dr)
            st.passes += 1
            st.drafted += len(dr)
            dirty = dirty or len(pending) + len(dr) > 1
            base, j, finished, rebuilt = len(pending) - 1, 0, False, False
            while True:
                row = rows[base + j]
                if dirty and margin > 0 and _gap(row) < margin:
                    # cache is exact up to out[:len(out)] only after a sequential rebuild
                    st.ties += 1
                    be.crop(0)                  # as a cold plain decode: whole prompt in one batch,
                    row = be.eval(ids)[-1]      # then one generated token per evaluation
                    st.reeval += len(ids)
                    for t in out:
                        row = be.eval([t])[0]
                        st.reeval += 1
                    dirty, rebuilt = False, True
                tok = con.pick(row)
                hit = j < len(dr) and tok == dr[j]
                if tok == be.eos:
                    st.finish, finished = "stop", True
                    break
                if want_logprobs:
                    r = row.astype(np.float64)
                    m = r.max()
                    lps.append(float(r[tok] - m - np.log(np.exp(r - m).sum())))
                con.accept(tok)
                out.append(tok)
                if hit:
                    st.accepted += 1
                    j += 1
                elif j == len(dr) and ptr is not None and ptr + j < len(old) and tok == old[ptr + j]:
                    st.accepted += 1            # extra token of a fully accepted chunk that also matches the draft
                else:
                    st.decoded += 1
                if len(out) >= max_tokens:
                    st.finish, finished = "length", True
                    break
                if rebuilt:
                    j -= 1 if hit else 0        # the pass ends here; `pos` below is this token's old position
                    break
                if not hit:
                    break
            if finished:
                break
            if j < len(dr) and not rebuilt:                         # drop the unverified rest of the draft chunk
                be.crop(be.n_tokens - (len(dr) - j))
            pending = [out[-1]]                     # the token that ended this pass has no KV yet
            if ptr is not None:
                pos = ptr + j                       # old position this token replaced (or extends past the chunk)
                expect = pos + 1
                ptr = pos + 1 if pos < len(old) and out[-1] == old[pos] else None
                if ptr is None and mode == "prefix":
                    old = []
            if ptr is None and mode == "resync" and old:
                new = _resync(old, out, n, expect)
                if new is not None and new < len(old):
                    st.resyncs += 1
                    ptr = new
    finally:
        con.close()
    st.wall_s = time.perf_counter() - t0
    return out, lps, st


class LlamaBackend:
    """llama-cpp-python Llama + a JSON-schema grammar. One grammar sampler per decode."""

    def __init__(self, llm, grammar) -> None:
        self.llm, self.grammar = llm, grammar
        self.eos = llm.token_eos()
        self.V = llm.n_vocab()

    @property
    def n_tokens(self) -> int:
        return self.llm.n_tokens

    def begin(self, ids: list[int]) -> int:
        llm = self.llm
        have = [int(t) for t in llm.input_ids[:llm.n_tokens]] if llm.n_tokens else []
        shared = 0
        while shared < min(len(ids), len(have)) and ids[shared] == have[shared]:
            shared += 1
        shared = min(shared, len(ids) - 1)
        self.crop(shared)
        if len(ids) - 1 > shared:
            llm.eval(ids[shared:-1])
        return shared

    def crop(self, n: int) -> None:
        llm = self.llm
        llm.n_tokens = n
        llm._ctx.kv_cache_seq_rm(-1, n, -1)

    def eval(self, ids: list[int]):
        llm = self.llm
        base, n = llm.n_tokens, len(ids)
        nb = llm.n_batch
        if n > nb:      # longer than one batch (a long prompt): chunk exactly as llm.eval does, keep logits of the last chunk
            split = ((n - 1) // nb) * nb
            llm._ctx.kv_cache_seq_rm(-1, base, -1)
            llm.eval(ids[:split])
            return self.eval(ids[split:])
        llm._ctx.kv_cache_seq_rm(-1, base, -1)
        llm._batch.set_batch(batch=ids, n_past=base, logits_all=True)
        llm._ctx.decode(llm._batch)
        llm.input_ids[base:base + n] = ids
        llm.n_tokens = base + n
        return np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(n, self.V))

    def constraint(self):
        return _GrammarConstraint(self.llm, self.grammar, self.V)


class _GrammarConstraint:
    """Greedy under a GBNF grammar. Fast path: the raw argmax is the masked argmax whenever the
    grammar allows it, and that is a one-candidate check; only otherwise is the vocabulary masked."""

    DT = np.dtype([("id", np.intc), ("logit", np.single), ("p", np.single)])

    def __init__(self, llm, grammar, V: int) -> None:
        import llama_cpp
        self.lc, self.V = llama_cpp, V
        self.smpl = llama_cpp.llama_sampler_init_grammar(
            llm._model.vocab, grammar._grammar.encode("utf-8"), grammar._root.encode("utf-8"))

    def _apply(self, arr) -> None:
        lc = self.lc
        da = lc.llama_token_data_array(
            data=arr.ctypes.data_as(__import__("ctypes").POINTER(lc.llama_token_data)),
            size=len(arr), selected=-1, sorted=False)
        lc.llama_sampler_apply(self.smpl, __import__("ctypes").byref(da))

    def pick(self, row) -> int:
        t = int(np.argmax(row))
        one = np.zeros(1, self.DT)
        one["id"], one["logit"] = t, row[t]
        self._apply(one)
        if one["logit"][0] != -np.inf:
            return t
        full = np.zeros(self.V, self.DT)
        full["id"], full["logit"] = np.arange(self.V), row
        self._apply(full)
        return int(full["id"][int(np.argmax(full["logit"]))])

    def accept(self, tok: int) -> None:
        self.lc.llama_sampler_accept(self.smpl, tok)

    def close(self) -> None:
        if self.smpl:
            self.lc.llama_sampler_free(self.smpl)
            self.smpl = None


class FakeBackend:
    """A tiny deterministic 'model': the logits after a context depend on its last 3 tokens, so
    batched and sequential evaluation agree exactly. `forbid` tokens are masked like a grammar."""

    def __init__(self, V: int = 24, eos: int = 0, forbid=(), salt: int = 0, mod=None) -> None:
        self.V, self.eos, self.forbid, self.salt = V, eos, set(forbid), salt
        self.mod = mod or {}                      # {(last token): bonus token} behaviour changes
        self.kv: list[int] = []
        self.evals: list[int] = []                # size of every eval call

    @property
    def n_tokens(self) -> int:
        return len(self.kv)

    def _row(self, ctx: list[int]) -> np.ndarray:
        key = hash((tuple(ctx[-3:]), self.salt)) & 0xFFFFFF
        rng = np.random.default_rng(key)
        row = rng.normal(size=self.V).astype(np.float32)
        row[self.eos] -= 6.0 if len(ctx) < 40 else 0.0
        return row

    def begin(self, ids):
        shared = 0
        while shared < min(len(ids), len(self.kv)) and ids[shared] == self.kv[shared]:
            shared += 1
        shared = min(shared, len(ids) - 1)
        self.crop(shared)
        self.kv += list(ids[shared:-1])
        return shared

    def crop(self, n):
        del self.kv[n:]

    def eval(self, ids):
        self.evals.append(len(ids))
        rows = []
        for t in ids:
            self.kv.append(t)
            rows.append(self._row(self.kv))
        return np.stack(rows)

    def constraint(self):
        outer = self

        class C:
            def pick(self, row):
                r = row.copy()
                for f in outer.forbid:
                    r[f] = -np.inf
                return int(np.argmax(r))

            def accept(self, tok):
                pass

            def close(self):
                pass
        return C()
