"""Micro-benchmark for llama-cpp-python: prefill/decode speed, draft
verification equivalence, KV crop. Prints and optionally writes JSON.

Usage: python scripts/bench_llm.py --model PATH [--out FILE.json]
"""
import argparse
import gc
import hashlib
import json
import resource
import sys
import time

import numpy as np

PROMPT_TEXT = (
    "The history of computing spans many decades, from mechanical calculators "
    "to modern distributed systems. Engineers built abstractions layer by layer, "
    "each hiding the complexity below it. "
)


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def greedy(llm, n):
    """Greedy-decode n tokens from the current cache state; return tokens."""
    out = []
    for _ in range(n):
        tok = int(np.argmax(llm.scores[llm.n_tokens - 1]))
        out.append(tok)
        llm.eval([tok])
    return out


def bench(path):
    from llama_cpp import Llama
    import llama_cpp

    llm = Llama(model_path=path, n_gpu_layers=-1, logits_all=True, n_ctx=4096,
                verbose=False, seed=0)
    ids = llm.tokenize((PROMPT_TEXT * 40).encode(), add_bos=True)[:512]
    assert len(ids) == 512, len(ids)

    # warmup
    llm.reset(); llm.eval(ids[:32])

    # prefill
    llm.reset()
    t = time.perf_counter(); llm.eval(ids); pre = time.perf_counter() - t
    prefill_tps = len(ids) / pre

    # decode 128 greedy
    t = time.perf_counter(); draft = greedy(llm, 128); dec = time.perf_counter() - t
    decode_tps = 128 / dec

    # verification: prompt + first 64 draft tokens in one batch
    draft64 = draft[:64]
    llm.reset()
    t = time.perf_counter(); llm.eval(ids + draft64); ver = time.perf_counter() - t
    verify_ms = ver * 1000
    # logits at position len(ids)-1+i predict draft[i]
    sc = np.asarray(llm.scores[len(ids) - 1: len(ids) - 1 + 64])
    equiv = int((sc.argmax(axis=1) == np.array(draft64)).sum())

    # KV crop: logits for the draft positions, crop to prompt, re-eval, compare
    first = np.array(llm.scores[len(ids): len(ids) + 64]).copy()
    llm.n_tokens = len(ids)  # crop (Llama truncates cache by resetting n_tokens)
    llama_cpp.llama_kv_self_seq_rm(llm._ctx.ctx, -1, len(ids), -1) if hasattr(
        llama_cpp, "llama_kv_self_seq_rm") else llm._ctx.kv_cache_seq_rm(-1, len(ids), -1)
    llm.scores[len(ids): len(ids) + 64] = np.nan  # poison: re-eval must overwrite
    llm.eval(draft64)
    second = np.array(llm.scores[len(ids): len(ids) + 64])
    max_diff = float(np.abs(first - second).max())
    same_argmax = bool((first.argmax(1) == second.argmax(1)).all())
    kv_crop_ok = same_argmax and max_diff < 1e-2

    res = dict(path=path, sha256=sha256(path), prefill_tps=round(prefill_tps, 2),
               decode_tps=round(decode_tps, 2), verify_ms_64=round(verify_ms, 1),
               equiv_tokens=equiv, equiv_rate=equiv / 64, kv_crop_ok=kv_crop_ok,
               kv_crop_max_abs_diff=max_diff, peak_rss_mb=round(peak_rss_mb(), 0))
    del llm; gc.collect()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    r = bench(a.model)
    print(json.dumps(r, indent=1))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
