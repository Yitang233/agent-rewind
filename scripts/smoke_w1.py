"""S3 smoke: N clean widened-W1 runs per model. Resumable by (model, variant); one model at a time.

Usage: python scripts/smoke_w1.py --models 1.5b 7b --runs 5 --k 3 [--no-cache] [--out runs/S3/clean.json]
"""
import argparse
import gc
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_rewind.llm import CachedLLM, LlamaCppLLM          # noqa: E402
from agent_rewind.workloads import booking                   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["1.5b", "7b"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/S3/clean.json"))
    a = ap.parse_args()
    env = json.load(open(os.path.join(ROOT, "runs/env.json")))["models"]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"cells": {}}
    raw = open(a.out.replace(".json", ".outputs.jsonl"), "a")
    for name in a.models:
        todo = [v for v in range(a.runs) if f"{name}/{v}" not in res["cells"]]
        if not todo:
            continue
        inner = LlamaCppLLM(env[name]["path"], model_sha=env[name]["sha256"],
                            cache_path=None if a.no_cache else os.path.join(ROOT, "runs/S3/calls.sqlite"))
        for v in todo:
            llm = CachedLLM(inner)
            before = dict(inner.totals)
            t0 = time.perf_counter()
            err, rows = None, []
            orig = inner.complete

            def spy(role, task, inputs, _o=orig):
                out = _o(role, task, inputs)
                rows.append({"task": task, "text": inner.last_text, "stats": inner.last_stats,
                             "conf": inner.last_conf})
                return out
            inner.complete = spy
            try:
                run, world = booking.run_meetings(llm, a.k, variant=v)
                ok = booking.meetings_ok(world, a.k)
            except Exception as e:                     # an unparseable output counts as a failure
                ok, err, world = False, repr(e)[:300], None
            inner.complete = orig
            delta = {k_: inner.totals[k_] - before[k_] for k_ in before}
            res["cells"][f"{name}/{v}"] = {
                "model": name, "variant": v, "success": ok, "error": err, "latency_s": time.perf_counter() - t0,
                "llm_calls": delta["calls"], "tokens": delta["prefill"] + delta["decoded"], "tokens_detail": delta,
                "outcome": world.outcome() if world else None, "peak_rss_mb": rss_mb(),
                "parse_ok": all(r["text"] for r in rows)}
            for r in rows:
                raw.write(json.dumps({"cell": f"{name}/{v}", **r}) + "\n")
            raw.flush()
            json.dump(res, open(a.out, "w"), indent=1)
            print(name, v, ok, err, f"{res['cells'][f'{name}/{v}']['latency_s']:.1f}s", flush=True)
        del inner
        gc.collect()


if __name__ == "__main__":
    main()
