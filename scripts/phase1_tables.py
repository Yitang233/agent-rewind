"""Recompute the numbers in Tables 1-3 of report/phase1.tex from the raw cells.

Usage: python scripts/phase1_tables.py [--dir results]

Reads <dir>/phase1/cells.jsonl (E1, Tables 1-2) and <dir>/phase1_baselines/cells.jsonl (E1b, Table 3);
`--dir runs` reads a fresh local run instead. Only 7B cells are used.
"""
import argparse
import json
import os
import statistics as st

ROOT = os.path.join(os.path.dirname(__file__), "..")
WLS = ["w1", "w3", "w4.byz", "w4.ipi"]


def load(path):
    return [r for r in (json.loads(ln) for ln in open(path) if ln.strip()) if r["spec"]["model"] == "7b"]


def mean(xs):
    xs = list(xs)
    return st.fmean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    e1 = [r for r in load(os.path.join(a.dir, "phase1", "cells.jsonl")) if r["spec"]["exp"] == "E1"]
    base = load(os.path.join(a.dir, "phase1_baselines", "cells.jsonl"))
    uncached = os.path.join(a.dir, "phase1", "w4byz_clean_uncached.json")

    print("Table 1  clean runs: success, steps, LLM calls, tokens, latency s (cache hits)")
    clean_tok = {}
    for w in WLS:
        F = [r["faulty"] for r in e1 if r["spec"]["workload"] == w and r["spec"]["fault"] == "clean"]
        hits = sum(f["llm"].get("cache_hits", 0) for f in F)
        tok = mean(f["tokens"] for f in F)
        if hits and w == "w4.byz" and os.path.exists(uncached):       # see the table's footnote
            tok = mean(x["tokens"] for x in json.load(open(uncached))["runs"])
        clean_tok[w] = tok
        print(f"  {w:7s} {sum(f['ok'] for f in F)}/{len(F)}  {mean(len(f['steps']) for f in F):.0f}  "
              f"{mean(f['llm_calls'] for f in F):.0f}  {tok:.0f}  "
              f"{'-' if hits else format(mean(f['wall_s'] for f in F), '.1f')}  ({hits})")

    unauth = {(r["spec"]["workload"], r["spec"]["fault"]): r.get("unauthorized", 0) for r in base}
    print("\nTable 2  propagation: changed, steps hit, irreversible calls, unauthorised")
    for w in WLS:
        steps = next(len(r["faulty"]["steps"]) for r in e1 if r["spec"]["workload"] == w)
        for fault in sorted({r["spec"]["fault"] for r in e1 if r["spec"]["workload"] == w} - {"clean"}):
            c = [r for r in e1 if r["spec"]["workload"] == w and r["spec"]["fault"] == fault]
            print(f"  {w:7s} {fault:15s} {mean(r['propagation']['outcome_changed'] for r in c):.2f}  "
                  f"{mean(len(r['propagation']['cone']) for r in c):.0f}/{steps}  "
                  f"{mean(r['irreversible_calls_faulty'] for r in c):.1f}  {unauth.get((w, fault), 0)}")

    print("\nTable 3  restart: rerun, waste, s, tok, irr, ok | always-verify: tok, x clean, det, FP")
    for w in WLS:
        rs = [r for r in base if r["spec"]["workload"] == w and r["spec"]["method"] == "restart"]
        R = [r for r in rs if r["spec"]["flags"]["audit"] == "none"]
        V = [r for r in rs if r["spec"]["flags"]["audit"] == "always"]
        av_tok = mean(r["faulty"]["audit"]["audit_tokens"] for r in V)
        print(f"  {w:7s} {mean(r['recovery']['n_rerun'] for r in R):.0f}  "
              f"{mean(r['recovery']['n_rerun'] - len(r['propagation']['cone']) - 1 for r in R):.1f}  "
              f"{mean(r['recovery']['wall_s'] for r in R):.1f}  {mean(r['recovery']['tokens'] for r in R):.0f}  "
              f"{mean(r['recovery']['irreversible_redone'] for r in R):.1f}  "
              f"{mean(r['recovery']['ok_recovered'] for r in R):.1f} | {av_tok:.0f}  {av_tok / clean_tok[w]:.2f}  "
              f"{mean(r['detected'] for r in V):.2f}  {mean(r['fp_flags'] for r in V):.2f}")


if __name__ == "__main__":
    main()
