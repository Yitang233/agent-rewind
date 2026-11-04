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

    # Older cells record unauthorised actions only after recovery (baseline grid, 1 repeat); newer
    # cells record them on the faulty run itself (`faulty.unauthorized`), which is what Table 2 means.
    base_unauth = {(r["spec"]["workload"], r["spec"]["fault"]): r.get("unauthorized", 0) for r in base
                   if r["spec"]["method"] == "restart" and r["spec"]["flags"]["audit"] == "none"}
    clean_s = {w: mean(r["faulty"]["wall_s"] for r in e1 if r["spec"]["workload"] == w and r["spec"]["fault"] == "clean"
                       and not r["faulty"]["llm"].get("cache_hits")) for w in WLS}
    faulty_cells = [r for r in e1 if r["spec"]["fault"] != "clean"]
    used = [r for r in faulty_cells if r["propagation"]["read_downstream"]]
    print(f"\nTable 2  propagation ({len(faulty_cells) - len(used)} of {len(faulty_cells)} faulty cells excluded: "
          "the injected value was never read)")
    print("  workload fault (class)            fail  steps hit  irrev  unauth  latency s (clean)   [* = unauth from baseline grid]")
    masked = []
    for w in WLS:
        steps = next(len(r["faulty"]["steps"]) for r in e1 if r["spec"]["workload"] == w)
        for fault in sorted({r["spec"]["fault"] for r in used if r["spec"]["workload"] == w}):
            c = [r for r in used if r["spec"]["workload"] == w and r["spec"]["fault"] == fault]
            kind = "adv" if c[0]["spec"]["fault"].startswith(("inject", "byz")) else "acc"
            fail = mean(r["propagation"]["outcome_changed"] for r in c)
            if "unauthorized" in c[0]["faulty"]:
                ua, mark = f"{mean(r['faulty']['unauthorized'] for r in c):.2f}", " "
            else:
                ua, mark = str(base_unauth.get((w, fault), "-")), "*"
            live = [r["faulty"]["wall_s"] for r in c if not r["faulty"]["llm"].get("cache_hits")]
            lat = f"{mean(live):.1f} ({clean_s[w]:.1f})" if live else "-"
            print(f"  {w:7s} {fault:15s} ({kind})  {fail:.2f}  {mean(len(r['propagation']['cone']) for r in c):.0f}/{steps}"
                  f"       {mean(r['irreversible_calls_faulty'] for r in c):.1f}    {ua}{mark}    {lat}")
            masked += [(w, fault, sum(not r["propagation"]["outcome_changed"] for r in c), len(c))]
    print("\n  read but masked (checker still passes): " +
          (", ".join(f"{w}/{f} {m}/{n}" for w, f, m, n in masked if m) or "none"))

    print("\nTable 3  restart: rerun, waste, s, tok, irr, ok | always-verify: tok, x clean, det, FP, ok")
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
              f"{mean(r['detected'] for r in V):.2f}  {mean(r['fp_flags'] for r in V):.2f}  "
              f"{mean(r['recovery']['ok_recovered'] for r in V):.2f}")


if __name__ == "__main__":
    main()
