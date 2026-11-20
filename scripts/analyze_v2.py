"""Phase 2 verdicts, figures and tables from the v2 cells.

    python scripts/analyze_v2.py            # reads runs/v2/cells.jsonl (+ runs/v2/E2v2, runs/v2/D3, runs/phase1)
                                            # writes runs/v2/figures/{F1..F8}.png/.json, T1-T3.md, analysis_v2.json

Applies the rules of runs/v2/preregistration.json as written (rule changes recorded there are the ones used).
Reuses helpers from analyze.py (ms, wilson, by, PAL).  Read-only on every runs/ folder except runs/v2/figures.
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as AN  # noqa: E402  (helpers; its main() is not run)
from analyze import PAL, ms, wilson  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from _common import ROOT, rec_llm  # noqa: E402,F401
V2 = ROOT / "runs" / "v2"
OUT = V2 / "figures"
CELLS = V2 / "cells.jsonl"
E2V2 = V2 / "E2v2" / "cells.jsonl"
PHASE1 = ROOT / "runs" / "phase1" / "cells.jsonl"
W3_FIX = ROOT / "runs" / "v2_fix" / "cells.jsonl"     # E8v2 W3 rerun after the undo-first and `-B` fixes (new folder; runs/v2 is not edited)
W3_RERUN_NOTE = "W3 rows: rerun after fix (runs/v2_fix)"
REAL = ["w1", "w3", "w4.ipi", "w4.byz"]
INJECT = {("w1", "inject_agenda"), ("w4.ipi", "inject_item")}
RNG = random.Random(291)
RES: dict = {}
NB = 2000


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def swap_w3(e8):
    """E8v2 with the W3 rows replaced by the v2_fix rerun, matched on (label, fault, repeat). Both sets are
    kept in RES["w3_rerun"]; the pre-fix rows stay in runs/v2/cells.jsonl and results/v2/v2_fix/compare.md."""
    fix = load(W3_FIX)
    key = lambda r: (sp(r)["workload"], sp(r)["label"], sp(r)["fault"], sp(r)["repeat"])   # noqa: E731
    new = {key(r): r for r in fix if "error" not in r}
    old = [r for r in e8 if sp(r)["workload"] == "w3"]
    missing = [k for k in map(key, old) if k not in new]
    if missing:
        raise SystemExit(f"v2_fix lacks {len(missing)} W3 cells, e.g. {missing[0]}")
    RES["w3_rerun"] = {"source": str(W3_FIX.relative_to(ROOT)), "note": W3_RERUN_NOTE, "cells": len(new),
                       "ok_recovered_before": {m: sum(bool(r["recovery"]["ok_recovered"]) for r in old if sp(r)["label"] == m) for m in sorted({sp(r)["label"] for r in old})},
                       "ok_recovered_after": {m: sum(bool(r["recovery"]["ok_recovered"]) for r in new.values() if sp(r)["label"] == m) for m in sorted({sp(r)["label"] for r in old})}}
    return [new.get(key(r), r) if sp(r)["workload"] == "w3" else r for r in e8]


def jdump(o):
    return json.dumps(o, indent=1, default=lambda x: ("inf" if x == math.inf else str(x)))


def sv(fig, name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=140)
    plt.close(fig)
    (OUT / f"{name}.json").write_text(jdump(data))
    return f"runs/v2/figures/{name}.png"


def EB(d):
    """asymmetric error bars from (p, lo, hi), clipped at 0"""
    return [[max(0.0, d[0] - d[1])], [max(0.0, d[2] - d[0])]]


def bk_ns(b):
    """bookkeeping ns = read + write + hashing outside a read/write."""
    return b["ns_read"] + b["ns_write"] + b.get("ns_hash_outside", 0)


def sp(r):
    return r["spec"]


def fl(r, k, d=None):
    return r["spec"]["flags"].get(k, d)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]


def med(v):
    return st.median(v) if v else None


def boot_ci(xs, f, n=NB):
    out = []
    for _ in range(n):
        out.append(f([RNG.choice(xs) for _ in xs]))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]


# ------------------------------------------------------------------------------ step 1: validator
def split_by_exp(rows):
    d = defaultdict(list)
    for r in rows:
        d[sp(r)["exp"]].append(r)
    return d


# ------------------------------------------------------------------------------ H7 / F1
def h7_f1(e2):
    sel = [r for r in e2 if sp(r)["label"] == "selective"]
    n_cells = len(sel)
    unexpl, static_miss, cls = 0, 0, defaultdict(int)
    for r in sel:
        p = r["propagation"]
        rc = p["rerun_class"]
        rer = set(r["recovery"]["rerun"])
        origin = {k for k, v in rc.items() if v == "origin"}
        explained = set(p["cone_dynamic"]) | set(p["new_steps"]) | origin
        unexpl += len(rer - explained)
        out_static = rer - set(p["cone_static"]) - origin
        static_miss += bool(out_static)
        for k in rer - set(p["cone_static"]) - origin:
            cls[rc.get(k, "UNCLASSIFIED")] += 1
    all_cls = defaultdict(int)
    for r in sel:
        for v in r["propagation"]["rerun_class"].values():
            all_cls[v] += 1
    vm = {m: [sum(1 for r in e2 if sp(r)["label"] == m and r["recovery"]["values_match"] is True),
              sum(1 for r in e2 if sp(r)["label"] == m)] for m in ["restart", "rollback", "selective", "selective-d2"]}
    ref_agree = sum(1 for r in sel if set(r["propagation"]["cone_dynamic"]) == set(r["propagation"].get("cone_dynamic_ref", r["propagation"]["cone_dynamic"])))
    h7 = {"selective_cells": n_cells, "unexplained_out_of_cone_reruns": unexpl,
          "cells_with_static_cone_miss": static_miss, "static_miss_steps_by_class": dict(cls),
          "all_rerun_steps_by_class": dict(all_cls), "values_match": vm,
          "cone_dynamic_equals_clean_ref_cone": [ref_agree, n_cells],
          "verdict": "supported" if unexpl == 0 else "not supported"}
    RES["H7"] = h7

    meths = ["restart", "rollback", "selective", "selective-d2"]
    col = {"restart": PAL["k"], "rollback": PAL["e"], "selective": PAL["a"], "selective-d2": PAL["c"]}
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    pts = {}
    for row, (fac, vals) in enumerate([("fan_in", [1, 2, 3]), ("shared_keys", [0, 3, 6])]):
        for j, v in enumerate(vals):
            a = ax[row][j]
            cells = [r for r in e2 if sp(r)["params"][fac] == v]
            pos = sorted({sp(r)["params"]["position"] for r in cells})
            for m in meths:
                mu, sd = [], []
                for p in pos:
                    x = [r["recovery"]["n_rerun"] for r in cells if sp(r)["label"] == m and sp(r)["params"]["position"] == p]
                    mm = ms(x)
                    mu.append(mm[0])
                    sd.append(mm[1])
                a.errorbar(pos, mu, yerr=sd, marker="o", ms=3, capsize=2, color=col[m], label=m)
                pts[f"{fac}={v}/{m}"] = dict(zip(map(str, pos), [round(x, 2) for x in mu]))
            for key, ls, lab in [("static", ":", "static cone + origin"), ("dyn", "-.", "dynamic cone + new steps"),
                                 ("suffix", "--", "time-order suffix")]:
                mu = []
                for p in pos:
                    xs = []
                    for r in cells:
                        if sp(r)["label"] != "selective" or sp(r)["params"]["position"] != p:
                            continue
                        pr = r["propagation"]
                        if key == "static":
                            xs.append(len(set(pr["cone_static"]) | {r["fault_at"][0]}))
                        elif key == "dyn":
                            xs.append(len(set(pr["cone_dynamic"]) | set(pr["new_steps"]) | {r["fault_at"][0]}))
                        else:
                            steps = r["faulty"]["steps"]
                            xs.append(len(steps) - steps.index(r["fault_at"][0]))
                    mu.append(st.fmean(xs))
                a.plot(pos, mu, ls=ls, color="#888" if key != "dyn" else PAL["d"], label=lab)
                pts[f"{fac}={v}/{key}"] = dict(zip(map(str, pos), [round(x, 2) for x in mu]))
            a.set_title(f"{fac} = {v}")
            a.set_xlabel("fault position (fraction of DAG)")
            if j == 0:
                a.set_ylabel("steps rerun (of 24)")
    ax[0][0].legend(fontsize=6)
    fig.suptitle(f"F1 (E2v2, W2 scripted, n={len(e2)} cells): steps rerun vs fault position; unexplained out-of-cone reruns = {unexpl}")
    RES["F1"] = {"path": sv(fig, "F1_cone_vs_suffix", {"points": pts, "H7": h7})}


# ------------------------------------------------------------------------------ H6 / F2
def h6_f2(e3, e5, e8, e6):
    tiers = ["agent", "step", "artifact", "field"]
    ns = [24, 200, 1000]
    tab = {}
    for n in ns:
        for t in tiers:
            rows = [r for r in e3 if sp(r)["params"]["n"] == n and fl(r, "depth") == t]
            bps = [r["faulty"]["bookkeeping"]["journal_bytes"] / r["faulty"]["bookkeeping"]["steps"] for r in rows]
            bpr = [r["recovery"]["bookkeeping"]["journal_bytes"] / r["recovery"]["bookkeeping"]["steps"] for r in rows
                   if r.get("recovery") and r["recovery"].get("bookkeeping") and r["recovery"]["bookkeeping"]["steps"]]
            nsr = [r["faulty"]["bookkeeping"]["ns_read"] / r["faulty"]["bookkeeping"]["n_read"]
                   for r in rows if r["faulty"]["bookkeeping"]["n_read"]]
            nsw = [r["faulty"]["bookkeeping"]["ns_write"] / r["faulty"]["bookkeeping"]["n_write"]
                   for r in rows if r["faulty"]["bookkeeping"]["n_write"]]
            tab[f"n={n}/{t}"] = {"n_cells": len(rows), "bytes_per_step": ms(bps)[:2], "rec_bytes_per_step": ms(bpr)[:2],
                                  "ns_per_read": ms(nsr)[:2], "ns_per_write": ms(nsw)[:2]}
    rises = {}
    for n in ns:
        b = [tab[f"n={n}/{t}"]["bytes_per_step"][0] for t in tiers]
        rises[f"n={n}"] = {"bytes_per_step_by_tier": [round(x, 1) for x in b],
                           "monotone_nondecreasing": all(b[i] <= b[i + 1] + 1e-9 for i in range(3)),
                           "field_over_agent": round(b[3] / b[0], 2) if b[0] else None}
    # real-run bookkeeping share of recovery wall, field depth (E5v2 + E8v2 rows with a recovery on a model workload)
    share = {}
    allsh = []
    for name, rows in [("E5v2", e5), ("E8v2", e8)]:
        for r in rows:
            rc = r.get("recovery")
            if not rc or not rc.get("bookkeeping") or sp(r)["model"] != "7b" or fl(r, "depth") != "field":
                continue
            s_ = bk_ns(rc["bookkeeping"]) / 1e9 / rc["wall_s"]
            allsh.append(s_)
            share.setdefault(f"{name}/{sp(r)['workload']}", []).append(s_)
    # clean-run tracking share (E6v2)
    cl = defaultdict(list)
    for r in e6:
        cl[sp(r)["workload"]].append(bk_ns(r["clean"]["bookkeeping"]) / 1e9 / r["clean"]["wall_s"])
    h6 = {"scripted_tiers": tab, "bytes_per_step_rise": rises,
          "real_share_of_recovery_wall": {"n_rows": len(allsh), "mean": st.fmean(allsh), "median": med(allsh),
                                           "max": max(allsh), "p95": q(allsh, 0.95),
                                           "by_group": {k: [st.fmean(v), max(v), len(v)] for k, v in sorted(share.items())}},
          "clean_run_share_of_wall_E6v2": {k: [st.fmean(v), max(v), len(v)] for k, v in sorted(cl.items())}}
    ok_rise = all(v["monotone_nondecreasing"] and (v["field_over_agent"] or 0) > 1 for v in rises.values())
    ok_share = st.fmean(allsh) < 0.02 and max(allsh) < 0.02
    h6["rule"] = {"bytes_rise_agent_to_field_all_n": ok_rise, "real_bookkeeping_lt_2pct_mean_and_max": ok_share,
                  "real_bookkeeping_lt_2pct_mean_only": st.fmean(allsh) < 0.02}
    h6["verdict"] = "supported" if (ok_rise and ok_share) else "not supported"
    RES["H6"] = h6

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for n, c in zip(ns, [PAL["a"], PAL["b"], PAL["c"]]):
        mu = [tab[f"n={n}/{t}"]["bytes_per_step"][0] for t in tiers]
        sd = [tab[f"n={n}/{t}"]["bytes_per_step"][1] for t in tiers]
        ax[0].errorbar(range(4), mu, yerr=sd, marker="o", capsize=3, color=c, label=f"n={n}")
        mu = [tab[f"n={n}/{t}"]["ns_per_read"][0] for t in tiers]
        sd = [tab[f"n={n}/{t}"]["ns_per_read"][1] for t in tiers]
        ax[1].errorbar(range(4), mu, yerr=sd, marker="o", capsize=3, color=c, label=f"n={n}")
    for a, ttl, yl in [(ax[0], "(a) journal bytes per step (scripted W2)", "bytes / step"),
                       (ax[1], "(b) ns per read (scripted W2, tracked run)", "ns / read")]:
        a.set_xticks(range(4), tiers)
        a.set_title(ttl, fontsize=9)
        a.set_ylabel(yl)
        a.set_yscale("log")
        a.legend(fontsize=8)
    keys = sorted(share)
    ax[2].boxplot([[x * 100 for x in share[k]] for k in keys], tick_labels=[k.replace("/", "\n") for k in keys], showfliers=True)
    ax[2].axhline(2, color=PAL["b"], ls="--", lw=1)
    ax[2].set_ylabel("bookkeeping % of recovery wall")
    ax[2].set_title("(c) real runs, field depth (dashed = 2% rule)", fontsize=9)
    ax[2].tick_params(axis="x", labelsize=6)
    fig.suptitle("F2 (E3v2 scripted; E5v2/E8v2 real): bookkeeping vs tracking depth")
    RES["F2"] = {"path": sv(fig, "F2_bookkeeping", {"table": tab, "real_share": h6["real_share_of_recovery_wall"]})}


# ------------------------------------------------------------------------------ E5v2: H1 / F4 / T3
def e5_index(e5):
    d = {}
    for r in e5:
        s = sp(r)
        d[(s["workload"], s["fault"], s["repeat"], fl(r, "style"), fl(r, "replay"))] = r
    return d


def call_map(r):
    return {(c["step"], c["call_index"]): c["sha256"] for c in r["recovery"]["calls"]}


def identity(a, b):
    """compare call sequences of two recoveries; returns (n_compared, n_identical, mismatches)"""
    ma, mb = call_map(a), call_map(b)
    keys = sorted(set(ma) | set(mb))
    mism = []
    ident = 0
    for k in keys:
        if ma.get(k) == mb.get(k) and k in ma:
            ident += 1
        else:
            mism.append({"step": k[0], "call_index": k[1], "in_off": k in ma, "in_arm": k in mb})
    return len(keys), ident, mism


def h1_f4_t3(e5):
    idx = e5_index(e5)
    res = {}
    mism_all = []
    comp = {"resync": [0, 0], "prefix": [0, 0]}
    okeq = {"resync": [0, 0], "prefix": [0, 0]}
    for w in REAL:
        cells = sorted({(k[1], k[2], k[3]) for k in idx if k[0] == w})
        for arm in ["prefix", "resync"]:
            wall, dec, wall_s = [], [], defaultdict(list)
            for (f, rp, sty) in cells:
                b, c = idx.get((w, f, rp, sty, "off")), idx.get((w, f, rp, sty, arm))
                if not b or not c:
                    continue
                lb, lc = rec_llm(b), rec_llm(c)
                wall.append(lc["wall_s"] / lb["wall_s"])
                wall_s[sty].append(lc["wall_s"] / lb["wall_s"])
                dec.append(lc["decoded"] / lb["decoded"] if lb["decoded"] else None)
                n, i, mm = identity(b, c)
                comp[arm][0] += n
                comp[arm][1] += i
                okeq[arm][0] += 1
                okeq[arm][1] += int(b["recovery"]["ok_recovered"] == c["recovery"]["ok_recovered"])
                for m in mm:
                    m.update({"workload": w, "fault": f, "repeat": rp, "style": sty, "arm": arm})
                    mism_all.append(m)
            dd = [x for x in dec if x is not None]
            res[f"{w}/{arm}"] = {"n_pairs": len(wall), "wall_ratio_median": med(wall), "wall_ratio_mean": st.fmean(wall),
                                 "wall_ratio_min_max": [min(wall), max(wall)],
                                 "wall_ratio_median_by_style": {s: [med(v), len(v)] for s, v in wall_s.items()},
                                 "decoded_ratio_median": med(dd), "decoded_ratio_mean": st.fmean(dd) if dd else None,
                                 "wall_ratio_median_ci95": list(boot_ci(wall, med))}
    n_ok = sum(1 for w in REAL if res[f"{w}/resync"]["wall_ratio_median"] <= 0.8)
    ident_all = comp["resync"][1] == comp["resync"][0]
    h1 = {"per_workload": res, "workloads_with_median_le_0.8": n_ok,
          "resync_calls_compared_identical": comp["resync"], "prefix_calls_compared_identical": comp["prefix"],
          "ok_recovered_equal_to_off": okeq,
          "mismatches": mism_all,
          "rule": {"median_ratio_le_0.8_on_ge_3_of_4": n_ok >= 3, "100pct_text_identical_resync": ident_all},
          "verdict": "supported" if (n_ok >= 3 and ident_all) else "not supported"}
    RES["H1"] = h1

    # F4
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    col = {"prefix": PAL["e"], "resync": PAL["a"]}
    for k, arm in enumerate(["prefix", "resync"]):
        for si, sty in enumerate(["cot", "atoms"]):
            for j, w in enumerate(REAL):
                dec, wall = [], []
                for (f, rp) in sorted({(k_[1], k_[2]) for k_ in idx if k_[0] == w and k_[3] == sty}):
                    b, c = idx.get((w, f, rp, sty, "off")), idx.get((w, f, rp, sty, arm))
                    if b and c:
                        dec.append(rec_llm(c)["decoded"] / rec_llm(b)["decoded"])
                        wall.append(rec_llm(c)["wall_s"] / rec_llm(b)["wall_s"])
                x = j + (k - 0.5) * 0.4 + (si - 0.5) * 0.18
                ax[0].bar(x, st.fmean(dec), 0.18, color=col[arm], hatch="//" if sty == "atoms" else None, edgecolor="white")
                ax[1].bar(x, med(wall), 0.18, color=col[arm], hatch="//" if sty == "atoms" else None, edgecolor="white")
                ax[1].scatter([x] * len(wall), wall, s=8, color="k", zorder=3)
    for a, ttl in [(ax[0], "decoded tokens / plain rerun (mean)"), (ax[1], "recovery LLM wall / plain rerun (bar = median, dots = cells)")]:
        a.set_xticks(range(4), REAL)
        a.axhline(1, color="#aaa", lw=0.8)
        a.set_title(ttl, fontsize=9)
    ax[1].axhline(0.8, color=PAL["b"], ls="--", lw=1)
    from matplotlib.patches import Patch
    ax[0].legend([Patch(color=col["prefix"]), Patch(color=col["resync"]), Patch(facecolor="w", hatch="//", edgecolor="k")],
                 ["prefix (default settings)", "resync-tuned", "hatch = atoms"], fontsize=7)
    d3 = json.load(open(V2 / "D3" / "heldout.json"))
    kk = '{"K": 64, "margin": 0.1, "mode": "resync", "tie": "local"}'
    pts3 = {}
    for mi, (mdl, c, mk) in enumerate([("7b", PAL["a"], "o"), ("1.5b", PAL["d"], "s")]):
        bw = d3[mdl][kk]["by_workload"]
        pts3[mdl] = {w: v["ratio"] for w, v in bw.items()}
        ax[2].scatter(range(len(bw)), [v["ratio"] for v in bw.values()], color=c, marker=mk, label=f"{mdl} (held-out, call-level)")
        for i, (w, v) in enumerate(bw.items()):
            ax[2].annotate(w, (i, v["ratio"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[2].axhline(1, color="#aaa", lw=0.8)
    ax[2].set_title("call-level held-out wall ratio (incl. 1.5B)", fontsize=9)
    ax[2].set_xticks([])
    ax[2].legend(fontsize=7)
    fig.suptitle("F4 (E5v2 7B held-out repeats 2-4, plus call-level 1.5B): replay vs plain recomputation")
    RES["F4"] = {"path": sv(fig, "F4_replay", {"H1": res, "D3_callevel_wall_ratio": pts3,
                                               "note": "1.5B has no full-cell data (cut, preregistration.json); points are call-level"})}

    # T3
    lines = ["# T3 replay identity (text-identical decoded calls)", "",
             "| source | arm | calls compared | identical | mismatches |", "|---|---|---|---|---|"]
    for arm in ["prefix", "resync"]:
        lines.append(f"| E5v2 full cells, 7B held-out (runs/v2/cells.jsonl) | {arm} | {comp[arm][0]} | {comp[arm][1]} | {comp[arm][0]-comp[arm][1]} |")
    for mdl in ["7b", "1.5b"]:
        for cfg, v in d3[mdl].items():
            if not cfg.startswith("{"):
                continue
            c = json.loads(cfg)
            lines.append(f"| held-out call-level, {mdl} (runs/v2/D3/heldout.json) | K={c['K']} margin={c['margin']} tie={c['tie']} | "
                         f"{v['calls']} | {v['identical']} | {len(v['mismatches'])} |")
    if mism_all:
        lines += ["", "E5v2 mismatches (position = workload/fault/repeat/style, step, call_index):", ""]
        for m in mism_all:
            lines.append(f"- {m['arm']}: {m['workload']}/{m['fault']}/rep{m['repeat']}/{m['style']} step={m['step']} call={m['call_index']} "
                         f"(in off: {m['in_off']}, in arm: {m['in_arm']})")
    (OUT / "T3_replay_identity.md").write_text("\n".join(lines) + "\n")
    RES["T3"] = "runs/v2/figures/T3_replay_identity.md"
    return idx


# ------------------------------------------------------------------------------ H2 / F5
def h2_f5(e5, e6, idx):
    out = {}
    for w in REAL:
        clean = defaultdict(dict)
        for r in e6:
            if sp(r)["workload"] == w:
                clean[sp(r)["repeat"]][fl(r, "style")] = r
        reps = sorted(r for r in clean if len(clean[r]) == 2)
        faults = sorted({k[1] for k in idx if k[0] == w})
        pairs = []                                # (fault, repeat): saving for each metric
        for f in faults:
            for rp in sorted({k[2] for k in idx if k[0] == w and k[1] == f}):
                c0 = idx.get((w, f, rp, "cot", "off"))
                a1 = idx.get((w, f, rp, "atoms", "resync"))
                if c0 and a1:
                    pairs.append((c0, a1))
        for metric, getc, getr in [("wall_s", lambda r: r["clean"]["llm"]["wall_s"], lambda r: rec_llm(r)["wall_s"]),
                                   ("decoded", lambda r: r["clean"]["llm"]["decoded"], lambda r: rec_llm(r)["decoded"])]:
            ov = [getc(clean[rp]["atoms"]) - getc(clean[rp]["cot"]) for rp in reps]
            sav = [getr(c0) - getr(a1) for c0, a1 in pairs]

            def pstar(o, s):
                e, g = st.fmean(o), st.fmean(s)
                return (max(0.0, e) / g) if g > 0 else math.inf

            ps = pstar(ov, sav)
            bs = sorted(pstar([RNG.choice(ov) for _ in ov], [RNG.choice(sav) for _ in sav]) for _ in range(NB))
            lo, hi = bs[int(0.025 * NB)], bs[int(0.975 * NB) - 1]
            out[f"{w}/{metric}"] = {"clean_overhead_atoms_minus_cot_per_run": [st.fmean(ov), st.stdev(ov) if len(ov) > 1 else 0, len(ov)],
                                    "recovery_saving_cot_off_minus_atoms_tuned": [st.fmean(sav), st.stdev(sav) if len(sav) > 1 else 0, len(sav)],
                                    "p_star": ps, "ci95": [lo, hi], "bounded": math.isfinite(hi),
                                    "share_boot_saving_le_0": sum(1 for b in bs if b == math.inf) / NB}
    bnd_w = [w for w in REAL if out[f"{w}/wall_s"]["bounded"]]
    bnd_t = [w for w in REAL if out[f"{w}/decoded"]["bounded"]]
    h2 = {"p_star": out, "bounded_on_wall": bnd_w, "bounded_on_decoded": bnd_t,
          "verdict": "supported" if len(bnd_w) >= 2 else "not supported",
          "verdict_decoded_metric": "supported" if len(bnd_t) >= 2 else "not supported"}
    # live clean costs
    cc = {}
    for w in REAL:
        for sty in ["cot", "atoms"]:
            rows = [r for r in e6 if sp(r)["workload"] == w and fl(r, "style") == sty]
            cc[f"{w}/{sty}"] = {"n": len(rows), "llm_wall_s": ms([r["clean"]["llm"]["wall_s"] for r in rows])[:2],
                                "decoded": ms([r["clean"]["llm"]["decoded"] for r in rows])[:2],
                                "ok_rate": sum(1 for r in rows if r["clean"]["ok"]) / len(rows),
                                "cache_hits": sum(r["clean"]["cache_hits"] for r in rows)}
    h2["live_clean_costs"] = cc
    RES["H2"] = h2

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for si, (sty, c) in enumerate([("cot", PAL["k"]), ("atoms", PAL["c"])]):
        ax[0].bar([i + (si - 0.5) * 0.35 for i in range(4)], [cc[f"{w}/{sty}"]["llm_wall_s"][0] for w in REAL], 0.35,
                  yerr=[cc[f"{w}/{sty}"]["llm_wall_s"][1] for w in REAL], capsize=2, color=c, label=sty)
        ax[1].bar([i + (si - 0.5) * 0.35 for i in range(4)], [cc[f"{w}/{sty}"]["decoded"][0] for w in REAL], 0.35,
                  yerr=[cc[f"{w}/{sty}"]["decoded"][1] for w in REAL], capsize=2, color=c, label=sty)
    ax[0].set_title("live clean run LLM wall (s), E6v2 n=5", fontsize=9)
    ax[1].set_title("live clean run decoded tokens, E6v2 n=5", fontsize=9)
    for a in ax[:2]:
        a.set_xticks(range(4), REAL)
        a.legend(fontsize=8)
    for mi, (metric, c) in enumerate([("wall_s", PAL["b"]), ("decoded", PAL["a"])]):
        for i, w in enumerate(REAL):
            d = out[f"{w}/{metric}"]
            p = min(1.0, d["p_star"])
            lo, hi = min(1.0, d["ci95"][0]), min(1.0, d["ci95"][1])
            ax[2].bar(i + (mi - 0.5) * 0.35, p, 0.35, color=c, label=f"p* by {metric}" if i == 0 else None)
            ax[2].errorbar(i + (mi - 0.5) * 0.35, p, yerr=[[max(0, p - lo)], [max(0, hi - p)]], color="k", capsize=2)
    ax[2].set_xticks(range(4), REAL)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("p* (clipped at 1.0 = never pays)")
    ax[2].set_title("p* with bootstrap 95% CI", fontsize=9)
    ax[2].legend(fontsize=8)
    fig.suptitle("F5 (E6v2 live clean + E5v2 held-out recovery): break-even fault rate of atoms vs free CoT")
    RES["F5"] = {"path": sv(fig, "F5_pstar", out | {"live_clean": cc})}


# ------------------------------------------------------------------------------ H3 / H4 / T2 / F6
ARMS = ["always", "m3", "m3t", "m3t+minicheck", "uniform0.1", "uniform0.25", "m3t+m4", "m3t+m4+gate"]


def aud_tok(r, ph="faulty"):
    return (r.get(ph) or {}).get("audit", {}).get("audit_tokens", 0) if ph == "faulty" else (r.get("recovery") or {}).get("audit", {}).get("audit_tokens", 0)


def h3_h4_t2_f6(e7):
    held = [r for r in e7 if sp(r)["repeat"] >= 2]
    tab = {}
    for w in REAL:
        for arm in ARMS:
            fa = [r for r in held if sp(r)["workload"] == w and sp(r)["label"] == arm and sp(r)["fault"] != "clean"]
            cl = [r for r in held if sp(r)["workload"] == w and sp(r)["label"] == arm and sp(r)["fault"] == "clean"]
            for f in sorted({sp(r)["fault"] for r in fa}) + ["ALL"]:
                sub = [r for r in fa if f == "ALL" or sp(r)["fault"] == f]
                k = sum(1 for r in sub if r.get("detected"))
                pr, lo, hi = wilson(k, len(sub))
                tab[f"{w}/{f}/{arm}"] = {
                    "n_faulty": len(sub), "detected": k, "det": [pr, lo, hi],
                    "audit_tok_faulty": ms([aud_tok(r) for r in sub])[:2],
                    "audit_tok_recovery": ms([aud_tok(r, "rec") for r in sub])[:2],
                    "n_clean": len(cl), "fp_per_clean": ms([r.get("fp_flags", 0) for r in cl])[:2],
                    "clean_runs_with_flag": sum(1 for r in cl if r.get("fp_flags", 0) > 0),
                    "audit_tok_clean": ms([aud_tok(r) for r in cl])[:2],
                    "unauthorized_faulty": sum(1 for r in sub if r.get("unauthorized_faulty")),
                    "unauthorized_final": sum(1 for r in sub if r.get("unauthorized"))}
    # H3
    h3 = {}
    ok_all = True
    for w, f in sorted(INJECT):
        m3t, alw = tab[f"{w}/{f}/m3t"], tab[f"{w}/{f}/always"]
        fp = tab[f"{w}/ALL/m3t"]["fp_per_clean"][0]
        c = {"m3t_detection": m3t["det"], "m3t_detected_of_n": [m3t["detected"], m3t["n_faulty"]],
             "m3t_fp_per_clean_run": fp, "m3t_clean_n": tab[f"{w}/ALL/m3t"]["n_clean"],
             "m3t_tokens": m3t["audit_tok_faulty"][0], "always_tokens": alw["audit_tok_faulty"][0],
             "det_ge_0.8": m3t["det"][0] >= 0.8, "fp_le_0.2": fp <= 0.2, "tokens_lt_always": m3t["audit_tok_faulty"][0] < alw["audit_tok_faulty"][0]}
        c["pass"] = c["det_ge_0.8"] and c["fp_le_0.2"] and c["tokens_lt_always"]
        ok_all &= c["pass"]
        h3[f"{w}/{f}"] = c
        for arm in ["m3t+minicheck", "m3t+m4", "m3"]:
            t = tab[f"{w}/{f}/{arm}"]
            c[f"beside_{arm}"] = {"det": t["det"], "detected_of_n": [t["detected"], t["n_faulty"]],
                                  "tokens": t["audit_tok_faulty"][0], "fp_per_clean_run": tab[f"{w}/ALL/{arm}"]["fp_per_clean"][0]}
    RES["H3"] = {"per_cell": h3, "verdict": "supported" if ok_all else "not supported"}

    # H4 frontier
    def pooled(arm, inj_only=False):
        fa = [r for r in held if sp(r)["label"] == arm and sp(r)["fault"] != "clean"
              and (not inj_only or (sp(r)["workload"], sp(r)["fault"]) in INJECT)]
        cl = [r for r in held if sp(r)["label"] == arm and sp(r)["fault"] == "clean"]
        k = sum(1 for r in fa if r.get("detected"))
        return {"n": len(fa), "detected": k, "det": list(wilson(k, len(fa))), "tok": st.fmean([aud_tok(r) for r in fa]),
                "fp": st.fmean([r.get("fp_flags", 0) for r in cl]), "n_clean": len(cl),
                "unauth_faulty_cells": sum(1 for r in fa if r.get("unauthorized_faulty")),
                "unauth_final_cells": sum(1 for r in fa if r.get("unauthorized"))}
    fr = {m: {a: pooled(a, inj) for a in ARMS} for m, inj in [("all_fault_cells", False), ("inject_cells", True)]}

    def dominated_by(a, b):          # b dominates a
        return b["det"][0] >= a["det"][0] and b["tok"] <= a["tok"] and (b["det"][0] > a["det"][0] or b["tok"] < a["tok"])
    dom = {}
    for m in fr:
        dom[m] = {u: dominated_by(fr[m]["m3t+m4"], fr[m][u]) for u in ["uniform0.1", "uniform0.25"]}
    dom["per_workload"] = {}
    for w in REAL:
        a = tab[f"{w}/ALL/m3t+m4"]
        dom["per_workload"][w] = {u: (lambda b: b["det"][0] >= a["det"][0] and b["audit_tok_faulty"][0] <= a["audit_tok_faulty"][0]
                                       and (b["det"][0] > a["det"][0] or b["audit_tok_faulty"][0] < a["audit_tok_faulty"][0]))(tab[f"{w}/ALL/{u}"])
                                  for u in ["uniform0.1", "uniform0.25"]}
    # gate
    gate = []
    for r in held:
        if sp(r)["label"] == "m3t+m4+gate" and (sp(r)["workload"], sp(r)["fault"]) in INJECT:
            rc = r.get("recovery") or {}
            g = r["gate"]
            ef = rc.get("effects", {})
            held_n = g["held"]
            acc = ef.get("withdrawn", 0) + ef.get("released", 0) + g["held_final"]
            gate.append({"w": sp(r)["workload"], "fault": sp(r)["fault"], "repeat": sp(r)["repeat"], "held": held_n,
                         "held_final": g["held_final"], "released_after_recovery": g["released_after_recovery"],
                         "withdrawn": ef.get("withdrawn", 0), "released_in_recovery": ef.get("released", 0), "leaked": g["leaked"],
                         "unauthorized_faulty": g["unauthorized_faulty"], "unauthorized_final": g["unauthorized"],
                         "ok_recovered": rc.get("ok_recovered"), "accounted": acc >= held_n and g["leaked"] == 0})
    ng0 = sum(1 for g in gate if g["unauthorized_faulty"] == 0)
    ng0f = sum(1 for g in gate if g["unauthorized_final"] == 0)
    gate_clean = [r for r in held if sp(r)["label"] == "m3t+m4+gate" and sp(r)["fault"] == "clean"]
    gfp = defaultdict(lambda: [0, 0])
    for r in gate_clean:
        gfp[sp(r)["workload"]][0] += int(r["gate"]["held"] > 0)
        gfp[sp(r)["workload"]][1] += 1
    nogate = {w: [sum(1 for r in held if sp(r)["label"] == "m3t+m4" and sp(r)["workload"] == w and sp(r)["fault"] == f and r.get("unauthorized_faulty")),
                  sum(1 for r in held if sp(r)["label"] == "m3t+m4" and sp(r)["workload"] == w and sp(r)["fault"] == f)] for w, f in sorted(INJECT)}
    # E8v2 gate arm (descriptive; repeats 0-4)
    # budget-matched random baseline (descriptive, NOT the pre-registered rule): piecewise-linear detection of random audits
    # through (0,0), uniform0.1, uniform0.25 and `always` (= p 1.0), read at the token budget of m3t+m4
    def interp(m, tok):
        pts = sorted([(0.0, 0.0)] + [(fr[m][a]["tok"], fr[m][a]["det"][0]) for a in ("uniform0.1", "uniform0.25", "always")])
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= tok <= x1:
                return y0 + (y1 - y0) * (tok - x0) / (x1 - x0)
        return None
    bm = {m: {"m3t+m4_tokens": fr[m]["m3t+m4"]["tok"], "m3t+m4_detection": fr[m]["m3t+m4"]["det"][0],
              "random_audit_detection_at_same_tokens": interp(m, fr[m]["m3t+m4"]["tok"])} for m in fr}
    h4 = {"budget_matched_random_baseline_descriptive": bm, "frontier": fr, "m3t+m4_dominated_by_uniform": dom, "gate_inject_cells": gate,
          "gate_unauthorized_faulty_zero": [ng0, len(gate)], "gate_unauthorized_final_zero": [ng0f, len(gate)],
          "all_held_accounted": all(g["accounted"] for g in gate),
          "m3t+m4_without_gate_unauthorized_faulty_cells": nogate,
          "gate_false_holds_on_clean_runs": {w: v for w, v in gfp.items()}}
    nd = not any(dom["all_fault_cells"].values())
    h4["rule"] = {"not_dominated_pooled_all_fault_cells": nd, "not_dominated_inject_cells": not any(dom["inject_cells"].values()),
                  "gate_unauthorized_zero_ge_80pct": ng0 / max(1, len(gate)) >= 0.8, "held_accounted": h4["all_held_accounted"]}
    h4["verdict"] = "supported" if (nd and h4["rule"]["gate_unauthorized_zero_ge_80pct"] and h4["all_held_accounted"]) else "not supported"
    RES["H4"] = h4
    RES["T2_cells"] = tab

    # F6
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    mk = {"always": ("D", PAL["k"]), "m3": ("v", PAL["e"]), "m3t": ("o", PAL["a"]), "m3t+minicheck": ("o", PAL["d"]),
          "m3t+m4": ("s", PAL["c"]), "m3t+m4+gate": ("P", PAL["b"])}
    for a, m in [(ax[0], "all_fault_cells"), (ax[1], "inject_cells")]:
        u = [fr[m]["uniform0.1"], fr[m]["uniform0.25"]]
        a.plot([0] + [x["tok"] for x in u], [0] + [x["det"][0] for x in u], "--", color="#888", label="uniform p = 0.1, 0.25 (+ origin)")
        for arm in ["uniform0.1", "uniform0.25"]:
            x = fr[m][arm]
            a.errorbar(x["tok"], x["det"][0], yerr=EB(x["det"]), fmt="x", color="#888", capsize=2)
            a.annotate(arm[-4:], (x["tok"], x["det"][0]), fontsize=6, xytext=(3, -8), textcoords="offset points")
        for arm, (mrk, c) in mk.items():
            x = fr[m][arm]
            a.errorbar(x["tok"], x["det"][0], yerr=EB(x["det"]), fmt=mrk, color=c, capsize=2,
                       mfc="none" if arm == "m3t+minicheck" else c, label=arm + (" (T5 tokens)" if arm == "m3t+minicheck" else ""))
        a.set_xlabel("audit tokens per faulty run (faulty phase)")
        a.set_ylabel("detection (Wilson 95%)")
        a.set_title(f"{m.replace('_', ' ')} (n={fr[m]['always']['n']} per arm)", fontsize=9)
    ax[0].legend(fontsize=6)
    for i, arm in enumerate(["m3t+m4", "m3t+m4+gate"]):
        for j, (w, f) in enumerate(sorted(INJECT)):
            n_u = sum(1 for r in held if sp(r)["label"] == arm and sp(r)["workload"] == w and sp(r)["fault"] == f and r.get("unauthorized_faulty"))
            n_t = sum(1 for r in held if sp(r)["label"] == arm and sp(r)["workload"] == w and sp(r)["fault"] == f)
            ax[2].bar(j + (i - 0.5) * 0.35, n_u, 0.35, color=PAL["b"] if i else PAL["k"], label=arm if j == 0 else None)
            ax[2].annotate(f"{n_u}/{n_t}", (j + (i - 0.5) * 0.35, n_u), ha="center", fontsize=8)
    ax[2].set_xticks(range(2), [f"{w}\n{f}" for w, f in sorted(INJECT)])
    ax[2].set_ylabel("cells with an unauthorized act in the faulty run")
    ax[2].set_ylim(0, 4)
    ax[2].legend(fontsize=7)
    ax[2].set_title("irreversible gate: unauthorized acts", fontsize=9)
    fig.suptitle("F6 (E7v2 7B held-out repeats 2-4): audit frontier (hollow = MiniCheck, tokens are T5 tokens)")
    RES["F6"] = {"path": sv(fig, "F6_audit_frontier", {"frontier": fr, "gate": gate})}

    # T2
    L = ["# T2 audits (E7v2, 7B, repeats 2-4, runs/v2/cells.jsonl)", "",
         "Detection = flag in cone(f) + {f} in the faulty run, Wilson 95%. FP = flags per clean run of that workload and arm. "
         "Tokens: Qwen audit tokens in the faulty phase; recovery column = audit tokens of the recovery phase. "
         "m3t+minicheck tokens are T5 tokens (not comparable). unauth = cells with `unauthorized_faulty` > 0.", "",
         "| workload | fault | audit | detected (n) | detection [Wilson] | FP/clean run (n clean) | audit tok / faulty run | recovery audit tok | unauth cells |",
         "|---|---|---|---|---|---|---|---|---|"]
    for w in REAL:
        for key in [k for k in tab if k.startswith(w + "/") and "/ALL/" not in k]:
            t = tab[key]
            _, f, arm = key.split("/")
            L.append(f"| {w} | {f} | {arm} | {t['detected']}/{t['n_faulty']} | {t['det'][0]:.2f} [{t['det'][1]:.2f},{t['det'][2]:.2f}] | "
                     f"{tab[f'{w}/ALL/{arm}']['fp_per_clean'][0]:.2f} ({tab[f'{w}/ALL/{arm}']['n_clean']}) | {t['audit_tok_faulty'][0]:.0f} | "
                     f"{t['audit_tok_recovery'][0]:.0f} | {t['unauthorized_faulty']} |")
    (OUT / "T2_audits.md").write_text("\n".join(L) + "\n")
    RES["T2"] = "runs/v2/figures/T2_audits.md"


# ------------------------------------------------------------------------------ H5 / T1 / F3 / F8
METH = ["restart", "rollback", "selective", "selective+M1", "selective+M1+M3t/M4+gate"]


def h5_t1_f3_f8(e8, p1):
    g = lambda w, m: [r for r in e8 if sp(r)["workload"] == w and sp(r)["label"] == m]   # noqa: E731
    keyed = {(sp(r)["workload"], sp(r)["label"], sp(r)["fault"], sp(r)["repeat"]): r for r in e8}
    t1 = {}
    for w in REAL:
        for m in METH:
            rows = g(w, m)
            rv = [x for r in rows for x in [r["recovery"].get("reuse_check")] if x]
            eq = sum(x["equal"] for x in rv)
            smp = sum(x["sampled"] for x in rv)
            ok = [bool(r["recovery"]["ok_recovered"]) for r in rows]
            t1[f"{w}/{m}"] = {
                "n": len(rows), "ok_recovered": [sum(ok), len(ok)], "ok_wilson": list(wilson(sum(ok), len(ok))),
                "values_match": [sum(1 for r in rows if r["recovery"]["values_match"]), len(rows)],
                "n_rerun": ms([r["recovery"]["n_rerun"] for r in rows])[:2],
                "recovery_wall_s": ms([r["recovery"]["wall_s"] for r in rows])[:2],
                "recovery_llm_wall_s": ms([rec_llm(r)["wall_s"] for r in rows])[:2],
                "decoded": ms([rec_llm(r)["decoded"] for r in rows])[:2],
                "prefill": ms([rec_llm(r)["prefill"] for r in rows])[:2],
                "bookkeeping_pct_recovery_wall": 100 * st.fmean([bk_ns(r["recovery"]["bookkeeping"]) / 1e9 / r["recovery"]["wall_s"] for r in rows]),
                "bookkeeping_pct_faulty_wall": 100 * st.fmean([bk_ns(r["faulty"]["bookkeeping"]) / 1e9 / r["faulty"]["wall_s"] for r in rows]),
                "reuse_valid": [eq, smp], "reuse_candidates": sum(x["candidates"] for x in rv),
                "duplicates": ms([r["duplicates"] for r in rows])[:2], "unauthorized": sum(1 for r in rows if r.get("unauthorized")),
                "irrev_redone": ms([r["recovery"]["irreversible_redone"] for r in rows])[0],
                "compensated": ms([r["recovery"]["compensated"] for r in rows])[0],
                "leaked": ms([r["recovery"]["effects"]["leaked"] for r in rows])[0],
                "audit_tok_recovery": ms([r["recovery"].get("audit", {}).get("audit_tokens", 0) for r in rows])[0],
                "cache_hits": sum(r["recovery"]["cache_hits"] for r in rows)}
    # rule
    per_w = {}
    n_better = 0
    for w in REAL:
        rs, se = t1[f"{w}/restart"], t1[f"{w}/selective"]
        diff_ok = abs(se["ok_recovered"][0] - rs["ok_recovered"][0])
        pairs = [(keyed[(w, "restart", f, rp)], keyed[(w, "selective", f, rp)]) for (_, m, f, rp) in keyed if m == "selective" and _ == w]
        fewer = st.fmean([a["recovery"]["n_rerun"] - b["recovery"]["n_rerun"] for a, b in pairs])
        wall_d = [b["recovery"]["wall_s"] / a["recovery"]["wall_s"] for a, b in pairs]
        better = fewer > 0 and st.fmean(b["recovery"]["wall_s"] for _, b in pairs) < st.fmean(a["recovery"]["wall_s"] for a, _ in pairs)
        n_better += better
        per_w[w] = {"ok_restart": rs["ok_recovered"], "ok_selective": se["ok_recovered"], "abs_diff_cells": diff_ok,
                    "within_one_cell": diff_ok <= 1, "mean_fewer_reruns_vs_restart": fewer,
                    "wall_ratio_selective_over_restart_median": med(wall_d), "wall_ratio_ci95": list(boot_ci(wall_d, med)),
                    "fewer_reruns_and_less_wall": better,
                    "variants_within_one_cell": {m: abs(t1[f"{w}/{m}"]["ok_recovered"][0] - rs["ok_recovered"][0]) <= 1 for m in METH[2:]}}
    rv_by = {}
    for m in METH[2:]:
        eq = sum(t1[f"{w}/{m}"]["reuse_valid"][0] for w in ("w1", "w4.ipi", "w4.byz", "w3"))
        sm = sum(t1[f"{w}/{m}"]["reuse_valid"][1] for w in ("w1", "w4.ipi", "w4.byz", "w3"))
        rv_by[m] = [eq, sm, list(wilson(eq, sm))]
    eq = sum(v[0] for v in rv_by.values())
    sm = sum(v[1] for v in rv_by.values())
    rv_all = [eq, sm, list(wilson(eq, sm))]
    exact = sum(x.get("exact_equal", False) for r in e8 for x in (r["recovery"].get("reuse_check") or {}).get("steps", []))
    h5 = {"table": t1, "per_workload_rule": per_w, "reuse_validity_by_method": rv_by, "reuse_validity_pooled": rv_all,
          "reuse_validity_exact_equal_steps": [exact, sm], "workloads_better_than_restart": n_better,
          "rule": {"ok_within_one_cell_all_workloads": all(v["within_one_cell"] for v in per_w.values()),
                   "reuse_validity_ge_95pct": rv_all[0] / max(1, rv_all[1]) >= 0.95,
                   "fewer_reruns_less_wall_on_ge_3_of_4": n_better >= 3}}
    h5["verdict"] = "supported" if all(h5["rule"].values()) else "not supported"
    RES["H5"] = h5

    # T1
    L = ["# T1 end-to-end (E8v2, atoms, 7B, runs/v2/cells.jsonl; repeats 0-4 x faults). " + W3_RERUN_NOTE, "",
         "Recovery wall = whole recovery phase; LLM wall = decode+prefill time inside it (cache off, 0 hits). "
         "bk% = (ns_read+ns_write+ns_hash_outside) / recovery wall. Reuse valid = canonical-equal sampled scratch reruns / sampled "
         "(only steps with model calls). Effect policy differs by method (restart/rollback naive, selective diff, M1 rows diff+outbox).", "",
         "| workload | method | n | ok_recovered | values_match | steps rerun | rec. wall s | LLM wall s | decoded | prefill | bk % | reuse valid | dup | unauth | leaked | audit tok |",
         "|" + "---|" * 16]
    for w in REAL:
        for m in METH:
            d = t1[f"{w}/{m}"]
            L.append(f"| {w} | {m} | {d['n']} | {d['ok_recovered'][0]}/{d['ok_recovered'][1]} | {d['values_match'][0]}/{d['n']} | "
                     f"{AN.fmt(d['n_rerun'] + [0], 1) if False else format(d['n_rerun'][0], '.1f')} | {d['recovery_wall_s'][0]:.1f} ± {d['recovery_wall_s'][1]:.1f} | "
                     f"{d['recovery_llm_wall_s'][0]:.1f} | {d['decoded'][0]:.0f} | {d['prefill'][0]:.0f} | {d['bookkeeping_pct_recovery_wall']:.2f} | "
                     f"{d['reuse_valid'][0]}/{d['reuse_valid'][1]} | {d['duplicates'][0]:.2f} | {d['unauthorized']} | {d['leaked']:.1f} | {d['audit_tok_recovery']:.0f} |")
    (OUT / "T1_end_to_end.md").write_text("\n".join(L) + "\n")
    RES["T1"] = "runs/v2/figures/T1_end_to_end.md"

    # F8
    short = {"restart": "restart", "rollback": "rollback", "selective": "sel", "selective+M1": "sel+M1", "selective+M1+M3t/M4+gate": "sel+M1+gate"}
    pal = [PAL["k"], PAL["e"], PAL["a"], PAL["c"], PAL["b"]]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for mi, m in enumerate(METH):
        for j, w in enumerate(REAL):
            d = t1[f"{w}/{m}"]
            x = j + (mi - 2) * 0.16
            p, lo, hi = d["ok_wilson"]
            ax[0].bar(x, p, 0.16, color=pal[mi], label=short[m] if j == 0 else None)
            ax[0].errorbar(x, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]], color="k", capsize=1, lw=0.7)
            vm = d["values_match"][0] / d["n"]
            ax[1].bar(x, vm, 0.16, color=pal[mi])
            if d["reuse_valid"][1]:
                p, lo, hi = wilson(*d["reuse_valid"])
                ax[2].bar(x, p, 0.16, color=pal[mi], label=short[m] if j in (0, 2) and mi >= 2 and j == 0 else None)
                ax[2].errorbar(x, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]], color="k", capsize=1, lw=0.7)
                ax[2].annotate(f"{d['reuse_valid'][0]}/{d['reuse_valid'][1]}", (x, 0.02), ha="center", fontsize=5, rotation=90, color="w")
    for a, ttl in [(ax[0], "ok_recovered (Wilson 95%)"), (ax[1], "final values == live clean reference"),
                   (ax[2], "reuse validity (canonical equal / sampled; w3 has none)")]:
        a.set_xticks(range(4), REAL)
        a.set_ylim(0, 1.08)
        a.set_title(ttl, fontsize=9)
    ax[2].axhline(0.95, color=PAL["b"], ls="--", lw=1)
    ax[0].legend(fontsize=6, ncol=2)
    fig.suptitle("F8 (E8v2, atoms, 7B): correctness by recovery method; " + W3_RERUN_NOTE)
    RES["F8"] = {"path": sv(fig, "F8_correctness", {k: {kk: v[kk] for kk in ("n", "ok_recovered", "values_match", "reuse_valid")} for k, v in t1.items()})}

    # F3: v1 E4 (read-only) + E8v2 duplicate / held counts
    E4 = [r for r in p1 if sp(r)["exp"] == "E4"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    pols = ["naive", "diff", "diff+outbox"]
    pc = {"naive": PAL["b"], "diff": PAL["e"], "diff+outbox": PAL["a"]}
    v1 = {}
    for ai, (metric, ttl) in enumerate([("leaked", "v1 E4: leaked (released, cannot undo)"), ("added", "v1 E4: irreversible calls added")]):
        for pi, p in enumerate(pols):
            mu = []
            for mi, m in enumerate(["restart", "rollback", "selective"]):
                x = [(r["irreversible_calls_total"] - r["irreversible_calls_faulty"]) if metric == "added" else r["recovery"]["effects"]["leaked"]
                     for r in E4 if sp(r)["label"] == m and fl(r, "effects") == p]
                mu.append(st.fmean(x) if x else 0)
                v1[f"{metric}/{m}/{p}"] = [mu[-1], len(x)]
            ax[ai].bar([i + (pi - 1) * 0.27 for i in range(3)], mu, 0.27, color=pc[p], label=p)
        ax[ai].set_xticks(range(3), ["restart", "rollback", "selective"])
        ax[ai].set_title(ttl + " (runs/phase1, unchanged)", fontsize=8)
    ax[0].legend(fontsize=7)
    dup = {}
    for mi, m in enumerate(METH):
        for j, w in enumerate(REAL):
            rows = g(w, m)
            dup[f"{w}/{m}"] = {"duplicates": sum(r["duplicates"] for r in rows), "n": len(rows),
                               "leaked": sum(r["recovery"]["effects"]["leaked"] for r in rows),
                               "held_faulty": sum((r["faulty"].get("gate") or {}).get("held", 0) for r in rows),
                               "withdrawn": sum(r["recovery"]["effects"]["withdrawn"] for r in rows),
                               "released_in_recovery": sum(r["recovery"]["effects"]["released"] for r in rows)}
            ax[2].bar(j + (mi - 2) * 0.16, dup[f"{w}/{m}"]["duplicates"] / len(rows), 0.16, color=pal[mi], label=short[m] if j == 0 else None)
            ax[3].bar(j + (mi - 2) * 0.16, dup[f"{w}/{m}"]["leaked"] / len(rows), 0.16, color=pal[mi])
    for a, ttl in [(ax[2], "E8v2: duplicate irreversible acts per cell"), (ax[3], "E8v2: leaked effects per cell")]:
        a.set_xticks(range(4), REAL)
        a.set_title(ttl, fontsize=8)
    ax[2].legend(fontsize=6)
    gh = {}
    for w in ("w1", "w4.ipi"):
        rows = [r for r in g(w, METH[4]) if (w, sp(r)["fault"]) in INJECT]
        gh[w] = {"cells": len(rows), "held": sum(r["gate"]["held"] for r in rows), "held_final": sum(r["gate"]["held_final"] for r in rows),
                 "withdrawn": sum(r["recovery"]["effects"]["withdrawn"] for r in rows),
                 "released_after_recovery": sum(r["gate"]["released_after_recovery"] for r in rows),
                 "unauthorized_faulty_cells": sum(1 for r in rows if r.get("unauthorized_faulty"))}
    fig.suptitle("F3 effects: v1 E4 (policy x method) and E8v2 duplicate / leaked counts")
    RES["F3"] = {"path": sv(fig, "F3_effects", {"v1_E4": v1, "E8v2": dup, "E8v2_gate_inject_cells": gh})}
    RES["E8v2_gate_inject_cells"] = gh


# ------------------------------------------------------------------------------ F7
def f7(e5, e8):
    d3 = json.load(open(V2 / "D3" / "profile.json"))
    rows = load(V2 / "D3" / "heldout_rows_7b.jsonl")
    phases = ["t_prefill", "t_verify", "t_tie", "t_decode", "t_pick", "t_resync"]

    def tot(cfgkey):
        sel = [r for r in rows if json.dumps(r["cfg"], sort_keys=True) == cfgkey]
        out = {p: sum(r["phases"].get(p, 0) for r in sel) for p in phases}
        out["wall"] = sum(r["wall_s"] for r in sel)
        out["other"] = max(0.0, out["wall"] - sum(out[p] for p in phases))
        return out, len(sel)
    tuned, n_t = tot('{"K": 64, "margin": 0.1, "mode": "resync", "tie": "local"}')
    off, n_o = tot('{"mode": "off"}')
    off_ = {p: 0 for p in phases}
    off_["t_decode"] = off["wall"]
    off_["other"] = 0
    bars = {"plain (held-out, 7B)": off_, "v1 replay K32 m0.3 full tie (profile)": {**{p: d3["phase_s"].get(p, 0) for p in phases}, "other": d3["phase_s"].get("t_other", 0)},
            "tuned (held-out, 7B)": tuned}
    cols = {"t_prefill": PAL["e"], "t_verify": PAL["a"], "t_tie": PAL["d"], "t_decode": PAL["k"], "t_pick": "#bbb", "t_resync": "#ddd", "other": "#eee"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    data = {}
    for i, (name, v) in enumerate(bars.items()):
        bot = 0
        for p in phases + ["other"]:
            ax[0].bar(i, v[p], 0.6, bottom=bot, color=cols[p], label=p if i == 0 else None)
            bot += v[p]
        data[name] = v
    ax[0].set_xticks(range(3), ["plain", "v1 replay\n(13 reruns)", "tuned\n(7 cases)"], fontsize=8)
    ax[0].set_ylabel("seconds (call level; prefill/verify/tie/decode)")
    ax[0].legend(fontsize=6)
    ax[0].set_title("(a) call-level phase breakdown", fontsize=9)
    # (b) full-cell recovery: LLM wall, bookkeeping, other (tools/audits/overhead)
    full = {}
    for arm in ["off", "prefix", "resync"]:
        rs = [r for r in e5 if fl(r, "replay") == arm and r.get("recovery")]
        llm_w = st.fmean(rec_llm(r)["wall_s"] for r in rs)
        bk = st.fmean(bk_ns(r["recovery"]["bookkeeping"]) / 1e9 for r in rs)
        tw = st.fmean(r["recovery"]["wall_s"] for r in rs)
        full[arm] = {"n": len(rs), "recovery_wall": tw, "llm_wall": llm_w, "bookkeeping": bk, "other_tools_etc": max(0.0, tw - llm_w - bk),
                     "prefill_tokens": st.fmean(rec_llm(r)["prefill"] for r in rs), "decoded_tokens": st.fmean(rec_llm(r)["decoded"] for r in rs),
                     "accepted_tokens": st.fmean(rec_llm(r)["accepted"] for r in rs)}
    for i, arm in enumerate(["off", "prefix", "resync"]):
        v = full[arm]
        ax[1].bar(i, v["llm_wall"], 0.6, color=PAL["a"], label="LLM (prefill+verify+tie+decode)" if i == 0 else None)
        ax[1].bar(i, v["bookkeeping"], 0.6, bottom=v["llm_wall"], color=PAL["c"], label="bookkeeping" if i == 0 else None)
        ax[1].bar(i, v["other_tools_etc"], 0.6, bottom=v["llm_wall"] + v["bookkeeping"], color="#ccc", label="other (tools, hashing, audits)" if i == 0 else None)
    ax[1].set_xticks(range(3), ["off", "prefix", "resync-tuned"])
    ax[1].set_ylabel("mean recovery wall (s), E5v2 all cells")
    ax[1].legend(fontsize=7)
    ax[1].set_title("(b) E5v2 full-cell recovery wall split (7B)", fontsize=9)
    fig.suptitle("F7 recovery latency breakdown (phase split exists only at call level; full cells record LLM wall + bookkeeping)")
    RES["F7"] = {"path": sv(fig, "F7_latency", {"D3_callevel": data, "E5v2_full": full, "D3_call_counts": {"tuned": n_t, "off": n_o}})}


# ------------------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load(CELLS)
    prog = json.load(open(V2 / "progress.json"))
    ex = split_by_exp(rows)
    e2 = load(E2V2)
    RES["grid"] = {"rows": len(rows), "by_exp": {k: len(v) for k, v in ex.items()}, "progress": prog,
                   "errors": sum(1 for r in rows if "error" in r), "peak_rss_mb_max": max(r["peak_rss_mb"] for r in rows if "peak_rss_mb" in r),
                   "cache_hits_in_measured_phases": sum(r[p].get("cache_hits", 0) for r in rows for p in ("clean", "faulty", "recovery", "audit_ref", "clean_ref") if isinstance(r.get(p), dict) and sp(r)["model"] == "7b"),
                   "E2v2_rows": len(e2)}
    e3, e5, e6, e7, e8 = (ex[k] for k in ("E3v2", "E5v2", "E6v2", "E7v2", "E8v2"))
    h7_f1(e2)
    h6_f2(e3, e5, e8, e6)
    idx = h1_f4_t3(e5)
    h2_f5(e5, e6, idx)
    h3_h4_t2_f6(e7)
    h5_t1_f3_f8(swap_w3(e8), load(PHASE1))
    f7(e5, e8)
    (OUT / "analysis_v2.json").write_text(jdump(RES))
    for h in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
        print(h, RES[h]["verdict"])
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
