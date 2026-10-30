#!/usr/bin/env python3
"""S9 analysis: runs/ cell files -> runs/figures/*.png, tables, analysis.json.

Reads only cell files (runs/phase1/cells.jsonl, runs/phase1/baselines/cells.jsonl,
runs/full/cells.jsonl). Mean +- sd is taken over the cells named in each caption
(repeats, and the other factors pooled where stated). Run inside the project env (conda activate agent-rewind):
    python scripts/analyze.py
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "figures"
P1 = RUNS / "phase1" / "cells.jsonl"
P1B = RUNS / "phase1" / "baselines" / "cells.jsonl"
FULL = RUNS / "full" / "cells.jsonl"

PAL = {"a": "#0072B2", "b": "#D55E00", "c": "#009E73", "d": "#CC79A7", "e": "#E69F00", "k": "#444444"}
REAL = ["w1", "w3", "w4.byz", "w4.ipi"]
RNG = random.Random(291)


def load(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return (None, None, 0)
    return (st.fmean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))


def fmt(m, d=2):
    if m[0] is None:
        return "n/a"
    return f"{m[0]:.{d}f} ± {m[1]:.{d}f}"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0, c - h), min(1, c + h))


def by(rows, *keyfn):
    g = defaultdict(list)
    for r in rows:
        g[tuple(f(r) for f in keyfn)].append(r)
    return g


S = lambda k: (lambda r: r["spec"][k])          # noqa: E731
FL = lambda k: (lambda r: r["spec"]["flags"][k])  # noqa: E731


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)
    return f"runs/figures/{name}"


A = {"sources": {"phase1": "runs/phase1/cells.jsonl", "baselines": "runs/phase1/baselines/cells.jsonl",
                 "full": "runs/full/cells.jsonl"}}

# ---------------------------------------------------------------- Figure 1 (E2) and P1 check


def fig1(p1):
    E2 = [r for r in p1 if r["spec"]["exp"] == "E2"]
    meths = ["restart", "rollback", "selective", "selective-d2"]
    col = {"restart": PAL["k"], "rollback": PAL["e"], "selective": PAL["a"], "selective-d2": PAL["c"]}
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    out = {}
    for row, (fac, vals) in enumerate([("fan_in", [1, 2, 3]), ("shared_keys", [0, 3, 6])]):
        for j, v in enumerate(vals):
            a = ax[row][j]
            cells = [r for r in E2 if r["spec"]["params"][fac] == v]
            pos = sorted({r["spec"]["params"]["position"] for r in cells})
            for m in meths:
                mu, sd = [], []
                for p in pos:
                    x = [r["recovery"]["n_rerun"] for r in cells if r["spec"]["label"] == m
                         and r["spec"]["params"]["position"] == p]
                    mm = ms(x)
                    mu.append(mm[0])
                    sd.append(mm[1])
                a.errorbar(pos, mu, yerr=sd, marker="o", ms=3, capsize=2, color=col[m], label=m)
                out[f"{fac}={v}/{m}"] = dict(zip(map(str, pos), [round(x, 2) for x in mu]))
            for key, ls, lab in [("cone", ":", "cone(f)"), ("suffix", "--", "time-order suffix")]:
                mu = []
                for p in pos:
                    xs = []
                    for r in cells:
                        if r["spec"]["label"] != "selective" or r["spec"]["params"]["position"] != p:
                            continue
                        if key == "cone":
                            xs.append(len(r["propagation"]["cone"]) + 1)
                        else:
                            steps = r["faulty"]["steps"]
                            xs.append(len(steps) - steps.index(r["fault_at"][0]))
                    mu.append(st.fmean(xs))
                a.plot(pos, mu, ls=ls, color="#888", label=lab)
            a.set_title(f"{fac} = {v}")
            a.set_xlabel("fault position (fraction of DAG)")
            if j == 0:
                a.set_ylabel("steps rerun (of 24)")
    ax[0][0].legend(fontsize=7)
    fig.suptitle("Figure 1 (E2, W2 scripted): steps rerun vs fault position; mean ± sd over 5 repeats, "
                 "other factor pooled; cone(f) includes origin")
    path = save(fig, "fig1_cone_vs_suffix.png")

    # P1 check on method=selective, exact cutoff, field depth
    sel = [r for r in E2 if r["spec"]["label"] == "selective"]
    viol, genuine, ctl = 0, 0, 0
    for r in sel:
        rer, cone, o = set(r["recovery"]["rerun"]), set(r["propagation"]["cone"]), r["fault_at"][0]
        extra = rer - cone - {o}
        if extra:
            viol += 1
            if extra - set(r["faulty"]["steps"]):
                ctl += 1           # extra steps never ran in the faulty run (control flow changed)
            else:
                genuine += 1
    brk = defaultdict(lambda: [0, 0])
    for r in sel:
        rer, cone, o = set(r["recovery"]["rerun"]), set(r["propagation"]["cone"]), r["fault_at"][0]
        extra = (rer - cone - {o}) & set(r["faulty"]["steps"])
        kind = "kv-path fault" if r["fault_at"][1].startswith("/kv") else "msg-path fault"
        brk[kind][0] += 1
        brk[kind][1] += int(bool(extra))
        brk[f"shared_keys={r['spec']['params']['shared_keys']}"][0] += 1
        brk[f"shared_keys={r['spec']['params']['shared_keys']}"][1] += int(bool(extra))
    over_suffix = sum(1 for r in sel if len(r["recovery"]["rerun"]) > len(r["faulty"]["steps"]) -
                      r["faulty"]["steps"].index(r["fault_at"][0]))
    red = {}
    for m in meths:
        x = [r["recovery"]["n_rerun"] for r in E2 if r["spec"]["label"] == m]
        red[m] = ms(x)[:2]
    ratio = [r["recovery"]["n_rerun"] / len(r["propagation"]["cone"] or [0]) if r["propagation"]["cone"] else None
             for r in sel]
    A["fig1"] = {"path": path, "cells_selective": len(sel), "rerun_not_subset_of_cone": viol,
                 "of_which_steps_not_in_faulty_run": ctl, "of_which_steps_that_ran_in_faulty_run": genuine, "genuine_by_kind[n,viol]": dict(brk),
                 "rerun_larger_than_suffix": over_suffix, "mean_sd_rerun_by_method": red, "points": out,
                 "selective_rerun_over_cone+1_mean": ms([x for x in ratio if x is not None])[:2]}


# ---------------------------------------------------------------- Figure 2 (E3)


def fig2(p1):
    E3 = [r for r in p1 if r["spec"]["exp"] == "E3"]
    tiers = ["agent", "step", "artifact", "field"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    res = {}
    for wi, (w, lab) in enumerate([("w2", "W2 (auto fault)"), ("w1", "W1 (3 faults, scripted)")]):
        for cut, c in [("exact", PAL["a"]), ("canonical", PAL["b"])]:
            mu, sd = [], []
            for t in tiers:
                x = [r["recovery"]["n_rerun"] for r in E3 if r["spec"]["workload"] == w
                     and r["spec"]["flags"]["depth"] == t and r["spec"]["flags"]["cutoff"] == cut]
                m = ms(x)
                mu.append(m[0])
                sd.append(m[1])
                res[f"{w}/{cut}/{t}"] = {"rerun": [round(m[0], 2), round(m[1], 2)], "n": m[2]}
            ax[wi].errorbar(range(4), mu, yerr=sd, marker="o", capsize=3, color=c, label=f"cutoff {cut}")
        ax[wi].set_xticks(range(4), tiers)
        ax[wi].set_title(lab)
        ax[wi].set_ylabel("steps rerun")
        ax[wi].set_xlabel("tracking depth cut")
    ax[0].legend(fontsize=8)
    for cut, c in [("exact", PAL["a"]), ("canonical", PAL["b"])]:
        mu, sd = [], []
        for t in tiers:
            x = [r["faulty"]["wall_s"] * 1000 for r in E3 if r["spec"]["workload"] == "w2"
                 and r["spec"]["flags"]["depth"] == t and r["spec"]["flags"]["cutoff"] == cut]
            m = ms(x)
            mu.append(m[0])
            sd.append(m[1])
            res[f"w2/{cut}/{t}/faulty_ms"] = [round(m[0], 3), round(m[1], 3)]
        ax[2].errorbar(range(4), mu, yerr=sd, marker="o", capsize=3, color=c, label=cut)
    ax[2].set_xticks(range(4), tiers)
    ax[2].set_title("W2 bookkeeping proxy: faulty-run wall (ms)")
    ax[2].set_ylabel("ms (scripted, tracked run)")
    ax[2].legend(fontsize=8)
    fig.suptitle("Figure 2 (E3): precision by depth cut; mean ± sd over 5 repeats × positions/faults pooled")
    for w in ("w2", "w1"):
        d = []
        for r in E3:
            fl = r["spec"]["flags"]
            if r["spec"]["workload"] == w and fl["depth"] == "agent" and fl["cutoff"] == "exact":
                o = [q for q in E3 if q["spec"]["workload"] == w and q["spec"]["flags"]["depth"] == "field"
                     and q["spec"]["flags"]["cutoff"] == "exact" and q["spec"]["params"] == r["spec"]["params"]
                     and q["spec"]["fault"] == r["spec"]["fault"] and q["spec"]["repeat"] == r["spec"]["repeat"]]
                if o:
                    d.append(r["recovery"]["n_rerun"] - o[0]["recovery"]["n_rerun"])
        res[f"{w}/paired_agent_minus_field"] = [ms(d)[0], ms(d)[1], len(d), sum(1 for x in d if x < 0)]
    can = [r["recovery"]["n_rerun"] for r in E3 if r["spec"]["flags"]["cutoff"] == "canonical"]
    exa = [r["recovery"]["n_rerun"] for r in E3 if r["spec"]["flags"]["cutoff"] == "exact"]
    res["canonical_vs_exact_total_rerun"] = [sum(can), sum(exa)]
    A["fig2"] = {"path": save(fig, "fig2_precision.png"), "points": res}


# ---------------------------------------------------------------- Figure 3 (E4)


def fig3(p1):
    E4 = [r for r in p1 if r["spec"]["exp"] == "E4"]
    meths = ["restart", "rollback", "selective"]
    pols = ["naive", "diff", "diff+outbox"]
    col = {"naive": PAL["b"], "diff": PAL["e"], "diff+outbox": PAL["a"]}
    wl = ["w1", "w2", "w3"]
    fig, ax = plt.subplots(3, 3, figsize=(12, 8.5), sharex=True)
    res = {}
    for j, w in enumerate(wl):
        for i, (metric, ttl) in enumerate([("added", "irreversible calls added in recovery"),
                                           ("leaked", "old irreversible effects not undoable (leaked)"),
                                           ("compensated", "compensations")]):
            a = ax[i][j]
            for k, p in enumerate(pols):
                mu, sd = [], []
                for m in meths:
                    x = [(r["irreversible_calls_total"] - r["irreversible_calls_faulty"] if metric == "added" else
                          r["recovery"]["effects"]["leaked"] if metric == "leaked" else r["recovery"]["compensated"])
                         for r in E4 if r["spec"]["workload"] == w and r["spec"]["label"] == m
                         and r["spec"]["flags"]["effects"] == p]
                    mm = ms(x)
                    mu.append(mm[0])
                    sd.append(mm[1])
                    res[f"{w}/{m}/{p}/{metric}"] = [round(mm[0], 2), round(mm[1], 2), mm[2]]
                a.bar([q + (k - 1) * 0.27 for q in range(3)], mu, 0.27, yerr=sd, capsize=2, color=col[p], label=p)
            a.set_xticks(range(3), meths)
            a.set_title(f"{w}: {ttl}", fontsize=9)
    ax[0][0].legend(fontsize=7)
    fig.suptitle("Figure 3 (E4): effect policy × recovery method; mean ± sd over repeats × faults")
    A["fig3"] = {"path": save(fig, "fig3_effects.png"), "points": res}


# ---------------------------------------------------------------- Figure 4 (E5) / Figure 5 (E5+E6)


def e5_pairs(full):
    E5 = [r for r in full if r["spec"]["exp"] == "E5"]
    return E5


def rec_llm(r):
    return r["recovery"]["llm"]


def fig4(full):
    E5 = e5_pairs(full)
    arms = ["off", "prefix", "resync"]
    col = {"off": PAL["k"], "prefix": PAL["e"], "resync": PAL["a"]}
    fig, ax = plt.subplots(2, 4, figsize=(14, 6.5))
    res = {}
    ident = {"cells": 0, "out_tokens_equal": 0, "ok_equal": 0}
    for j, w in enumerate(REAL):
        for si, style in enumerate(["cot", "atoms"]):
            rows = {(r["spec"]["fault"], r["spec"]["repeat"], r["spec"]["flags"]["replay"]): r for r in E5
                    if r["spec"]["workload"] == w and r["spec"]["flags"]["style"] == style}
            keys = sorted({(f, rp) for (f, rp, _) in rows})
            for k, arm in enumerate(arms):
                dec, wall, acc = [], [], []
                for key in keys:
                    base, cur = rows.get(key + ("off",)), rows.get(key + (arm,))
                    if not base or not cur:
                        continue
                    lb, lc = rec_llm(base), rec_llm(cur)
                    if lb["decoded"] <= 0:
                        continue
                    dec.append(lc["decoded"] / lb["decoded"])
                    wall.append(lc["wall_s"] / lb["wall_s"] if lb["wall_s"] else None)
                    tot = lc["accepted"] + lc["decoded"]
                    acc.append(lc["accepted"] / tot if tot else None)
                    if arm != "off":
                        ident["cells"] += 1
                        ident["out_tokens_equal"] += int(tot == lb["decoded"])
                        ident["ok_equal"] += int(cur["recovery"]["ok_recovered"] == base["recovery"]["ok_recovered"])
                md, mw, ma = ms(dec), ms(wall), ms(acc)
                res[f"{w}/{style}/{arm}"] = {"decoded_ratio": [md[0], md[1]], "wall_ratio": [mw[0], mw[1]],
                                             "token_acceptance": [ma[0], ma[1]], "n": md[2]}
                ax[0][j].bar(k + si * 3.6, md[0], 0.8, yerr=md[1], capsize=2, color=col[arm],
                             hatch="//" if style == "atoms" else None, edgecolor="white")
                ax[1][j].bar(k + si * 3.6, mw[0] if mw[0] is not None else 0, 0.8, yerr=mw[1] or 0, capsize=2,
                             color=col[arm], hatch="//" if style == "atoms" else None, edgecolor="white")
        for r_ in (0, 1):
            ax[r_][j].set_xticks([1, 4.6], ["free CoT", "atoms"])
            ax[r_][j].axhline(1, color="#aaa", lw=0.8)
            ax[r_][j].set_title(w)
        ax[0][j].set_ylabel("decoded tokens / plain rerun")
        ax[1][j].set_ylabel("recovery LLM wall / plain rerun")
    from matplotlib.patches import Patch
    ax[0][0].legend([Patch(color=col[a]) for a in arms], arms, fontsize=8)
    fig.suptitle("Figure 4 (E5, 7B): M1 replay vs step-level recomputation (off); paired per fault × repeat, "
                 "mean ± sd; hatched = atoms")
    A["fig4"] = {"path": save(fig, "fig4_replay.png"), "points": res, "equality_from_e5": ident}
    return res


def p_star(full):
    """Break-even fault rate of atoms vs free CoT under resync replay.

    Clean-run extra cost = (clean calls) x (atoms - cot cost per live call, plain-rerun arm).
    Saving per fault = recovery cost (cot, resync) - recovery cost (atoms, resync).
    p* = extra / saving; inf when no saving. Bootstrap over cells.
    """
    E5 = e5_pairs(full)
    E6 = [r for r in full if r["spec"]["exp"] == "E6"]
    out = {}
    for w in REAL:
        calls = ms([r["faulty"]["llm_calls"] for r in E6 if r["spec"]["workload"] == w])[0]

        def cells(style, arm):
            return [r for r in E5 if r["spec"]["workload"] == w and r["spec"]["flags"]["style"] == style
                    and r["spec"]["flags"]["replay"] == arm]

        for metric in ["decoded", "wall_s"]:
            def stat(sample):
                (oc, oa, rc, ra) = sample
                per = lambda rows: sum(rec_llm(r)[metric] for r in rows) / max(1, sum(r["recovery"]["llm_calls"]
                                                                                       for r in rows))  # noqa: E731
                extra = calls * (per(oa) - per(oc))
                rcm = st.fmean(rec_llm(r)[metric] for r in rc)
                ram = st.fmean(rec_llm(r)[metric] for r in ra)
                save_ = rcm - ram
                return extra, save_

            sets = (cells("cot", "off"), cells("atoms", "off"), cells("cot", "resync"), cells("atoms", "resync"))
            if not all(sets):
                continue
            extra, save_ = stat(sets)
            ps = (max(0.0, extra) / save_) if save_ > 0 else math.inf
            bs = []
            for _ in range(1000):
                smp = tuple([RNG.choice(s) for _ in s] for s in sets)
                e, sv = stat(smp)
                bs.append((max(0.0, e) / sv) if sv > 0 else math.inf)
            bs.sort()
            out[f"{w}/{metric}"] = {"clean_calls": calls, "extra_per_clean_run": extra, "saving_per_fault": save_,
                                    "p_star": ps, "ci95": [bs[25], bs[974]], "n_cells": [len(s) for s in sets]}
    return out


def fig5(full, f4):
    E6 = [r for r in full if r["spec"]["exp"] == "E6"]
    E5 = e5_pairs(full)
    ps = p_star(full)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    res = {}
    # (a) acceptance of replay (resync/prefix) by style
    for k, (arm, c) in enumerate([("prefix", PAL["e"]), ("resync", PAL["a"])]):
        for si, style in enumerate(["cot", "atoms"]):
            mu = [f4[f"{w}/{style}/{arm}"]["token_acceptance"][0] for w in REAL]
            sd = [f4[f"{w}/{style}/{arm}"]["token_acceptance"][1] for w in REAL]
            ax[0].bar([i + (2 * k + si - 1.5) * 0.2 for i in range(4)], mu, 0.2, yerr=sd, capsize=2, color=c,
                      hatch="//" if style == "atoms" else None, edgecolor="white")
    ax[0].set_xticks(range(4), REAL)
    ax[0].set_ylabel("accepted / output tokens")
    ax[0].set_title("(a) acceptance (hatch = atoms)")
    # (b) structure cost per live call (plain-rerun arm)
    for si, style in enumerate(["cot", "atoms"]):
        mu, sd = [], []
        for w in REAL:
            x = [rec_llm(r)["decoded"] / r["recovery"]["llm_calls"] for r in E5 if r["spec"]["workload"] == w
                 and r["spec"]["flags"]["style"] == style and r["spec"]["flags"]["replay"] == "off"
                 and r["recovery"]["llm_calls"]]
            m = ms(x)
            mu.append(m[0])
            sd.append(m[1])
            res[f"{w}/{style}/decoded_per_call_plain"] = [m[0], m[1], m[2]]
        ax[1].bar([i + (si - 0.5) * 0.35 for i in range(4)], mu, 0.35, yerr=sd, capsize=2,
                  color=PAL["k"] if style == "cot" else PAL["c"], label=style)
    ax[1].set_xticks(range(4), REAL)
    ax[1].set_ylabel("decoded tokens per live call (plain rerun)")
    ax[1].set_title("(b) decode cost per live call")
    ax[1].legend(fontsize=8)
    # (c) p*
    for mi, (metric, c) in enumerate([("decoded", PAL["a"]), ("wall_s", PAL["b"])]):
        vals, lo, hi = [], [], []
        for w in REAL:
            d = ps.get(f"{w}/{metric}")
            p = min(1.0, d["p_star"]) if d else 0
            vals.append(p)
            lo.append(max(0, p - min(1.0, d["ci95"][0])) if d else 0)
            hi.append(max(0, min(1.0, d["ci95"][1]) - p) if d else 0)
        ax[2].bar([i + (mi - 0.5) * 0.35 for i in range(4)], vals, 0.35, yerr=[lo, hi], capsize=2, color=c,
                  label=f"p* by {metric}")
    ax[2].set_xticks(range(4), REAL)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("p* (1.0 = never pays, clipped)")
    ax[2].set_title("(c) p*, atoms vs CoT (resync), 95% CI")
    ax[2].legend(fontsize=8)
    fig.suptitle("Figure 5 (E5 + E6): acceptance, structure cost, break-even p* (7B)")
    A["fig5"] = {"path": save(fig, "fig5_structure_pstar.png"), "p_star": ps, "per_call": res}
    # E6 clean-run caution: cache hits
    cl = {}
    for w in REAL:
        for style in ["cot", "atoms"]:
            rows = [r for r in E6 if r["spec"]["workload"] == w and r["spec"]["flags"]["style"] == style]
            hits = sum(r["faulty"]["llm"]["cache_hits"] for r in rows)
            calls = sum(r["faulty"]["llm"]["calls"] for r in rows)
            cl[f"{w}/{style}"] = {"cache_hit_calls": hits, "calls": calls,
                                  "nominal_tokens": ms([r["faulty"]["tokens"] for r in rows])[:2],
                                  "success": ms([1.0 if r["faulty"]["ok"] else 0.0 for r in rows])[0]}
    A["e6_clean"] = cl


# ---------------------------------------------------------------- Table 1 (E8) and noise floor


def table1(full, p1):
    E8 = [r for r in full if r["spec"]["exp"] == "E8"]
    meths = ["restart", "rollback", "selective", "selective+M1+M2", "selective+M1+M2+M3/M4"]
    lines = ["| workload | method | n | ok_recovered | unauthorized | steps rerun | steps reused | irrev redone | "
             "compensated | leaked | recovery wall s | decoded tok | prefill tok | audit tok (recovery) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for w in REAL:
        for m in meths:
            rows = [r for r in E8 if r["spec"]["workload"] == w and r["spec"]["label"] == m]
            if not rows:
                continue
            g = lambda f: ms([f(r) for r in rows])  # noqa: E731
            d = {
                "n": len(rows),
                "ok_recovered": g(lambda r: float(r["recovery"]["ok_recovered"]) if r["recovery"]["ok_recovered"]
                                  is not None else None),
                "unauthorized": g(lambda r: r.get("unauthorized")),
                "n_rerun": g(lambda r: r["recovery"]["n_rerun"]),
                "n_reused": g(lambda r: len(r["recovery"]["reused"])),
                "irrev_redone": g(lambda r: r["recovery"]["irreversible_redone"]),
                "compensated": g(lambda r: r["recovery"]["compensated"]),
                "leaked": g(lambda r: r["recovery"]["effects"]["leaked"]),
                "wall": g(lambda r: r["recovery"]["llm"]["wall_s"]),
                "decoded": g(lambda r: r["recovery"]["llm"]["decoded"]),
                "prefill": g(lambda r: r["recovery"]["llm"]["prefill"]),
                "audit_tok": g(lambda r: r["recovery"].get("audit", {}).get("audit_tokens", 0)),
                "cache_hits_in_recovery": sum(r["recovery"]["llm"]["cache_hits"] for r in rows),
            }
            res[f"{w}/{m}"] = d
            lines.append(f"| {w} | {m} | {d['n']} | {fmt(d['ok_recovered'])} | {fmt(d['unauthorized'])} | "
                         f"{fmt(d['n_rerun'], 1)} | {fmt(d['n_reused'], 1)} | {fmt(d['irrev_redone'], 1)} | "
                         f"{fmt(d['compensated'], 1)} | {fmt(d['leaked'], 1)} | {fmt(d['wall'], 1)} | "
                         f"{fmt(d['decoded'], 0)} | {fmt(d['prefill'], 0)} | {fmt(d['audit_tok'], 0)} |")
    # per fault, ok_recovered (for non-propagating checks)
    pf = {}
    for k, v in by(E8, S("workload"), S("label"), S("fault")).items():
        pf["/".join(k)] = ms([float(r["recovery"]["ok_recovered"]) for r in v if r["recovery"]["ok_recovered"] is not None])[:2]
    # noise floor: clean-vs-clean spread over repeats
    nf = {}
    for w in REAL:
        c1 = [r for r in p1 if r["spec"]["exp"] == "E1" and r["spec"]["fault"] == "clean" and r["spec"]["model"] == "7b"
              and r["spec"]["workload"] == w]
        c6 = [r for r in full if r["spec"]["exp"] == "E6" and r["spec"]["workload"] == w]
        nf[w] = {"E1_clean_success": ms([float(r["faulty"]["ok"]) for r in c1])[0], "E1_n": len(c1),
                 "E1_wall_s": ms([r["faulty"]["wall_s"] for r in c1])[:2],
                 "E1_wall_uncached_only": ms([r["faulty"]["wall_s"] for r in c1 if r["faulty"]["llm"].get("cache_hits", 0) == 0])[:3],
                 "E6_success": ms([float(r["faulty"]["ok"]) for r in c6])[0], "E6_n": len(c6)}
    # deterministic replay of an identical cell? same prompt twice gives the cache, so noise floor of
    # decoding itself is 0 by construction (greedy) -- recorded in results.md
    A["table1"] = {"rows": res, "per_fault_ok_recovered": pf, "noise_floor": nf}
    md = ["# Table 1 (E8, 7B, runs/full/cells.jsonl): mean ± sd over faults × repeats pooled", "",
          "Recovery wall = LLM wall of the recovery phase (live decode, no cache hits). ok_recovered, not ok_final.",
          "", *lines, ""]
    (OUT / "table1.md").write_text("\n".join(md))
    return res


# ---------------------------------------------------------------- Table 2 and Figure 6 (E7)


def table2(full):
    E7 = [r for r in full if r["spec"]["exp"] == "E7"]
    pols = ["always", "uniform", "m3", "m3+cache", "m3+m4"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    lines = ["| workload | audit | faulty runs | detected (Wilson 95%) | clean runs | FP flags/clean run | "
             "audit tok/faulty run | audit tok/clean run | recovery re-audits | recovery audit tok | "
             "recovery reused verdicts | residual flags after recovery |", "|" + "---|" * 12]
    res = {}
    irr_equal = True
    for j, w in enumerate(REAL):
        for pi, p in enumerate(pols):
            fa = [r for r in E7 if r["spec"]["workload"] == w and r["spec"]["flags"]["audit"] == p
                  and r["spec"]["fault"] != "clean"]
            cl = [r for r in E7 if r["spec"]["workload"] == w and r["spec"]["flags"]["audit"] == p
                  and r["spec"]["fault"] == "clean"]
            k = sum(1 for r in fa if r.get("detected"))
            pr, lo, hi = wilson(k, len(fa))
            fatok = ms([r["faulty"].get("audit", {}).get("audit_tokens", 0) for r in fa])
            fatok_c = ms([r["faulty"].get("audit", {}).get("audit_tokens", 0) for r in cl])
            fp = ms([r.get("fp_flags", 0) for r in cl])
            fpf = ms([r.get("fp_flags", 0) for r in fa])
            ra = ms([r["recovery"].get("audit", {}).get("audits_run", 0) for r in fa if "recovery" in r])
            rt = ms([r["recovery"].get("audit", {}).get("audit_tokens", 0) for r in fa if "recovery" in r])
            rr = ms([r["recovery"].get("audit", {}).get("reused", 0) for r in fa if "recovery" in r])
            rf = ms([len(r["recovery"].get("audit", {}).get("flagged", [])) for r in fa if "recovery" in r])
            by_fault = {f: wilson(sum(1 for r in fa if r["spec"]["fault"] == f and r.get("detected")),
                                  sum(1 for r in fa if r["spec"]["fault"] == f))[0]
                        for f in sorted({r["spec"]["fault"] for r in fa})}
            res[f"{w}/{p}"] = {"n_faulty": len(fa), "detected": [pr, lo, hi], "n_clean": len(cl),
                               "fp_per_clean": fp[:2], "fp_per_faulty": fpf[:2],
                               "audit_tok_faulty": fatok[:2], "audit_tok_clean": fatok_c[:2],
                               "recovery_audits": ra[:2], "recovery_tok": rt[:2], "recovery_reused": rr[:2],
                               "residual_flags": rf[:2], "detect_by_fault": by_fault}
            lines.append(f"| {w} | {p} | {len(fa)} | {pr:.2f} [{lo:.2f},{hi:.2f}] | {len(cl)} | {fmt(fp)} | "
                         f"{fmt(fatok, 0)} | {fmt(fatok_c, 0)} | {fmt(ra, 1)} | {fmt(rt, 0)} | {fmt(rr, 1)} | {fmt(rf, 2)} |")
            ax[j].errorbar(fatok[0], 1 - pr, xerr=fatok[1], yerr=[[max(0, (1 - pr) - (1 - hi))], [max(0, (1 - lo) - (1 - pr))]],
                           fmt="o", capsize=3, label=p, color=[PAL["k"], PAL["b"], PAL["e"], PAL["c"], PAL["a"]][pi])
        ax[j].set_title(w)
        ax[j].set_xlabel("audit tokens per faulty run (mean ± sd)")
    ax[0].set_ylabel("missed faults (1 - detection), Wilson 95%")
    ax[0].legend(fontsize=8)
    # audits do not gate effects: irreversible calls identical across audit policies?
    g = by(E7, S("workload"), S("label"), S("fault"), S("repeat"))
    for k, v in g.items():
        if len({r["irreversible_calls_faulty"] for r in v}) > 1:
            irr_equal = False
    fig.suptitle("Figure 6 (E7, 7B): missed faults vs audit tokens per audit policy")
    A["table2"] = {"rows": res, "irreversible_calls_identical_across_policies": irr_equal}
    A["fig6"] = {"path": save(fig, "fig6_audit_placement.png")}
    (OUT / "table2.md").write_text("# Table 2 (E7, 7B, runs/full/cells.jsonl)\n\nDetection = flag inside cone(f) ∪ {f}; "
                                   "faulty-run audits only count toward detection.\n\n" + "\n".join(lines) + "\n")


# ---------------------------------------------------------------- anomalies


def anomalies(p1, p1b, full):
    out = {}
    w2 = [r for r in p1 if r["spec"]["workload"] == "w2" and "unauthorized" in r]
    out["w2_unauthorized_equals_total_irreversible"] = {
        "cells": len(w2), "equal": sum(1 for r in w2 if r["unauthorized"] == r["irreversible_calls_total"])}
    for name, rows in [("full", full), ("baselines", p1b)]:
        un = [r for r in rows if r.get("unauthorized")]
        out[f"{name}_unauthorized_nonzero"] = [
            {"w": r["spec"]["workload"], "m": r["spec"]["label"], "f": r["spec"]["fault"], "u": r["unauthorized"]}
            for r in un[:40]]
    # W3 restart vs selective ok_recovered (S7 flag)
    w3 = {}
    for name, rows in [("full/E8", [r for r in full if r["spec"]["exp"] == "E8"]), ("baselines", p1b)]:
        for k, v in by([r for r in rows if r["spec"]["workload"] == "w3"], S("label"), S("fault")).items():
            w3[f"{name}/{'/'.join(k)}"] = [ms([float(r["recovery"]["ok_recovered"]) for r in v])[0], len(v)]
    out["w3_ok_recovered"] = w3
    # w3 restart: does the final world still hold the fault?
    out["clean_failures_e6"] = [(r["spec"]["workload"], r["spec"]["flags"]["style"], r["spec"]["repeat"])
                                for r in full if r["spec"]["exp"] == "E6" and not r["faulty"]["ok"]]
    out["errors"] = sum(1 for rows in (p1, p1b, full) for r in rows if "error" in r)
    # E8 recoveries that hit the cache
    out["recovery_cache_hits"] = sum(r["recovery"]["llm"]["cache_hits"] for r in full if "recovery" in r and r["spec"]["model"] == "7b")
    # peak rss
    out["peak_rss_mb_full"] = max(r["peak_rss_mb"] for r in full)
    # w1 inject / wrong faults ok_recovered by method (E8)
    ex = {}
    for k, v in by([r for r in full if r["spec"]["exp"] == "E8"], S("workload"), S("fault"), S("label")).items():
        ex["/".join(k)] = [ms([float(r["recovery"]["ok_recovered"]) for r in v])[0], len(v)]
    out["e8_ok_recovered_by_cell"] = ex
    # ok_final vs ok_recovered disagreement
    out["e8_ok_final_vs_recovered_disagree"] = sum(
        1 for r in full if r["spec"]["exp"] == "E8" and r["recovery"]["ok_final"] != r["recovery"]["ok_recovered"])
    # faults not propagating (outcome unchanged in E8 faulty run)
    npr = by([r for r in full if r["spec"]["exp"] == "E8" and r["spec"]["label"] == "restart"], S("workload"), S("fault"))
    out["e8_faulty_run_outcome_changed"] = {"/".join(k): ms([float(r["propagation"]["outcome_changed"]) for r in v])[0]
                                            for k, v in npr.items()}
    # replay: draft acceptance zero rows, resync slower than off in wall
    A["anomalies"] = out


def main():
    p1, p1b, full = load(P1), load(P1B), load(FULL)
    print(f"cells: phase1={len(p1)} baselines={len(p1b)} full={len(full)}")
    OUT.mkdir(parents=True, exist_ok=True)
    fig1(p1)
    fig2(p1)
    fig3(p1)
    f4 = fig4(full)
    fig5(full, f4)
    table1(full, p1)
    table2(full)
    anomalies(p1, p1b, full)
    (OUT / "analysis.json").write_text(json.dumps(A, indent=1, default=lambda o: None if (isinstance(o, float) and math.isinf(o)) else str(o)))
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
