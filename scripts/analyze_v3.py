"""v3 tables and figures: the always-verify arm, the timeout faults, the two post-hoc fixes, the fault stage.

    python scripts/analyze_v3.py [V3_DIR] [--tex PATH]
    python scripts/analyze_v3.py runs/v3_prep --stage-only      # T7 from runs/v2 alone (today)

V3_DIR   folder with cells.jsonl (default runs/v3; runs/v3_dry for the scripted dry run)
inputs   runs/v2/cells.jsonl (E8v2, E7v2), runs/v2_fix/cells.jsonl (E8v2 W3 after the harness fixes), V3_DIR/cells.jsonl
outputs  V3_DIR/figures/{T1v3,T4_faults,T5_stale_code,T6_gate,T7_fault_stage}.md, F9_correctness.png/.json,
         analysis_v3.json, and the report macros (default report/generated/v3.tex)

Scripted rows (a dry run) are analysed in shape only: the report macros then go to V3_DIR/figures/v3_draft.tex,
never to report/generated/, and every number is marked DRY. Pre-registered folders (runs/v2, runs/v2_fix) are read only.
analyze_v2.py and its outputs are not touched. Every v3 cell is matched to its v2 cell on (workload, fault, repeat);
a missing partner raises.

Post-hoc fixes (named so everywhere): revert=on (E8v3r, engine fix for W3 stale_code) and release=on (E8v3g, gate
release rule). Their rows are shown next to the unfixed rows, never instead of them.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v2 as A2  # noqa: E402  (helpers only; its main() is not run)
from analyze import PAL, ms, wilson  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = A2.ROOT
V2_CELLS, V2_FIX = A2.CELLS, A2.W3_FIX
GENERATED = ROOT / "report" / "generated" / "v3.tex"
sp, load, jdump = A2.sp, A2.load, A2.jdump

ARMS = ["restart", "rollback", "always-verify", "selective", "selective+M1", "selective+M1+M3t/M4+gate"]
SHORT = {"restart": "restart", "rollback": "rollback", "always-verify": "always-verify", "selective": "selective",
         "selective+M1": "sel.+replay", "selective+M1+M3t/M4+gate": "full stack"}
WORK = ["w1", "w3", "w4.ipi", "w4.byz"]
WNAME = {"w1": "W1", "w3": "W3", "w4.ipi": "W4.ipi", "w4.byz": "W4.byz"}
TEXW = {"w1": "Wone", "w3": "Wthree", "w4.ipi": "Wfouripi", "w4.byz": "Wfourbyz"}
RES: dict = {}
TEX: dict[str, str] = {}


class Mismatch(SystemExit):
    pass


def cell(r):
    return (sp(r)["workload"], sp(r)["fault"], sp(r)["repeat"])


def index(rows, *, label=None):
    d = {}
    for r in rows:
        if "error" in r or (label and sp(r)["label"] != label):
            continue
        k = (sp(r)["workload"], sp(r)["fault"], sp(r)["repeat"], sp(r)["label"])
        if k in d:
            raise Mismatch(f"duplicate cell {k}")
        d[k] = r
    return d


def audit_tok(r):
    return (r["recovery"].get("audit") or {}).get("audit_tokens", 0)


def metrics(rows):
    """T1's metrics for a set of E8 rows (recovery phase), plus total inference tokens = decoded + prefill + audit."""
    n = len(rows)
    ok = sum(bool(r["recovery"]["ok_recovered"]) for r in rows)
    llm = [A2.rec_llm(r) for r in rows]
    rv = [x for r in rows for x in [r["recovery"].get("reuse_check")] if x]
    return {
        "n": n, "ok_recovered": [ok, n], "ok_wilson": list(wilson(ok, n)),
        "values_match": [sum(1 for r in rows if r["recovery"]["values_match"]), n],
        "n_rerun": ms([r["recovery"]["n_rerun"] for r in rows])[:2],
        "recovery_wall_s": ms([r["recovery"]["wall_s"] for r in rows])[:2],
        "decoded": ms([x.get("decoded", 0) for x in llm])[:2], "prefill": ms([x.get("prefill", 0) for x in llm])[:2],
        "audit_tokens": ms([audit_tok(r) for r in rows])[0],
        "inference_tokens": ms([x.get("decoded", 0) + x.get("prefill", 0) + audit_tok(r) for x, r in zip(llm, rows)])[0],
        "duplicates": ms([r["duplicates"] for r in rows])[0],
        "unauthorized_cells": sum(1 for r in rows if r.get("unauthorized")),
        "leaked": ms([r["recovery"]["effects"]["leaked"] for r in rows])[0],
        "reuse_valid": [sum(x["equal"] for x in rv), sum(x["sampled"] for x in rv)],
    }


def f1(x, d=1):
    return "n/a" if x is None else f"{x:.{d}f}"


def frac(d):
    return f"{d[0]}/{d[1]}"


# ------------------------------------------------------------------------------ inputs and matching
def inputs(v3dir: Path):
    v2 = load(V2_CELLS)
    fix = load(V2_FIX)
    v3 = load(v3dir / "cells.jsonl")
    errs = [r for r in v3 if "error" in r]
    if errs:
        raise Mismatch(f"{len(errs)} error rows in {v3dir}/cells.jsonl")
    e8 = [r for r in v2 if sp(r)["exp"] == "E8v2" and sp(r)["workload"] != "w3"] + \
         [r for r in fix if "error" not in r and sp(r)["exp"] == "E8v2"]
    e7 = [r for r in v2 if sp(r)["exp"] == "E7v2"]
    by = defaultdict(list)
    for r in v3:
        by[sp(r)["exp"]].append(r)
    return e8, e7, v2, by, v3


def assert_matches(a, b, what):
    """every cell of `a` has a partner in `b` (dict keys (workload, fault, repeat))"""
    missing = sorted(set(a) - set(b))
    if missing:
        raise Mismatch(f"{what}: {len(missing)} cell(s) without a partner, e.g. {missing[0]}")


# ------------------------------------------------------------------------------ T1v3
def t1v3(e8, e83):
    base = {m: {(w, f, rp): r for (w, f, rp, lb), r in index(e8).items() if lb == m}
            for m in ARMS if m != "always-verify"}
    av = {(w, f, rp): r for (w, f, rp, lb), r in index(e83).items() if lb == "always-verify"}
    assert_matches(av, base["restart"], "E8v3 vs E8v2 restart")
    for m, d in base.items():                                # same six fault cells in every arm
        for k in av:
            if k not in d:
                raise Mismatch(f"E8v2 arm {m} lacks {k}")
    for k in base["restart"]:
        if k[0] in WORK and k not in av:
            raise Mismatch(f"E8v3 lacks the always-verify cell {k}")
    tab = {}
    for w in WORK:
        for m in ARMS:
            src = av if m == "always-verify" else base[m]
            rows = [r for k, r in src.items() if k[0] == w]
            tab[f"{w}/{m}"] = metrics(rows)
    L = ["# T1v3 end-to-end with the always-verify arm (E8v2 + E8v3, atoms, 7B). W3 baseline rows: rerun after fix (runs/v2_fix)",
         "", "Recovery wall = whole recovery phase (always-verify: the audits of the rerun steps are inside it). "
         "Inference tokens = decoded + prefill of the recovery + audit tokens. always-verify = selective, effects=diff, audit=always.", "",
         "| workload | method | n | ok_recovered | values_match | steps rerun | rec. wall s | decoded | inference tok | dup | unauth cells | leaked | audit tok |",
         "|" + "---|" * 13]
    for w in WORK:
        for m in ARMS:
            d = tab[f"{w}/{m}"]
            L.append(f"| {w} | {m} | {d['n']} | {frac(d['ok_recovered'])} | {frac(d['values_match'])} | {f1(d['n_rerun'][0])} | "
                     f"{f1(d['recovery_wall_s'][0])} ± {f1(d['recovery_wall_s'][1])} | {f1(d['decoded'][0], 0)} | {f1(d['inference_tokens'], 0)} | "
                     f"{d['duplicates']:.2f} | {d['unauthorized_cells']} | {d['leaked']:.1f} | {f1(d['audit_tokens'], 0)} |")
    write("T1v3_end_to_end.md", L)
    for w in WORK:
        d = tab[f"{w}/always-verify"]
        TEX[f"Av{TEXW[w]}"] = (f"always-verify & {f1(d['n_rerun'][0])} & {f1(d['recovery_wall_s'][0])} & {f1(d['decoded'][0], 0)} & "
                               f"{frac(d['ok_recovered'])} \\\\")
        TEX[f"AvOk{TEXW[w]}"] = frac(d["ok_recovered"])
        TEX[f"AvWall{TEXW[w]}"] = f1(d["recovery_wall_s"][0])
        TEX[f"AvAudTok{TEXW[w]}"] = f1(d["audit_tokens"], 0)
        s = tab[f"{w}/selective+M1"]
        TEX[f"SelMoneWall{TEXW[w]}"] = f1(s["recovery_wall_s"][0])
        TEX[f"AvOverMone{TEXW[w]}"] = f"{d['recovery_wall_s'][0] / s['recovery_wall_s'][0]:.1f}"
    rowname = {"restart": "Restart", "rollback": "Rollback", "always-verify": "Av", "selective": "Sel", "selective+M1": "SelM",
               "selective+M1+M3t/M4+gate": "Gate"}
    for w in WORK:                                    # one table row per (workload, arm): steps rerun, decoded tokens, inference tokens
        for m in ARMS:
            d = tab[f"{w}/{m}"]
            TEX[f"Row{TEXW[w]}{rowname[m]}"] = (f"{SHORT[m]} & {f1(d['n_rerun'][0])} & {f1(d['decoded'][0], 0)} & "
                                                f"{f1(d['inference_tokens'], 0)} \\\\")
    toks = [(tab[f"{w}/always-verify"]["audit_tokens"], w) for w in WORK]
    lo, hi = min(toks), max(toks)
    TEX["AvAudTokMin"], TEX["AvAudTokMax"] = f1(lo[0], 0), f1(hi[0], 0)
    TEX["AvAudTokMinWl"], TEX["AvAudTokMaxWl"] = WNAME[lo[1]], WNAME[hi[1]]
    av_all = [r for r in av.values()]
    TEX["AvOkAll"] = frac([sum(bool(r["recovery"]["ok_recovered"]) for r in av_all), len(av_all)])
    TEX["AvCells"] = str(len(av_all))
    RES["T1v3"] = tab
    return tab


# ------------------------------------------------------------------------------ T4: fault kinds
def t4(e83f):
    idx = index(e83f)
    L = ["# T4 fault kinds not in E8v2 (E8v3f, atoms, 7B): timeout_team (W1) and timeout_item (W4.ipi), six arms, 5 repeats", "",
         "| workload | fault | method | n | ok_recovered | values_match | steps rerun | rec. wall s | decoded | dup | unauth cells | leaked | audit tok | held |",
         "|" + "---|" * 14]
    out = {}
    for w, f in (("w1", "timeout_team"), ("w4.ipi", "timeout_item")):
        for m in ARMS:
            rows = [r for (w2, f2, rp, lb), r in idx.items() if (w2, f2, lb) == (w, f, m)]
            if not rows:
                raise Mismatch(f"E8v3f lacks the {w}/{f}/{m} cells")
            d = metrics(rows)
            d["held"] = sum(len(r["recovery"].get("held", [])) for r in rows)
            out[f"{w}/{f}/{m}"] = d
            L.append(f"| {w} | {f} | {m} | {d['n']} | {frac(d['ok_recovered'])} | {frac(d['values_match'])} | {f1(d['n_rerun'][0])} | "
                     f"{f1(d['recovery_wall_s'][0])} ± {f1(d['recovery_wall_s'][1])} | {f1(d['decoded'][0], 0)} | {d['duplicates']:.2f} | "
                     f"{d['unauthorized_cells']} | {d['leaked']:.1f} | {f1(d['audit_tokens'], 0)} | {d['held']} |")
    write("T4_faults.md", L)
    for w, f, tag in (("w1", "timeout_team", "Team"), ("w4.ipi", "timeout_item", "Item")):
        for m, mt in (("restart", "Restart"), ("rollback", "Rollback"), ("selective", "Sel"), ("selective+M1", "SelMone"),
                      ("always-verify", "Av"), ("selective+M1+M3t/M4+gate", "Gate")):
            d = out[f"{w}/{f}/{m}"]
            TEX[f"T{tag}{mt}Ok"] = frac(d["ok_recovered"])
            TEX[f"T{tag}{mt}Rerun"] = f1(d["n_rerun"][0])
            TEX[f"T{tag}{mt}Wall"] = f1(d["recovery_wall_s"][0])
    RES["T4"] = out
    return out


# ------------------------------------------------------------------------------ T5: stale_code
def t5(fix, e83r):
    old = {(sp(r)["label"], sp(r)["repeat"]): r for r in fix if "error" not in r and sp(r)["fault"] == "stale_code"}
    new = {(sp(r)["label"], sp(r)["repeat"]): r for r in e83r}
    assert_matches(new, old, "E8v3r vs v2_fix")
    arms = ["selective", "selective+M1", "selective+M1+M3t/M4+gate"]
    L = ["# T5 W3 stale_code: before (runs/v2_fix) and after the early-revert fix (E8v3r, revert=on, POST-HOC), atoms, 7B", "",
         "| method | fix | n | ok_recovered | values_match | steps rerun | rec. wall s | failing repeats |", "|" + "---|" * 8]
    out = {}
    rs = [r for (lb, rp), r in old.items() if lb == "restart"]
    if rs:
        d = metrics(rs)
        out["restart"] = d
        L.append(f"| restart | (reference) | {d['n']} | {frac(d['ok_recovered'])} | {frac(d['values_match'])} | {f1(d['n_rerun'][0])} | "
                 f"{f1(d['recovery_wall_s'][0])} | {[r_[1] for r_, r in old.items() if r_[0] == 'restart' and not r['recovery']['ok_recovered']]} |")
    for m in arms:
        for name, src in (("none (v2_fix)", old), ("revert=on (post-hoc)", new)):
            rows = [r for (lb, rp), r in src.items() if lb == m]
            d = metrics(rows)
            fails = sorted(rp for (lb, rp), r in src.items() if lb == m and not r["recovery"]["ok_recovered"])
            d["failing_repeats"] = fails
            out[f"{m}/{'revert' if 'revert' in name else 'v2_fix'}"] = d
            L.append(f"| {m} | {name} | {d['n']} | {frac(d['ok_recovered'])} | {frac(d['values_match'])} | {f1(d['n_rerun'][0])} | "
                     f"{f1(d['recovery_wall_s'][0])} | {fails} |")
    write("T5_stale_code.md", L)
    tot = lambda tag: [sum(out[f"{m}/{tag}"]["ok_recovered"][0] for m in arms), sum(out[f"{m}/{tag}"]["ok_recovered"][1] for m in arms)]  # noqa: E731
    TEX["StaleBefore"], TEX["StaleAfter"] = frac(tot("v2_fix")), frac(tot("revert"))
    TEX["StaleSelBefore"], TEX["StaleSelAfter"] = frac(out["selective/v2_fix"]["ok_recovered"]), frac(out["selective/revert"]["ok_recovered"])
    TEX["StaleRestart"] = frac(out["restart"]["ok_recovered"]) if "restart" in out else "n/a"
    RES["T5"] = out
    return out


# ------------------------------------------------------------------------------ T6: gate release rule
def gate_cell(r):
    g = (r.get("recovery") or {}).get("gate", {})
    return {"ok": bool(r["recovery"]["ok_recovered"]) if "recovery" in r else None,
            "held_final": g.get("held", None), "released": (r.get("recovery") or {}).get("effects", {}).get("released"),
            "unauthorized": r.get("unauthorized"), "flags_faulty": (r["faulty"].get("audit") or {}).get("flagged", []),
            "residual_flags": g.get("residual_flags"), "wall_s": r["recovery"]["wall_s"] if "recovery" in r else None}


def t6(e8, e7, e83g):
    v2g = {rp: r for (w, f, rp, lb), r in index(e8).items() if (w, f, lb) == ("w4.ipi", "inject_item", "selective+M1+M3t/M4+gate")}
    new = {sp(r)["repeat"]: r for r in e83g if sp(r)["fault"] == "inject_item"}
    clean_new = {sp(r)["repeat"]: r for r in e83g if sp(r)["fault"] == "clean"}
    clean_old = {sp(r)["repeat"]: r for r in e7 if (sp(r)["workload"], sp(r)["fault"], sp(r)["flags"].get("audit")) == ("w4.ipi", "clean", "m3t+m4+gate")}
    assert_matches(new, v2g, "E8v3g vs E8v2 gate arm")
    assert_matches(clean_new, clean_old, "E8v3g clean vs E7v2 clean")
    L = ["# T6 gate release rule on W4.ipi (full stack = M1 + m3t/m4 + gate; release=on is POST-HOC)", "",
         "Fault cells (inject_item): E8v2 gate arm (release off) against E8v3g (release=on). `same faulty flags` says whether the faulty run of "
         "the pair raised the same flags (a clean A/B needs it).", "",
         "| repeat | release | ok_recovered | held after recovery | released at commit | unauthorized | residual flags (reused steps) | same faulty flags |",
         "|" + "---|" * 8]
    rows = []
    for rp in sorted(new):
        a, b = gate_cell(v2g[rp]), gate_cell(new[rp])
        same = a["flags_faulty"] == b["flags_faulty"]
        for tag, d in (("off", a), ("on", b)):
            L.append(f"| {rp} | {tag} | {d['ok']} | {d['held_final']} | {d['released']} | {d['unauthorized']} | {d['residual_flags']} | {same if tag == 'on' else ''} |")
        rows.append({"repeat": rp, "off": a, "on": b, "same_faulty_flags": same})
    n = len(rows)
    okA, okB = sum(x["off"]["ok"] for x in rows), sum(x["on"]["ok"] for x in rows)
    heldA, heldB = sum(bool(x["off"]["held_final"]) for x in rows), sum(bool(x["on"]["held_final"]) for x in rows)
    unauth = sum(1 for x in rows if x["on"]["unauthorized"])
    rst = {rp: r for (w, f, rp, lb), r in index(e8).items() if (w, f, lb) == ("w4.ipi", "inject_item", "restart")}
    okR = sum(bool(r["recovery"]["ok_recovered"]) for r in rst.values())
    L += ["", f"ok_recovered: release off {okA}/{n}, release on {okB}/{n}, restart {okR}/{len(rst)}. Cells still holding after recovery: off {heldA}, on {heldB}. "
          f"Unauthorized actions with the rule: {unauth} cells.", "",
          "Clean runs (no fault, no recovery, so the rule cannot act): E7v2 clean cell against E8v3g clean cell, same repeat.", "",
          "| repeat | E7v2 held | E8v3g held | flags (E7v2) | flags (E8v3g) |", "|---|---|---|---|---|"]
    clean = []
    for rp in sorted(clean_new):
        a, b = clean_old[rp]["faulty"]["gate"]["held"], clean_new[rp]["faulty"]["gate"]["held"]
        fa, fb = clean_old[rp]["faulty"]["audit"]["flagged"], clean_new[rp]["faulty"]["audit"]["flagged"]
        L.append(f"| {rp} | {a} | {b} | {fa} | {fb} |")
        clean.append({"repeat": rp, "held_e7v2": a, "held_e8v3g": b, "flags_e7v2": fa, "flags_e8v3g": fb})
    hA, hB = sum(bool(c["held_e7v2"]) for c in clean), sum(bool(c["held_e8v3g"]) for c in clean)
    L.append(f"\nClean runs that hold a post: E7v2 {hA}/{len(clean)}, E8v3g {hB}/{len(clean)}.")
    write("T6_gate.md", L)
    TEX.update({"GateOkOff": f"{okA}/{n}", "GateOkOn": f"{okB}/{n}", "GateOkRestart": f"{okR}/{len(rst)}",
                "GateHeldOff": str(heldA), "GateHeldOn": str(heldB), "GateUnauth": str(unauth), "GateN": str(n),
                "GateCleanOff": f"{hA}/{len(clean)}", "GateCleanOn": f"{hB}/{len(clean)}",
                "GateResidual": str(sum(len(x["on"]["residual_flags"] or []) for x in rows))})
    RES["T6"] = {"fault_cells": rows, "clean_cells": clean, "ok_off": okA, "ok_on": okB, "ok_restart": okR}
    return RES["T6"]


# ------------------------------------------------------------------------------ T7 / P6: fault stage
def stage_table(e8, e83f):
    """Reruns and recovery wall by the faulted step, from E8v2 cells (+E8v3f timeout cells when present). No new run."""
    from agent_rewind.bench import build_case
    pos = {}
    for w in WORK:
        c = build_case(w, 0, {})
        names = [s.name for s in c.workflow.steps]
        for f, ft in c.faults.items():
            pos[(w, f)] = (ft.step, names.index(ft.step), len(names))
    rows = list(e8) + list(e83f)
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s = sp(r)
        if s["label"] in ("restart", "selective", "selective+M1"):
            cells[(s["workload"], s["fault"])][s["label"]].append(r)
    L = ["# T7 fault stage from existing cells (no new run): reruns and recovery wall by the faulted step",
         "", "Stage = position of the faulted step in the workflow (0 = first). Live workloads have 1-2 fixed fault positions per workload; "
         "the position sweep 0.1-0.9 exists only on scripted W2 (E2v2). Rows from E8v2 (W3: runs/v2_fix) and, when present, E8v3f.", "",
         "| workload | fault | faulted step | stage | steps | restart rerun | selective rerun | restart wall s | sel.+M1 wall s | n |", "|" + "---|" * 10]
    out = {}
    for (w, f) in sorted(cells, key=lambda k: (WORK.index(k[0]), pos[k][1])):
        c = cells[(w, f)]
        if not all(c.get(m) for m in ("restart", "selective", "selective+M1")):
            continue
        step, i, n = pos[(w, f)]
        a, b, d = (ms([r["recovery"]["n_rerun"] for r in c["restart"]])[0], ms([r["recovery"]["n_rerun"] for r in c["selective"]])[0],
                   None)
        wr, wm = ms([r["recovery"]["wall_s"] for r in c["restart"]])[0], ms([r["recovery"]["wall_s"] for r in c["selective+M1"]])[0]
        out[f"{w}/{f}"] = {"step": step, "stage": i, "steps": n, "rerun_restart": a, "rerun_selective": b, "wall_restart": wr,
                           "wall_selective_m1": wm, "n": len(c["restart"])}
        L.append(f"| {w} | {f} | {step} | {i} | {n} | {f1(a)} | {f1(b)} | {f1(wr)} | {f1(wm)} | {len(c['restart'])} |")
    write("T7_fault_stage.md", L)
    tag = {"w1/wrong_capacity": "WoneCap", "w1/inject_agenda": "WoneAgenda", "w1/timeout_team": "WoneTeam",
           "w3/wrong_issue": "WthreeIssue", "w3/stale_code": "WthreeCode", "w4.ipi/inject_item": "FouripiItem",
           "w4.ipi/timeout_item": "FouripiTimeout", "w4.byz/byz_factcheck": "FourbyzFact"}
    for k, d in out.items():
        t = tag.get(k)
        if t:
            TEX.update({f"Stage{t}Pos": str(d["stage"]), f"Stage{t}Steps": str(d["steps"]), f"Stage{t}Rerun": f1(d["rerun_selective"]),
                        f"Stage{t}Restart": f1(d["rerun_restart"]), f"Stage{t}WallR": f1(d["wall_restart"]), f"Stage{t}WallM": f1(d["wall_selective_m1"])})
    RES["T7"] = out
    return out


# ------------------------------------------------------------------------------ figure
def f9(t1, t4_, t5_, t6_):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    pal = [PAL["k"], PAL["e"], PAL["d"], PAL["a"], PAL["c"], PAL["b"]]
    for mi, m in enumerate(ARMS):
        for j, w in enumerate(WORK):
            p, lo, hi = t1[f"{w}/{m}"]["ok_wilson"]
            x = j + (mi - 2.5) * 0.14
            ax[0].bar(x, p, 0.14, color=pal[mi], label=SHORT[m] if j == 0 else None)
            ax[0].errorbar(x, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]], color="k", capsize=1, lw=0.7)
    ax[0].set_xticks(range(4), [WNAME[w] for w in WORK])
    ax[0].set_title("ok_recovered, E8v2 + always-verify (Wilson 95%)", fontsize=9)
    ax[0].legend(fontsize=6, ncol=2)
    labels = []
    for k, (w, f) in enumerate((("w1", "timeout_team"), ("w4.ipi", "timeout_item"))):
        for mi, m in enumerate(ARMS):
            p, lo, hi = t4_[f"{w}/{f}/{m}"]["ok_wilson"]
            x = k + (mi - 2.5) * 0.14
            ax[1].bar(x, p, 0.14, color=pal[mi])
            ax[1].errorbar(x, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]], color="k", capsize=1, lw=0.7)
        labels.append(f"{WNAME[w]} {f}")
    ax[1].set_xticks(range(2), labels, fontsize=7)
    ax[1].set_title("ok_recovered, timeout faults (E8v3f)", fontsize=9)
    bars = [("stale_code\nsel. v2_fix", t5_["selective/v2_fix"]["ok_recovered"]), ("stale_code\nsel. revert=on*", t5_["selective/revert"]["ok_recovered"]),
            ("W4.ipi gate\nrelease off", [t6_["ok_off"], len(t6_["fault_cells"])]), ("W4.ipi gate\nrelease on*", [t6_["ok_on"], len(t6_["fault_cells"])]),
            ("W4.ipi\nrestart", [t6_["ok_restart"], len(t6_["fault_cells"])])]
    for i, (lb, d) in enumerate(bars):
        p, lo, hi = wilson(*d)
        ax[2].bar(i, p, 0.6, color=[PAL["k"], PAL["c"], PAL["k"], PAL["c"], PAL["e"]][i])
        ax[2].errorbar(i, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]], color="k", capsize=2, lw=0.8)
        ax[2].annotate(f"{d[0]}/{d[1]}", (i, 0.03), ha="center", fontsize=7, color="w")
    ax[2].set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=6)
    ax[2].set_title("post-hoc fixes (*) next to the unfixed rows", fontsize=9)
    for a in ax:
        a.set_ylim(0, 1.08)
    fig.suptitle("F9 correctness (ok_recovered), v3 additions" + (" - DRY (scripted)" if DRY else ""))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / "F9_correctness.png", dpi=140)
    plt.close(fig)
    (FIG / "F9_correctness.json").write_text(jdump({"bars": {b[0].replace("\n", " "): b[1] for b in bars}}))


# ------------------------------------------------------------------------------ output
FIG: Path
DRY = False


def write(name, lines):
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / name).write_text("\n".join(lines) + "\n")


def tex_out(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ["% Generated by scripts/analyze_v3.py. Do not edit. " + ("DRY VALUES (scripted cells): layout checks only." if DRY else "")]
    body = [f"\\newcommand{{\\vthree{k}}}{{{v}}}" for k, v in sorted(TEX.items())]
    path.write_text("\n".join(head + body) + "\n")


def main(argv=None) -> int:
    global FIG, DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("v3dir", nargs="?", default=str(ROOT / "runs" / "v3"))
    ap.add_argument("--tex", default=None)
    ap.add_argument("--stage-only", action="store_true", help="T7 (fault stage) from runs/v2 + v2_fix alone; needs no v3 cells")
    a = ap.parse_args(argv)
    v3dir = Path(a.v3dir).resolve()
    FIG = v3dir / "figures"
    if a.stage_only:
        e8 = [r for r in load(V2_CELLS) if sp(r)["exp"] == "E8v2" and sp(r)["workload"] != "w3"] + \
             [r for r in load(V2_FIX) if "error" not in r and sp(r)["exp"] == "E8v2"]
        stage_table(e8, [])
        print((FIG / "T7_fault_stage.md").read_text())
        return 0
    e8, e7, v2, by, v3 = inputs(v3dir)
    DRY = any(sp(r)["model"] == "scripted" for r in v3)
    default_tex = (FIG / "v3_draft.tex") if DRY else GENERATED
    tex = Path(a.tex).resolve() if a.tex else default_tex
    if DRY and GENERATED.parent in tex.parents:
        raise SystemExit("refusing to write report inputs from scripted cells (a dry run); use the default draft path")
    for e in ("E8v3", "E8v3f", "E8v3r", "E8v3g"):
        if not by[e]:
            raise Mismatch(f"no {e} rows in {v3dir}/cells.jsonl")
    fix = [r for r in load(V2_FIX) if "error" not in r]
    t1 = t1v3(e8, by["E8v3"])
    t4_ = t4(by["E8v3f"])
    t5_ = t5(fix, by["E8v3r"])
    t6_ = t6(e8, e7, by["E8v3g"])
    stage_table(e8, by["E8v3f"])
    f9(t1, t4_, t5_, t6_)
    RES["meta"] = {"v3_dir": str(v3dir), "rows": len(v3), "dry": DRY, "inputs": [str(V2_CELLS.relative_to(ROOT)), str(V2_FIX.relative_to(ROOT))]}
    (FIG / "analysis_v3.json").write_text(jdump(RES))
    tex_out(tex)
    print(("DRY " if DRY else "") + "wrote", FIG, "and", tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
