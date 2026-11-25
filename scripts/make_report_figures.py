"""Figures of the final report, from the analysis files and the cells. One style, vector PDF, no in-figure titles.

    python scripts/make_report_figures.py [--out report/figures]

Inputs (read only): results/v2/figures/analysis_v2.json, results/v2/figures/F1_cone_vs_suffix.json,
results/v2/figures/F4_replay.json, results/v3/figures/analysis_v3.json, results/v2/cells.jsonl (worked example).
Outputs: fig_headline, fig_example, fig_cone, fig_replay (.pdf and .json each; the json holds every plotted value).
Figures are drawn at their printed width (column 3.35 in, page 7 in), so text sizes are print sizes (>= 7 pt).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
V3 = ROOT / "results" / "v3"

# one palette for every figure (Okabe-Ito, colour-blind safe); one colour per method
COL = {"restart": "#7A7A7A", "rollback": "#CC79A7", "always-verify": "#E69F00", "selective": "#56B4E9",
       "selective+M1": "#0072B2", "selective+M1+M3t/M4+gate": "#009E73"}
NAME = {"restart": "restart", "rollback": "rollback", "always-verify": "always-verify", "selective": "selective",
        "selective+M1": "selective + replay", "selective+M1+M3t/M4+gate": "full stack (gate)"}
WL = [("w1", "W1"), ("w3", "W3"), ("w4.ipi", "W4.ipi"), ("w4.byz", "W4.byz")]
FS = 7.5

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "axes.titlesize": FS, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, out: Path, name: str, data: dict):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    (out / f"{name}.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    plt.close(fig)


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=FS + 1, fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------------------------------------- (a)+(b) headline
def fig_headline(out: Path):
    t = json.load(open(V3 / "figures" / "analysis_v3.json"))["T1v3"]
    wall_arms = ["restart", "always-verify", "selective", "selective+M1", "selective+M1+M3t/M4+gate"]
    all_arms = ["restart", "rollback", "always-verify", "selective", "selective+M1", "selective+M1+M3t/M4+gate"]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.05), gridspec_kw={"width_ratios": [3.3, 1.6, 1.25], "wspace": 0.42})
    data = {"wall": {}, "pooled": {}}
    # (a) recovery wall per workload
    a = ax[0]
    w = 0.17
    for i, (wk, wn) in enumerate(WL):
        top = max(t[f"{wk}/{m}"]["recovery_wall_s"][0] + t[f"{wk}/{m}"]["recovery_wall_s"][1] for m in wall_arms)
        for j, m in enumerate(wall_arms):
            d = t[f"{wk}/{m}"]
            x = i + (j - 2) * w
            mean, sd = d["recovery_wall_s"]
            a.bar(x, mean, w * 0.92, color=COL[m], yerr=sd, error_kw={"lw": 0.5, "capsize": 1.0, "capthick": 0.5})
            ok = f"{d['ok_recovered'][0]}/{d['n']}"
            a.text(x, top + 1.5, ok, ha="center", va="bottom", fontsize=6.5, rotation=90,
                   color="#B03000" if d["ok_recovered"][0] < t[f"{wk}/restart"]["ok_recovered"][0] - 1 else "black")
            data["wall"][f"{wk}/{m}"] = {"wall_mean_s": mean, "wall_sd_s": sd, "ok_recovered": d["ok_recovered"][0], "n": d["n"]}
    a.set_xticks(range(4))
    a.set_xticklabels([n for _, n in WL])
    a.set_ylabel("recovery wall time (s)")
    a.set_ylim(0, 62)
    panel_label(a, "(a)")
    # (b) duplicates and leaked per recovery, pooled over the 30 fault cells (n-weighted mean of the workload means)
    def pooled(m, key):
        n = sum(t[f"{wk}/{m}"]["n"] for wk, _ in WL)
        return sum(t[f"{wk}/{m}"][key] * t[f"{wk}/{m}"]["n"] for wk, _ in WL) / n
    b = ax[1]
    wb = 0.13
    for j, m in enumerate(all_arms):
        for k, key in enumerate(["duplicates", "leaked"]):
            v = pooled(m, key)
            b.bar(k + (j - 2.5) * wb, v, wb * 0.92, color=COL[m])
            data["pooled"].setdefault(m, {})[key] = v
    b.set_xticks([0, 1])
    b.set_xticklabels(["duplicate\neffects", "leaked\neffects"])
    b.set_ylabel("per recovery")
    panel_label(b, "(b)")
    # (c) cells with an unauthorized action, of all fault cells
    c = ax[2]
    ntot = sum(t[f"{wk}/restart"]["n"] for wk, _ in WL)
    for j, m in enumerate(all_arms):
        v = sum(t[f"{wk}/{m}"]["unauthorized_cells"] for wk, _ in WL)
        c.bar(j, v, 0.8, color=COL[m])
        c.text(j, v + 0.3, str(v), ha="center", va="bottom", fontsize=6.5)
        data["pooled"][m]["unauthorized_cells"] = v
    c.set_xticks([])
    c.set_ylabel(f"cells with an unauthorized\naction (of {ntot})")
    c.set_ylim(0, 14.5)
    panel_label(c, "(c)")
    handles = [Patch(color=COL[m], label=NAME[m]) for m in all_arms]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.16), handlelength=1.0,
               columnspacing=1.0, handletextpad=0.4, fontsize=7)
    data["note"] = ("bars (a): mean over fault cells, whiskers sd, label = ok_recovered cells in one row above each workload group (red when two or more miss); "
                    "rollback omitted in (a): its wall time equals restart's; (b),(c): pooled over all 30 fault cells per method")
    save(fig, out, "fig_headline", data)


# ---------------------------------------------------------------------------------------------- (c) worked example
LANES = [0, 1, 2]


def cell_for(rows, fault):
    for r in rows:
        s = r["spec"]
        if (s.get("exp") == "E8v2" and s.get("workload") == "w1" and s.get("fault") == fault and s.get("label") == "selective+M1"
                and s.get("repeat") == 0):
            return r
    raise SystemExit(f"no E8v2 cell for {fault}")


def draw_graph(ax, r, origin, title, effects_note):
    rer, reu = set(r["recovery"]["rerun"]), set(r["recovery"]["reused"])
    comp = r["recovery"]["compensated"]
    cs = {"origin": "#D55E00", "rerun": "#F0B27A", "reused": "#CFE8CF"}

    def state(s):
        return "origin" if s == origin else "rerun" if s in rer else "reused"
    pos = {"research.team": (0.55, 3.2), "research.capacity": (1.75, 3.2)}
    for i in LANES:
        pos[f"research.meeting.{i}"] = (2.95 + 1.2 * i, 3.2)
    for i in LANES:
        y = 2.1 - 0.85 * i
        pos[f"plan.{i}"] = (1.0, y)
        pos[f"execute.book.{i}"] = (2.6, y)
        pos[f"execute.email.{i}"] = (4.2, y)
    short = {"research.team": "team", "research.capacity": "capacity"}
    for i in LANES:
        short[f"research.meeting.{i}"] = f"meeting {i}"
        short[f"plan.{i}"] = f"plan {i}"
        short[f"execute.book.{i}"] = f"book {i}"
        short[f"execute.email.{i}"] = f"e-mail {i}"
    bw, bh = 1.08, 0.5

    def edge(a, b):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.annotate("", xy=(x2 - bw / 2, y2), xytext=(x1 + bw / 2, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color="#555555", shrinkA=0, shrinkB=0, mutation_scale=5))
    for i in LANES:
        edge(f"plan.{i}", f"execute.book.{i}")
        edge(f"execute.book.{i}", f"execute.email.{i}")
    for s_, (x, y) in pos.items():
        st = state(s_)
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    fc=cs[st], ec="#D55E00" if st == "origin" else "#555555", lw=1.2 if st == "origin" else 0.5))
        ax.text(x, y, short[s_], ha="center", va="center", fontsize=7, fontweight="bold" if st == "origin" else "normal")
        if st == "rerun" and s_.startswith("execute.email"):
            ax.text(x + bw / 2 - 0.02, y + bh / 2 - 0.02, "!", color="#B00000", fontsize=8, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.08", fc="white", ec="#B00000", lw=0.6))
        if st == "rerun" and s_.startswith("execute.book") and comp:
            ax.text(x + bw / 2 - 0.02, y + bh / 2 - 0.02, "\u00d7", color="#B00000", fontsize=8, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.08", fc="white", ec="#B00000", lw=0.6))
    ax.set_xlim(0.0, 6.05)
    ax.set_ylim(-0.95, 4.1)
    ax.axis("off")
    ax.text(0.0, 4.05, title, fontsize=FS + 0.5, fontweight="bold", va="top")
    ax.text(0.0, -0.45, effects_note, fontsize=7, va="top", ha="left", linespacing=1.3)
    return {s_: state(s_) for s_ in pos}


def fig_example(out: Path):
    rows = [json.loads(l) for l in open(V2 / "cells.jsonl")]
    cap = cell_for(rows, "wrong_capacity")
    ag = cell_for(rows, "inject_agenda")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"wspace": 0.0})
    rw = json.load(open(V3 / "figures" / "analysis_v3.json"))["T1v3"]["w1/restart"]["recovery_wall_s"][0]
    e1, e2 = cap["recovery"]["effects"], ag["recovery"]["effects"]
    n1 = (f"effects kept {e1['retained']}, cancelled {e1['compensated']}, re-sent {e1['released']}, leaked {e1['leaked']}\n"
          f"recovery {cap['recovery']['wall_s']:.1f} s (restart: {rw:.1f} s)")
    n2 = (f"effects kept {e2['retained']}, cancelled {e2['compensated']}, re-sent {e2['released']}, leaked {e2['leaked']}\n"
          f"recovery {ag['recovery']['wall_s']:.1f} s (restart: {rw:.1f} s)")
    d1 = draw_graph(ax[0], cap, "research.capacity", "(a) wrong room capacity", n1)
    d2 = draw_graph(ax[1], ag, "research.meeting.0", "(b) injected agenda item", n2)
    handles = [Patch(fc="#D55E00", ec="#D55E00", label="faulty result"), Patch(fc="#F0B27A", ec="#555555", label="rerun"),
               Patch(fc="#CFE8CF", ec="#555555", label="reused, no model call"),
               plt.Line2D([], [], marker="none", color="none", label="\u00d7 booking cancelled"),
               plt.Line2D([], [], marker="none", color="none", label="! old e-mail already sent")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=7,
               columnspacing=1.0, handletextpad=0.4)
    data = {"cells": {"wrong_capacity": {"spec": cap["spec"], "states": d1, "effects": e1, "recovery_wall_s": cap["recovery"]["wall_s"]},
                      "inject_agenda": {"spec": ag["spec"], "states": d2, "effects": e2, "recovery_wall_s": ag["recovery"]["wall_s"]}},
            "note": ("E8v2 W1 selective+M1 repeat 0. Arrows drawn: plan -> booking -> e-mail within each meeting only; the markers follow "
                     "from which steps were rerun and the effect counts of the cell (cancelled 2 = the two rerun bookings; leaked = old e-mails)")}
    save(fig, out, "fig_example", data)


# ---------------------------------------------------------------------------------------------- (d) cone
def fig_cone(out: Path):
    d = json.load(open(V2 / "figures" / "F1_cone_vs_suffix.json"))["points"]
    panels = [("fan_in=1", "fan-in 1"), ("fan_in=3", "fan-in 3"), ("shared_keys=6", "6 shared keys")]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.6), sharey=True, gridspec_kw={"wspace": 0.08})
    data = {}
    for a, (key, ttl) in zip(ax, panels):
        for m, ls, mk in [("restart", "-", "s"), ("rollback", "-", "^"), ("selective", "-", "o")]:
            pts = d[f"{key}/{m}"]
            xs = sorted(pts, key=float)
            ys = [pts[x] for x in xs]
            a.plot([float(x) for x in xs], ys, ls, marker=mk, ms=3.2, lw=1.1, color=COL[m], label=NAME[m])
            data[f"{key}/{m}"] = dict(zip(xs, ys))
        a.set_xlabel("fault position in the workflow")
        a.set_title(ttl, fontsize=FS, pad=2)
        a.set_ylim(0, 25)
        a.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax[0].set_ylabel("steps rerun (of 24)")
    ax[2].legend(loc="upper right", bbox_to_anchor=(1.0, 0.90), frameon=False, handlelength=1.4)
    data["note"] = "W2 synthetic DAGs, scripted model, 24 steps; mean of 5 repeats; the other fan-in and shared-key panels are in results/v2/figures/F1"
    save(fig, out, "fig_cone", data)


# ---------------------------------------------------------------------------------------------- (e) replay
def fig_replay(out: Path):
    h = json.load(open(V2 / "figures" / "F4_replay.json"))["H1"]
    fig, ax = plt.subplots(1, 2, figsize=(3.35, 1.75), gridspec_kw={"wspace": 0.5})
    data = {}
    w = 0.36
    for k, (key, ylab) in enumerate([("decoded_ratio_median", "decoded tokens,\nrelative to a plain rerun"),
                                     ("wall_ratio_median", "LLM wall time,\nrelative to a plain rerun")]):
        a = ax[k]
        for j, (kind, col, lab) in enumerate([("prefix", "#BBBBBB", "prefix draft"), ("resync", COL["selective+M1"], "resync draft")]):
            xs, ys, lo, hi = [], [], [], []
            for i, (wk, wn) in enumerate(WL):
                r = h[f"{wk}/{kind}"]
                v = r[key]
                xs.append(i + (j - 0.5) * w)
                ys.append(v)
                if key == "wall_ratio_median":
                    ci = r["wall_ratio_median_ci95"]
                    lo.append(v - ci[0])
                    hi.append(ci[1] - v)
                data[f"{wk}/{kind}/{key}"] = v
            a.bar(xs, ys, w * 0.92, color=col, label=lab,
                  yerr=[lo, hi] if lo else None, error_kw={"lw": 0.5, "capsize": 1.0, "capthick": 0.5})
        a.axhline(1.0, color="black", lw=0.5, ls=":")
        a.set_xticks(range(4))
        a.set_xticklabels([n for _, n in WL], rotation=45, ha="right", rotation_mode="anchor")
        a.set_ylabel(ylab)
        a.set_ylim(0, 1.35)
    ax[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.0, fontsize=7)
    data["note"] = "E5v2 7B held-out repeats; medians; whiskers on wall: 95% CI of the median; dotted line = plain rerun"
    save(fig, out, "fig_replay", data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "report" / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    fig_headline(out)
    fig_example(out)
    fig_cone(out)
    fig_replay(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
