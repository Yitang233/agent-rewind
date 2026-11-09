"""Make speculative replay (M1) faster in wall time. Background, resumable, one model at a time.

Sub-commands (all write under runs/v2/D3/):
  record   run a workload with a fault live (prompt cache off), recover it with replay=off and store every
           recovery model call (prompt, schema, old output = draft, plain output) in cases/<key>.json
  replay   replay the recorded call sequences under a list of configs: wall, identity with the plain arm of the
           same sequence, and the per-phase times of replay.ReplayStats -> <out>.jsonl
  profile  phase split of the v1 replay (tie=full, K=32, margin=0.3) vs plain decode on recorded reruns
  tune     the tuning grid layout x K x margin on the tuning cases (variants 0-1), then picks the configuration
  heldout  the chosen configuration on variants 2-4 (+ synthetic 1,500-token prompt); identity and ratio; also
           tie=full vs tie=local on every recorded call
  guard    clean-success guard for layout=stable
Call sequences are replayed in recovery order with a cold KV cache at the start of every sequence, so each
arm sees the same state. Greedy decoding throughout; identity is token-id equality with the plain arm.
"""
import argparse
import gc
import hashlib
import json
import os
import resource
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "runs/v2/D3")
CASES = os.path.join(OUT, "cases")
FAULT = {"w1": "wrong_capacity", "w3": "wrong_issue", "w4.ipi": "inject_item", "w4.byz": "byz_factcheck"}


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)


def busy():
    import subprocess
    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    return [ln.strip() for ln in ps.splitlines() if ln.split() and ln.split()[0] != str(os.getpid())
            and "replay_tune" not in ln and "pgrep" not in ln
            and any(x in ln for x in ("agent_rewind.bench", "replay_check", "audit_check", "bench_llm"))
            and os.path.basename(ln.split()[1]).startswith("python")]


def load_model(name, **kw):
    from agent_rewind.llm import LlamaCppLLM
    if busy():
        raise SystemExit("blocked: another model process is running: " + busy()[0])
    env = json.load(open(os.path.join(ROOT, "runs/env.json")))["models"][name]
    return LlamaCppLLM(env["path"], model_sha=env["sha256"], cache_path=None, n_ctx=4096, **kw)


def jl_read(path):
    return [json.loads(ln) for ln in open(path) if ln.strip()] if os.path.exists(path) else []


def jl_append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def cold(inner):
    inner.llm.n_tokens = 0
    inner.llm._ctx.kv_cache_seq_rm(-1, 0, -1)


def case_path(model, wl, variant, layout, style):
    return os.path.join(CASES, f"{model}_{wl}_{variant}_{layout}_{style}.json")


# ---- record ------------------------------------------------------------------------------------

def record(inner, model, wl, variant, layout, style):
    from agent_rewind import bench
    from agent_rewind.deps import depth_of
    from agent_rewind.engine import execute, producers
    from agent_rewind.llm import CachedLLM
    path = case_path(model, wl, variant, layout, style)
    if os.path.exists(path):
        return path
    inner.style, inner.layout, inner.tie = style, layout, "full"
    inner.replay, inner.drafts, inner.history = "off", {}, {}
    case = bench.build_case(wl, variant, {})
    log = []
    orig = inner.generate

    def spy(prompt, schema, step=None):
        g = orig(prompt, schema, step)
        log.append({"step": step, "prompt": prompt, "schema": schema, "ids": g["ids"], "stats": dict(g["stats"])})
        return g
    inner.generate = spy
    try:
        wf, world = case.workflow, case.world
        fault = case.faults[FAULT[wl]]
        kw = dict(effects="naive", depth=depth_of("field"))
        t0 = time.perf_counter()
        cold(inner)
        faulty = execute(wf, world, CachedLLM(inner, dedupe=False), overrides={(fault.step, wf.path(fault.name)): fault.value}, **kw)
        faulty_s, n_f = time.perf_counter() - t0, len(log)
        drafts = {k: list(v) for k, v in inner.history.items()}
        dirty = producers(faulty, {wf.path(fault.name)})
        inner.history = {}
        t0 = time.perf_counter()
        execute(wf, world, CachedLLM(inner, dedupe=False), prior=faulty, dirty=dirty, policy="selective", **kw)
        rec_s = time.perf_counter() - t0
        calls = [dict(c, draft=drafts.get(c["step"])) for c in log[n_f:]]
        ok = None
        res = {"model": model, "workload": wl, "variant": variant, "layout": layout, "style": style,
               "fault": fault.name, "faulty_wall_s": faulty_s, "faulty_calls": n_f, "recovery_wall_s": rec_s,
               "dirty": sorted(dirty), "calls": calls, "ok": ok}
    finally:
        inner.generate = orig
        case.close()
    os.makedirs(CASES, exist_ok=True)
    json.dump(res, open(path, "w"))
    return path


# ---- replay ------------------------------------------------------------------------------------

PHASES = ("t_prefill", "t_verify", "t_decode", "t_tie", "t_resync", "t_pick")


def run_seq(inner, calls, cfg):
    """Replay a recorded call sequence under cfg = {mode, K, margin, tie}. Returns per-call dicts."""
    inner.replay = cfg["mode"]
    inner.K, inner.margin, inner.tie = cfg.get("K", 32), cfg.get("margin", 0.3), cfg.get("tie", "local")
    inner.resync_n = cfg.get("n", 4)
    cold(inner)
    out = []
    for c in calls:
        inner.drafts = {c["step"]: c["draft"]} if c.get("draft") is not None and c["step"] else {}
        g = inner.generate(c["prompt"], c["schema"], step=c["step"])
        st = g["stats"]
        out.append({"step": c["step"], "ids": g["ids"], "wall_s": st["wall_s"], "stats": st,
                    "had_draft": c.get("draft") is not None})
    return out


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def compare(res, ref):
    """-> (identical, compared, mismatches) of res against the plain arm `ref` call by call."""
    n_id, mism = 0, []
    for i, (r, p) in enumerate(zip(res, ref)):
        if r["ids"] == p["ids"]:
            n_id += 1
        else:
            mism.append({"call": i, "step": r["step"], "first_diff": first_diff(r["ids"], p["ids"])})
    return n_id, len(ref), mism


def gap_at(inner, c, ids_plain, pos):
    """Raw logit gap between the plain token and the replay token cannot be known here; report the plain
    run's top-2 gap at `pos`, computed with one-token evaluations (how a plain decode sees it)."""
    import numpy as np

    from agent_rewind import replay as R
    ids = inner.llm.tokenize(c["prompt"].encode(), add_bos=False, special=True)
    be = R.LlamaBackend(inner.llm, None)
    cold(inner)
    sh, row = be.prefill(ids)
    for t in ids_plain[:pos]:
        row = be.eval([t])[0]
    top = np.sort(row)[-2:]
    return float(top[1] - top[0])


def cmd_replay(inner, model, case_files, cfgs, out, tag, extra=None, plain_cache=None):
    """Run every cfg on every case file; the plain arm is run once per case (in this process)."""
    done = {r["key"] for r in jl_read(out)}
    for cf in case_files:
        case = json.load(open(cf))
        key0 = os.path.basename(cf)
        inner.style, inner.layout = case["style"], case["layout"]
        calls = case["calls"]
        plain = None
        for cfg in [{"mode": "off"}] + cfgs:
            key = f"{tag}|{key0}|{json.dumps(cfg, sort_keys=True)}"
            if key in done:
                continue
            if plain is None:
                plain = run_seq(inner, calls, {"mode": "off"})
            res = plain if cfg["mode"] == "off" else run_seq(inner, calls, cfg)
            n_id, n, mism = compare(res, plain)
            for m in mism:
                m["plain_top2_gap"] = gap_at(inner, calls[m["call"]], plain[m["call"]]["ids"], m["first_diff"]) \
                    if m["first_diff"] < len(plain[m["call"]]["ids"]) else None
            row = {"key": key, "tag": tag, "case": key0, "model": model, "workload": case["workload"],
                   "variant": case["variant"], "layout": case["layout"], "style": case["style"], "cfg": cfg,
                   "n_calls": n, "identical": n_id, "mismatches": mism,
                   "wall_s": sum(r["wall_s"] for r in res), "plain_wall_s": sum(r["wall_s"] for r in plain),
                   "decoded": sum(r["stats"]["decoded"] for r in res), "prefill": sum(r["stats"]["prefill"] for r in res),
                   "out_tokens": sum(len(r["ids"]) for r in res), "prompt_tokens": [r["stats"]["prompt"] for r in res],
                   "ties": sum(r["stats"].get("ties", 0) for r in res), "reeval": sum(r["stats"].get("reeval", 0) for r in res),
                   "passes": sum(r["stats"].get("passes", 0) for r in res),
                   "phases": {k: sum(r["stats"].get(k, 0.0) for r in res) for k in PHASES},
                   "per_call_wall": [r["wall_s"] for r in res], "peak_rss_gb": rss_gb(), **(extra or {})}
            if plain_cache is not None:
                plain_cache[key0] = plain
            jl_append(out, row)
            print(f"{key0} {cfg} wall={row['wall_s']:.1f}s plain={row['plain_wall_s']:.1f}s id={n_id}/{n}", flush=True)


# ---- grids ---------------------------------------------------------------------------------------

def write_profile():
    rows = [r for r in jl_read(os.path.join(OUT, "profile_rows.jsonl")) if r["tag"] == "profile"]
    arms = {}
    for r in rows:
        arms.setdefault(r["cfg"]["mode"], []).append(r)
    rep = arms.get("resync", [])
    tot = {k: sum(r["phases"][k] for r in rep) for k in PHASES}
    wall = sum(r["wall_s"] for r in rep)
    tot["t_other"] = wall - sum(tot.values())
    plain = sum(r["plain_wall_s"] for r in rep)
    prof = {"config": "v1 replay: layout=v1 resync K=32 margin=0.3 tie=full", "model": "7b", "style": "atoms",
            "cases": [r["case"] for r in rep], "recovery_reruns": sum(r["n_calls"] for r in rep),
            "plain_wall_s": plain, "replay_wall_s": wall, "ratio": wall / plain if plain else None,
            "phase_s": tot, "phase_share": {k: (v / wall if wall else None) for k, v in tot.items()},
            "ties": sum(r["ties"] for r in rep), "reeval_tokens": sum(r["reeval"] for r in rep),
            "passes": sum(r["passes"] for r in rep), "decoded": sum(r["decoded"] for r in rep),
            "plain_decoded": sum(r["out_tokens"] for r in arms.get("off", [])),
            "prompt_tokens": [t for r in rep for t in r["prompt_tokens"]],
            "per_case": [{"case": r["case"], "calls": r["n_calls"], "wall_s": r["wall_s"], "plain_wall_s": r["plain_wall_s"],
                          "ties": r["ties"], "reeval": r["reeval"], "phases": r["phases"], "identical": r["identical"]} for r in rep]}
    json.dump(prof, open(os.path.join(OUT, "profile.json"), "w"), indent=1)
    print(json.dumps({k: prof[k] for k in ("plain_wall_s", "replay_wall_s", "ratio", "phase_share", "ties", "reeval_tokens")}), flush=True)


def grid_cfgs():
    return [{"mode": "resync", "K": K, "margin": m, "tie": "local"} for K in (16, 32, 64) for m in (0.0, 0.1, 0.3)]


def summarize_tune(rows):
    """Pooled per (layout, K, margin): wall ratio vs off and identity over tuning rows."""
    groups = {}
    for r in rows:
        if r["cfg"]["mode"] == "off":
            continue
        k = (r["layout"], r["cfg"]["K"], r["cfg"]["margin"])
        groups.setdefault(k, []).append(r)
    out = []
    for (lay, K, m), rs in sorted(groups.items()):
        wall, pw = sum(r["wall_s"] for r in rs), sum(r["plain_wall_s"] for r in rs)
        n_id, n = sum(r["identical"] for r in rs), sum(r["n_calls"] for r in rs)
        out.append({"layout": lay, "K": K, "margin": m, "cases": len(rs), "calls": n, "identical": n_id,
                    "wall_s": wall, "plain_wall_s": pw, "ratio": wall / pw if pw else None,
                    "per_case_ratio": [round(r["wall_s"] / r["plain_wall_s"], 3) for r in rs],
                    "ties": sum(r["ties"] for r in rs), "reeval": sum(r["reeval"] for r in rs),
                    "decoded": sum(r["decoded"] for r in rs)})
    return out


def pick_best(summ):
    ok = [s for s in summ if s["identical"] == s["calls"]]
    return min(ok, key=lambda s: (s["ratio"], s["wall_s"])) if ok else None


# ---- main ----------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "profile", "tune", "heldout", "guard", "synth", "choose", "report"])
    ap.add_argument("--model", default="7b")
    ap.add_argument("--workloads", nargs="+", default=["w1", "w3"])
    ap.add_argument("--variants", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--layouts", nargs="+", default=["v1", "stable"])
    ap.add_argument("--style", default="atoms")
    ap.add_argument("--layout", default="v1")
    ap.add_argument("--freeze", choices=["rule", "fallback"], default="rule",
                    help="choose: freeze the rule's pick, or the best configuration with margin > 0")
    ap.add_argument("--config", default=os.path.join(ROOT, "bench/configs/replay_tuned.json"))
    a = ap.parse_args()
    os.makedirs(CASES, exist_ok=True)
    if a.cmd == "choose":
        return choose(a)
    if a.cmd == "report":
        return report(a)
    t_start = time.perf_counter()
    inner = load_model(a.model)
    try:
        if a.cmd == "record":
            for lay in a.layouts:
                for wl in a.workloads:
                    for v in a.variants:
                        t0 = time.perf_counter()
                        try:
                            p = record(inner, a.model, wl, v, lay, a.style)
                        except Exception as e:               # e.g. unparseable output of the small model
                            print(f"record failed {a.model} {wl} {v} {lay}: {e!r}"[:300], flush=True)
                            inner.generate = type(inner).generate.__get__(inner)
                            continue
                        c = json.load(open(p))
                        print(f"recorded {os.path.basename(p)} calls={len(c['calls'])} faulty={c['faulty_wall_s']:.1f}s "
                              f"recovery={c['recovery_wall_s']:.1f}s ({time.perf_counter() - t0:.0f}s) rss={rss_gb():.1f}GB", flush=True)
        elif a.cmd == "profile":
            files = [case_path(a.model, wl, v, "v1", a.style) for wl in ("w1", "w3") for v in (0, 1)]
            cfgs = [{"mode": "resync", "K": 32, "margin": 0.3, "tie": "full"}]
            cmd_replay(inner, a.model, files, cfgs, os.path.join(OUT, "profile_rows.jsonl"), "profile")
            write_profile()
        elif a.cmd == "tune":
            files = [case_path(a.model, wl, v, lay, a.style) for lay in a.layouts for wl in a.workloads for v in a.variants]
            cmd_replay(inner, a.model, files, grid_cfgs(), os.path.join(OUT, "tune_rows.jsonl"), "tune")
        elif a.cmd == "heldout":
            tuned = json.load(open(a.config))
            fl = tuned["flags"]
            cfg = {"mode": fl["replay"], "K": fl["replay_K"], "margin": fl["margin"], "tie": fl["tie"]}
            files = [case_path(a.model, wl, v, lay, a.style) for lay in a.layouts for wl in a.workloads for v in a.variants]
            if a.model == "7b":
                files += [case_path(a.model, "synth", 0, lay, a.style) for lay in a.layouts]
            files = [f for f in files if os.path.exists(f)]
            fb = tuned.get("fallback_flags") or {}
            cfgs = [cfg,                                                           # the frozen configuration
                    {"mode": "resync", "K": fb.get("replay_K", 32), "margin": fb.get("margin", 0.1), "tie": "local"},   # fallback (margin > 0)
                    {"mode": "resync", "K": 32, "margin": 0.3, "tie": "full"},    # v1 re-check
                    {"mode": "resync", "K": 32, "margin": 0.3, "tie": "local"}]   # local re-check: must equal the line above
            cmd_replay(inner, a.model, files, cfgs, os.path.join(OUT, f"heldout_rows_{a.model}.jsonl"), "heldout")
        elif a.cmd == "guard":
            guard(inner, a)
        elif a.cmd == "synth":
            synth(inner, a)
    finally:
        print(f"model time {time.perf_counter() - t_start:.0f}s peak rss {rss_gb():.1f}GB", flush=True)
        del inner
        gc.collect()


def choose(a):
    """Apply the pre-stated rule to the tuning rows: fastest (pooled wall ratio vs off, same layout) with 100% identity.
    Writes runs/v2/D3/tune.json and bench/configs/replay_tuned.json."""
    rows = [r for r in jl_read(os.path.join(OUT, "tune_rows.jsonl")) if r["tag"] == "tune"]
    summ = summarize_tune(rows)
    best = pick_best(summ)
    safe = pick_best([s_ for s_ in summ if s_["margin"] > 0])
    frozen = best if a.freeze == "rule" else safe
    mism = [{"case": r["case"], "cfg": r["cfg"], "mismatches": r["mismatches"]} for r in rows if r["mismatches"]]
    off = {lay: sum(r["plain_wall_s"] for r in rows if r["layout"] == lay and r["cfg"]["mode"] == "off") for lay in ("v1", "stable")}
    json.dump({"rule": "fastest pooled wall ratio vs replay=off (same layout) among configs with 100% identity on tuning cells",
               "cases": sorted({r["case"] for r in rows}), "grid": summ, "chosen": best, "fallback_margin_gt_0": safe,
               "mismatches": mism, "plain_wall_s_by_layout": off}, open(os.path.join(OUT, "tune.json"), "w"), indent=1)
    def flags(c):
        return {"replay": "resync", "layout": c["layout"], "replay_K": c["K"], "margin": c["margin"], "tie": "local"}
    cfg = {"name": "replay_tuned",
           "note": "replay settings frozen for the v2 grid, chosen on tuning cells (variants 0-1; W1 wrong_capacity, W3 wrong_issue; 7B atoms) "
                   "by runs/v2/D3/tune.json. Splice `flags` into the E5v2/E8v2 experiments. Held-out evidence: runs/v2/D3/heldout.json.",
           "flags": flags(frozen), "frozen": a.freeze,
           "rule_pick_flags": flags(best), "fallback_flags": flags(safe) if safe else None,
           "tuning": {"ratio_vs_off": frozen["ratio"], "identity": [frozen["identical"], frozen["calls"]]},
           "defaults_v1": {"layout": "v1", "replay_K": 32, "margin": 0.3, "tie": "full"}}
    json.dump(cfg, open(a.config, "w"), indent=1)
    print(json.dumps(cfg, indent=1))


def report(a):
    """Write runs/v2/D3/heldout.json from the held-out rows (7b and 1.5b) and the guard rows."""
    out = {}
    for m in ("7b", "1.5b"):
        rows = [r for r in jl_read(os.path.join(OUT, f"heldout_rows_{m}.jsonl")) if r["tag"] == "heldout"]
        if not rows:
            continue
        arms = {}
        for r in rows:
            arms.setdefault(json.dumps(r["cfg"], sort_keys=True), []).append(r)
        res = {}
        for k, rs in arms.items():
            if rs[0]["cfg"]["mode"] == "off":
                continue
            by_wl = {}
            for r in rs:
                by_wl.setdefault(r["workload"], []).append(r)
            res[k] = {"identical": sum(r["identical"] for r in rs), "calls": sum(r["n_calls"] for r in rs),
                      "wall_s": sum(r["wall_s"] for r in rs), "plain_wall_s": sum(r["plain_wall_s"] for r in rs),
                      "ratio": sum(r["wall_s"] for r in rs) / sum(r["plain_wall_s"] for r in rs),
                      "decoded": sum(r["decoded"] for r in rs), "plain_decoded": sum(r["out_tokens"] for r in arms.get(
                          json.dumps({"mode": "off"}), []) if r["case"] in {x["case"] for x in rs}),
                      "ties": sum(r["ties"] for r in rs), "reeval": sum(r["reeval"] for r in rs),
                      "mismatches": [{"case": r["case"], **mm} for r in rs for mm in r["mismatches"]],
                      "by_workload": {w: {"ratio": sum(r["wall_s"] for r in x) / sum(r["plain_wall_s"] for r in x),
                                          "per_case_ratio": [round(r["wall_s"] / r["plain_wall_s"], 3) for r in x],
                                          "identical": [sum(r["identical"] for r in x), sum(r["n_calls"] for r in x)],
                                          "prompt_tokens": sorted({t for r in x for t in r["prompt_tokens"]})}
                                      for w, x in sorted(by_wl.items())}}
        out[m] = res
    out["tie_recheck_full_vs_local"] = tie_equivalence(a)
    g = jl_read(os.path.join(OUT, "guard_rows.jsonl"))
    out["guard"] = {}
    for r in g:
        d = out["guard"].setdefault(f"{r['layout']}/{r['workload']}", {"ok": 0, "n": 0, "failed": []})
        d["n"] += 1
        d["ok"] += int(r["ok"])
        if not r["ok"]:
            d["failed"].append({"variant": r["variant"], "error": r["error"]})
    out["note"] = ("Held-out = variants 2-4 (+ one synthetic 1,607-token prompt, 7B). Call-level replay of recorded recovery "
                   "sequences (cold KV at the start of each sequence); wall = sum of LLM call wall; identity = token ids equal to "
                   "the plain arm of the same sequence. cfg with margin 0.0 is the tuning rule's pick; margin 0.1 is the frozen one.")
    json.dump(out, open(os.path.join(OUT, "heldout.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:6000])


def tie_equivalence(a):
    """Rows of the held-out run where margin 0.3 was used with tie=full and tie=local: per case, are the outputs the same
    token by token? (Both are compared to the plain arm, so identical counts that agree prove equality on the plain-identical calls;
    a call where either differs from plain is listed.)"""
    out = {}
    for m in ("7b", "1.5b"):
        rows = [r for r in jl_read(os.path.join(OUT, f"heldout_rows_{m}.jsonl")) if r["tag"] == "heldout" and r["cfg"]["mode"] == "resync"
                and r["cfg"]["margin"] == 0.3 and r["cfg"]["K"] == 32]
        out[m] = {"cases": len({r["case"] for r in rows}),
                  "full": [sum(r["identical"] for r in rows if r["cfg"]["tie"] == "full"), sum(r["n_calls"] for r in rows if r["cfg"]["tie"] == "full")],
                  "local": [sum(r["identical"] for r in rows if r["cfg"]["tie"] == "local"), sum(r["n_calls"] for r in rows if r["cfg"]["tie"] == "local")],
                  "reeval_full": sum(r["reeval"] for r in rows if r["cfg"]["tie"] == "full"),
                  "reeval_local": sum(r["reeval"] for r in rows if r["cfg"]["tie"] == "local"),
                  "t_tie_full": sum(r["phases"]["t_tie"] for r in rows if r["cfg"]["tie"] == "full"),
                  "t_tie_local": sum(r["phases"]["t_tie"] for r in rows if r["cfg"]["tie"] == "local"),
                  "ties_full": sum(r["ties"] for r in rows if r["cfg"]["tie"] == "full"),
                  "ties_local": sum(r["ties"] for r in rows if r["cfg"]["tie"] == "local")}
    return out


def guard(inner, a):
    """Clean runs (cache off) with layout=stable: success per workload vs v1."""
    from agent_rewind import bench
    from agent_rewind.deps import depth_of
    from agent_rewind.engine import execute
    from agent_rewind.llm import CachedLLM
    out = os.path.join(OUT, "guard_rows.jsonl")
    done = {r["key"] for r in jl_read(out)}
    for lay in a.layouts:
        for wl in a.workloads:
            for v in a.variants:
                key = f"{lay}|{wl}|{v}"
                if key in done:
                    continue
                inner.style, inner.layout, inner.replay, inner.tie = a.style, lay, "off", "local"
                inner.drafts, inner.history = {}, {}
                case = bench.build_case(wl, v, {})
                try:
                    cold(inner)
                    t0 = time.perf_counter()
                    err = None
                    try:
                        execute(case.workflow, case.world, CachedLLM(inner, dedupe=False), effects="naive", depth=depth_of("field"))
                        ok = bool(case.check(case.world))
                    except Exception as e:                                   # unparseable output
                        ok, err = False, repr(e)[:200]
                    jl_append(out, {"key": key, "layout": lay, "workload": wl, "variant": v, "ok": ok, "error": err,
                                    "wall_s": time.perf_counter() - t0})
                    print(key, ok, err, flush=True)
                finally:
                    case.close()


def synth_prompt(layout, issue):
    """W3's plan_patch prompt with the Code group padded by 60 dummy helpers (about 1,500 tokens in all)."""
    from agent_rewind import atoms as A
    from agent_rewind import llm as L
    from agent_rewind.workloads import codefix
    base = codefix.listing(codefix.source_for(0))
    pad = [ln for i in range(60) for ln in (f"def helper_{i}(x):", f"    return x + {i}", "", "")]
    code = list(base)[:-1] + ["", ""] + pad
    groups = [A.Group("Code", {codefix.CODE_PATH: code}),
              A.Group("Tests", {codefix.TEST_PATH: ["import unittest", "import cart"]}),
              A.Group("Issue", {codefix.ISSUE_PATH: issue})]
    ins, ans, hint = L.ATOM_TASKS["plan_patch"]
    system, user = L.stable_prompt("planner", ins, groups, hint) if layout == "stable" else \
        A.build_prompt("planner", ins, groups, hint)
    return A.chatml(system, user), A.atom_schema(groups, ans)


def synth(inner, a):
    """Synthetic long-prompt case: the draft is the plain output for the 'old' issue, the rerun prompt has the 'new' one."""
    for lay in a.layouts:
        path = case_path(a.model, "synth", 0, lay, a.style)
        if os.path.exists(path):
            continue
        old_p, schema = synth_prompt(lay, "# Bug in `bulk_discount`\n\nOrdering exactly 10 items should get the bulk discount.\n")
        new_p, _ = synth_prompt(lay, "# Bug in `shipping`\n\nAn order of exactly 50.0 should ship free, but pays 5.0.\n")
        inner.style, inner.layout, inner.replay, inner.drafts = a.style, lay, "off", {}
        cold(inner)
        g = inner.generate(old_p, schema, step=None)
        calls = [{"step": "synth", "prompt": new_p, "schema": schema, "draft": g["ids"], "ids": None, "stats": {}}]
        json.dump({"model": a.model, "workload": "synth", "variant": 0, "layout": lay, "style": a.style,
                   "calls": calls, "old_prompt_tokens": g["stats"]["prompt"]}, open(path, "w"))
        print("synth", lay, "old prompt tokens", g["stats"]["prompt"], "draft", len(g["ids"]), flush=True)


if __name__ == "__main__":
    main()
