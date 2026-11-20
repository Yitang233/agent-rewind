"""Execute one cell: faulty run, recovery, clean reference and the result row."""
from __future__ import annotations

import json
import random
import time

from .. import audit as AU
from ..cutoff import Canonical
from ..deps import Meter, depth_of
from ..engine import dynamic_cone, execute, explain_reruns, producers, run_step
from ..store import canon
from .cases import _ok, affected_steps, build_case, choose_w2_fault, n_irreversible, unauthorized_acts
from .config import flag
from .grid import cell_key
from .metrics import (MeteredCutoff, RecordingLLM, _bookkeeping, _checker_counts, _delta, _outcome, _totals,
                      rss_mb)
from .models import Models, _audit_args


def clean_reference(spec: dict, models: "Models", inner, live: bool) -> dict:
    """A live (cache off) clean run of this variant: outcome-path hashes, memoised per out dir by
    (workload, params, repeat, style, model). Cost fields are those of the run that computed it."""
    fl, real = spec["flags"], spec["model"] != "scripted"
    key = json.dumps([spec["workload"], spec["params"], spec["repeat"], fl["style"], spec["model"]], sort_keys=True)
    refs = models.refs()
    if key in refs:
        return dict(refs[key], source="memo")
    case = build_case(spec["workload"], spec["repeat"], spec["params"])
    db = getattr(inner, "db", None) if real else None
    try:
        backend = inner if real else case.scripted()
        if real:
            backend.style = fl["style"]
            backend.replay, backend.drafts, backend.history = "off", {}, {}
            backend.db = None                                   # the reference always decodes live
        llm = RecordingLLM(backend, dedupe=False)
        t0, tot0 = time.perf_counter(), _totals(backend)
        run = execute(case.workflow, case.world, llm, effects=fl["effects"], depth=depth_of(fl["depth"]))
        delta = _delta(tot0, _totals(backend))
        ref = {"key": key, "outcome": _outcome(run.values()), "wall_s": time.perf_counter() - t0,
               "tokens": sum(r.cost.tokens for r in run.records), "llm": delta,
               "cache_hits": int(delta.get("cache_hits", 0)), "memo_hits": llm.hits, "ok": bool(case.check(case.world))}
    finally:
        if real:
            inner.db = db
        case.close()
    models.save_ref(ref)
    return dict(ref, source="computed")


def reuse_check(spec: dict, rec, wf, inner, real: bool) -> dict:
    """Sample up to 2 reused model steps (seeded by the repeat), re-execute each live in a scratch branch of the
    recovered store with a scratch world, and compare its writes canonically with the reused ones."""
    cands = sorted(r.step for r in rec.records if r.reused and r.cost.llm_calls > 0)
    pick = random.Random(spec["repeat"]).sample(cands, min(2, len(cands)))
    out = {"sampled": len(pick), "equal": 0, "candidates": len(cands), "steps": []}
    canon_eq = Canonical()
    by_step = {r.step: r for r in rec.records}
    for name in pick:
        scratch = build_case(spec["workload"], spec["repeat"], spec["params"])
        t0 = time.perf_counter()
        try:
            store = rec.store
            branch = store.new_branch(parent=rec.branch)
            backend = inner if real else scratch.scripted()
            llm = RecordingLLM(backend, dedupe=False)
            step = next(s for s in wf.steps if s.name == name)
            new = run_step(step, store, branch, scratch.world, llm, None, rec.depth)
            old = by_step[name]
            same_paths = set(new.values) == set(old.values)
            equal = same_paths and all(canon_eq.value(p, new.values[p]) == canon_eq.value(p, old.values[p])
                                       for p in old.values)
            exact = same_paths and all(canon(new.values[p]) == canon(old.values[p]) for p in old.values)
            out["steps"].append({"step": name, "equal": bool(equal), "exact_equal": bool(exact),
                                 "tokens": new.cost.tokens, "wall_s": time.perf_counter() - t0,
                                 "calls": llm.log})
            out["equal"] += int(equal)
        except Exception as e:                                   # a step that cannot be re-executed counts as unequal
            out["steps"].append({"step": name, "equal": False, "error": f"{type(e).__name__}: {e}"[:200]})
        finally:
            scratch.close()
    return out


def run_cell(spec: dict, models: Models) -> dict:
    t_cell = time.perf_counter()
    fl, model = spec["flags"], spec["model"]
    scripted = model == "scripted"
    cache_off = flag(fl, "cache") == "off"
    case = build_case(spec["workload"], spec["repeat"], spec["params"])
    row = {"spec": spec, "key": cell_key(spec), "cache": flag(fl, "cache")}
    inner = models.inner if not scripted else case.scripted()
    real = not scripted
    live = real and cache_off                     # no prompt cache, no in-run call memo, no checker memo
    is_w2 = spec["workload"] == "w2"
    db0 = getattr(inner, "db", None) if real else None

    def new_llm():
        return None if is_w2 else RecordingLLM(inner, dedupe=not live)

    try:
        if real:
            inner.style = fl["style"]
            inner.layout, inner.K, inner.margin, inner.tie = (flag(fl, "layout"), flag(fl, "replay_K"),
                                                              flag(fl, "margin"), flag(fl, "tie"))
            inner.replay, inner.drafts, inner.history = "off", {}, {}
            if live:
                inner.db = None
        wf, world = case.workflow, case.world
        kw = dict(effects=fl["effects"], depth=depth_of(fl["depth"]),
                  cutoff=None if fl["cutoff"] == "exact" else fl["cutoff"])
        aud_policy = fl["audit"]
        trust = aud_policy in AU.TRUST_POLICIES
        if aud_policy.endswith("+gate") and fl["effects"] != "diff+outbox":
            raise ValueError("audit policy m3t+m4+gate needs effects=diff+outbox")
        if flag(fl, "release") == "on" and not aud_policy.endswith("+gate"):
            raise ValueError("release=on needs an audit policy with +gate")
        if trust:
            kw["untrusted"] = list(case.meta.get("untrusted", ()))
        if real and aud_policy != "none":
            models.auditor_checker(flag(fl, "checker")).live = live

        def auditor(trace=None):
            if aud_policy == "none" or not real and is_w2:
                return None
            return AU.Auditor(models.auditor_checker(flag(fl, "checker")) if real else AU.RuleChecker(), aud_policy,
                              **_audit_args(aud_policy, scripted, trace, spec["repeat"], flag(fl, "audit_p"),
                                            case.meta.get("untrusted", ()), flag(fl, "checker")))

        # --- the run with the fault (or the clean run)
        fault, overrides, clean_ref = None, None, None
        if is_w2:
            clean_ref = execute(wf, case.meta["dag"].world(), None, **kw)
            fault = choose_w2_fault(case, clean_ref)
            row["fault_at"] = [fault.step, fault.name]
        elif spec["fault"] != "clean":
            fault = case.faults[spec["fault"]]
        if fault:
            overrides = {(fault.step, wf.path(fault.name)): fault.value}
        trace_for_audit = None
        if "m4" in aud_policy:                        # cone costs come from a clean reference run
            ref = build_case(spec["workload"], spec["repeat"], spec["params"])
            try:
                t0, tot0 = time.perf_counter(), _totals(inner)
                ref_llm = RecordingLLM(inner, dedupe=not live)
                trace_for_audit = execute(ref.workflow, ref.world, ref_llm, **kw)
                d = _delta(tot0, _totals(inner))
                row["audit_ref"] = {"wall_s": time.perf_counter() - t0, "llm": d, "cache_hits": int(d.get("cache_hits", 0)),
                                    "memo_hits": ref_llm.hits}
            finally:
                ref.close()
        if real and fl["replay"] != "off":
            inner.db = None                           # the faulty run must decode, so its tokens exist as drafts
        meter, mc = Meter(), MeteredCutoff(kw["cutoff"])
        llm_f = new_llm()
        c0 = _checker_counts(models, real)
        t0, tot0 = time.perf_counter(), _totals(inner)
        faulty = execute(wf, world, llm_f, overrides=overrides, audit=auditor(trace_for_audit), meter=meter,
                         **{**kw, "cutoff": mc})
        c1 = _checker_counts(models, real)
        d = _delta(tot0, _totals(inner))
        row["faulty"] = {"wall_s": time.perf_counter() - t0, "tokens": sum(r.cost.tokens for r in faulty.records),
                         "llm_calls": sum(r.cost.llm_calls for r in faulty.records), "ok": _ok(case, world, 0),
                         "llm": d, "steps": [r.step for r in faulty.records], "cache_hits": int(d.get("cache_hits", 0)),
                         "memo_hits": llm_f.hits if llm_f else 0, "bookkeeping": _bookkeeping(meter, mc),
                         "audit_checker": {"live_evals": c1[0] - c0[0], "memo_hits": c1[1] - c0[1]}}
        if hasattr(world, "unauthorized"):            # measured on the faulty run itself, before any recovery
            row["faulty"]["unauthorized"] = world.unauthorized()
        if fault is None:
            row["clean"] = {k: row["faulty"][k] for k in ("wall_s", "tokens", "llm_calls", "ok", "llm", "cache_hits",
                                                           "memo_hits", "bookkeeping")}
        if aud_policy != "none" and faulty.audits:
            row["faulty"]["audit"] = AU.Auditor.summary(faulty)
        if fault:
            path = wf.path(fault.name)
            cone = affected_steps(faulty, path)
            row["propagation"] = {"cone": sorted(cone), "read_downstream": bool(cone),
                                  "effects_reached": sorted({e.tool for r in faulty.records if r.step in cone
                                                             for e in r.intents if e.cls != "read"}),
                                  "outcome_changed": not row["faulty"]["ok"] if not is_w2 else None}
            if aud_policy != "none" and faulty.audits:
                fl_nodes = row["faulty"]["audit"]["flagged"]
                row["detected"] = any(n.split("/")[0] in cone | {fault.step} for n in fl_nodes)
                row["fp_flags"] = sum(1 for n in fl_nodes if n.split("/")[0] not in cone | {fault.step})
        elif aud_policy != "none" and faulty.audits:
            row["fp_flags"] = len(row["faulty"]["audit"]["flagged"])
        row["irreversible_calls_faulty"] = n_irreversible(world)
        held_f = list(faulty.report["held_steps"])
        if trust:
            row["unauthorized_faulty"] = unauthorized_acts(case, world)
            row["faulty"]["gate"] = {"held": len(held_f), "held_steps": held_f, "effects": faulty.report["effects"]}

        # --- recovery
        if fault and spec["method"] != "none":
            dirty = {fault.step} if flag(fl, "dirty") == "origin" else producers(faulty, {wf.path(fault.name)})
            if real:
                inner.db = None                       # recovery decodes live: honest tokens and time
                inner.replay, inner.drafts, inner.history = fl["replay"], {k: list(v) for k, v in inner.history.items()}, {}
            mark = case.mark(world)
            meter_r, mc_r = Meter(), MeteredCutoff(mc.base)
            llm_r = new_llm()
            c0 = _checker_counts(models, real)
            t0, tot0 = time.perf_counter(), _totals(inner)
            rec = execute(wf, world, llm_r, prior=faulty, dirty=dirty, policy=spec["method"],
                          audit=auditor(faulty) if aud_policy != "none" else None, cutoff=mc_r, meter=meter_r,
                          untrusted=kw.get("untrusted", ()), revert_early=flag(fl, "revert") == "on",
                          release_gate=flag(fl, "release") == "on")
            wall = time.perf_counter() - t0
            c1 = _checker_counts(models, real)
            d = _delta(tot0, _totals(inner))
            rep = rec.report
            dyn = dynamic_cone(faulty, rec)
            row["propagation"].update(
                cone_static=row["propagation"]["cone"], cone_dynamic=sorted(dyn["cone"] | {fault.step}),
                new_steps=sorted(dyn["new"]), dropped_steps=sorted(dyn["dropped"]),
                rerun_class=explain_reruns(faulty, rec, {fault.step}, cone))
            if clean_ref is not None:
                row["propagation"]["cone_dynamic_ref"] = sorted(dynamic_cone(faulty, clean_ref)["cone"] | {fault.step})
            row["recovery"] = {
                "wall_s": wall, "tokens": rep["tokens"], "llm_calls": rep["llm_calls"],
                "rerun": rep["rerun"], "reused": rep["reused"], "n_rerun": len(rep["rerun"]),
                "irreversible_redone": len(rep["irreversible_redone"]), "effects": rep["effects"],
                "compensated": rep["compensated"], "held": rep["held"], "latency_s": rep["latency_s"],
                "ok_final": _ok(case, world, 0), "ok_recovered": _ok(case, world, mark),
                "llm": d, "cache_hits": int(d.get("cache_hits", 0)), "memo_hits": llm_r.hits if llm_r else 0,
                "calls": llm_r.log if llm_r else [], "bookkeeping": _bookkeeping(meter_r, mc_r),
                "audit_checker": {"live_evals": c1[0] - c0[0], "memo_hits": c1[1] - c0[1]},
                "values_match": (rec.values() == clean_ref.values()) if clean_ref is not None else None}
            if aud_policy != "none" and rec.audits:
                row["recovery"]["audit"] = AU.Auditor.summary(rec)
            if trust:
                rel = sum(1 for r in rec.last().values() for i in r.intents
                          if i.cls == "irreversible" and i.released and r.step in held_f and not r.reused)
                row["recovery"]["gate"] = {"held": len(rep["held"]), "held_steps": list(rep["held_steps"]),
                                           "released_after_recovery": rel, "leaked": rep["effects"]["leaked"],
                                           "withdrawn": rep["effects"]["withdrawn"], "effects": rep["effects"]}
                if flag(fl, "release") == "on":      # only with the release rule, so rows of other cells keep their keys
                    row["recovery"]["gate"]["released_at_commit"] = rep["effects"]["released"]
                    row["recovery"]["gate"]["residual_flags"] = [n for n in rec.report["audit"]["flagged"]
                                                                 if n.split("/")[0] not in rep["rerun"]]
            row["irreversible_calls_total"] = n_irreversible(world)
            if real:
                inner.replay, inner.drafts = "off", {}
                inner.db = None                       # checks below decode live too
            if not is_w2:
                if flag(fl, "reuse_check") == "on":
                    row["recovery"]["reuse_check"] = reuse_check(spec, rec, wf, inner, real)
                ref = clean_reference(spec, models, inner, live)
                row["clean_ref"] = {k: ref[k] for k in ("source", "wall_s", "tokens", "llm", "cache_hits", "memo_hits", "ok")}
                row["recovery"]["values_match"] = _outcome(rec.values()) == ref["outcome"]
        row["duplicates"] = world.duplicates_irreversible()
        if is_w2:
            row["unauthorized"] = None                # W2 has no authorization policy: its emails are not violations
        elif trust:
            row["unauthorized"] = unauthorized_acts(case, world)       # workload-aware (W4 posts, W3 files)
        elif hasattr(world, "unauthorized"):
            row["unauthorized"] = world.unauthorized()
        if trust:
            g = row.get("recovery", {}).get("gate", {})
            row["gate"] = {"held": len(held_f), "held_final": g.get("held", len(held_f)),
                           "released_after_recovery": g.get("released_after_recovery", 0),
                           "leaked": g.get("leaked", 0), "unauthorized": row["unauthorized"],
                           "unauthorized_faulty": row["unauthorized_faulty"]}
    finally:
        if real:
            inner.db = db0
        case.close()
    row["wall_s"] = time.perf_counter() - t_cell
    row["peak_rss_mb"] = rss_mb()
    return row
