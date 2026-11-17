"""Validate a v2 cells.jsonl before it is analysed.

    python scripts/check_cells.py runs/v2/<dir>/cells.jsonl --exp E5v2|E6v2|E7v2|E8v2|E3v2|E2v2
    python scripts/check_cells.py runs/v3/cells.jsonl --exp E8v3|E8v3f|E8v3r|E8v3g \\
           --ref runs/v2/cells.jsonl --ref runs/v2_fix/cells.jsonl     [--allow-scripted: runs/v3_dry only]

Only rows whose spec.exp equals --exp are checked (a file may hold several experiments).

Checks, for every row:
  * no error rows, and the required keys of the experiment are present;
  * every measured phase (clean, faulty, recovery, audit_ref, clean_ref) has `cache_hits` and `memo_hits`;
  * model (non-scripted) cells of E3v2, E5v2, E6v2, E7v2 and E8v2 carry flag cache=off, and every measured phase of
    a cache=off cell has 0 cache hits, 0 call-memo hits and 0 checker-memo hits;
  * recovery cells of model workloads have a clean_ref and a computed values_match;
  * each cell has the partner or reference its experiment needs:
      E5v2  replay=off partner for every replay=prefix|resync cell (same workload, params, fault, repeat, other flags)
      E6v2  both styles for every (workload, repeat)
      E7v2  a clean run with the same audit setting for every faulty cell; `detected` / `fp_flags` present
      E8v2  a restart cell in the same style for every other method
      E2v2  a restart cell for every other method
v3 experiments (rows of runs/v3 against the pre-registered v2 rows, matched on workload, params, fault, repeat, style):
      E8v3   restart reference (E8v2 file, W3: runs/v2_fix) for every cell; scripted rows are refused
      E8v3f  a restart cell of its own in the same file (all six arms are run together)
      E8v3r  an E8v2 `selective` partner (same label, flag revert off) in the --ref files, i.e. the v2_fix row
      E8v3g  fault cells: the E8v2 gate arm (flag release off); clean cells: the E7v2 clean cell of the same repeat
      A v3 cell of a model workload is refused when its model is `scripted` unless --allow-scripted is given.
Exit code 1 lists the failures (at most 60 are printed); exit code 0 prints a one-line summary.
"""
from __future__ import annotations

import argparse
import json
import sys

PHASES = ("clean", "faulty", "recovery", "audit_ref", "clean_ref")
V3 = ("E8v3", "E8v3f", "E8v3r", "E8v3g")
LIVE_EXPS = {"E3v2", "E5v2", "E6v2", "E7v2", "E8v2"} | set(V3)
BASE_KEYS = ("spec", "key", "faulty", "wall_s")
REQUIRED = {                  # (keys of the row, keys of row["recovery"] when a recovery ran)
    "E2v2": ((), ("n_rerun", "rerun", "reused", "values_match")),
    "E3v2": ((), ("wall_s",)),
    "E5v2": ((), ("calls", "wall_s", "llm", "tokens", "ok_recovered")),
    "E6v2": ((), ()),
    "E7v2": ((), ()),
    "E8v2": (("duplicates",), ("ok_recovered", "n_rerun", "wall_s", "tokens", "values_match", "calls")),
}
for _e in V3:
    REQUIRED[_e] = REQUIRED["E8v2"]
TUNED = ("layout", "replay_K", "margin", "tie")        # replay settings carried by the resync-tuned arm
HOUSEKEEPING = ("replay", "audit", "effects", "cutoff", "depth", "cache", "reuse_check", "audit_p", "revert", "release") + TUNED


def load(path: str) -> list[dict]:
    rows = []
    for i, ln in enumerate(open(path), 1):
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except ValueError:
                rows.append({"error": f"line {i}: not JSON", "spec": {}, "key": f"line{i}"})
    return rows


def group_key(spec: dict, drop=(), fault: bool = True, method: bool = False) -> str:
    """Identity of a cell up to the dropped flags (and optionally up to the fault and the method)."""
    fl = {k: v for k, v in spec.get("flags", {}).items() if k not in drop}
    d = {k: spec.get(k) for k in ("workload", "params", "repeat", "model")}
    if fault:
        d["fault"] = spec.get("fault")
    if method:
        d["method"] = spec.get("method")
    return json.dumps([d, fl], sort_keys=True)


def _rows_of(rows: list[dict], exp: str) -> list[dict]:
    names = {exp, exp.removesuffix("v2")}                 # the E2 file was written with exp "E2"
    return [r for r in rows if "error" in r or r.get("spec", {}).get("exp", exp) in names]


def ident(spec: dict) -> tuple:
    """Cell identity for the v3 partners: it leaves out the model, so a scripted dry run can be checked in shape."""
    return (spec.get("workload"), json.dumps(spec.get("params"), sort_keys=True), spec.get("fault"), spec.get("repeat"),
            spec.get("flags", {}).get("style"))


def check_v3(ok: list[dict], exp: str, refs: list[dict], allow_scripted: bool) -> list[str]:
    bad: list[str] = []

    def fail(r, msg):
        bad.append(f"{r.get('key', '?')[:110]}: {msg}")

    # W3 references come from the rerun after the teammates' fixes (runs/v2_fix); the W3 rows of runs/v2 predate them
    refs = [r for r in refs if "error" not in r and "spec" in r
            and (r["spec"].get("workload") != "w3" or "v2_fix" in r.get("_src", ""))]
    restart = {ident(r["spec"]) for r in refs if r["spec"].get("exp") == "E8v2" and r["spec"]["method"] == "restart"}
    for r in ok:
        s = r["spec"]
        if s.get("model") == "scripted" and not allow_scripted:
            fail(r, "scripted cell (a model run is expected; --allow-scripted is for runs/v3_dry)")
    if exp == "E8v3f":
        own = {ident(r["spec"]) for r in ok if r["spec"]["method"] == "restart"}
        for r in ok:
            if r["spec"]["method"] not in ("none", "restart") and ident(r["spec"]) not in own:
                fail(r, "no restart cell of its own")
    elif exp == "E8v3":
        for r in ok:
            if ident(r["spec"]) not in restart:
                fail(r, "no restart reference in --ref (E8v2 cell; W3: runs/v2_fix)")
    elif exp == "E8v3r":
        part = {(ident(r["spec"]), r["spec"]["label"]) for r in refs if r["spec"].get("exp") == "E8v2"
                and r["spec"]["method"] == "selective" and r["spec"]["flags"].get("revert", "off") == "off"}
        for r in ok:
            if r["spec"]["flags"].get("revert") != "on":
                fail(r, "E8v3r cell without revert=on")
            if (ident(r["spec"]), r["spec"]["label"]) not in part:
                fail(r, "no E8v2 partner with the same label (the v2_fix row) in --ref")
            if ident(r["spec"]) not in restart:
                fail(r, "no restart reference in --ref")
    elif exp == "E8v3g":
        gate = {ident(r["spec"]) for r in refs if r["spec"].get("exp") == "E8v2" and r["spec"]["flags"].get("audit") == "m3t+m4+gate"
                and r["spec"]["flags"].get("release", "off") == "off"}
        clean = {ident(r["spec"]) for r in refs if r["spec"].get("exp") == "E7v2" and r["spec"]["fault"] == "clean"
                 and r["spec"]["flags"].get("audit") == "m3t+m4+gate"}
        for r in ok:
            s = r["spec"]
            if s["flags"].get("release") != "on":
                fail(r, "E8v3g cell without release=on")
            if ident(s) not in (clean if s["fault"] == "clean" else gate):
                fail(r, "no E8v2 gate cell (fault) / E7v2 clean cell (clean) to compare with in --ref")
            if s["fault"] != "clean":
                if ident(s) not in restart:
                    fail(r, "no restart reference in --ref")
                g = (r.get("recovery") or {}).get("gate", {})
                for k in ("released_at_commit", "residual_flags"):
                    if k not in g:
                        fail(r, f"recovery.gate.{k} missing")
    return bad


def check(rows: list[dict], exp: str, refs: list[dict] | None = None, allow_scripted: bool = False) -> list[str]:
    bad: list[str] = []
    rows = _rows_of(rows, exp)

    def fail(r, msg):
        bad.append(f"{r.get('key', '?')[:110]}: {msg}")

    if not rows:
        return ["no rows"]
    row_keys, rec_keys = REQUIRED[exp]
    for r in rows:
        if "error" in r:
            fail(r, "error row: " + str(r["error"])[:120])
            continue
        spec = r["spec"]
        fl, model = spec.get("flags", {}), spec.get("model")
        for k in BASE_KEYS + row_keys:
            if k not in r:
                fail(r, f"missing key {k}")
        if "faulty" not in r:
            continue
        if spec.get("fault") == "clean" and "clean" not in r:
            fail(r, "clean cell without a `clean` phase")
        if spec.get("method", "none") != "none" and "recovery" not in r:
            fail(r, "recovery method but no recovery phase")
        live = fl.get("cache", "on") == "off"
        if exp in LIVE_EXPS and model != "scripted" and not live:
            fail(r, "model cell without flag cache=off")
        for ph in PHASES:
            p = r.get(ph)
            if p is None:
                continue
            for k in ("cache_hits", "memo_hits"):
                if k not in p:
                    fail(r, f"{ph}: missing {k}")
            if live and model != "scripted":
                if p.get("cache_hits"):
                    fail(r, f"{ph}: cache_hits={p['cache_hits']} with cache=off")
                if p.get("memo_hits"):
                    fail(r, f"{ph}: memo_hits={p['memo_hits']} with cache=off")
                if (p.get("audit_checker") or {}).get("memo_hits"):
                    fail(r, f"{ph}: checker memo hits with cache=off")
        if live and model != "scripted" and (r.get("audit_ref") or {}).get("cache_hits"):
            fail(r, "audit_ref cache hits with cache=off")
        if "bookkeeping" not in r["faulty"]:
            fail(r, "faulty: missing bookkeeping")
        rec = r.get("recovery")
        if rec is not None:
            for k in rec_keys:
                if k not in rec:
                    fail(r, f"recovery: missing {k}")
            if "bookkeeping" not in rec:
                fail(r, "recovery: missing bookkeeping")
            if spec.get("workload") != "w2":
                if rec.get("values_match") is None:
                    fail(r, "recovery: values_match not computed")
                if "clean_ref" not in r:
                    fail(r, "no clean_ref")
                if fl.get("reuse_check") == "on" and "reuse_check" not in rec:
                    fail(r, "reuse_check=on but no reuse_check record")

    ok = [r for r in rows if "error" not in r and "faulty" in r]
    if exp == "E5v2":
        groups: dict[str, set] = {}
        for r in ok:
            groups.setdefault(group_key(r["spec"], ("replay", "cache") + TUNED), set()).add(r["spec"]["flags"].get("replay", "off"))
        for k, reps in groups.items():
            if "off" not in reps or len(reps) < 2:
                bad.append(f"no replay pair (have {sorted(reps)}): {k[:110]}")
    elif exp == "E6v2":
        groups = {}
        for r in ok:
            groups.setdefault(group_key(r["spec"], ("style",)), set()).add(r["spec"]["flags"].get("style"))
        for k, st in groups.items():
            if st != {"atoms", "cot"}:
                bad.append(f"needs both styles (have {sorted(map(str, st))}): {k[:110]}")
    elif exp == "E7v2":
        clean = {group_key(r["spec"], fault=False) for r in ok if r["spec"]["fault"] == "clean"}
        for r in ok:
            s = r["spec"]
            if s["flags"].get("audit", "none") == "none":
                fail(r, "cell without an audit policy")
            if s["fault"] == "clean":
                if "fp_flags" not in r:
                    fail(r, "clean cell without fp_flags")
            else:
                if "detected" not in r:
                    fail(r, "faulty cell without `detected`")
                if group_key(s, fault=False) not in clean:
                    fail(r, "no clean run with the same audit setting")
    elif exp in ("E8v2", "E2v2"):
        drop = HOUSEKEEPING if exp == "E8v2" else ()
        have: dict[str, set] = {}
        for r in ok:
            have.setdefault(group_key(r["spec"], drop), set()).add(r["spec"]["method"])
        for r in ok:
            s = r["spec"]
            if s["method"] not in ("none", "restart") and "restart" not in have[group_key(s, drop)]:
                fail(r, "no restart reference" + (" in the same style" if exp == "E8v2" else ""))
    elif exp == "E3v2":
        for r in ok:
            if "depth" not in r["spec"]["flags"]:
                fail(r, "no depth flag")
    elif exp in V3:
        bad += check_v3(ok, exp, refs or [], allow_scripted)
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells")
    ap.add_argument("--exp", required=True, choices=sorted(REQUIRED))
    ap.add_argument("--ref", action="append", default=[], help="v2 cells file with the reference cells (E8v3*), repeatable")
    ap.add_argument("--allow-scripted", action="store_true", help="accept scripted rows of an E8v3* file (dry run)")
    a = ap.parse_args(argv)
    rows = load(a.cells)
    refs = [dict(r, _src=f) for f in a.ref for r in load(f)]
    if a.exp in V3 and not refs:
        print(f"[check_cells] {a.exp}: needs --ref (E8v2 cells; W3: runs/v2_fix)")
        return 1
    bad = check(rows, a.exp, refs, a.allow_scripted)
    if bad:
        print(f"[check_cells] {a.exp}: {len(bad)} failure(s) in {len(rows)} rows")
        for b in bad[:60]:
            print("  FAIL", b)
        return 1
    print(f"[check_cells] {a.exp}: {len(_rows_of(rows, a.exp))} rows ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
