"""Grid expansion: a config -> the list of cell specs and their keys."""
from __future__ import annotations

import itertools
import json

from .cases import build_case
from .config import DEFAULTS, _check_flags


def _entries(cfg_exp: dict):
    for w in cfg_exp["workloads"]:
        w = {"name": w} if isinstance(w, str) else dict(w)
        grid = w.get("grid", {})
        keys = sorted(grid)
        for combo in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
            yield w["name"], {**w.get("params", {}), **dict(zip(keys, combo))}, w.get("faults")


def _flag_sets(exp: dict) -> list[tuple[str, str, dict]]:
    if "variants" in exp:
        return [(v["label"], v["method"], _check_flags({**DEFAULTS, **v.get("flags", {})})) for v in exp["variants"]]
    grid = exp.get("flags", {})
    keys = sorted(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
        fl = _check_flags({**DEFAULTS, **dict(zip(keys, combo))})
        for m in exp.get("methods", ["selective"]):
            out.append((m, m, fl))
    return out


def cell_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def expand(cfg: dict, dry: bool = False, only: set[str] | None = None) -> list[dict]:
    cells, seen = [], set()
    for exp in cfg["experiments"]:
        if only and exp["exp"] not in only:
            continue
        for wname, params, wfaults in _entries(exp):
            faults = wfaults or exp.get("faults", ["*"])
            names = None
            for f in faults:
                if f == "*":
                    names = names or _fault_names(wname)
                fl = names if f == "*" else [f]
                for fault in fl:
                    for label, method, flags in _flag_sets(exp):
                        if fault == "clean" and method != "none" and "variants" not in exp:
                            continue
                        for model in exp.get("models", ["scripted"]):
                            model = "scripted" if dry else model
                            for r in range(exp.get("repeat_from", 0), exp.get("repeats", 1)):
                                spec = dict(exp=exp["exp"], workload=wname, params=params, fault=fault, label=label,
                                            method=("none" if fault == "clean" else method), flags=flags,
                                            model=model, repeat=r)
                                k = cell_key(spec)
                                if k not in seen:
                                    seen.add(k)
                                    cells.append(spec)
    order = []
    for c in cells:
        if c["model"] not in order:
            order.append(c["model"])
    return sorted(cells, key=lambda c: order.index(c["model"]))     # stable: one model per pass


def _fault_names(workload: str) -> list[str]:
    if workload == "w2":
        return ["auto"]
    return list(build_case_names(workload))


def build_case_names(workload: str) -> list[str]:
    c = build_case(workload, 0, {})
    try:
        return list(c.faults)
    finally:
        c.close()
