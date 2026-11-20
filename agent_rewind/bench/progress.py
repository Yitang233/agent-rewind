"""Result rows, progress file and the hours estimate."""
from __future__ import annotations

import json
import os
import time

from .grid import expand


def load_rows(path: str) -> dict[str, dict]:
    rows = {}
    if os.path.exists(path):
        for ln in open(path):
            if ln.strip():
                try:
                    r = json.loads(ln)
                    rows[r["key"]] = r
                except ValueError:
                    pass
    return rows


def write_progress(path: str, done: int, total: int, t_start: float, rows_new: list[dict], pending: list[dict]) -> dict:
    elapsed = time.perf_counter() - t_start
    per = {}
    for r in rows_new:
        per.setdefault(r["spec"]["model"], []).append(r["wall_s"])
    eta = sum(sum(per.get(c["model"], [x for v in per.values() for x in v] or [0])) /
              max(1, len(per.get(c["model"], [x for v in per.values() for x in v] or [0]))) for c in pending)
    prog = {"done": done, "total": total, "elapsed_s": round(elapsed, 1), "eta_s": round(eta, 1),
            "eta_h": round(eta / 3600, 2), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = path + ".tmp"
    json.dump(prog, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
    return prog


def estimate(cfg: dict, measured_path: str) -> dict:
    """Hours for the config from measured cell times: mean wall by (model, workload, method, replay), with
    fallbacks to coarser keys. Scripted cells cost ~0."""
    rows = list(load_rows(measured_path).values())
    keys = [lambda s: (s["model"], s["workload"], s["method"], s["flags"]["replay"], s["flags"]["audit"]),
            lambda s: (s["model"], s["workload"], s["method"]), lambda s: (s["model"], s["workload"]),
            lambda s: (s["model"],)]
    tables = []
    for kf in keys:
        t = {}
        for r in rows:
            t.setdefault(kf(r["spec"]), []).append(r["wall_s"])
        tables.append({k: sum(v) / len(v) for k, v in t.items()})
    cells = expand(cfg)
    total, by_model, unknown = {}, {}, 0
    for c in cells:
        if c["model"] == "scripted":
            continue
        for kf, t in zip(keys, tables):
            if kf(c) in t:
                by_model[c["model"]] = by_model.get(c["model"], 0) + t[kf(c)]
                break
        else:
            unknown += 1
    total = {m: round(s / 3600, 2) for m, s in by_model.items()}
    return {"cells": len(cells), "real_cells": sum(c["model"] != "scripted" for c in cells), "hours_by_model": total,
            "hours": round(sum(total.values()), 2), "unestimated_cells": unknown}
