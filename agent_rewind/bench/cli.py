"""Command line of the grid runner (`python -m agent_rewind.bench`)."""
from __future__ import annotations

import argparse
import json
import os
import time

from .cell import run_cell
from .config import ROOT
from .grid import cell_key, expand
from .metrics import rss_mb
from .models import Models
from .progress import estimate, load_rows, write_progress


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--estimate", default=None, help="cells.jsonl with measured real cells")
    ap.add_argument("--resume", action="store_true", help="accepted; finished cells are always skipped by key")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    only = {x for x in a.only.split(",") if x} or None
    if a.estimate:
        print(json.dumps(estimate(cfg, a.estimate), indent=1))
        return 0
    name = cfg["name"] + ("_dry" if a.dry_run else "")
    out_dir = a.out or os.path.join(ROOT, "runs", name)
    os.makedirs(out_dir, exist_ok=True)
    cells_path, prog_path = os.path.join(out_dir, "cells.jsonl"), os.path.join(out_dir, "progress.json")
    cells = expand(cfg, a.dry_run, only)
    done = load_rows(cells_path)
    pending = [c for c in cells if cell_key(c) not in done]
    if a.limit:
        pending = pending[:a.limit]
    max_h = a.max_hours if a.max_hours is not None else cfg.get("max_hours")
    t_start = time.perf_counter()
    deadline = t_start + max_h * 3600 if max_h else None
    models = Models(out_dir, a.dry_run)
    new_rows, errors = [], 0
    print(f"[bench] {cfg['name']}: {len(cells)} cells, {len(done)} done, {len(pending)} pending", flush=True)
    write_progress(prog_path, len(done), len(cells), t_start, new_rows, pending)
    with open(cells_path, "a") as f:
        for i, spec in enumerate(pending):
            if deadline and time.perf_counter() > deadline:
                print("[bench] --max-hours reached; stopping (resume to continue)", flush=True)
                break
            try:
                models.load(spec["model"])
                row = run_cell(spec, models)
            except Exception as e:          # a failing cell is recorded, not retried until its row is removed
                errors += 1
                models.reset()
                row = {"spec": spec, "key": cell_key(spec), "error": f"{type(e).__name__}: {e}"[:300], "wall_s": 0.0}
                if "another model process" in str(e):
                    print("[bench] blocked:", e, flush=True)
                    break
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            new_rows.append(row)
            rest = pending[i + 1:]
            prog = write_progress(prog_path, len(done) + len(new_rows), len(cells), t_start, new_rows, rest)
            if len(new_rows) % 25 == 0 or spec["model"] != "scripted":
                print(f"[bench] {prog['done']}/{prog['total']} eta {prog['eta_h']}h last={row['wall_s']:.1f}s", flush=True)
    models.free()
    print(f"[bench] finished: {len(new_rows)} new cells, {errors} errors, peak_rss_mb {rss_mb():.0f}", flush=True)
    return 0
