"""Resumable grid runner for every experiment (E1-E8).

    python -m agent_rewind.bench --config bench/configs/full.json [--dry-run] [--max-hours H]
                                 [--only E5,E8] [--limit N] [--estimate runs/S6/measure/cells.jsonl]

A config is {"name", "max_hours", "experiments": [...]}. An experiment is a grid
    exp        label written into every row (E1..E8)
    workloads  "w1" | "w2" | "w3" | "w4.ipi" | "w4.byz", or {"name", "params", "grid", "faults"}:
               `params` go to the builder, `grid` is a cartesian product over more params,
               `faults` overrides the experiment's list for that workload
    faults     names from the workload's fault table, "*" (all of them), "clean", or "auto" (w2: a seeded
               fault at the workload's `position`)
    methods    none | restart | rollback | selective | selective-d2   (none = faulty run only)
    flags      {flag: [values]} (cartesian), over DEFAULTS below
    variants   alternative to methods x flags: [{"label", "method", "flags"}] (E8's named systems)
    models     scripted | 1.5b | 7b  (one model is loaded per pass, in the order of first appearance)
    repeats    r = repeat_from..repeats-1 is the workload variant / seed (`repeat_from` default 0; 2 = held-out split)
A cell is one (experiment, workload, params, fault, method, flags, model, repeat). Its key is the sorted
JSON of that spec; finished keys in runs/<name>/cells.jsonl are skipped, so the run resumes.
runs/<name>/progress.json holds done, total, eta. --dry-run runs every cell with the scripted model in
runs/<name>_dry/. Flags:
    style   atoms | cot            prompt layout (M2)
    replay  off | prefix | resync  speculative replay of the recovery's model calls (M1)
    audit   none | always | uniform | m3 | m3+cache | m3+m4   (M3/M4; model workloads with atoms)
            m3t | m3t+cache | m3t+m4 | m3t+m4+gate            (trust-aware; declared untrusted sources, runs/v2/D4/calibration.json;
                                                              the gate needs effects=diff+outbox)
    checker llama | minicheck      which model scores audits (minicheck: runs/v2/D5/calibration.json; absent unless set)
    cutoff  exact | canonical      effects  naive | diff | diff+outbox      depth  agent|step|artifact|field
v3 flags (absent from a cell's spec unless a config sets them):
    revert  off | on       compensate a later step's faulty-run effects before a rerun step reads the world (engine revert_early)
    release off | on       recovery with an audit +gate: a flag on a step the recovery reused no longer holds an irreversible
                           intent (audit.Auditor.gate(recovered=...)); rows then carry recovery.gate.released_at_commit, residual_flags
Hygiene flags. They are absent from a cell's spec unless a config sets them, so v1 cell keys are unchanged:
    cache        on | off        off: every phase (clean reference, faulty, recovery, audits) decodes live: no SQLite
                                 prompt cache, no in-run call memo, no checker memo. Default on.
    reuse_check  off | on        recovery cells of model workloads: re-run up to 2 reused LLM steps live in a scratch
                                 branch and compare them canonically with the reused output. Default off.
    audit_p      float           probability of the `uniform` audit policy. Default 1.0 (v1: uniform == m3).
Dynamic-cone flag:
    dirty        producers | origin   which steps start dirty in recovery: every step that wrote the faulty path
                                 (v1; on a shared key that includes innocent shadow writers) or only the fault step.
Recovery rows also carry propagation.cone_static (the path-overlap cone), cone_dynamic (the engine's closure over
recorded read hashes vs the recovered run; w2 also cone_dynamic_ref vs a clean reference), new_steps and rerun_class.
Each phase of a row records `cache_hits` (SQLite hits), `memo_hits` (in-run call memo) and `bookkeeping`
({ns_read, ns_write, ns_hash, ns_hash_outside, n_read, n_write, n_hash, journal_bytes, steps}). `recovery.calls` lists
(step, call_index, sha256 of the output text, n_tokens) per model call. `recovery.values_match` compares the
recovered outcome paths (everything outside /msg and /audit) canonically with a live clean reference run
(`clean_ref`, memoised in <out>/clean_refs.jsonl per workload, params, repeat, style, model).
"""
from __future__ import annotations

from ..deps import Meter  # noqa: F401  (re-exported: tests and scripts read bench.Meter)
from .cases import _ok, affected_steps, build_case, choose_w2_fault, n_irreversible, spec_is_dag, unauthorized_acts
from .cell import clean_reference, reuse_check, run_cell
from .cli import main
from .config import CHOICES, DEFAULTS, METHODS, MODEL_KEYS, NEW_DEFAULTS, ROOT, _check_flags, flag
from .grid import _entries, _fault_names, _flag_sets, build_case_names, cell_key, expand
from .metrics import MeteredCutoff, RecordingLLM, _bookkeeping, _checker_counts, _delta, _outcome, _sha, _totals, rss_mb
from .models import Models, _audit_args, _calibration, calibration_d4
from .progress import estimate, load_rows, write_progress

__all__ = ['CHOICES', 'DEFAULTS', 'METHODS', 'MODEL_KEYS', 'Meter', 'MeteredCutoff', 'Models', 'NEW_DEFAULTS', 'ROOT', 'RecordingLLM', '_audit_args', '_bookkeeping', '_calibration', '_check_flags', '_checker_counts', '_delta', '_entries', '_fault_names', '_flag_sets', '_ok', '_outcome', '_sha', '_totals', 'affected_steps', 'build_case', 'build_case_names', 'calibration_d4', 'cell_key', 'choose_w2_fault', 'clean_reference', 'estimate', 'expand', 'flag', 'load_rows', 'main', 'n_irreversible', 'reuse_check', 'rss_mb', 'run_cell', 'spec_is_dag', 'unauthorized_acts', 'write_progress']
