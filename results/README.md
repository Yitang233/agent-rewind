# Results

Phase 2 (the code and results graded in the final report) is first; the Phase 1 snapshot follows.

## Phase 2: requirement-to-evidence table

Every number in `report/main.pdf` is in one of these files. Setup for the commands (the analysis scripts read `runs/`, which is not shared):

```bash
mkdir -p runs && cp -R results/v2 runs/v2 && cp -R results/v2/v2_fix runs/v2_fix && cp -R results/v3 runs/v3
cp -R results/phase1 runs/phase1 && cp -R results/phase1_baselines runs/phase1_baselines
python scripts/analyze_v2.py          # rewrites runs/v2/figures (T1-T3, F1-F8, analysis_v2.json)
python scripts/analyze_v3.py runs/v3  # rewrites runs/v3/figures (T1v3, T4-T7, F9, analysis_v3.json) and report/generated/v3.tex
python scripts/make_report_figures.py # report/figures/fig_*.pdf and .json
python -m unittest -q                 # 169 tests; 7 model-dependent ones skip without a model file
```

| Option 5 requirement or metric | Evidence (shared copy) | Result | Recomputed by |
|---|---|---|---|
| One focused mechanism | `agent_rewind/engine.py`, `store.py`, `deps.py`; report section 3 | selective recovery by verifying read hashes | unit tests |
| Every method gets the faulty result's identity | `agent_rewind/bench/` (`cell.py`) derives the dirty set from the injected fault for every method (`dirty` flag); the same fault is injected in every method's cell | met | `scripts/check_cells.py` |
| Full restart under the same fault schedule and budget | `v2/figures/T1_end_to_end.md`, `v3/figures/T1v3_end_to_end.md`, report Fig. 3 and Table 2 | restart 38.9 / 9.4 / 13.7 / 13.5 s against 9.3 / 6.8 / 6.8 / 3.8 s (W1 / W3 / W4.ipi / W4.byz) | `analyze_v2.py`, `analyze_v3.py` |
| Verification (always-verify) under the same schedule | `v3/figures/T1v3_end_to_end.md` (arm `always-verify`) | 27/30 correct, same as selective; 1.5-2.7x the wall time of selective + replay | `analyze_v3.py` |
| Vary the fault stage | scripted sweep `v2/E2v2/cells.jsonl`, `v2/figures/F1_cone_vs_suffix.*`; live: `v3/figures/T7_fault_stage.md` | restart 22.5, rollback 12.2, selective 4.7 of 24 steps; W1 selective reruns 6.0 / 9.2 / 3.4 of 14 by faulted step | `analyze_v2.py`, `analyze_v3.py` |
| Fault kinds | wrong, stale, injected, Byzantine in `v2/cells.jsonl`; timeout in `v3/figures/T4_faults.md` | planner faults do not exist in the workloads (stated in the report) | `analyze_v3.py` |
| Shared state | W1 shared team keys; scripted `shared_keys` panels of `v2/figures/F1_cone_vs_suffix.*` | covered | `analyze_v2.py` |
| Non-LLM state (files) | W3 git workspace; `v2/v2_fix/cells.jsonl`, `v3/figures/T5_stale_code.md` | restart and rollback 10/10, selective 9/10; 15/15 with the post-hoc `revert=on` fix. No browser or VM snapshot | `analyze_v3.py` |
| Irreversible actions | `v3/figures/T1v3_end_to_end.md` (unauth cells, leaked, dup), `v3/figures/T6_gate.md`, report Fig. 3b,c | unauthorized 0 of 30 with the gate, 5 without, 7 with always-verify; the gate holds good work (W4.ipi 2/5; 4/5 with the post-hoc release rule) | `analyze_v3.py` |
| Correctness | `ok_recovered` in T1v3, report Fig. 3a | within one cell of restart except the gate arm on W4.ipi | `analyze_v3.py` |
| Recomputed stages, retained work | `steps rerun` in T1v3, `retained` in each cell's `recovery.effects` | selective reruns 2.0-6.3 steps against 5-14 for restart | `analyze_v3.py` |
| Latency | recovery wall in T1v3 | see above | `analyze_v3.py` |
| Inference cost | `inference tok` column of T1v3 (decoded + prefill + audit tokens), `v2/figures/F4_replay.*` | replay decodes 19-64% of a plain rerun's tokens, 81/81 identical | `analyze_v2.py` |
| Tracking overhead | `v2/figures/F2_bookkeeping.*`, `RESULTS.md` section H6 | 0.005% of recovery wall (max 0.057%) | `analyze_v2.py` |
| Reuse validity | `reuse_valid` in T1v3, `RESULTS.md` section H5 | 60/60 sampled decisions equal a scratch rerun; no W3 samples | `analyze_v2.py` |
| Detection false positives and misses (optional) | `v2/figures/T2_audits.md`, `F6_audit_frontier.*` | trust-aware audits detect 8/15 against 15/15 for always-verify; negative result | `analyze_v2.py` |

Verdict text for every pre-registered claim: [`v2/RESULTS.md`](v2/RESULTS.md) and [`v3/preregistration.md`](v3/preregistration.md); `v3/README.md` lists the v3 files. Post-hoc changes (`revert=on`, `release=on`, replay margin 0.1) are marked as such.

---

# Phase 1 results and logs

These are the raw cells and logs behind Tables 1–3 of `report/phase1.tex`. They were produced on an Apple M5 (32 GB) with Qwen2.5-7B-Instruct Q4_K_M; model hashes and speeds are in `models.json`. `runs/` stays untracked, and this folder is a snapshot of it.

To recompute every number in the three tables from these files, run:

```bash
python scripts/phase1_tables.py
```

| File | Grid | Used for |
|---|---|---|
| `phase1/cells.jsonl` | `bench/configs/phase1.json` (1,563 cells; 54 use the 7B model) | Table 1 (clean runs) and Table 2 (propagation) |
| `phase1/w4byz_clean_uncached.json` | three W4.byz clean runs with the prompt cache off | Table 1, W4.byz tokens |
| `phase1_baselines/cells.jsonl` | `bench/configs/phase1_baselines.json` (112 cells) | Table 3 and the unauthorised column of Table 2 |
| `phase1_baselines/cells.before_undo_fix.jsonl` | same grid, before the restart undo-order fix | W3 restart 0/6 before the fix (observation 3) |
| `logs/*.log` | progress logs of the grid runs | |

## Notes

- **Reproduced numbers.** Every success rate, step count, LLM call count, cone size, irreversible-call count and unauthorised count in Tables 1–3 matches these files.
- **Machine-dependent numbers.** Wall time differs because the other rows in the PDF were measured on an M5 Pro. Token counts can differ by a few percent between runs, because they count computed tokens (prefill not covered by the KV prefix, plus decoded) and prefix reuse depends on which call ran before.
- **W4.byz clean runs.** In the grid, these were all prompt-cache hits, so Table 1 uses three uncached runs instead.
- **Rerun cells.** The 32 7B cells in `phase1/` that hit the prompt cache were rerun after the cache-hit token fix in `llm.py`; that resume run is not in `logs/`.
