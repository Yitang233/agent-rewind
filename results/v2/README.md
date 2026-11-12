# v2 results

These are the raw cells behind the v2 experiments (`bench/configs/v2.json`, `e2v2.json`, `v2_fix.json`). `runs/` stays untracked, and this folder is a snapshot of it. Phase 1 lives in `results/phase1/` and `results/phase1_baselines/`; this folder does not change them.

To recompute the v2 tables from these files, point `scripts/analyze_v2.py` at them (it reads the `runs/v2` layout, so copy or link this folder to `runs/v2` first).

| File | Grid | Used for |
|---|---|---|
| `cells.jsonl` | `bench/configs/v2.json` (574 rows run, with the budget cuts listed in `preregistration.json`) | main v2 experiments (E5v2–E8v2) |
| `E2v2/cells.jsonl` | `bench/configs/e2v2.json` (900 cells) | E2v2 |
| `v2_fix/cells.jsonl` | `bench/configs/v2_fix.json` (50 cells) | rerun of E8v2 W3 (7B, atoms) after the undo-first and `-B` fixes |
| `v2_fix/compare.md` | comparison of `v2_fix` with the E8v2 W3 cells in `cells.jsonl` | what changed, per fault and arm |
| `RESULTS.md` | | verdicts H1-H7 with every number  |
| `figures/` | | `analyze_v2.py` output: F1-F8 (png + json), T1-T3, `analysis_v2.json` (copy of `runs/v2/figures/`) |
| `preregistration.json` | | hypotheses, config, disclosures and budget cuts written before the run, plus `rerun_after_fix` (appended 2026-11-04) |

## Notes

- **Not included.** `*.sqlite` prompt caches, `clean_refs.jsonl`, `patches/` and `grid/` stay in `runs/v2` and are not tracked.
- **Rerun cells.** The 50 cells in `v2_fix/` cover the same E8v2 W3 cells as in `cells.jsonl`. Whether the analysis replaces the old rows or reports both is not decided.
- **Open.** `stale_code` with the three selective arms still fails at repeat 0 after the fix; the cells store no unittest output, so the cause is not determined.
