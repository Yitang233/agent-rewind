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
