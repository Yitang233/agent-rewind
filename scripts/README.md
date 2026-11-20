# Scripts

Run from the repository root inside the env: `python scripts/<name>.py`. The analysis scripts read `runs/` (local, not
shared; see the setup block of [`results/README.md`](../results/README.md) to copy `results/` into it).

## Analysis and checks (what a reader runs)

| Script | Input | Output |
|---|---|---|
| `check_cells.py` | one `cells.jsonl`, `--exp E5v2\|E6v2\|E7v2\|E8v2\|E3v2\|E2v2\|E8v3...`, optional `--ref` files | pass/fail list of cell checks (no error rows, cache flags, hit counters, partner cells) |
| `analyze.py` | `runs/phase1`, `runs/phase1_baselines`, `runs/full` | Phase 1 figures and tables in `runs/figures/`; also provides the helpers (`ms`, `wilson`, `by`, `PAL`) of the two scripts below |
| `analyze_v2.py` | `runs/v2/cells.jsonl`, `runs/v2/E2v2`, `runs/v2/D3`, `runs/phase1` | `runs/v2/figures/` (T1-T3, F1-F8, `analysis_v2.json`) |
| `analyze_v3.py [runs/v3]` | `runs/v3/cells.jsonl`, `runs/v2`, `runs/v2_fix` | `runs/v3/figures/` (T1v3, T4-T7, F9, `analysis_v3.json`) and `report/generated/v3.tex` |
| `make_report_figures.py` | `results/v2/figures`, `results/v3/figures`, `results/v2/cells.jsonl` | `report/figures/fig_*.pdf` and `.json` |
| `phase1_tables.py [--dir results]` | `results/phase1`, `results/phase1_baselines` | the numbers of Tables 1-3 of `report/phase1.tex` on stdout |
| `_common.py` | - | helpers shared by `analyze.py`, `analyze_v2.py` (`ROOT`, `rec_llm`); not run directly |

These file names are cited by the report, the README and `results/`; they do not change.

## `dev/`: one-off tools of the model runs (need a local model, not part of the results table)

| Script | What it did | Writes |
|---|---|---|
| `dev/audit_check.py` | S5: scoped, cached audits and their placement on W1 (7B); calibration of the audit thresholds | `runs/S5/` |
| `dev/replay_check.py` | S4: speculative replay against plain decoding on W1 recoveries | `runs/S4/` |
| `dev/replay_tune.py` | tuning of replay settings (record, replay, profile, tune, heldout, guard) | `runs/v2/D3/` |
| `dev/bench_llm.py` | micro-benchmark of llama-cpp-python (prefill, decode, KV crop) | optional JSON file |

The grid runner itself is a package, `python -m agent_rewind.bench` (see [`bench/configs/README.md`](../bench/configs/README.md)).
