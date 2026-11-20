# Grid configs

Each file is read by `python -m agent_rewind.bench --config <file>` (flags and the config format: docstring of
[`agent_rewind/bench/__init__.py`](../../agent_rewind/bench/__init__.py)). Cells go to `runs/<name>/cells.jsonl` unless
`--out` says otherwise (`runs/` is local and not shared; the shared copies are in `results/`). Names follow history and are
quoted in `runs/*/command.txt` and the pre-registrations, so no file is renamed or moved.

| Config | Experiment | Cells | Default output | Shared copy in `results/` | Used by |
|---|---|---|---|---|---|
| `phase1.json` | Phase 1 E1-E4 (propagation, cone vs suffix, precision, effects) | 1,563 | `runs/phase1` | `phase1/cells.jsonl` | `scripts/analyze.py`, `scripts/phase1_tables.py`; Phase 1 report Tables 1-2 |
| `phase1_baselines.json` | Phase 1 E1b (restart, always-verify baselines) | 112 | `runs/phase1_baselines` | `phase1_baselines/cells.jsonl` | `scripts/phase1_tables.py`; Phase 1 report Table 3 |
| `full.json` | E5-E8 on the 7B model (Phase 1 pilot of the final grid) | 540 | `runs/full` | not shared | `scripts/analyze.py` (`FULL`) |
| `s6_measure.json` | S6 timing sample used to estimate `full.json` (`--estimate`) | - | `runs/S6/measure` | not shared | `--estimate` |
| `replay_tuned.json` | Not a grid: the replay settings frozen for v2 (`flags` spliced into E5v2/E8v2) | - | - | evidence in `v2/D3/` | `v2/preregistration.json` |
| `v2.json` | Phase 2 E3v2, E5v2, E6v2, E7v2, E8v2 (7B, cache off, held-out repeats) | 574 | `runs/v2` | `v2/cells.jsonl` | `scripts/check_cells.py`, `scripts/analyze_v2.py`; report via `analysis_v2.json` |
| `e2v2.json` | Phase 2 E2v2 (scripted W2: fault stage, fan-in, shared keys) | 900 | `runs/E2v2` (kept at `runs/v2/E2v2`) | `v2/E2v2/cells.jsonl` | `scripts/analyze_v2.py` (F1) (`fig_cone` of the report) |
| `v2_fix.json` | E8v2 W3 cells rerun after the harness fixes (PR #2) | 50 | `runs/v2_fix` | `v2/v2_fix/cells.jsonl` | `scripts/analyze_v3.py` (T5 reference) |
| `v3.json` | v3: always-verify arm, timeout faults, post-hoc `revert=on` and `release=on` (E8v3, E8v3f, E8v3g, E8v3r) | 113 | `runs/v3` | `v3/cells.jsonl` | `scripts/check_cells.py`, `scripts/analyze_v3.py` ; report via `analysis_v3.json` |
| `dev/calibrate_trust.json` | Calibration of the trust-aware audit (m3t), record-only | - | `runs/v2/D4/calib_cells` | `runs/v2/D4/calibration.json` (read by `bench.models.calibration_d4`) | pre-registration of v2 |
| `dev/calibrate_minicheck.json` | The same for the MiniCheck checker | - | `runs/v2/D5/calib_cells` | `runs/v2/D5/calibration.json` | pre-registration of v2 |

Cell counts are those of the README run table and `results/v2/preregistration.json`; `python -m agent_rewind.bench --config <file> --dry-run`
prints the count a config expands to. Report tables and figures come from the analysis files, see
[`results/README.md`](../../results/README.md) for the requirement-to-file table.
