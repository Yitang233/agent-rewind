# v3 results (run 2026-11-22, 113 cells, 0 error rows)

`bench/configs/v3.json` is the last model run of Phase 2 (113 cells, 7B, cache off, about 1.4 h). It adds the
always-verify arm and the timeout faults, and tests two post-hoc fixes. Until it has run this folder holds only this
file and the pre-registration. After the run these files are copied here from `runs/v3/`:

| File | What it is |
|---|---|
| `cells.jsonl` | every v3 cell (E8v3, E8v3f, E8v3r, E8v3g), 0 error rows |
| `figures/T1v3_end_to_end.md` | T1 with the always-verify arm: restart, rollback, always-verify, selective, +M1, full stack |
| `figures/T4_faults.md` | `timeout_team` (W1) and `timeout_item` (W4.ipi), six arms |
| `figures/T5_stale_code.md` | W3 `stale_code`: `v2_fix` rows and the `revert=on` rows (post-hoc engine fix), side by side |
| `figures/T6_gate.md` | W4.ipi gate: release off against `release=on` (post-hoc release rule), and the clean-run holds |
| `figures/T7_fault_stage.md` | reruns and recovery wall by the faulted step (existing cells, plus `timeout_team`) |
| `figures/F9_correctness.png`, `.json` | `ok_recovered` for the v3 additions |
| `figures/analysis_v3.json` | every number of the tables |
| `preregistration.md` | what v3 tests and the two post-hoc fixes, written before the run |

Reference cells: E8v2 in `../v2/cells.jsonl`; W3 in `../v2/v2_fix/cells.jsonl`; the clean cells of E7v2 in `../v2/cells.jsonl`.
Recompute: copy this folder to `runs/v3` and run `python scripts/analyze_v3.py runs/v3`.

Two fixes are **post hoc** (made after seeing v2 results) and are always shown next to the unfixed rows:
`revert=on` (engine: compensate a later step's faulty-run effects before a rerun step reads the world) and
`release=on` (gate: a flag on a step the recovery reused no longer holds an irreversible intent).
Both are off by default, so `runs/v2` stays reproducible by the code.
