# v3 pre-registration (written 2026-11-22, before `bench/configs/v3.json` is run)

Config: `bench/configs/v3.json`, 113 cells, Qwen2.5-7B Q4_K_M, greedy, `cache=off`, one run, output `runs/v3`.
Cells are matched to the pre-registered `runs/v2` cells on (workload, fault, repeat, style); W3 references are the
`runs/v2_fix` rows. Nothing is rerun or re-tuned after the results are seen; a failed cell stays in the tables.

| Experiment | Cells | Question | Rule written before the run |
|---|---|---|---|
| `E8v3` | 30 | How does an always-verify arm compare with restart and selective recovery end to end, on the six E8v2 fault cells? | Descriptive. The row is added to T1 with its audit tokens counted in inference cost. No verdict. |
| `E8v3f` | 60 | Do the timeout faults (`timeout_team` W1, `timeout_item` W4.ipi) behave like the other faults? | `ok_recovered` of each arm within one cell of restart (of 5) counts as the same as restart. Descriptive otherwise. |
| `E8v3r` | 15 | Is the W3 `stale_code` failure the engine ordering bug? (`revert=on`, **post-hoc fix**) | Supported if the three selective arms recover 5/5 (v2_fix: 4/5). If a repeat still fails, it is reported as unexplained. |
| `E8v3g` | 8 | Does the gate release rule (`release=on`, **post-hoc fix**) restore W4.ipi full-stack correctness? | Supported if `ok_recovered` is within one cell of restart (>= 4/5) with 0 unauthorized actions. Clean runs have no recovery, so the rule cannot change their holds; the 3 clean cells check that. |

Both fixes are made after the v2 results and are named so in every table; `runs/v2` and `runs/v2_fix` are never edited.
The release rule trusts the faulty identity that every method is given: a flag on a step the recovery reused is ignored
because that step's reads did not change. It would miss a second, unrelated injection in an unchanged source; the grid has none.
