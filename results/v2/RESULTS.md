# Phase 2 results (v2): verdicts under the pre-registered rules

Source: `runs/v2/cells.jsonl` (574 rows, v2 grid, 0 errors, 0 cells trimmed, `runs/v2/progress.json`: 18475 s, peak RSS 5.9 GB) and
`runs/v2/E2v2/cells.jsonl` (900 rows). Script: `scripts/analyze_v2.py` -> `runs/v2/figures/analysis_v2.json` (all numbers below),
figures `F1..F8`, tables `T1..T3`. Rules: `runs/v2/preregistration.json`. Qwen2.5-7B Q4_K_M, greedy, cache off, held-out =
repeats 2-4 (n per cell group is 3). Cells are scenario variants, not sampling repeats; intervals are Wilson or bootstrap (2000).
Notes for readers: no measured phase has a cache hit (`grid.cache_hits_in_measured_phases` = 0).

Validator (step 1): `check_cells.py` passes for E2v2 (900), E3v2 (60), E5v2 (108), E6v2 (40), E7v2 (216), E8v2 (150) when run per
experiment on a split file (the validator does not filter by `--exp`; run on the mixed file it fails with false pairing errors).
Nothing was trimmed by `--max-hours`; no verdict is "incomplete" for that reason.

## Summary

| H | verdict | one line |
|---|---|---|
| H1 | **supported** | median wall ratio 0.38 / 0.68 / 0.70 / 0.50 (W1/W3/W4.ipi/W4.byz); 81/81 calls identical; margin disclosure |
| H2 | **supported as written** (weak) | bounded CI on 3 of 4 by wall, but only W1 has a non-trivial p*; W4.* have p* = 0 |
| H3 | **not supported** | detection 3/3 on both, but tokens > always on both and FP 0.67/clean run on W4.ipi |
| H4 | **supported as written** (weak) | not dominated by uniform 0.1/0.25 (they are cheaper, less sensitive); gate 0 unauthorized in 6/6 |
| H5 | **supported** (exception: gate arm on W4.ipi) | ok within one cell, reuse valid 60/60, fewer reruns and less wall on 4/4 (W3 marginal) |
| H6 | **supported** (small effect) | bytes/step rise 12-13% agent->field; real bookkeeping 0.005% (max 0.057%) of recovery wall |
| H7 | **supported** | 0 unexplained of 225 selective cells; 30 exception steps classified |

## H1 supported: replay cuts recovery LLM wall (E5v2, held-out)

Paired resync-tuned / off recovery LLM wall, per (fault, repeat, style), pooled over styles (`H1.per_workload`, F4):

| workload | n pairs | median ratio | bootstrap 95% CI of median | median, cot / atoms | decoded-token ratio (mean) |
|---|---|---|---|---|---|
| W1 | 12 | 0.382 | 0.348-0.437 | 0.38 / 0.38 | 0.205 cot, 0.165 atoms |
| W3 | 12 | 0.683 | 0.559-0.773 | 0.72 / 0.62 | 0.655, 0.413 |
| W4.ipi | 6 | 0.701 | 0.520-0.875 | 0.71 / 0.69 | 0.519, 0.346 |
| W4.byz | 6 | 0.502 | 0.408-0.665 | 0.52 / 0.49 | 0.352, 0.127 |

Rule: median <= 0.8 on 4 of 4 (>= 3 needed), and text identity: 81 of 81 compared calls identical (resync), `mismatches` = []; also
`ok_recovered` equal to off in 36/36. Default-setting `prefix` arm: medians 0.89/0.96/0.94/0.99 (no wall gain), 81/81 identical (T3).
Caveats: (1) margin 0.1 was chosen during replay tuning after the held-out call-level failure at margin 0.0 (`runs/v2/D3/heldout.json`): disclosed, and
this grid is a separate full-cell test; (2) resync per-cell ratios range 0.25-1.34 (W3 max 1.34, W1 0.71), so single recoveries can still be slower;
(3) off/prefix use default replay settings, only resync-tuned carries the tuned settings; (4) 1.5B not run in the v2 grid (F4 panel c shows call-level 1.5B ratios
0.22-0.46, 7B 0.41-1.05, descriptive). vs v1: v1 wall ratios were 0.65-1.43 (not supported); the change is the local tie re-check.

## H2 supported as written, weakly: break-even fault rate (E6v2 live clean + E5v2)

p* = (clean atoms - cot per run) / (cot-off recovery - atoms-resync-tuned recovery); bootstrap over 5 clean variants and the
held-out recovery pairs (`H2.p_star`, F5). By LLM wall:

| workload | clean overhead s (n=5) | saving per fault s (n pairs) | p* | 95% CI | bounded |
|---|---|---|---|---|---|
| W1 | +3.51 | 7.18 (6) | 0.49 | 0.33-0.73 | yes |
| W3 | +1.95 | 0.72 (6) | 2.71 | 0.49-inf (saving <= 0 in 28.5% of resamples) | no |
| W4.ipi | -1.24 | 6.12 (3) | 0 | 0-0 | yes (degenerate) |
| W4.byz | -0.77 | 9.86 (3) | 0 | 0-0 | yes (degenerate) |

By decoded tokens all four are bounded (W1 0.54 [0.42,0.72], W3 0.82 [0.47,2.16], W4 0). Rule (bounded CI on >= 2 workloads): met. Caveats: only W1
has a positive, bounded p* (a strict reading, "non-degenerate bounded", gives 1 workload by wall, 2 by tokens); on W4.* atoms are cheaper than
cot even without faults, so there is nothing to break even on; the saving mixes the style effect with the replay effect (as pre-registered);
W3 is unbounded because its saving is about zero; n is 3-6. Clean success (E6v2): W1 cot 0/5 vs atoms 4/5, W3 4/5 vs 5/5, W4 5/5 both.

## H3 not supported: trust-aware audits (E7v2 held-out, `H3`, T2, F6)

| cell | m3t detection (n) | m3t FP / clean run | m3t tokens | always tokens | rule |
|---|---|---|---|---|---|
| W1 inject_agenda | 3/3 (Wilson 0.44-1.0) | 0.00 (3 clean) | 1588 | 1263 | fails tokens |
| W4.ipi inject_item | 3/3 (0.44-1.0) | 0.67 (3 clean) | 1188 | 738 | fails FP and tokens |

m3 (no trust) detects W1 inject_agenda 0/3 (so trust does fix detection there) but the probe's tokens make m3t dearer than always.
Beside m3t: m3t+minicheck W1 3/3, 948 T5 tokens, FP 0; W4.ipi 2/3, FP 0.67 (T5 tokens are not comparable with Qwen tokens). m3t+m4: W1 3/3 at 901 tokens
(below always), W4.ipi 3/3 at 1082 (above). Pooled over all 15 fault cells m3t detects 8/15 (0.53), always 15/15. m3t misses W3 wrong_issue 0/3 and W4.byz 0/3
(the probe cannot see wrong values, see the audit caveats). Disclosures: m3t uses a model probe on untrusted-only atoms; tau from tuning cells; n=3.

## H4 supported as written, weakly: placement and gate (E7v2 held-out, `H4`, F6)

Pooled frontier over 15 fault cells (detection, mean audit tokens): uniform0.1 0.20 / 111; uniform0.25 0.40 / 320; m3t+m4 0.53 / 734; m3 0.73 / 839; always 1.00 / 889.
Inject cells (n=6): uniform 0.33 / 133 and 341; m3t+m4 1.00 / 992; always 1.00 / 1001.
Rule part 1: m3t+m4 is not dominated by uniform 0.1 or 0.25 (they have fewer tokens and lower detection): met. This is weak: a budget-matched random
baseline interpolated through uniform and always (descriptive, `budget_matched_random_baseline_descriptive`) would detect 0.84 at m3t+m4's pooled 734 tokens
(m3t+m4: 0.53) and 0.99 on inject cells (m3t+m4: 1.00). So the placement does not beat budget-matched random on the pooled frontier; p = 0.5 was cut, so the
test could not see this. Per workload, m3t+m4 is dominated by uniform0.1 on W3 and by both on W4.byz (no detection there).
Rule part 2: gate `unauthorized_faulty` = 0 in 6/6 inject cells (final `unauthorized` also 0/6); without the gate m3t+m4 has it in 3/3 (W1) and 1/3 (W4.ipi).
Every held action is accounted for by count (held <= withdrawn + released + still held, leaked 0 in all six; counts, not per-intent identities).
Costs: the gate holds a clean W4.ipi post in 2/3 clean runs, and in 2/3 W4.ipi inject cells the post stays held, `ok_recovered` False; W1 inject
`ok_recovered` 2/3.

## H5 supported: selective recovery end to end (E8v2, atoms, 5 repeats, T1, F8)

| workload | ok restart / selective | rerun restart -> selective | recovery wall restart -> selective -> +M1 (s) | wall ratio sel/restart (median, CI) |
|---|---|---|---|---|
| W1 (n=10) | 8/10, 8/10 | 14.0 -> 6.3 | 38.9 -> 24.6 -> 9.3 | 0.58 (0.33-1.00) |
| W3 (n=10, rerun after fix, `runs/v2_fix`) | 10/10, 9/10 | 5.0 -> 4.0 | 9.4 -> 8.6 -> 6.8 | 0.94 (0.80-1.06) |
| W4.ipi (n=5) | 5/5, 5/5 | 6.0 -> 3.8 | 13.7 -> 11.0 -> 6.8 | 0.70 (0.68-1.00) |
| W4.byz (n=5) | 5/5, 5/5 | 6.0 -> 2.0 | 13.5 -> 9.5 -> 3.8 | 0.71 (0.67-0.71) |

Rule: ok within one cell (all 4); reuse validity 60/60 canonical-equal (Wilson 0.94-1.0; per method 20/20, 0.84-1.0; the three selective arms sample overlapping
steps, so the effective n is about 20; w3 has no candidates); fewer reruns and less wall on 4 of 4 (W3 is marginal: 1 step, CI includes 1; the other three are clear). W3 rows are the rerun after the undo-first and `-B` fixes (`runs/v2_fix`, `results/v2/v2_fix/compare.md`); the pre-fix W3 rows in `runs/v2/cells.jsonl` are not used.
v3 update (post hoc, `../v3/figures/T6_gate.md`): with the gate release rule (`release=on`) the W4.ipi full stack recovers 4/5 with 0 unauthorized (restart 5/5), within one cell; clean runs still hold a post in 2/3 and on `timeout_item` the gate arm is 2/5. Exception: the full-stack gate arm has ok_recovered 2/5 on W4.ipi (held post never released), outside one cell of restart. Other findings: W3 after the fix: restart and rollback recover and match the clean reference 10/10, selective 9/10 (`stale_code` repeat 0 fails for all three selective arms; cause below), so W3 no longer favours selective on correctness (the earlier 4/10 for restart was the harness bug); duplicates restart 0.6-2.0 per cell vs 0 for selective; selective+M1 cuts decoded tokens to 85 (W1), 66 (W3), 71, 18. Effect policy differs by method
(restart/rollback naive), as in v1. W1 unauthorized 5/10 cells for every non-gate method vs 0 with the gate.

W3 `stale_code` repeat 0, cause found (2026-11-22, from `runs/v2_fix/cells.jsonl` and a scripted reproduction): with effects `diff`, selective reruns `research.code` before `execute.apply` has been settled, so the researcher lists a `cart.py` that still has the faulty run's edit. The plan then differs from the clean plan (165 vs 145 decoded tokens, a different plan hash), while restart and rollback undo first. It is an engine ordering bug, not noise: it only bites when the faulty plan edits a real line (repeat 0). Fix: `execute(..., revert_early=True)` / bench flag `revert` compensates the later steps' effects before a rerun reads the world and applies them again if the step is reused (`tests/test_engine.py::RerunReadsCleanWorld`). It is off by default, so `runs/v2` stays reproducible by the code. The 15 affected cells (3 selective arms x 5 repeats of `stale_code`) were rerun in v3 (E8v3r, `revert=on`, post hoc): **15/15 recovered** against 12/15 without the fix (see `../v3/figures/T5_stale_code.md`); the rows above are without the fix.

## H6 supported: bookkeeping vs depth (E3v2 scripted; real rows from E5v2/E8v2)

Scripted W2, 5 seeds per cell (F2): journal bytes/step agent -> field: n=24 228 -> 255, n=200 225 -> 255, n=1000 228 -> 258 (x1.12-1.13, monotone at every n). ns per read
grows with n (6-7k at n=24, 40-45k at 200, 200-221k at 1000) but not with depth (field is lower than agent); ns per write ~15k flat. Real runs (258 recovery rows, field depth): bookkeeping share of recovery wall
mean 0.005%, max 0.057%; clean runs ~0 (E6v2). Rule met with room. Caveat: the real depth-tier cells were cut (preregistration rule change), so the rise in bytes is a scripted-only
result, and its size is small (12-13%).

## H7 supported: minimality (E2v2)

Selective, 225 cells: 0 unexplained out-of-cone reruns (`H7`); 14 cells rerun steps outside the static cone, all classified (18 new steps, 12 steps downstream of control-flow changes).
Final values match the clean reference in 225/225 (selective-d2: 195/225). Caveat: the dynamic cone is defined from recorded read hashes, near-tautological for rerun steps; the independent check is
the clean-run comparison (225/225 agree). The 8 v1 exceptions caused by the bench seeding `dirty=producers` are removed by `dirty=origin`.

## Other panels

F3: v1 E4 (unchanged) plus E8v2 duplicates/leaks/gate counts. F7: call-level phase split (call-level rows) and full-cell recovery wall split (E5v2); full cells carry only LLM wall, bookkeeping and the remainder (phase timers exist only at call level).

## W1 repeat 4 (cause found, 2026-11-23, from existing cells)

W1 `ok_recovered` is 8/10 for every method because repeat 4 fails in all ten E8v2 cells (5 methods x 2 faults). In those cells the
clean reference run itself fails the workload's check (`clean_ref.ok` False in 10/10; True in all 40 cells of repeats 0-3), while
every recovery equals that reference (`values_match` 10/10). So the repeat is a scenario variant on which the 7B model's clean run is wrong
(why it is wrong was not examined); it does not separate the methods.
