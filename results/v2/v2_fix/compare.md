# runs/v2_fix vs runs/v2 (E8v2, W3, 7B, atoms)

Cells compared: 50. Cells with any difference in (ok_recovered, n_rerun, compensated, values_match, tokens): 12.

## ok_recovered per (fault, label): old -> new (repeats 0-4 passing)

| fault | label | v2 | v2_fix |
|---|---|---|---|
| stale_code | restart | 4/5 | 5/5 |
| stale_code | rollback | 4/5 | 5/5 |
| stale_code | selective | 4/5 | 4/5 |
| stale_code | selective+M1 | 4/5 | 4/5 |
| stale_code | selective+M1+M3t/M4+gate | 4/5 | 4/5 |
| wrong_issue | restart | 4/5 | 5/5 |
| wrong_issue | rollback | 4/5 | 5/5 |
| wrong_issue | selective | 5/5 | 5/5 |
| wrong_issue | selective+M1 | 5/5 | 5/5 |
| wrong_issue | selective+M1+M3t/M4+gate | 5/5 | 5/5 |

## Failing repeats (v2_fix)

- stale_code / selective / r0: v2 ok=False, v2_fix ok=False
- stale_code / selective+M1 / r0: v2 ok=False, v2_fix ok=False
- stale_code / selective+M1+M3t/M4+gate / r0: v2 ok=False, v2_fix ok=False

## All cells (v2 -> v2_fix)

| fault | label | r | ok | n_rerun | compensated | values_match | tokens |
|---|---|---|---|---|---|---|---|
| stale_code | restart | 0 | **False -> True** | 5 | 1 | **False -> True** | **588 -> 557** |
| stale_code | restart | 1 | True | 5 | 1 | True | 571 |
| stale_code | restart | 2 | True | 5 | 1 | True | 559 |
| stale_code | restart | 3 | True | 5 | 1 | True | 519 |
| stale_code | restart | 4 | True | 5 | 1 | True | 578 |
| stale_code | rollback | 0 | **False -> True** | 4 | 1 | **False -> True** | **588 -> 557** |
| stale_code | rollback | 1 | True | 4 | 1 | True | 571 |
| stale_code | rollback | 2 | True | 4 | 1 | True | 559 |
| stale_code | rollback | 3 | True | 4 | 1 | True | 519 |
| stale_code | rollback | 4 | True | 4 | 1 | True | 578 |
| stale_code | selective | 0 | False | 4 | 1 | False | 588 |
| stale_code | selective | 1 | True | 4 | 1 | True | 571 |
| stale_code | selective | 2 | True | 4 | 1 | True | 559 |
| stale_code | selective | 3 | True | 4 | 1 | True | 519 |
| stale_code | selective | 4 | True | 4 | 1 | True | 578 |
| stale_code | selective+M1 | 0 | False | 4 | 1 | False | 446 |
| stale_code | selective+M1 | 1 | True | 4 | 1 | True | 439 |
| stale_code | selective+M1 | 2 | True | 4 | 1 | True | 414 |
| stale_code | selective+M1 | 3 | True | 4 | 1 | True | 358 |
| stale_code | selective+M1 | 4 | True | 4 | 1 | True | 425 |
| stale_code | selective+M1+M3t/M4+gate | 0 | False | 4 | 1 | False | 704 |
| stale_code | selective+M1+M3t/M4+gate | 1 | True | 4 | 1 | True | 697 |
| stale_code | selective+M1+M3t/M4+gate | 2 | True | 4 | 1 | True | 727 |
| stale_code | selective+M1+M3t/M4+gate | 3 | True | 4 | 1 | True | 692 |
| stale_code | selective+M1+M3t/M4+gate | 4 | True | 4 | 1 | True | 700 |
| wrong_issue | restart | 0 | **False -> True** | 5 | 1 | **False -> True** | **556 -> 195** |
| wrong_issue | restart | 1 | True | 5 | 1 | **False -> True** | **521 -> 209** |
| wrong_issue | restart | 2 | True | 5 | 1 | **False -> True** | **538 -> 247** |
| wrong_issue | restart | 3 | True | 5 | 1 | **False -> True** | **486 -> 233** |
| wrong_issue | restart | 4 | True | 5 | 1 | **False -> True** | **577 -> 238** |
| wrong_issue | rollback | 0 | **False -> True** | 5 | 1 | **False -> True** | **556 -> 195** |
| wrong_issue | rollback | 1 | True | 5 | 1 | **False -> True** | **521 -> 209** |
| wrong_issue | rollback | 2 | True | 5 | 1 | **False -> True** | **538 -> 247** |
| wrong_issue | rollback | 3 | True | 5 | 1 | **False -> True** | **486 -> 233** |
| wrong_issue | rollback | 4 | True | 5 | 1 | **False -> True** | **577 -> 238** |
| wrong_issue | selective | 0 | True | 4 | 1 | True | 195 |
| wrong_issue | selective | 1 | True | 4 | 1 | True | 209 |
| wrong_issue | selective | 2 | True | 4 | 1 | True | 247 |
| wrong_issue | selective | 3 | True | 4 | 1 | True | 233 |
| wrong_issue | selective | 4 | True | 4 | 1 | True | 238 |
| wrong_issue | selective+M1 | 0 | True | 4 | 1 | True | 136 |
| wrong_issue | selective+M1 | 1 | True | 4 | 1 | True | 150 |
| wrong_issue | selective+M1 | 2 | True | 4 | 1 | True | 182 |
| wrong_issue | selective+M1 | 3 | True | 4 | 1 | True | 167 |
| wrong_issue | selective+M1 | 4 | True | 4 | 1 | True | 189 |
| wrong_issue | selective+M1+M3t/M4+gate | 0 | True | 4 | 1 | True | 756 |
| wrong_issue | selective+M1+M3t/M4+gate | 1 | True | 4 | 1 | True | 770 |
| wrong_issue | selective+M1+M3t/M4+gate | 2 | True | 4 | 1 | True | 807 |
| wrong_issue | selective+M1+M3t/M4+gate | 3 | True | 4 | 1 | True | 787 |
| wrong_issue | selective+M1+M3t/M4+gate | 4 | True | 4 | 1 | True | 189 |

## Open: stale_code repeat 0 still fails for the three selective arms

The v2_fix cells for stale_code / selective / r0 (n_rerun 4, tokens 588, ok_final False, values_match False)
store no unittest output, so the test tail cannot be shown from the cell and the cause is not determined here.
`unittest -B` is therefore not shown to be the cause for these arms. Needs a live repro of that one cell
with test output captured.

Not done: the `rerun_after_fix` key in `runs/v2/preregistration.json` (LAUNCH step 5), because the owner said not to touch `runs/v2`.
