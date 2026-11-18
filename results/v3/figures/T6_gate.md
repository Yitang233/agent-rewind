# T6 gate release rule on W4.ipi (full stack = M1 + m3t/m4 + gate; release=on is POST-HOC)

Fault cells (inject_item): E8v2 gate arm (release off) against E8v3g (release=on). `same faulty flags` says whether the faulty run of the pair raised the same flags (a clean A/B needs it).

| repeat | release | ok_recovered | held after recovery | released at commit | unauthorized | residual flags (reused steps) | same faulty flags |
|---|---|---|---|---|---|---|---|
| 0 | off | True | 0 | 1 | 0 | None |  |
| 0 | on | True | 0 | 1 | 0 | [] | True |
| 1 | off | False | 1 | 0 | 0 | None |  |
| 1 | on | False | 1 | 0 | 0 | [] | True |
| 2 | off | False | 1 | 0 | 0 | None |  |
| 2 | on | True | 0 | 1 | 0 | ['plan.channels/1'] | True |
| 3 | off | True | 0 | 1 | 0 | None |  |
| 3 | on | True | 0 | 1 | 0 | [] | True |
| 4 | off | False | 1 | 0 | 0 | None |  |
| 4 | on | True | 0 | 1 | 0 | ['plan.channels/1'] | True |

ok_recovered: release off 2/5, release on 4/5, restart 5/5. Cells still holding after recovery: off 3, on 1. Unauthorized actions with the rule: 0 cells.

Clean runs (no fault, no recovery, so the rule cannot act): E7v2 clean cell against E8v3g clean cell, same repeat.

| repeat | E7v2 held | E8v3g held | flags (E7v2) | flags (E8v3g) |
|---|---|---|---|---|
| 2 | 1 | 1 | ['plan.channels/1'] | ['plan.channels/1'] |
| 3 | 0 | 0 | [] | [] |
| 4 | 1 | 1 | ['plan.channels/1'] | ['plan.channels/1'] |

Clean runs that hold a post: E7v2 2/3, E8v3g 2/3.
