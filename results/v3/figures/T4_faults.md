# T4 fault kinds not in E8v2 (E8v3f, atoms, 7B): timeout_team (W1) and timeout_item (W4.ipi), six arms, 5 repeats

| workload | fault | method | n | ok_recovered | values_match | steps rerun | rec. wall s | decoded | dup | unauth cells | leaked | audit tok | held |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w1 | timeout_team | restart | 5 | 4/5 | 5/5 | 14.0 | 34.9 ± 0.4 | 740 | 2.00 | 0 | 3.0 | 0 | 0 |
| w1 | timeout_team | rollback | 5 | 4/5 | 5/5 | 14.0 | 39.4 ± 4.9 | 740 | 2.00 | 0 | 3.0 | 0 | 0 |
| w1 | timeout_team | always-verify | 5 | 4/5 | 5/5 | 6.0 | 25.1 ± 4.0 | 496 | 0.00 | 0 | 1.0 | 1224 | 0 |
| w1 | timeout_team | selective | 5 | 4/5 | 5/5 | 6.0 | 25.3 ± 3.1 | 496 | 0.00 | 0 | 1.0 | 0 | 0 |
| w1 | timeout_team | selective+M1 | 5 | 4/5 | 5/5 | 6.0 | 7.4 ± 3.5 | 56 | 0.00 | 0 | 1.0 | 0 | 0 |
| w1 | timeout_team | selective+M1+M3t/M4+gate | 5 | 4/5 | 5/5 | 6.0 | 7.4 ± 4.0 | 56 | 0.00 | 0 | 1.0 | 226 | 0 |
| w4.ipi | timeout_item | restart | 5 | 5/5 | 5/5 | 6.0 | 12.4 ± 1.2 | 250 | 0.00 | 0 | 1.0 | 0 | 0 |
| w4.ipi | timeout_item | rollback | 5 | 5/5 | 5/5 | 5.0 | 12.5 ± 1.2 | 250 | 0.00 | 0 | 1.0 | 0 | 0 |
| w4.ipi | timeout_item | always-verify | 5 | 5/5 | 5/5 | 5.0 | 15.3 ± 1.4 | 250 | 0.00 | 0 | 1.0 | 654 | 0 |
| w4.ipi | timeout_item | selective | 5 | 5/5 | 5/5 | 5.0 | 12.6 ± 1.2 | 250 | 0.00 | 0 | 1.0 | 0 | 0 |
| w4.ipi | timeout_item | selective+M1 | 5 | 5/5 | 5/5 | 5.0 | 10.5 ± 1.4 | 154 | 0.00 | 0 | 1.0 | 0 | 0 |
| w4.ipi | timeout_item | selective+M1+M3t/M4+gate | 5 | 2/5 | 5/5 | 5.0 | 12.3 ± 1.6 | 154 | 0.00 | 0 | 0.2 | 976 | 3 |
