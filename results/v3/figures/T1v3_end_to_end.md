# T1v3 end-to-end with the always-verify arm (E8v2 + E8v3, atoms, 7B). W3 baseline rows: rerun after fix (runs/v2_fix)

Recovery wall = whole recovery phase (always-verify: the audits of the rerun steps are inside it). Inference tokens = decoded + prefill of the recovery + audit tokens. always-verify = selective, effects=diff, audit=always.

| workload | method | n | ok_recovered | values_match | steps rerun | rec. wall s | decoded | inference tok | dup | unauth cells | leaked | audit tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w1 | restart | 10 | 8/10 | 10/10 | 14.0 | 38.9 ± 0.3 | 740 | 2423 | 1.20 | 5 | 3.0 | 0 |
| w1 | rollback | 10 | 8/10 | 10/10 | 12.5 | 38.9 ± 0.4 | 740 | 2423 | 1.20 | 5 | 3.0 | 0 |
| w1 | always-verify | 10 | 8/10 | 10/10 | 6.3 | 25.1 ± 11.7 | 470 | 2780 | 0.00 | 5 | 1.8 | 1224 |
| w1 | selective | 10 | 8/10 | 10/10 | 6.3 | 24.6 ± 12.5 | 470 | 1520 | 0.00 | 5 | 1.8 | 0 |
| w1 | selective+M1 | 10 | 8/10 | 10/10 | 6.3 | 9.3 ± 5.7 | 85 | 1135 | 0.00 | 5 | 1.8 | 0 |
| w1 | selective+M1+M3t/M4+gate | 10 | 8/10 | 10/10 | 6.3 | 10.3 ± 6.4 | 85 | 1691 | 0.00 | 0 | 1.1 | 520 |
| w3 | restart | 10 | 10/10 | 10/10 | 5.0 | 9.4 ± 1.5 | 168 | 391 | 0.00 | 0 | 0.0 | 0 |
| w3 | rollback | 10 | 10/10 | 10/10 | 4.5 | 9.2 ± 1.0 | 168 | 391 | 0.00 | 0 | 0.0 | 0 |
| w3 | always-verify | 10 | 9/10 | 9/10 | 4.0 | 10.3 ± 0.7 | 170 | 1436 | 0.00 | 0 | 0.0 | 589 |
| w3 | selective | 10 | 9/10 | 9/10 | 4.0 | 8.6 ± 0.8 | 170 | 394 | 0.00 | 0 | 0.0 | 0 |
| w3 | selective+M1 | 10 | 9/10 | 9/10 | 4.0 | 6.8 ± 4.0 | 66 | 291 | 0.00 | 0 | 0.0 | 0 |
| w3 | selective+M1+M3t/M4+gate | 10 | 9/10 | 9/10 | 4.0 | 7.5 ± 3.7 | 66 | 837 | 0.00 | 0 | 0.0 | 154 |
| w4.ipi | restart | 5 | 5/5 | 5/5 | 6.0 | 13.7 ± 1.3 | 250 | 1060 | 0.60 | 0 | 1.4 | 0 |
| w4.ipi | rollback | 5 | 5/5 | 5/5 | 5.0 | 13.7 ± 1.3 | 250 | 1060 | 0.60 | 0 | 1.4 | 0 |
| w4.ipi | always-verify | 5 | 5/5 | 5/5 | 3.8 | 11.1 ± 1.7 | 202 | 1510 | 0.00 | 2 | 0.8 | 654 |
| w4.ipi | selective | 5 | 5/5 | 5/5 | 3.8 | 11.0 ± 1.9 | 202 | 854 | 0.00 | 0 | 0.8 | 0 |
| w4.ipi | selective+M1 | 5 | 5/5 | 5/5 | 3.8 | 6.8 ± 2.1 | 71 | 723 | 0.00 | 0 | 0.8 | 0 |
| w4.ipi | selective+M1+M3t/M4+gate | 5 | 2/5 | 5/5 | 3.8 | 8.2 ± 2.4 | 71 | 1480 | 0.00 | 0 | 0.0 | 755 |
| w4.byz | restart | 5 | 5/5 | 5/5 | 6.0 | 13.5 ± 0.8 | 249 | 1054 | 2.00 | 0 | 2.0 | 0 |
| w4.byz | rollback | 5 | 5/5 | 5/5 | 4.0 | 13.5 ± 0.8 | 249 | 1054 | 2.00 | 0 | 2.0 | 0 |
| w4.byz | always-verify | 5 | 5/5 | 5/5 | 2.0 | 9.5 ± 0.7 | 175 | 1370 | 0.00 | 0 | 0.0 | 647 |
| w4.byz | selective | 5 | 5/5 | 5/5 | 2.0 | 9.5 ± 0.7 | 175 | 721 | 0.00 | 0 | 0.0 | 0 |
| w4.byz | selective+M1 | 5 | 5/5 | 5/5 | 2.0 | 3.8 ± 1.2 | 18 | 564 | 0.00 | 0 | 0.0 | 0 |
| w4.byz | selective+M1+M3t/M4+gate | 5 | 5/5 | 5/5 | 2.0 | 4.0 ± 1.2 | 18 | 744 | 0.00 | 0 | 0.0 | 179 |
