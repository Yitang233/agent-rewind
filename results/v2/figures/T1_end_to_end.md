# T1 end-to-end (E8v2, atoms, 7B, runs/v2/cells.jsonl; repeats 0-4 x faults). W3 rows: rerun after fix (runs/v2_fix)

Recovery wall = whole recovery phase; LLM wall = decode+prefill time inside it (cache off, 0 hits). bk% = (ns_read+ns_write+ns_hash_outside) / recovery wall. Reuse valid = canonical-equal sampled scratch reruns / sampled (only steps with model calls). Effect policy differs by method (restart/rollback naive, selective diff, M1 rows diff+outbox).

| workload | method | n | ok_recovered | values_match | steps rerun | rec. wall s | LLM wall s | decoded | prefill | bk % | reuse valid | dup | unauth | leaked | audit tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w1 | restart | 10 | 8/10 | 10/10 | 14.0 | 38.9 ± 0.3 | 38.9 | 740 | 1683 | 0.00 | 0/0 | 1.20 | 5 | 3.0 | 0 |
| w1 | rollback | 10 | 8/10 | 10/10 | 12.5 | 38.9 ± 0.4 | 38.9 | 740 | 1683 | 0.00 | 0/0 | 1.20 | 5 | 3.0 | 0 |
| w1 | selective | 10 | 8/10 | 10/10 | 6.3 | 24.6 ± 12.5 | 24.6 | 470 | 1050 | 0.00 | 12/12 | 0.00 | 5 | 1.8 | 0 |
| w1 | selective+M1 | 10 | 8/10 | 10/10 | 6.3 | 9.3 ± 5.7 | 9.2 | 85 | 1050 | 0.01 | 12/12 | 0.00 | 5 | 1.8 | 0 |
| w1 | selective+M1+M3t/M4+gate | 10 | 8/10 | 10/10 | 6.3 | 10.3 ± 6.4 | 9.2 | 85 | 1085 | 0.01 | 12/12 | 0.00 | 0 | 1.1 | 520 |
| w3 | restart | 10 | 10/10 | 10/10 | 5.0 | 9.4 ± 1.5 | 9.1 | 168 | 223 | 0.01 | 0/0 | 0.00 | 0 | 0.0 | 0 |
| w3 | rollback | 10 | 10/10 | 10/10 | 4.5 | 9.2 ± 1.0 | 8.9 | 168 | 223 | 0.00 | 0/0 | 0.00 | 0 | 0.0 | 0 |
| w3 | selective | 10 | 9/10 | 9/10 | 4.0 | 8.6 ± 0.8 | 8.4 | 170 | 224 | 0.00 | 0/0 | 0.00 | 0 | 0.0 | 0 |
| w3 | selective+M1 | 10 | 9/10 | 9/10 | 4.0 | 6.8 ± 4.0 | 6.5 | 66 | 224 | 0.01 | 0/0 | 0.00 | 0 | 0.0 | 0 |
| w3 | selective+M1+M3t/M4+gate | 10 | 9/10 | 9/10 | 4.0 | 7.5 ± 3.7 | 6.8 | 66 | 616 | 0.01 | 0/0 | 0.00 | 0 | 0.0 | 154 |
| w4.ipi | restart | 5 | 5/5 | 5/5 | 6.0 | 13.7 ± 1.3 | 13.6 | 250 | 810 | 0.00 | 0/0 | 0.60 | 0 | 1.4 | 0 |
| w4.ipi | rollback | 5 | 5/5 | 5/5 | 5.0 | 13.7 ± 1.3 | 13.7 | 250 | 810 | 0.01 | 0/0 | 0.60 | 0 | 1.4 | 0 |
| w4.ipi | selective | 5 | 5/5 | 5/5 | 3.8 | 11.0 ± 1.9 | 11.0 | 202 | 652 | 0.00 | 3/3 | 0.00 | 0 | 0.8 | 0 |
| w4.ipi | selective+M1 | 5 | 5/5 | 5/5 | 3.8 | 6.8 ± 2.1 | 6.8 | 71 | 652 | 0.00 | 3/3 | 0.00 | 0 | 0.8 | 0 |
| w4.ipi | selective+M1+M3t/M4+gate | 5 | 2/5 | 5/5 | 3.8 | 8.2 ± 2.4 | 6.8 | 71 | 654 | 0.00 | 3/3 | 0.00 | 0 | 0.0 | 755 |
| w4.byz | restart | 5 | 5/5 | 5/5 | 6.0 | 13.5 ± 0.8 | 13.5 | 249 | 805 | 0.00 | 0/0 | 2.00 | 0 | 2.0 | 0 |
| w4.byz | rollback | 5 | 5/5 | 5/5 | 4.0 | 13.5 ± 0.8 | 13.5 | 249 | 805 | 0.00 | 0/0 | 2.00 | 0 | 2.0 | 0 |
| w4.byz | selective | 5 | 5/5 | 5/5 | 2.0 | 9.5 ± 0.7 | 9.5 | 175 | 546 | 0.00 | 5/5 | 0.00 | 0 | 0.0 | 0 |
| w4.byz | selective+M1 | 5 | 5/5 | 5/5 | 2.0 | 3.8 ± 1.2 | 3.8 | 18 | 546 | 0.01 | 5/5 | 0.00 | 0 | 0.0 | 0 |
| w4.byz | selective+M1+M3t/M4+gate | 5 | 5/5 | 5/5 | 2.0 | 4.0 ± 1.2 | 3.8 | 18 | 547 | 0.01 | 5/5 | 0.00 | 0 | 0.0 | 179 |
