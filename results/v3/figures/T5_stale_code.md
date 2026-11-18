# T5 W3 stale_code: before (runs/v2_fix) and after the early-revert fix (E8v3r, revert=on, POST-HOC), atoms, 7B

| method | fix | n | ok_recovered | values_match | steps rerun | rec. wall s | failing repeats |
|---|---|---|---|---|---|---|---|
| restart | (reference) | 5 | 5/5 | 5/5 | 5.0 | 8.6 | [] |
| selective | none (v2_fix) | 5 | 4/5 | 4/5 | 4.0 | 8.9 | [0] |
| selective | revert=on (post-hoc) | 5 | 5/5 | 5/5 | 4.0 | 9.5 | [] |
| selective+M1 | none (v2_fix) | 5 | 4/5 | 4/5 | 4.0 | 5.4 | [0] |
| selective+M1 | revert=on (post-hoc) | 5 | 5/5 | 5/5 | 4.0 | 5.4 | [] |
| selective+M1+M3t/M4+gate | none (v2_fix) | 5 | 4/5 | 4/5 | 4.0 | 6.0 | [0] |
| selective+M1+M3t/M4+gate | revert=on (post-hoc) | 5 | 5/5 | 5/5 | 4.0 | 5.7 | [] |
