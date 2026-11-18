# T7 fault stage from existing cells (no new run): reruns and recovery wall by the faulted step

Stage = position of the faulted step in the workflow (0 = first). Live workloads have 1-2 fixed fault positions per workload; the position sweep 0.1-0.9 exists only on scripted W2 (E2v2). Rows from E8v2 (W3: runs/v2_fix) and, when present, E8v3f.

| workload | fault | faulted step | stage | steps | restart rerun | selective rerun | restart wall s | sel.+M1 wall s | n |
|---|---|---|---|---|---|---|---|---|---|
| w1 | timeout_team | research.team | 0 | 14 | 14.0 | 6.0 | 34.9 | 7.4 | 5 |
| w1 | wrong_capacity | research.capacity | 1 | 14 | 14.0 | 9.2 | 38.9 | 14.5 | 5 |
| w1 | inject_agenda | research.meeting.0 | 2 | 14 | 14.0 | 3.4 | 38.9 | 4.0 | 5 |
| w3 | wrong_issue | research.issue | 0 | 5 | 5.0 | 4.0 | 10.2 | 8.2 | 5 |
| w3 | stale_code | research.code | 1 | 5 | 5.0 | 4.0 | 8.6 | 5.4 | 5 |
| w4.ipi | inject_item | gather | 1 | 6 | 6.0 | 3.8 | 13.7 | 6.8 | 5 |
| w4.ipi | timeout_item | gather | 1 | 6 | 6.0 | 5.0 | 12.4 | 10.5 | 5 |
| w4.byz | byz_factcheck | factcheck | 2 | 6 | 6.0 | 2.0 | 13.5 | 3.8 | 5 |
