# Tests

A small offline suite for the claims the report makes: recovery restores the clean
result, reruns stay inside the dynamic cone, no irreversible effect is delivered
twice, audits are reused only when their cited sources are unchanged, the gate holds
what it should and releases what it should, and replay decodes the same tokens as a
plain decode. Everything runs on the scripted model; a source module does not need a
matching test.

| File | Retained evidence |
|---|---|
| `test_store.py` | Merkle hashes, MVCC reads, branches, depth cuts, cones, SQLite round trip |
| `test_engine.py` | The four method-design cases (write-set change, shadowed write, branch flip, read order), restart and rollback, out-of-cone reruns explained by the dynamic cone, `revert_early` on W3, the Phase 1 booking numbers |
| `test_property.py` | Seeded W2 DAGs: final(selective) = final(restart) = clean, rerun chain selective ⊆ rollback ⊆ restart, no duplicate under `diff+outbox`, chained and uncommitted recoveries, reruns equal the dynamic cone |
| `test_effects.py` | Intent keys and the reconcile table, naive duplicates vs `diff`, outbox commit/withdraw, trust holds, equality tiers, world latency clock |
| `test_atoms.py` | Atom parsing, schema, confidence spans, cited cutoff, scripted W1 variants solvable |
| `test_audit.py` | M3 cache keys (re-audit iff cited sources change), always-verify, placement rule, M4 skips, threshold calibration |
| `test_trust.py`, `test_release.py` | M3t flags, the irreversible gate, the v3 release rule and its default-off equality |
| `test_replay.py` | Replay = plain decode for any draft, tie handling under noisy batches, local vs full re-check, stable prompt layout |
| `test_workloads.py` | W3 code fix (faults, compensation, unauthorized edits) and W4 TAMAS scenarios |
| `test_recovery.py` | Phase 1 interface (`experiment.py`, `recovery.py`): every method restores every booking fault |
| `test_overhead.py` | Metering does not change a run; coarser tiers record fewer reads and bytes |
| `test_bench.py`, `test_hygiene.py` | Grid expansion, dry-run resume, cell keys and flag checks, live-cost flags, call hashes, reuse check, `scripts/check_cells.py` |
| `test_llm.py` | A cache hit is charged the original tokens |

Run from the repository root in the `agent-rewind` env (about 20 s):

```bash
python -m unittest -q
```

`TamasTest` and nothing else skips: it needs the TAMAS clone (`~/src/TAMAS` or
`TAMAS_DIR`). Real-model behaviour (replay equality on llama.cpp, MiniCheck scores,
speed) is evidence in `results/` from `scripts/dev/`, not part of this suite.

Keep a case when it catches a concrete wrong result that a nearby case cannot. Avoid
static inventories (flag lists, task registries, step counts), wall-clock
thresholds, prompt wording, and tests that need a model file.
