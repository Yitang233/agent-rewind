# mas-recovery

CS 291A project (Option 5): how failures propagate through a
researcher → planner → executor workflow, and how to recover by rerunning only
the steps that depend on a faulty result. See [DESIGN.md](DESIGN.md).

Python 3.10+, standard library only.

```sh
python3 -m mas_recovery.demo                  # all faults, scripted model
python3 -m mas_recovery.demo --fault inject_agenda
python3 -m mas_recovery.demo --llm ollama:qwen2.5:7b   # local model via Ollama
python3 -m unittest -v                        # tests
```

The demo prints, per fault, how it propagated and a table comparing recovery
methods, and writes traces (`*.jsonl`, one commit per line) and `results.json`
to `runs/`.

## Layout
| File | Role |
|---|---|
| `mas_recovery/trace.py` | commits and versioned artifacts (messages and shared state) |
| `mas_recovery/world.py` | simulated tools with effect classes |
| `mas_recovery/llm.py` | scripted / Ollama backends, call cache and token counts |
| `mas_recovery/workflow.py` | the booking workflow and step context |
| `mas_recovery/faults.py` | fault schedule (accidental vs adversarial) |
| `mas_recovery/recovery.py` | restart, rollback, selective recovery |
| `mas_recovery/experiment.py` | propagation measurement and method comparison |

## Next
Phase 1 (due 2026-11-04): more workflows, real-model runs, the 1–2 page report.
Phase 2 (due 2026-12-10): git/container workspace snapshots, a fault detector,
the full experiment grid and the final report.
