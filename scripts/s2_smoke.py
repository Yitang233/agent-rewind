"""Rerun counts for one W2 DAG under all methods -> runs/S2/smoke.json."""
import copy, json, random, sys
from pathlib import Path
from mas_recovery.engine import execute
from mas_recovery.workloads.dag import generate

dag = generate(n=40, layers=5, fan_in=2, shared_keys=3, seed=7)
clean = execute(dag.workflow, dag.world(), None, effects="diff+outbox")
step, path, wrong = dag.fault_for(clean, random.Random(7))
world = dag.world()
faulty = execute(dag.workflow, world, None, overrides={(step, path): wrong}, effects="diff+outbox")
out = {"dag": dag.spec, "fault": {"step": step, "path": path}, "steps_run": len(clean.records), "methods": {}}
for method in ("restart", "rollback", "selective", "selective-d2"):
    w = copy.deepcopy(world)
    r = execute(dag.workflow, w, None, prior=faulty, dirty={step}, policy=method)
    out["methods"][method] = {"reruns": len(r.report["rerun"]), "reused": len(r.report["reused"]),
                              "final_ok": r.values() == clean.values(), "effects": r.report["effects"],
                              "duplicate_sends": w.duplicates()}
Path(sys.argv[1] if len(sys.argv) > 1 else "runs/S2/smoke.json").write_text(json.dumps(out, indent=1, default=str))
print({m: v["reruns"] for m, v in out["methods"].items()})
