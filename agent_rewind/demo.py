"""python -m agent_rewind.demo [--fault LABEL|all] [--llm scripted|ollama:<model>] [--out runs]"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from .experiment import clean_reference, compare, propagation
from .faults import FAULTS
from .llm import OllamaLLM

COLUMNS = ["method", "correct", "values_match", "rerun", "reused", "llm_calls", "tokens",
           "irreversible_redone", "unauthorized", "latency_ms"]


def table(rows: list[dict]) -> str:
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in COLUMNS}
    line = lambda r: "  ".join(str(r[c]).ljust(widths[c]) for c in COLUMNS)
    return "\n".join([line({c: c for c in COLUMNS})] + [line(r) for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", default="all", choices=["all", *FAULTS])
    ap.add_argument("--llm", default="scripted", help="scripted, or ollama:<model>")
    ap.add_argument("--verify-miss-rate", type=float, default=0.0)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    backend = None
    if args.llm.startswith("ollama:"):
        backend = partial(OllamaLLM, args.llm.split(":", 1)[1])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clean = clean_reference(backend)
    clean["trace"].dump(out / "clean.jsonl")
    print(f"clean run: outcome={clean['outcome']['rooms']} tokens={clean['tokens']}\n")

    labels = FAULTS if args.fault == "all" else [args.fault]
    results = []
    for label in labels:
        fault = FAULTS[label]
        prop = propagation(fault, clean, backend)
        prop["trace"].dump(out / f"{label}.faulty.jsonl")
        print(f"== {label} ({fault.kind}{', adversarial' if fault.adversarial else ''}) "
              f"at {fault.step} -> {fault.name}")
        print(f"   read downstream: {prop['read_downstream']}  outcome changed: {prop['outcome_changed']}  "
              f"affected: {prop['affected_steps']}  effects: {prop['effects_reached']}  "
              f"unauthorized: {prop['unauthorized']}")
        rows = compare(fault, clean, backend, args.verify_miss_rate)
        print(table(rows) + "\n")
        results.append({k: v for k, v in prop.items() if k not in ("trace", "world")} | {"recovery": rows})
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"traces and results in {out}/")


if __name__ == "__main__":
    main()
