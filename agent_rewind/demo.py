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


LEGEND = """columns: correct = final outcome matches the clean run; values_match = every namespace value does;
         rerun / reused = steps; irreversible_redone = irreversible calls made again by recovery;
         unauthorized = actions the clean run never takes. always-verify is a fresh run that audits
         every step: rerun counts the steps it flagged and redid, tokens cover the whole run."""


def table(rows: list[dict]) -> str:
    """Method left-aligned, numbers right-aligned, no trailing blanks."""
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in COLUMNS}

    def cell(c, v):
        return str(v).ljust(widths[c]) if c == "method" else str(v).rjust(widths[c])
    line = lambda r: "  ".join(cell(c, r[c]) for c in COLUMNS).rstrip()
    return "\n".join([line({c: c for c in COLUMNS})] + [line(r) for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m agent_rewind.demo",
        description="Inject one fault into the W1 booking workflow (single meeting), show how it "
                    "propagates, and compare every recovery policy on the same faulty run.")
    ap.add_argument("--fault", default="all", choices=["all", *FAULTS], help="fault to inject (default: all)")
    ap.add_argument("--llm", default="scripted",
                    help="scripted (default, deterministic, about 1 s) or ollama:<model> for a local Ollama model")
    ap.add_argument("--verify-miss-rate", type=float, default=0.0,
                    help="probability that the always-verify audit misses a faulty step (default: 0)")
    ap.add_argument("--out", default="runs", help="folder for traces and results.json (default: runs)")
    args = ap.parse_args()

    backend = None
    if args.llm.startswith("ollama:"):
        backend = partial(OllamaLLM, args.llm.split(":", 1)[1])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clean = clean_reference(backend)
    clean["trace"].dump(out / "clean.jsonl")
    print(f"clean run: outcome={clean['outcome']['rooms']} tokens={clean['tokens']}\n")

    print(LEGEND + "\n")
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
    print(f"traces and results in {out.resolve()}/")


if __name__ == "__main__":
    main()
