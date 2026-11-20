"""Model loading (one at a time), audit checkers and audit/calibration arguments."""
from __future__ import annotations

import gc
import json
import math
import os
import re
import subprocess

from .. import audit as AU
from .config import ROOT


class Models:
    """Loads one model at a time; `llm_for(model)` returns the inner backend for a cell."""

    def __init__(self, out_dir: str, dry: bool) -> None:
        self.out_dir, self.name, self.inner, self.checker = out_dir, None, None, None
        self.db = None
        self.mini = None                      # MiniCheck stays loaded across 7B swaps (small, HF/MPS)
        self._refs: dict | None = None

    def load(self, name: str):
        if self.name == name:
            return
        self.free()
        if name == "scripted":
            self.name = name
            return
        env = json.load(open(os.path.join(ROOT, "runs/env.json")))["models"][name]
        ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
        busy = [ln.strip() for ln in ps.splitlines() if ln.split() and ln.split()[0] != str(os.getpid())
                and os.path.basename(ln.split()[1]).startswith("python")
                and re.search(r"agent_rewind[.]bench|replay_check|audit_check|bench_llm", ln)]
        if busy:
            raise RuntimeError("another model process is running: " + busy[0])
        from ..llm import LlamaCppLLM
        self.inner = LlamaCppLLM(env["path"], model_sha=env.get("sha256"), n_ctx=4096,
                                 cache_path=os.path.join(self.out_dir, f"cache_{name}.sqlite"))
        self.db = self.inner.db
        self.name = name

    def refs(self) -> dict:
        """Memoised clean references of this out dir (<out>/clean_refs.jsonl), keyed by clean_reference()."""
        if self._refs is None:
            self._refs = {}
            p = os.path.join(self.out_dir, "clean_refs.jsonl")
            if os.path.exists(p):
                for ln in open(p):
                    if ln.strip():
                        r = json.loads(ln)
                        self._refs[r["key"]] = r
        return self._refs

    def save_ref(self, ref: dict) -> None:
        self.refs()[ref["key"]] = ref
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "clean_refs.jsonl"), "a") as f:
            f.write(json.dumps(ref, default=str) + "\n")

    def free(self) -> None:
        self.inner = self.checker = None
        gc.collect()
        self.name = None

    def reset(self) -> None:
        """After a failed cell: drop the KV cache so the next cell starts from a clean context, and
        reattach the call cache that run_cell may have detached for a live decode."""
        if self.inner is not None:
            self.inner.db = self.db
            self.inner.llm.reset()
            self.inner.replay, self.inner.drafts, self.inner.history = "off", {}, {}

    def auditor_checker(self, kind: str = "llama"):
        if self.inner is None:
            return AU.RuleChecker()
        if kind == "minicheck":
            if self.mini is None:
                self.mini = AU.MiniCheckChecker()
            self.checker = self.mini
            return self.mini
        if self.checker is None or isinstance(self.checker, AU.MiniCheckChecker):
            self.checker = AU.LlamaChecker(self.inner)
        return self.checker


def _calibration() -> dict:
    p = os.path.join(ROOT, "runs/S5/summary.json")
    try:
        cal = json.load(open(p))["calibration"]
        return {"tau_atom": cal["tau_atom"], "tau_out": cal["tau_out"], "mean_audit_tokens": cal["mean_audit_tokens"]}
    except (OSError, KeyError, ValueError):
        return {"tau_atom": -0.0005, "tau_out": -0.03, "mean_audit_tokens": 80.0}


def calibration_d4(checker: str = "llama") -> dict:
    """tau_atom (support) and tau_inj (trust probe) of the m3t policies, from runs/v2/D4/calibration.json.
    D4_RECORD_ONLY=1 (calibration runs): nothing is flagged, every score is still recorded."""
    if os.environ.get("D4_RECORD_ONLY"):
        return {"tau_atom": -math.inf, "tau_inj": math.inf}
    p = os.path.join(ROOT, "runs/v2/D5/calibration.json" if checker == "minicheck" else "runs/v2/D4/calibration.json")
    try:
        cal = json.load(open(p))
        return {"tau_atom": cal["tau_atom"], "tau_inj": cal["tau_inj"]}
    except (OSError, KeyError, ValueError):
        raise RuntimeError(f"{p} missing: calibrate the m3t policies first (runs/v2/D4/calibrate.py)")


def _audit_args(policy: str, scripted: bool, trace=None, seed: int = 0, p: float = 1.0, untrusted=(),
                checker: str = "llama") -> dict:
    cal = _calibration()
    kw = dict(threshold=-1.0 if scripted else cal["tau_atom"], out_threshold=-1.0 if scripted else cal["tau_out"],
              trace=trace, seed=seed)
    if policy == "uniform":
        kw["p"] = float(p)                            # p=1.0 keeps v1's uniform == m3
    if "m4" in policy:
        kw["w_irr"] = 0.5 * cal["mean_audit_tokens"]
    if policy in AU.TRUST_POLICIES:
        kw["untrusted"] = list(untrusted)
        if scripted:
            kw["tau_inj"] = -1.0
        else:
            c4 = calibration_d4(checker)
            kw["threshold"], kw["tau_inj"] = c4["tau_atom"], c4["tau_inj"]
    return kw
