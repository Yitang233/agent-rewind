# Design: commit-graph recovery for multi-agent workflows

CS 291A project, Option 5 (failure propagation and recovery in multi-agent systems).

**Question.** Can tracking dependencies between agent results support efficient
recovery while keeping task execution correct and safe?

## Idea
Every agent step is a **commit**. It records the exact artifact versions it read,
the artifacts it wrote, its tool effects and its model cost. Messages and
shared-state keys are both versioned artifacts, so the "read from" edges form a
dependency graph that is finer than time order.

When a result is found faulty (its identity is given, so recovery is judged
apart from detection):
1. Rerun the step that produced it.
2. Rerun a later step only if it read a **changed** unit.
3. **Early cutoff:** if a rerun's output hash matches the old one, stop
   invalidating there (as build systems like Bazel do).
4. Before rerunning a step, undo its effects by **effect class**: `read` (none),
   `compensable` (e.g. cancel a booking), `irreversible` (an email stays sent:
   reported, never counted as recovered).

## Precision vs bookkeeping
`granularity` sets the dependency unit: `fact` (artifact name), `message` (a
step's whole output) or `agent` (everything an agent produced). Coarser units
store less but invalidate more. Early cutoff can be switched off to measure its
value.

## Methods compared (same fault schedule, fresh model cache each)
| Method | Behaviour |
|---|---|
| restart | undo what is possible, rerun every step |
| rollback | time order: rerun everything from the first faulty step |
| selective/{agent,message,fact}[/no-cutoff] | this design |
| always-verify | an extra model check on every researcher output; retry on a flag |

## Metrics
Recovered correctness (outcome and every final artifact equal the clean run's,
which also checks that reused results are really valid), steps rerun and
reused, model calls and tokens, irreversible actions redone, unauthorized
actions (they stay), recovery latency. Propagation: whether the fault is read
downstream at all, whether the outcome changes, affected steps and effects.
Accidental faults (wrong, timeout, stale) and adversarial ones (malicious) are
reported apart; a fault that is read but changes nothing (`masked_capacity`) is
not counted as robustness.

## Known limits of the starting version
- One workflow (room booking). Planned: a code-fix workflow over files and a
  report workflow over a larger shared store.
- Environment state is a simulated in-memory world. Planned: tools in a
  container whose workspace is a git repo, committed after each step, so
  rollback of files is a checkout.
- Dependencies are recorded per step read, so a planner that reads three facts
  depends on all three even if it used one. Model-cited fact ids would be finer
  but can under-report; the final-value check catches that.
- The scripted verifier is an oracle with a configurable miss rate; swap in a
  real model for real numbers.
- Recovery reruns the faulty producer with the fault removed, i.e. the source
  is assumed fixed.

## Related work
TAMAS and A2ASecBench (threats in multi-agent systems), course readings TClone
(forking live environments), DeepSeek harness (session replay vs environment
snapshot) and Agentix; build-system early cutoff; data provenance; sagas and
compensating transactions. Check the 2025 *AgentGit* paper (git-style rollback
and branching for agents) and state the difference: dependency-driven selective
invalidation, shared state and effect classes.
