# Think Before You Trust

**Research questions:** 
1. Does the model's self-reported confidence actually help identify which claims to verfiy under limited budget?
2. How robust is that signal when confidence is unreliable - and is there a structural signal, which the model cannot misreport, that performs comparably?

We model each of the LLM's dependent reasoning claims as nodes on a tree (or DAG) structure. In a realistic claim tree/DAG, the errors propagate along with dependencies. We don't crown a winning heuristic; we isolate why verification helps — gating, timing, or structural prioritization — and pin each heuristic between a random floor and a truth-conditioned ceiling. 

## TL;DR

- Reasoning = a claim tree/DAG. One false premise can contaminate its whole
  downstream subtree, so a well-placed check is worth many bad ones.
- The core score is `ExpectedDamage(v) = p_false(v) · D̂(v)`, where `D̂` is the
  expected descendant count estimated *before* expansion. Most adaptive policies
  act on this; the learned value policy is our attempt to beat it.
- Headline metric is `reliability = 1 − undetected_false / total_nodes`, benchmarked
  against DP Oracle (the truth-conditioned ceiling).

## Verification Strategies

| Strategy | Scores a node by |
|---|---|
| Random | baseline |
| Uncertainty-first | lowest stated confidence *(self-report)* |
| Dependency-aware | `p_false · D̂` *(mixed)* |
| Greedy-MC | submodular subtree coverage *(structure)* |
| Betweenness | centrality only *(pure structure)* |

## Two kinds of signal

Every strategy scores a claim to decide what to verify.

**Self-reported** (things the model controls):

1. **Stated confidence** — from the model's self-report, `p_false = 1 − conf`
2. **Predicted error** — inferred risk from ancestors' verified state

**Structural** (things the model cannot control):

1. **Descendant count / `D̂`** — number of reachable downstream nodes
2. **Expected damage** — `p_false · D̂`
3. **Betweenness centrality** — claims that many reasoning paths run through

## Results

Each entry-point script writes to a matching `results*/` directory. The figures that
carry the argument:

- **`results_adaptive_dag/adaptive_vs_baseline_calibration.png`** — the central result.
  Expected-damage-greedy vs a minimal random stop/expand baseline on identical hidden
  ground-truth DAGs. *[reliability: adaptive ___ vs baseline ___ at budget ___]*
- **`results_tree/heuristic_gap.png`** (and the DAG/adaptive variants) — every heuristic
  placed between Random and DP Oracle. The gap between **DP Optimal** (optimal w.r.t. the
  model's own `p_false`) and **DP Oracle** (optimal w.r.t. truth) is the number worth
  reporting: it's the ceiling that better confidence calibration, not better allocation,
  would have to close. *[DP Optimal ___ vs DP Oracle ___]*
- **`*/adaptive_sigma_sweep.png`** — contamination vs over-quarantine as the threshold
  moves. *[best sigma ≈ ___]*
- **`results_warmup_checkpoint*/policy_comparison.png`** — warm-up-then-checkpoint vs the
  Dependency-aware composite. *[___]*
- **`results_*/ablation.png`** — mechanism decomposition: how much of the advantage
  survives when gating / timing / structural prioritization is removed one at a time.
  *[summary of what drops out]*

Confidence-calibration findings for real models (whether verbalized confidence tracks
actual correctness on number-theory claims) live in `christestv1/`. *[Claude / Gemini /
GPT calibration summary]*
