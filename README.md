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

(to be filled in) 

Confidence-calibration findings for real models (whether verbalized confidence tracks
actual correctness on number-theory claims) live in `christestv1/`. *[Claude / Gemini /
GPT calibration summary]*

## Repository Structure

```
tree codes/            # single-parent claim tree model
  tree_model.py           strategies, ClaimTree, DP oracle, evaluation
  tree_simulation.py       simulation / rollout logic
  tree_results.py          plotting for tree experiments
  damage_core.py           shared expected-damage scoring
  tree_main.py             entry point → results_tree/

dag codes/              # multi-parent claim DAG model
  dag_model.py             strategies, ClaimDAG, evaluation
  dag_visualization.py     plotting for DAG experiments
  dag_main.py              entry point (fixed budgets) → results_dag/
  adaptive_dag_main.py     entry point (adaptive/CLI budgets) → results_adaptive_dag/

data/runs/              # verbalized-confidence transcripts from real models
  claude/, gemini/, deepseek/, gptoss-120b/, gptoss-20b/, llama-70b/, llama-8b/, qwen-80b/
                           per-model, per-seed JSON runs used for calibration analysis

results_tree/            figures + pickled artifacts from tree_main.py
results_adaptive_dag/    figures from adaptive_dag_main.py
```

## Running the Experiments

Dependencies: `numpy`, `matplotlib`, `scikit-learn`.

```bash
pip install numpy matplotlib scikit-learn

# Claim-tree experiments (fixed budget + DP oracle, calibration comparison)
cd "tree codes" && python tree_main.py

# Claim-DAG experiments (fixed budget)
cd "dag codes" && python dag_main.py

# Claim-DAG experiments with adaptive/configurable budgets
cd "dag codes" && python adaptive_dag_main.py \
  --max-nodes 80 --trials 50 --budgets 5,10,15,20,25,30,40,50 \
  --output-dir ../results_adaptive_dag --seed 2
```

Each script prints a per-strategy precision/recall/reliability/cascade-prevented table
for a reference tree or DAG, then writes its plots to the corresponding `results*/`
directory (created if missing).
