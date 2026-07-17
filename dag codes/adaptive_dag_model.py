"""
Adaptive Verification Model For DAGs: 
  - Expanding graph gradually based on expected damage estimates
  - expected_damage threshold (sigma) to decide when to stop expanding
    and backtrack to verify
  - Local probability propagation after each verification
  - Sigma calibration against static ground-truth DAG
  - Adaptive baseline: random stop/expand policy

The static DAG is used as ground truth for calibration. 
The adaptive model ONLY sees nodes it has expanded.
"""

import numpy as np
import random
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from dag_model import ClaimDAG, compute_predicted_error, evaluate

# 1. FIND EXPECTED DAMAGE

def expected_damage(
    node_id: int,
    dag: ClaimDAG,
    verified_set: set[int],
    remaining_depth: int,
    branching_lambda: float,
) -> float:
    """
    Expected damage estimates how much damage node v causes if it's false and never gets caught.
    (expected_damage(v) = P(v false) * propagation_rate * E[future_descendants])

    E[future_descendants] sums branching_lambda^d over d = 1..remaining_depth,
    (how many nodes downstream would end up contaminated)

    remaining_depth is how many more levels could still branch out below v.
    """
    p_false = compute_predicted_error(node_id, dag, verified_set)

    if remaining_depth <= 0:
        e_future = 0.0
    else:
        # lambda + lambda^2 + ... + lambda^d
        e_future = sum(
            branching_lambda ** d for d in range(1, remaining_depth + 1)
        )

    return p_false * dag.propagation_rate * e_future

# 2. INTEGRATE LOCAL PROBABILITY PROPAGATION

def propagate_verification(dag: ClaimDAG, verified_node_id: int) -> None:
    """
    After verifying a node, update confidence of its immediate downstream neighbors.

    Triggers dag._compute_confidence(), which handles the coverage/consistency blend.
    """
    node = dag.nodes[verified_node_id]
    node.is_verified = True  # mark so _compute_confidence sees it

    for child_id in node.children:
        if child_id not in dag.nodes:
            continue
        child = dag.nodes[child_id]
        # Recompute intrinsic from existing confidence as baseline
        intrinsic = child.confidence
        child.confidence = dag._compute_confidence(child_id, intrinsic)

# 3. ADAPTIVE EXPANSION STATE

@dataclass
class AdaptiveState:
    """
    What the adaptive agent has seen and done so far.

    expanded_ids: nodes it has expanded (revealed downstream neighbors of)
    verified_set: nodes it has explicitly verified
    verify_budget_used: verifications consumed
    expand_budget_used: expansions consumed (token cost)
    backtrack_path: current path being backtracked for verification
    """
    expanded_ids: set = field(default_factory=set)
    verified_set: set = field(default_factory=set)
    verify_budget_used: int = 0
    expand_budget_used: int = 0
    backtrack_path: list = field(default_factory=list)

# 4. ADAPTIVE LOOP

def run_adaptive(
    dag: ClaimDAG,
    sigma: float,
    verify_budget: int,
    expand_budget: int = None,
    branching_lambda: float = None,
    rng_seed: int = 0,
) -> AdaptiveState:
    """
    Grow-threshold-backtrack-verify loop.

    Starts at the root and expands outward. For each newly expanded node,
    checks expected_damage against sigma. If damage is too high, stops
    expanding that branch, backtracks toward root verifying nodes as it
    goes (assuming sufficient budget), and updates node probabilities after 
    each verification. If the updated damage drops back under sigma, expansion
    resumes; otherwise the branch is abandoned. 
    
    Continue until the expand budget runs out or there's nothing left to expand.

    dag           : full static DAG, used as oracle — the agent only sees
                    nodes it has explicitly expanded
    sigma         : damage threshold; higher = expand more, verify less
    verify_budget : max verifications allowed
    expand_budget : max expansions allowed (None = unlimited)
    branching_lambda: feeds the E[future_descendants] estimate
    """
    rng = np.random.default_rng(rng_seed)
    if branching_lambda is None:
        branching_lambda = dag.branching_lambda
    if expand_budget is None:
        expand_budget = len(dag.nodes)  # effectively unlimited

    state = AdaptiveState()
    max_depth = dag.max_depth_actual

    # BFS frontier — (node_id, path_from_root)
    frontier = deque([(0, [0])])  # start at root

    while frontier and state.expand_budget_used < expand_budget:
        node_id, path = frontier.popleft()

        if node_id in state.expanded_ids:
            continue
        if node_id not in dag.nodes:
            continue

        # expand this node
        state.expanded_ids.add(node_id)
        state.expand_budget_used += 1

        node = dag.nodes[node_id]
        remaining_depth = max_depth - node.depth

        dmg = expected_damage(
            node_id, dag, state.verified_set, remaining_depth, branching_lambda
        )

        if dmg > sigma:
            # backtrack and verify, most recent node first
            backtrack = list(reversed(path))

            for vid in backtrack:
                if state.verify_budget_used >= verify_budget:
                    break
                if vid in state.verified_set:
                    continue

                state.verified_set.add(vid)
                state.verify_budget_used += 1
                propagate_verification(dag, vid)

            # check if verification brought damage back under threshold
            dmg_updated = expected_damage(
                node_id, dag, state.verified_set, remaining_depth, branching_lambda
            )

            if dmg_updated <= sigma:
                for child_id in node.children:
                    if child_id not in state.expanded_ids:
                        frontier.append((child_id, path + [child_id]))
            # else: branch stays abandoned, children never queued

        else:
            for child_id in node.children:
                if child_id not in state.expanded_ids:
                    frontier.append((child_id, path + [child_id]))

    return state

# 5. ADAPTIVE BASELINE — random stop/expand

def run_adaptive_random(
    dag: ClaimDAG,
    verify_budget: int,
    expand_budget: int = None,
    stop_prob: float = 0.3,
    rng_seed: int = 0,
) -> AdaptiveState:
    """
    Baseline: stop expanding at random nodes, expand at random nodes.

    stop_prob: probability of stopping expansion at any given node
               (and triggering a random verification along current path)
    """
    rng = np.random.default_rng(rng_seed)
    if expand_budget is None:
        expand_budget = len(dag.nodes)

    state = AdaptiveState()
    frontier = deque([(0, [0])])

    while frontier and state.expand_budget_used < expand_budget:
        node_id, path = frontier.popleft()

        if node_id in state.expanded_ids:
            continue
        if node_id not in dag.nodes:
            continue

        state.expanded_ids.add(node_id)
        state.expand_budget_used += 1

        if rng.random() < stop_prob:
            # Random stop: verify a random node from the current path
            unverified_path = [n for n in path if n not in state.verified_set]
            if unverified_path and state.verify_budget_used < verify_budget:
                vid = int(rng.choice(unverified_path))
                state.verified_set.add(vid)
                state.verify_budget_used += 1
                propagate_verification(dag, vid)
        else:
            node = dag.nodes[node_id]
            for child_id in node.children:
                if child_id not in state.expanded_ids:
                    frontier.append((child_id, path + [child_id]))

    return state

## 6. COMPARE ADAPTIVE RESULT TO STATIC GROUND TRUTH

def compare_to_static(state: AdaptiveState, dag: ClaimDAG) -> dict:
    """
    Scores the adaptive run against full static ground truth.

    Only looks at expanded nodes, since that's all the agent actually saw.
    Checks how many of the false nodes it expanded into got caught (verified
    directly or quarantined as a descendant of something it verified) versus
    slipped through undetected. That ratio is agreement_rate — basically how
    close the run got to verifying all the right nodes.
    """
    static_metrics = evaluate(dag, state.verified_set)

    false_ids = set(dag.false_ids)
    expanded = state.expanded_ids

    # false nodes among expanded ones — how many did we catch or quarantine?
    expanded_false = expanded & false_ids
    caught = state.verified_set & expanded_false

    quarantined = set()
    for nid in caught:
        quarantined |= dag.descendants_ids(nid)

    undetected = expanded_false - caught - quarantined
    agreement_rate = 1.0 - len(undetected) / len(expanded) if expanded else 1.0

    return {
        "agreement_rate":    agreement_rate,
        "expanded_nodes":    len(expanded),
        "verified_nodes":    len(state.verified_set),
        "expand_budget_used": state.expand_budget_used,
        **static_metrics,
    }
# 7. SIGMA CALIBRATION EXPERIMENT

def calibrate_sigma(
    sigmas: list,
    verify_budgets: list,
    n_trials: int = 50,
    max_nodes: int = 80,
    expand_budget: int = None,
) -> dict:
    """
    For each (sigma, verify_budget) pair, run n_trials adaptive simulations
    against static ground-truth DAGs and measure agreement_rate.

    Returns nested dict:
      results[sigma][verify_budget] = {"mean": {...}, "std": {...}}
    """
    raw = {s: {b: [] for b in verify_budgets} for s in sigmas}

    for trial in range(n_trials):
        # Static DAG = ground truth oracle
        dag = ClaimDAG(max_nodes=max_nodes, seed=trial)

        for sigma in sigmas:
            for vb in verify_budgets:
                state = run_adaptive(
                    dag, sigma=sigma, verify_budget=vb,
                    expand_budget=expand_budget, rng_seed=trial,
                )
                metrics = compare_to_static(state, dag)
                raw[sigma][vb].append(metrics)

    out = {}
    for sigma in sigmas:
        out[sigma] = {}
        for vb in verify_budgets:
            trials = raw[sigma][vb]
            keys = trials[0].keys()
            out[sigma][vb] = {
                "mean": {k: float(np.mean([t[k] for t in trials])) for k in keys},
                "std":  {k: float(np.std( [t[k] for t in trials])) for k in keys},
            }
    return out


def calibrate_sigma_vs_baseline(
    sigmas: list,
    verify_budgets: list,
    n_trials: int = 50,
    max_nodes: int = 80,
    stop_prob: float = 0.3,
) -> tuple[dict, dict]:
    """
    Run both adaptive and random-baseline experiments for the same trials.
    Returns (adaptive_results, baseline_results) with identical structure.
    """
    adaptive_raw  = {s: {b: [] for b in verify_budgets} for s in sigmas}
    baseline_raw  = {b: [] for b in verify_budgets}

    for trial in range(n_trials):
        dag = ClaimDAG(max_nodes=max_nodes, seed=trial)

        # Baseline (same budgets, no sigma — random policy)
        for vb in verify_budgets:
            state = run_adaptive_random(
                dag, verify_budget=vb, stop_prob=stop_prob, rng_seed=trial
            )
            baseline_raw[vb].append(compare_to_static(state, dag))

        # Adaptive
        for sigma in sigmas:
            for vb in verify_budgets:
                state = run_adaptive(
                    dag, sigma=sigma, verify_budget=vb, rng_seed=trial
                )
                adaptive_raw[sigma][vb].append(compare_to_static(state, dag))

    def _aggregate(raw_dict, keys_level1, keys_level2):
        out = {}
        for k1 in keys_level1:
            out[k1] = {}
            for k2 in keys_level2:
                trials = raw_dict[k1][k2]
                ks = trials[0].keys()
                out[k1][k2] = {
                    "mean": {k: float(np.mean([t[k] for t in trials])) for k in ks},
                    "std":  {k: float(np.std( [t[k] for t in trials])) for k in ks},
                }
        return out

    adaptive_results = _aggregate(adaptive_raw, sigmas, verify_budgets)

    baseline_results = {}
    for vb in verify_budgets:
        trials = baseline_raw[vb]
        ks = trials[0].keys()
        baseline_results[vb] = {
            "mean": {k: float(np.mean([t[k] for t in trials])) for k in ks},
            "std":  {k: float(np.std( [t[k] for t in trials])) for k in ks},
        }

    return adaptive_results, baseline_results