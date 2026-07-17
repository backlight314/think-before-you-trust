"""
Hallucination Propagation Simulation — DAG Model

Same mechanics as tree_model.py, just ported onto a DAG topology instead
of a tree.

Carried over unchanged:
  _make_intrinsic_confidence()   same Beta distributions / miscalibration
  _compute_confidence()          structural confidence from ancestor
                                  coverage and consistency, now using ALL
                                  parents as ancestors
  compute_predicted_error()      three-state propagation model with
                                  delta_unc / delta_cont, adapted for
                                  multi-parent nodes (worst-case parent
                                  rule — most pessimistic verified parent
                                  wins)
  strategy_dependency_aware()    same composite score and default weights,
                                  built on compute_predicted_error instead
                                  of raw confidence
  evaluate()                     quarantine now uses descendants_ids
                                  instead of subtree_ids

What's different on a DAG:
  Node.parents: list format, multiple parents allowed
  Node.parent_id returns the first parent error propagation triggers 
  on any false parent (max rule)
  _compute_confidence(): "ancestors" means everything reachable backward
    through parent edges (transitive closure, not just one path up)
  compute_predicted_error() takes the worst case across parents — 
  parent verified false → full contamination penalty
  parent unchecked → uncertainty penalty
  DP does not work on DAGs, so we replace strategy_optimal() with
    strategy_greedy_mc_lazy() (CELF, gives the (1-1/e) submodular guarantee)
  descendants_ids() follows directed forward edges for DAG reachability
"""

import numpy as np
import random
import os
import sys
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional

OUT_DIR = "results_dag"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. DATA STRUCTURES

@dataclass
class Node:
    id: int
    depth: int
    parents: list          # list of parent IDs (multiple allowed in DAG)
    truth: int             # hidden ground truth: 0=false, 1=true
    confidence: float      # AI's stated P(true) in [0,1]
    children: list = field(default_factory=list)
    is_verified: bool = False
    is_corrupted: bool = False

    # adaptive/online simulation state (ported from tree_model.Node)
    p_false: float = 0.0   # current estimated P(node is false)
    status: str = "active"  # active | expanded | verified_true |
                             # verified_false | quarantined | terminal
    expanded: bool = False  # whether expand_node() has been called on this node

    @property
    def num_descendants(self):
        return self._num_descendants

    @num_descendants.setter
    def num_descendants(self, v):
        self._num_descendants = v

    def __post_init__(self):
        self._num_descendants = 0

    # convenience: single parent_id for backward compat with strategies
    @property
    def parent_id(self):
        return self.parents[0] if self.parents else None

# 2. DAG GENERATION

class ClaimDAG:
    """
    Builds a DAG by:
      1. Growing a base tree (BFS, same stopping criterion as tree version)
      2. Adding cross edges: each node at depth d has a chance to gain
         an additional parent from depth < d (guarantees acyclicity)

    Error propagation (multi-parent, max rule):
      - If ALL parents are true  → error at base_error_rate
      - If ANY parent is false   → error at propagation_rate

    Confidence (structural, same formula as tree version):
      - Intrinsic self-report at build time (no ancestors verified)
      - Recomputable post-verification via _compute_confidence()
    """

    def __init__(
        self,
        max_nodes: int = 140,
        base_error_rate: float = 0.15,
        propagation_rate: float = 0.85,
        extra_edge_prob: float = 0.20,   # P(each node gains an extra parent)
        max_depth: Optional[int] = None,
        branching_lambda: Optional[float] = None,
        seed: int = 42,
        build_full: bool = True,
        overconfident: bool = False,
    ):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.max_nodes = max_nodes
        self.base_error_rate = base_error_rate
        self.propagation_rate = propagation_rate
        self.extra_edge_prob = extra_edge_prob
        self.overconfident = overconfident
        self.max_depth = (
            max_depth if max_depth is not None else int(self.rng.integers(4, 8))
        )
        self.branching_lambda = (
            branching_lambda
            if branching_lambda is not None
            else float(self.rng.uniform(1.5, 3.0))
        )

        self.nodes: dict[int, Node] = {}
        self._id_counter = 0
        self._initialize_root()
        if build_full:
            self._build_full()

    def _new_id(self):
        i = self._id_counter
        self._id_counter += 1
        return i

    def _make_intrinsic_confidence(self, truth: int) -> float:
        """
        AI's raw self-certainty at generation time (miscalibrated and noisy).
        (Identical to tree_model version)
        False claims are overconfident (Beta skewed high).
        True claims are occasionally underconfident.
        """
        if truth == 1:
            return float(np.clip(self.rng.beta(5, 2), 0.40, 0.99))
        else:
            if self.overconfident:
                return float(np.clip(self.rng.beta(8, 2), 0.55, 0.99))
            else:
                return float(np.clip(self.rng.beta(2, 5), 0.05, 0.80))

    def _ancestors_of(self, nid: int) -> list[int]:
        """
        All ancestors of nid reachable by following parent edges backward.
        Returns them in BFS order (closest first).
        """
        visited, queue = [], deque(self.nodes[nid].parents)
        seen = set(self.nodes[nid].parents)
        while queue:
            cur = queue.popleft()
            visited.append(cur)
            for p in self.nodes[cur].parents:
                if p not in seen:
                    seen.add(p)
                    queue.append(p)
        return visited

    def _compute_confidence(self, nid: int, intrinsic: float) -> float:
        """
        Structural confidence for a node: ported directly from tree_model.

        coverage    = fraction of ancestors that are verified
        consistency = 1 - 2*std(truth values of verified ancestors)
                      (all same → 1; half-half → 0)
        confidence = coverage^2 * consistency + (1 - coverage^2) * intrinsic
        (confidence is same as tree version)

        At build time nothing is verified, so coverage=0 and this returns
        intrinsic — correct behaviour.
        """
        ancestors = self._ancestors_of(nid)
        if not ancestors:
            return intrinsic   # root: pure intrinsic

        verified_ancestors = [a for a in ancestors if self.nodes[a].is_verified]
        coverage = len(verified_ancestors) / len(ancestors)

        if coverage == 0.0:
            return intrinsic

        truth_vals = [self.nodes[a].truth for a in verified_ancestors]
        p = float(np.mean(truth_vals))
        std = float(np.sqrt(p * (1.0 - p)))
        consistency = 1.0 - 2.0 * std   # ∈ [0, 1]

        # Mirrors tree formula: coverage^2 * consistency + (1-coverage^2) * intrinsic
        structural = coverage * consistency + (1.0 - coverage) * intrinsic
        confidence = coverage * structural + (1.0 - coverage) * intrinsic
        return float(np.clip(confidence, 0.01, 0.99))

    def _any_parent_false(self, parent_ids: list) -> bool:
        return any(self.nodes[p].truth == 0 for p in parent_ids)

    def _initialize_root(self):
        """DAG is not fully built here, only the root node. 
        Children are generated only when expand_node() is called"""
        root_intrinsic = self._make_intrinsic_confidence(1)
        root = Node(
            id=self._new_id(),
            depth=0,
            parents=[],
            truth=1,
            confidence=root_intrinsic,
        )
        root.p_false = float(np.clip(1.0 - root.confidence, 0.0, 1.0))
        self.nodes[root.id] = root
        self.by_depth: dict[int, list] = defaultdict(list)
        self.by_depth[0].append(root.id)

    def expand_node(
        self,
        node_id: int,
        max_new_children: Optional[int] = None,
        add_cross_edges: bool = True,
    ) -> list[int]:
        """
        Reveals the children of node_id, if it's eligible to expand. With
        add_cross_edges on, each new child also has an extra_edge_prob
        chance of picking up an extra parent from an already-revealed node
        at a strictly smaller depth (maintains acyclic structure).
        Used for online/adaptive generation.

        add_cross_edges = False is what _build_full()'s first pass uses, to
        reproduce the original two-phase offline build: grow the tree
        first, then do one cross-edge pass over the whole thing.

        Returns the ids of any newly created children.
        """
        node = self.nodes[node_id]
        new_children: list[int] = []

        if node.status == "quarantined":
            return new_children
        if len(self.nodes) >= self.max_nodes:
            return new_children
        if node.depth >= self.max_depth:
            node.expanded = True
            if node.status == "active":
                node.status = "terminal"
            return new_children

        n_children = min(
            int(self.rng.poisson(self.branching_lambda)),
            self.max_nodes - len(self.nodes),
        )
        if max_new_children is not None:
            n_children = min(n_children, max_new_children)

        for _ in range(n_children):
            # Ground truth: max rule (any false parent → propagation_rate)
            if self._any_parent_false([node_id]):
                child_truth = int(self.rng.random() > self.propagation_rate)
                corrupted = child_truth == 0
            else:
                child_truth = int(self.rng.random() > self.base_error_rate)
                corrupted = False

            intrinsic = self._make_intrinsic_confidence(child_truth)

            child = Node(
                id=self._new_id(),
                depth=node.depth + 1,
                parents=[node_id],
                truth=child_truth,
                confidence=intrinsic,   # placeholder, overwritten below
                is_corrupted=corrupted,
            )
            self.nodes[child.id] = child
            node.children.append(child.id)
            self.by_depth[child.depth].append(child.id)
            new_children.append(child.id)

            # Overwrite with structural confidence (accounts for any
            # ancestors verified so far)
            child.confidence = self._compute_confidence(child.id, intrinsic)

            # ── cross edge (extra parent at strictly smaller depth) ────
            # Only nodes already revealed (in self.by_depth) are candidates —
            # online generation has no knowledge of unrevealed nodes.
            if add_cross_edges and child.depth >= 2 and self.rng.random() < self.extra_edge_prob:
                candidates = [
                    v
                    for d in range(0, child.depth)
                    for v in self.by_depth[d]
                    if v not in child.parents and v != child.id
                ]
                if candidates:
                    extra_parent_id = int(self.rng.choice(candidates))
                    child.parents.append(extra_parent_id)
                    self.nodes[extra_parent_id].children.append(child.id)

                    # Re-evaluate truth given new parent (max rule)
                    if child.truth == 1 and self.nodes[extra_parent_id].truth == 0:
                        if self.rng.random() < self.propagation_rate:
                            child.truth = 0
                            child.is_corrupted = True
                            intrinsic = self._make_intrinsic_confidence(0)
                            child.confidence = self._compute_confidence(child.id, intrinsic)

            # Initial p_false estimate: with no verified ancestors this
            # reduces to 1 - confidence (matches compute_predicted_error()).
            child.p_false = float(np.clip(1.0 - child.confidence, 0.0, 1.0))

        node.expanded = True
        if node.status == "active":
            node.status = "expanded" if new_children else "terminal"

        return new_children

    def _build_full(self):
        """
        Build the whole DAG (offline/legacy behavior), reproducing
        the original two-phase construction exactly:
          1. BFS tree growth (no cross edges yet)
          2. a single cross-edge pass over every node, in creation order
        """
        #Phase 1: grow base tree
        queue = deque([0])
        while queue and len(self.nodes) < self.max_nodes:
            pid = queue.popleft()
            new_children = self.expand_node(pid, add_cross_edges=False)
            queue.extend(new_children)

        #Phase 2: add cross edges (makes it a DAG)
        # Extra parent must be at strictly smaller depth → no cycles.
        for nid, node in list(self.nodes.items()):
            if node.depth < 2:
                continue
            if self.rng.random() > self.extra_edge_prob:
                continue

            candidates = [
                v
                for d in range(0, node.depth)
                for v in self.by_depth[d]
                if v not in node.parents and v != nid
            ]
            if not candidates:
                continue

            extra_parent_id = int(self.rng.choice(candidates))
            node.parents.append(extra_parent_id)
            self.nodes[extra_parent_id].children.append(nid)

            # Re-evaluate truth given new parent (max rule)
            if node.truth == 1 and self.nodes[extra_parent_id].truth == 0:
                if self.rng.random() < self.propagation_rate:
                    node.truth = 0
                    node.is_corrupted = True
                    intrinsic = self._make_intrinsic_confidence(0)
                    node.confidence = self._compute_confidence(nid, intrinsic)
                    node.p_false = float(np.clip(1.0 - node.confidence, 0.0, 1.0))

        self._compute_descendants()

    def _compute_descendants(self):
        """
        Descendants in a DAG are everything reachable by following directed edges forward. 
        Uses iterative DFS so nothing gets double-counted.
        """
        for nid in self._topological_order():
            visited = set()
            stack = list(self.nodes[nid].children)
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                stack.extend(self.nodes[cur].children)
            self.nodes[nid].num_descendants = len(visited)

    def _topological_order(self) -> list:
        """Kahn's algorithm for topological sort."""
        in_degree = {nid: len(node.parents) for nid, node in self.nodes.items()}
        queue = deque([nid for nid, d in in_degree.items() if d == 0])
        order = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for child in self.nodes[nid].children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order

    #public properties

    @property
    def all_ids(self):
        return list(self.nodes.keys())

    @property
    def false_ids(self):
        return [nid for nid, n in self.nodes.items() if n.truth == 0]

    @property
    def max_depth_actual(self):
        return max(n.depth for n in self.nodes.values())

    @property
    def max_descendants(self):
        return max(n.num_descendants for n in self.nodes.values()) or 1

    @property
    def num_extra_edges(self):
        return sum(1 for n in self.nodes.values() if len(n.parents) > 1)

    def descendants_ids(self, nid) -> set[int]:
        """All nodes reachable from nid following directed edges (inclusive of nid)."""
        visited, stack = set(), [nid]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(self.nodes[cur].children)
        return visited

# 2b. PREDICTED ERROR MODEL  (ported from tree_model)

def compute_predicted_error(
    node_id: int,
    dag: "ClaimDAG",
    verified_set: set[int],
    p_base: float = None,
    delta_unc: float = None,
    delta_cont: float = None,
) -> float:
    """
    The model's estimate of P(node is false), given how much of the node's
    ancestry has been verified so far. Same as tree_model, with one change:
    multi-parent nodes use a worst-case parent rule.

    Three-state propagation:
      p_error = p_base                  if all direct parents verified-true
              = p_base + delta_unc      if a direct parent is still unchecked
              = p_base + delta_cont     if a direct parent is verified-false

    Multi-parent adaptation:
      - If ANY verified parent is FALSE  → full contamination penalty (delta_cont)
      - Else if ANY parent is unchecked  → uncertainty penalty (delta_unc)
      - Else (all parents verified-TRUE) → clean propagation (p_base)

    delta_unc = delta_cont * pi_hat, where pi_hat comes from the
    unverified ancestors' confidence scores, same as the tree version.

    alpha (fraction of ancestors verified) blends the propagation model
    with the AI's self-reported confidence.
    """
    node = dag.nodes[node_id]

    if p_base is None:
        p_base = dag.base_error_rate
    if delta_cont is None:
        delta_cont = max(dag.propagation_rate - p_base, 0.0)

    ancestors = dag._ancestors_of(node_id)

    if not ancestors:
        return 1.0 - node.confidence   # root node

    unverified_ancestors = [a for a in ancestors if a not in verified_set]
    if unverified_ancestors:
        pi_hat = float(np.mean(
            [1.0 - dag.nodes[a].confidence for a in unverified_ancestors]
        ))
    else:
        pi_hat = 0.0

    if delta_unc is None:
        delta_unc = delta_cont * pi_hat

    #Multi-parent worst-case rule
    direct_parents = node.parents

    # Check if ANY verified parent is false → contamination penalty
    any_verified_false = any(
        (p in verified_set and dag.nodes[p].truth == 0)
        for p in direct_parents
    )
    # Check if ANY direct parent is unchecked
    any_unchecked = any(p not in verified_set for p in direct_parents)

    if any_verified_false:
        p_prop = p_base + delta_cont
    elif any_unchecked:
        p_prop = p_base + delta_unc
    else:
        # All direct parents verified-true
        p_prop = p_base

    #Blend propagation model with AI's self-reported confidence
    n_verified = len(ancestors) - len(unverified_ancestors)
    alpha = n_verified / len(ancestors)   # fraction verified

    p_ai = 1.0 - node.confidence
    p_predicted = alpha * p_prop + (1.0 - alpha) * p_ai

    return float(np.clip(p_predicted, 0.0, 1.0))

# 2c. ADAPTIVE ONLINE VERIFICATION  (ported from tree_model)

def estimate_future_descendants(
    node: Node, dag: "ClaimDAG", remaining_node_budget: int
) -> float:
    """
    Expected number of descendants `node` could still generate if expansion
    continues down to dag.max_depth, capped by the remaining node budget.
    Identical to tree_model's version.
    """
    remaining_depth = dag.max_depth - node.depth
    b = dag.branching_lambda

    if remaining_depth <= 0:
        return 0.0

    if b == 1:
        structural = float(remaining_depth)
    else:
        structural = (b ** (remaining_depth + 1) - b) / (b - 1)

    return min(structural, remaining_node_budget)


def expected_damage(
    node_id: int, dag: "ClaimDAG", rho: float, remaining_node_budget: int
) -> float:
    """
    ExpectedDamage(v) = P(v false) * rho * E[future descendants of v]
    """
    node = dag.nodes[node_id]
    d_hat = estimate_future_descendants(node, dag, remaining_node_budget)
    return node.p_false * rho * d_hat


def update_descendant_probabilities(
    dag: "ClaimDAG", verified_node_id: int, epsilon: float, rho: float
) -> None:
    """
    After `verified_node_id` has been verified (its p_false already set to
    0.0 or 1.0), propagate the update to its existing descendants via BFS:

        child.p_false = clip(epsilon + rho * max(parent.p_false for parents), 0, 1)

    Multi-parent adaptation: use the most pessimistic (max) p_false among
    a child's direct parents, mirroring the worst-case parent rule used by
    compute_predicted_error(). Only existing nodes are touched.
    """
    queue = deque(dag.nodes[verified_node_id].children)

    while queue:
        child_id = queue.popleft()
        child = dag.nodes[child_id]
        parent_p_false = max(dag.nodes[p].p_false for p in child.parents)

        old = child.p_false
        child.p_false = float(np.clip(epsilon + rho * parent_p_false, 0.0, 1.0))

        if abs(child.p_false - old) > 1e-4:
            queue.extend(child.children)


def verify_node(
    dag: "ClaimDAG",
    node_id: int,
    verified_set: set[int],
    epsilon: Optional[float] = None,
    rho: Optional[float] = None,
) -> int:
    """
    Reveal the ground truth of `node_id`, p_false (0 or 1), and
    propagate the update to its existing descendants.
    """
    if epsilon is None:
        epsilon = dag.base_error_rate
    if rho is None:
        rho = max(dag.propagation_rate - dag.base_error_rate, 0.0)

    node = dag.nodes[node_id]
    verified_set.add(node_id)
    node.is_verified = True

    if node.truth == 1:
        node.status = "verified_true"
        node.p_false = 0.0
    else:
        node.status = "verified_false"
        node.p_false = 1.0

    update_descendant_probabilities(dag, node_id, epsilon, rho)

    return node.truth


def quarantine_node(dag: "ClaimDAG", node_id: int, frontier: set[int]) -> None:
    """
    Mark a verified-false node as quarantined
    """
    node = dag.nodes[node_id]
    node.status = "quarantined"
    node.expanded = True
    frontier.discard(node_id)


class AdaptiveSigma:
    """Empirical-quantile adaptive threshold. Tracks observed damages
    and sets sigma to the quantile that matches the current spend rate."""

    def __init__(self, verify_budget: int, max_nodes: int):
        self.verify_budget = verify_budget
        self.max_nodes = max_nodes
        self.damage_history: list[float] = []

    def update(self, damage: float):
        self.damage_history.append(damage)

    def get(self, B_remaining: int, n_revealed: int) -> float:
        if len(self.damage_history) < 5:
            return 0.0
        N_remaining = max(self.max_nodes - n_revealed, 1)
        spend_rate = B_remaining / N_remaining
        quantile = 1.0 - min(spend_rate, 1.0)
        return float(np.quantile(self.damage_history, quantile))


class FixedSigma:
    def __init__(self, value: float):
        self.value = value

    def update(self, damage: float):
        pass

    def get(self, B_remaining: int, n_revealed: int) -> float:
        return self.value

# 3. VERIFICATION STRATEGIES

def strategy_oracle(dag: ClaimDAG, budget: int, **_) -> set[int]:
    return set(dag.all_ids)


def strategy_random(dag: ClaimDAG, budget: int, **_) -> set[int]:
    pool = dag.all_ids
    return set(random.sample(pool, min(budget, len(pool))))


def strategy_level(dag: ClaimDAG, budget: int, **_) -> set[int]:
    by_depth = defaultdict(list)
    for nid, node in dag.nodes.items():
        by_depth[node.depth].append(nid)
    depths = sorted(by_depth.keys())
    n_levels = max(1, budget // max(len(v) for v in by_depth.values()))
    step = max(1, len(depths) // n_levels)
    selected_depths = set(depths[::step])
    candidates = [nid for d in selected_depths for nid in by_depth[d]]
    return set(random.sample(candidates, min(budget, len(candidates))))


def strategy_recent(dag: ClaimDAG, budget: int, **_) -> set[int]:
    sorted_nodes = sorted(dag.nodes.items(), key=lambda x: -x[1].depth)
    return {nid for nid, _ in sorted_nodes[:budget]}


def strategy_uncertainty(dag: ClaimDAG, budget: int, **_) -> set[int]:
    """
    Uses the three-state propagation model, not raw confidence.
    """
    scored = sorted(
        dag.nodes.items(),
        key=lambda x: compute_predicted_error(x[0], dag, set()),
        reverse = True,
    )
    return {nid for nid, _ in scored[:budget]}


def strategy_dependency_aware(
    dag: ClaimDAG,
    budget: int,
    w_conf: float = 0.35,
    w_depth: float = 0.20,
    w_desc: float = 0.30,
    w_parent: float = 0.15,
) -> set[int]:
    """
    Same as tree_model; multi-parent adaptation: parent score uses
    the maximum predicted error across all direct parents (most pessimistic).

    score(v) = w_conf  * predicted_p_error(v)
             + w_depth * (depth / max_depth)
             + w_desc  * (descendants / max_descendants)
             + w_parent* max(predicted_p_error(p) for p in parents(v))
    """
    max_depth = dag.max_depth_actual or 1
    max_desc = dag.max_descendants
    empty_verified: set[int] = set()

    scores = {}
    for nid, node in dag.nodes.items():
        s_conf = compute_predicted_error(nid, dag, empty_verified)
        s_depth = node.depth / max_depth
        s_desc = node.num_descendants / max_desc
        if node.parents:
            s_parent = max(
                compute_predicted_error(p, dag, empty_verified)
                for p in node.parents
            )
        else:
            s_parent = 0.0
        scores[nid] = (
            w_conf  * s_conf
            + w_depth * s_depth
            + w_desc  * s_desc
            + w_parent * s_parent
        )
    top = sorted(scores, key=scores.__getitem__, reverse=True)
    return set(top[:budget])

# GREEDY MC (CELF) — replaces DP Optimal on DAGs

def _sample_assignment(dag: ClaimDAG, rng: np.random.Generator) -> dict:
    """
    One top-down sampled truth assignment (independent-cascade style),
    over the whole DAG in topological order.

      - root(s): always true
      - otherwise: error rate = propagation_rate if any parent sampled
        false, else base_error_rate
    """
    sampled = {}
    for nid in dag._topological_order():
        node = dag.nodes[nid]
        if not node.parents:
            sampled[nid] = 1   # root always true
        else:
            any_parent_false = any(sampled[p] == 0 for p in node.parents)
            rate = dag.propagation_rate if any_parent_false else dag.base_error_rate
            sampled[nid] = int(rng.random() > rate)
    return sampled


def _safe_count(dag: ClaimDAG, verify_set: set[int], sampled: dict) -> int:
    """
    Safe-node count on a FIXED assignment (deterministic):
      - Verified-false nodes quarantine their entire descendant set
      - Safe = true nodes + quarantined nodes
    """
    quarantined = set()
    for nid in verify_set:
        if sampled[nid] == 0:
            quarantined |= dag.descendants_ids(nid)
    return sum(1 for nid in dag.nodes if sampled[nid] == 1 or nid in quarantined)


def strategy_greedy_mc_lazy(dag: ClaimDAG, budget: int, n_simulations: int = 200) -> set[int]:
    """
    CELF greedy for submodular maximisation, maintains (1 - 1/e) approximation.

    Marginal gains only decrease as the selected set grows (submodularity).
    If a node was not the best last round it won't become better --> store in max-heap (defer re-checking)

    Uses common random numbers: a fixed bank of sampled truth assignments is
    drawn once up front, and every marginal-gain estimate (for every
    candidate node, at every step of the greedy loop) is evaluated against
    that same bank. 
    
    S and S|{v} are compared on identical realizations, the difference is a paired, 
    low-variance estimate (instead of redrawing fresh samples for each side, which 
    lets sampling noise swamp the true marginal gain.)
    """
    import heapq

    # Pre-sample a FIXED bank of assignments (common random numbers).
    assignments = [
        _sample_assignment(dag, np.random.default_rng(42 + i))
        for i in range(n_simulations)
    ]

    def estimate_marginal(S: set, v: int) -> float:
        Sv = S | {v}
        diff = 0.0
        for a in assignments:
            diff += _safe_count(dag, Sv, a) - _safe_count(dag, S, a)
        return diff / len(assignments)

    # Initialise heap with marginal gains from empty set
    heap = []
    for nid in dag.all_ids:
        gain = estimate_marginal(set(), nid)
        heapq.heappush(heap, (-gain, nid))   # max-heap via negation

    selected = set()
    for _ in range(min(budget, len(dag.all_ids))):
        while True:
            neg_gain, nid = heapq.heappop(heap)
            if nid in selected:
                continue
            fresh_gain = estimate_marginal(selected, nid)
            top_neg = heap[0][0] if heap else 0
            if fresh_gain >= -top_neg:
                selected.add(nid)
                break
            else:
                heapq.heappush(heap, (-fresh_gain, nid))

    return selected


#Ablation variants

def _make_ablation(w_conf, w_depth, w_desc, w_parent, name):
    def fn(dag, budget, **_):
        return strategy_dependency_aware(
            dag, budget,
            w_conf=w_conf, w_depth=w_depth, w_desc=w_desc, w_parent=w_parent,
        )
    fn.__name__ = name
    return fn


ABLATIONS = {
    "Conf only":        _make_ablation(1.0, 0.0, 0.0, 0.0, "Conf only"),
    "Descendants only": _make_ablation(0.0, 0.0, 1.0, 0.0, "Descendants only"),
    "Depth only":       _make_ablation(0.0, 1.0, 0.0, 0.0, "Depth only"),
    "Conf + Desc":      _make_ablation(0.5, 0.0, 0.5, 0.0, "Conf + Desc"),
    "Full composite":   _make_ablation(0.35, 0.20, 0.30, 0.15, "Full composite"),
}

STRATEGIES = {
    "Random":            strategy_random,
    "Level-sampling":    strategy_level,
    "Recent (deep)":     strategy_recent,
    "Uncertainty":       strategy_uncertainty,
    "Dependency-aware":  strategy_dependency_aware,
    "Greedy MC":         strategy_greedy_mc_lazy,
}

# 4. EVALUATION

def evaluate(dag: ClaimDAG, verify_set: set[int]) -> dict:
    """
    DAG evaluation: quarantine uses descendants_ids (reachable nodes).
    A node is quarantined if any verified-false ancestor can reach it.
    Metrics mirror tree_model.evaluate() exactly.
    """
    false_ids = set(dag.false_ids)
    total = len(dag.nodes)

    TP = verify_set & false_ids
    precision = len(TP) / len(verify_set) if verify_set else 0.0
    recall    = len(TP) / len(false_ids)  if false_ids  else 1.0

    quarantined = set()
    for nid in TP:
        quarantined |= dag.descendants_ids(nid)

    undetected_false = false_ids - verify_set - quarantined
    reliability      = 1.0 - len(undetected_false) / total
    cascade_prevented = len(quarantined - TP)

    return {
        "precision":         precision,
        "recall":            recall,
        "reliability":       reliability,
        "cascade_prevented": cascade_prevented,
        "true_positives":    len(TP),
        "false_nodes":       len(false_ids),
        "undetected_false":  len(undetected_false),
        "verified_false":    len(TP),
        "verified_true":     len(verify_set - false_ids),
    }

# 5. EXPERIMENTS

def run_experiment(
    budgets: list[int],
    n_trials: int = 50,
    max_nodes: int = 140,
    base_error_rate: float = 0.15,
    propagation_rate: float = 0.85,
    extra_edge_prob: float = 0.15,
    strategy_set: dict = None,
) -> dict:
    """
    Returns nested dict:
      results[strategy][budget] = {"mean": {...}, "std": {...}}
    """
    if strategy_set is None:
        strategy_set = STRATEGIES

    raw = {s: {b: [] for b in budgets} for s in strategy_set}

    for trial in range(n_trials):
        dag = ClaimDAG(
            max_nodes=max_nodes,
            base_error_rate=base_error_rate,
            propagation_rate=propagation_rate,
            extra_edge_prob=extra_edge_prob,
            seed=trial,
        )
        for budget in budgets:
            for name, fn in strategy_set.items():
                vset    = fn(dag, budget)
                metrics = evaluate(dag, vset)
                raw[name][budget].append(metrics)

    out = {}
    for strat in strategy_set:
        out[strat] = {}
        for budget in budgets:
            trials = raw[strat][budget]
            keys   = trials[0].keys()
            out[strat][budget] = {
                "mean": {k: np.mean([t[k] for t in trials]) for k in keys},
                "std":  {k: np.std( [t[k] for t in trials]) for k in keys},
            }
    return out


def run_sensitivity(budgets, n_trials=50, max_nodes=140):
    regimes = {
        "Low  (ε=0.05, ρ=0.40)": {"base_error_rate": 0.05, "propagation_rate": 0.40},
        "Med  (ε=0.15, ρ=0.70)": {"base_error_rate": 0.15, "propagation_rate": 0.70},
        "High (ε=0.30, ρ=0.90)": {"base_error_rate": 0.30, "propagation_rate": 0.90},
    }
    return {
        label: run_experiment(budgets, n_trials, max_nodes, **params)
        for label, params in regimes.items()
    }

# 6. ADAPTIVE ONLINE SIMULATION  (ported from tree_model)
"""
    Offline (above): build full DAG, then pick a fixed verify_set.
    Adaptive (below): generate DAG incrementally, pick frontier node 
    with highest expected_damage at each step.
    If damage exceeds sigma and verify budget remains, verify (maybe quarantine) it;
    otherwise expand. Repeat until the node budget or frontier is exhausted.       
"""

def run_adaptive_simulation(
    max_nodes: int = 140,
    verify_budget: int = 20,
    sigma = 5.0,
    seed: int = 0,
    max_depth: int = 6,
    branching_lambda: float = 2.2,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
    extra_edge_prob: float = 0.15,
    on_step=None,
) -> tuple["ClaimDAG", set[int]]:
    """
    Generate a claim DAG online, interleaving expansion, verification, and
    quarantine decisions based on expected_damage(v) vs. sigma.

    propagation_rate = base_error_rate + rho 
    -->compute_predicted_error's delta_cont matches rho.

    on_step (optional): callback fired after each step with a dict
    describing what happened (only used for animation/diagnostics)

    Returns (dag, verified_set).
    """
    dag = ClaimDAG(
        max_nodes=max_nodes,
        base_error_rate=base_error_rate,
        propagation_rate=base_error_rate + rho,
        extra_edge_prob=extra_edge_prob,
        max_depth=max_depth,
        branching_lambda=branching_lambda,
        seed=seed,
        build_full=False,
    )

    frontier = {0}
    verified_set: set[int] = set()
    verify_budget_remaining = verify_budget

    if sigma == "adaptive":
        sigma_policy = AdaptiveSigma(verify_budget, max_nodes)
    else:
        sigma_policy = FixedSigma(float(sigma))

    while frontier and len(dag.nodes) < max_nodes:
        remaining_node_budget = max_nodes - len(dag.nodes)

        scored = sorted(
            frontier,
            key=lambda nid: expected_damage(nid, dag, rho, remaining_node_budget),
            reverse=True,
        )
        node_id = scored[0]
        damage = expected_damage(node_id, dag, rho, remaining_node_budget)
        sigma_policy.update(damage)
        current_sigma = sigma_policy.get(verify_budget_remaining, len(dag.nodes))
        p_false_before = dag.nodes[node_id].p_false
        base_event = {
            "node_id": node_id, "damage": damage, "sigma": current_sigma,
            "p_false_before": p_false_before,
            "frontier_before": set(frontier),
        }

        if damage > current_sigma and verify_budget_remaining > 0:
            truth = verify_node(
                dag, node_id, verified_set, epsilon=base_error_rate, rho=rho
            )
            verify_budget_remaining -= 1

            if truth == 0:
                quarantine_node(dag, node_id, frontier)
                if on_step:
                    on_step({**base_event, "action": "verify_quarantine",
                             "truth": truth, "new_children": []})
                continue
            else:
                # Verified true — safe to expand.
                new_children = dag.expand_node(node_id)
                frontier.discard(node_id)
                frontier.update(new_children)
                if on_step:
                    on_step({**base_event, "action": "verify_expand",
                             "truth": truth, "new_children": new_children})
        else:
            # Risk below threshold — expand without verification.
            new_children = dag.expand_node(node_id)
            frontier.discard(node_id)
            frontier.update(new_children)
            if on_step:
                on_step({**base_event, "action": "expand",
                         "truth": None, "new_children": new_children})

    return dag, verified_set


def evaluate_adaptive(dag: "ClaimDAG", verified_set: set[int]) -> dict:
    """
    Metrics for an adaptively-generated DAG (compare to evaluate() for the
    offline/fixed-budget case).
    """
    false_ids = set(dag.false_ids)
    total = len(dag.nodes)

    verified_false = {nid for nid in verified_set if dag.nodes[nid].truth == 0}
    verified_true = {nid for nid in verified_set if dag.nodes[nid].truth == 1}

    quarantined = {nid for nid, n in dag.nodes.items() if n.status == "quarantined"}
    true_branches_blocked = {nid for nid in quarantined if dag.nodes[nid].truth == 1}
    false_branches_blocked = {nid for nid in quarantined if dag.nodes[nid].truth == 0}

    undetected_false = false_ids - verified_set - quarantined
    contamination_rate = len(undetected_false) / total if total else 0.0
    reliability = 1.0 - contamination_rate

    return {
        "total_nodes": total,
        "false_nodes": len(false_ids),
        "undetected_false": len(undetected_false),
        "quarantined": len(quarantined),
        "verified_false": len(verified_false),
        "verified_true": len(verified_true),
        "verify_budget_used": len(verified_set),
        "contamination_rate": contamination_rate,
        "reliability": reliability,
        "true_branches_blocked": len(true_branches_blocked),
        "false_branches_blocked": len(false_branches_blocked),
    }


def run_adaptive_experiment(
    sigmas: list[float] = (0, 1, 2, 5, 10, 20, 50),
    n_trials: int = 50,
    max_nodes: int = 140,
    verify_budget: int = 20,
    max_depth: int = 6,
    branching_lambda: float = 2.2,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
    extra_edge_prob: float = 0.15,
) -> dict:
    """
    Sigma sweep: for each sigma, run n_trials adaptive simulations and
    record mean/std of evaluate_adaptive()'s metrics.

    Returns nested dict: results[sigma] = {"mean": {...}, "std": {...}}
    """
    raw = {sigma: [] for sigma in sigmas}

    for sigma in sigmas:
        for trial in range(n_trials):
            dag, verified_set = run_adaptive_simulation(
                max_nodes=max_nodes,
                verify_budget=verify_budget,
                sigma=sigma,
                seed=trial,
                max_depth=max_depth,
                branching_lambda=branching_lambda,
                base_error_rate=base_error_rate,
                rho=rho,
                extra_edge_prob=extra_edge_prob,
            )
            raw[sigma].append(evaluate_adaptive(dag, verified_set))

    out = {}
    for sigma in sigmas:
        trials = raw[sigma]
        keys = trials[0].keys()
        out[sigma] = {
            "mean": {k: float(np.mean([t[k] for t in trials])) for k in keys},
            "std": {k: float(np.std([t[k] for t in trials])) for k in keys},
        }
    return out


def run_adaptive_sigma_experiment(
    fixed_sigmas: list[float] = (1, 2, 5, 10, 20),
    n_trials: int = 50,
    max_nodes: int = 140,
    verify_budget: int = 20,
    max_depth: int = 6,
    branching_lambda: float = 2.2,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
    extra_edge_prob: float = 0.15,
) -> dict:
    """
    Runs n_trials for each fixed sigma AND for sigma="adaptive".
    Returns results[key] = {"mean": {...}, "std": {...}}
    where key is the float sigma or the string "adaptive".
    """
    all_sigmas = list(fixed_sigmas) + ["adaptive"]
    raw = {s: [] for s in all_sigmas}

    for s in all_sigmas:
        for trial in range(n_trials):
            dag, verified_set = run_adaptive_simulation(
                max_nodes=max_nodes,
                verify_budget=verify_budget,
                sigma=s,
                seed=trial,
                max_depth=max_depth,
                branching_lambda=branching_lambda,
                base_error_rate=base_error_rate,
                rho=rho,
                extra_edge_prob=extra_edge_prob,
            )
            raw[s].append(evaluate_adaptive(dag, verified_set))

    out = {}
    for s in all_sigmas:
        trials = raw[s]
        keys = trials[0].keys()
        out[s] = {
            "mean": {k: float(np.mean([t[k] for t in trials])) for k in keys},
            "std": {k: float(np.std([t[k] for t in trials])) for k in keys},
        }
    return out

# 7. WARM-UP + SUBTREE-CHECKPOINT ADAPTIVE POLICY (ported from tree_model)
"""
    Same policy as tree_model's section 7, applied to the DAG. 
    "Subtree" means descendants_ids(root_id): every node forward-reachable from the
    checkpoint via directed edges. Due to cross-edges, a quarantined
    descendant might be reachable from a non-quarantined node elsewhere in
    the DAG; quarantine_subtree() still marks it quarantined (its own
    generation stops), matching the "quarantine everything downstream of this
    checkpoint" intent.
"""

def expand_to_depth(dag: "ClaimDAG", frontier: set[int], target_depth: int) -> None:
    """
    Expand active frontier nodes, shallowest first, until no
    active frontier node remains with depth < target_depth. 
    No verification happens here.
    """
    while len(dag.nodes) < dag.max_nodes:
        shallow = [
            nid for nid in frontier
            if dag.nodes[nid].depth < target_depth and dag.nodes[nid].status == "active"
        ]
        if not shallow:
            break
        node_id = min(shallow, key=lambda nid: (dag.nodes[nid].depth, nid))
        new_children = dag.expand_node(node_id)
        frontier.discard(node_id)
        frontier.update(new_children)


def select_subtree_root(
    dag: "ClaimDAG", frontier: set[int], rho: float, remaining_node_budget: int
) -> int:
    """Pick the frontier node with the highest expected_damage()."""
    return max(
        frontier,
        key=lambda nid: expected_damage(nid, dag, rho, remaining_node_budget),
    )


def expand_subtree_for_k_levels(
    dag: "ClaimDAG", root_id: int, k: int, frontier: set[int]
) -> list[int]:
    """
    BFS-expands the subtree rooted at root_id, up to k levels past
    root_id's depth. Stops early if it hits a quarantined node or the
    node budget runs out.
    Returns the ids of the nodes that were expanded.
    """
    start_depth = dag.nodes[root_id].depth
    local_frontier = deque([root_id])
    expanded: list[int] = []

    while local_frontier:
        node_id = local_frontier.popleft()
        node = dag.nodes[node_id]

        if node.depth >= start_depth + k:
            continue
        if node.status == "quarantined":
            continue
        if len(dag.nodes) >= dag.max_nodes:
            break

        new_children = dag.expand_node(node_id)
        frontier.discard(node_id)
        frontier.update(new_children)
        expanded.append(node_id)
        local_frontier.extend(new_children)

    return expanded


def quarantine_subtree(dag: "ClaimDAG", root_id: int, frontier: set[int]) -> None:
    """
    Quarantine `root_id` and every node already generated downstream of it.
    """
    for nid in dag.descendants_ids(root_id):
        node = dag.nodes[nid]
        node.status = "quarantined"
        node.expanded = True
        frontier.discard(nid)


def run_warmup_checkpoint_simulation(
    max_nodes: int = 140,
    verify_budget: int = 20,
    sigma_low: float = 3.0,
    sigma_high: float = 8.0,
    warmup_depth: int = 2,
    subtree_expand_depth: int = 2,
    seed: int = 0,
    max_depth: int = 6,
    branching_lambda: float = 2.2,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
    extra_edge_prob: float = 0.15,
    on_step=None,
) -> tuple["ClaimDAG", set[int]]:
    """
    Warm-up + subtree-checkpoint adaptive policy (DAG version).

      damage > sigma_high  (high risk): verify the root immediately;
        if false, quanrantine its subtree; else, expand it one more level.
      sigma_low < damage <= sigma_high (medium risk): let the subtree grow
          for `subtree_expand_depth` levels, then verify the root as a
          checkpoint; if false, quarantine the whole grown subtree.
      damage <= sigma_low (low risk): expand normally without verification.

    If out of verification budget, a node that would have been verified 
    (damage > sigma_high) is quarantined instead of being expanded further.

    Returns (dag, verified_set).
    """
    dag = ClaimDAG(
        max_nodes=max_nodes,
        base_error_rate=base_error_rate,
        propagation_rate=base_error_rate + rho,
        extra_edge_prob=extra_edge_prob,
        max_depth=max_depth,
        branching_lambda=branching_lambda,
        seed=seed,
        build_full=False,
    )

    frontier = {0}
    verified_set: set[int] = set()
    verify_budget_remaining = verify_budget

    # 1. Warm-up — expand the first `warmup_depth` levels, no verification.
    expand_to_depth(dag, frontier, warmup_depth)

    # 2-8. Subtree-checkpoint loop.
    while frontier and len(dag.nodes) < max_nodes:
        remaining_node_budget = max_nodes - len(dag.nodes)
        root_id = select_subtree_root(dag, frontier, rho, remaining_node_budget)
        damage = expected_damage(root_id, dag, rho, remaining_node_budget)
        base_event = {
            "node_id": root_id, "damage": damage,
            "sigma_low": sigma_low, "sigma_high": sigma_high,
            "p_false_before": dag.nodes[root_id].p_false,
            "frontier_before": set(frontier),
        }

        if damage > sigma_high:
            if verify_budget_remaining > 0:
                truth = verify_node(dag, root_id, verified_set,
                                     epsilon=base_error_rate, rho=rho)
                verify_budget_remaining -= 1
                if truth == 0:
                    quarantine_subtree(dag, root_id, frontier)
                    action = "checkpoint_quarantine"
                else:
                    expand_subtree_for_k_levels(dag, root_id, 1, frontier)
                    action = "checkpoint_verify_expand"
            else:
                # No budget left for a high-risk node — quarantine rather
                # than risk expanding it further.
                quarantine_subtree(dag, root_id, frontier)
                action = "budget_exhausted_quarantine"

        elif damage > sigma_low:
            expand_subtree_for_k_levels(dag, root_id, subtree_expand_depth, frontier)
            if verify_budget_remaining > 0:
                truth = verify_node(dag, root_id, verified_set,
                                     epsilon=base_error_rate, rho=rho)
                verify_budget_remaining -= 1
                if truth == 0:
                    quarantine_subtree(dag, root_id, frontier)
                    action = "subtree_checkpoint_quarantine"
                else:
                    action = "subtree_checkpoint_verify"
            else:
                action = "subtree_expand_no_budget"

        else:
            new_children = dag.expand_node(root_id)
            frontier.discard(root_id)
            frontier.update(new_children)
            action = "expand"

        # Safety net: every branch above is expected to remove root_id from
        # the frontier (directly or via quarantine_subtree /
        # expand_subtree_for_k_levels), but discard to avoid an
        # infinite loop on degenerate parameters (e.g. subtree_expand_depth=0).
        frontier.discard(root_id)

        if on_step:
            p_false_after = {nid: n.p_false for nid, n in dag.nodes.items()}
            on_step({**base_event, "action": action, "p_false_after": p_false_after})

    return dag, verified_set


def run_warmup_checkpoint_experiment(
    sigma_lows: list[float] = (1, 2, 3, 5),
    sigma_highs: list[float] = (5, 8, 10, 15),
    n_trials: int = 50,
    max_nodes: int = 140,
    verify_budget: int = 20,
    warmup_depth: int = 2,
    subtree_expand_depth: int = 2,
    max_depth: int = 6,
    branching_lambda: float = 2.2,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
    extra_edge_prob: float = 0.15,
) -> dict:
    """
    Sweep (sigma_low, sigma_high) pairs with sigma_high > sigma_low. 
    For each pair, run n_trials warm-up + checkpoint simulations and record
    mean/std of evaluate_adaptive()'s metrics.

    Returns results[(sigma_low, sigma_high)] = {"mean": {...}, "std": {...}}
    """
    pairs = [(lo, hi) for lo in sigma_lows for hi in sigma_highs if hi > lo]
    out = {}

    for sigma_low, sigma_high in pairs:
        trials = []
        for trial in range(n_trials):
            dag, verified_set = run_warmup_checkpoint_simulation(
                max_nodes=max_nodes,
                verify_budget=verify_budget,
                sigma_low=sigma_low,
                sigma_high=sigma_high,
                warmup_depth=warmup_depth,
                subtree_expand_depth=subtree_expand_depth,
                seed=trial,
                max_depth=max_depth,
                branching_lambda=branching_lambda,
                base_error_rate=base_error_rate,
                rho=rho,
                extra_edge_prob=extra_edge_prob,
            )
            trials.append(evaluate_adaptive(dag, verified_set))

        keys = trials[0].keys()
        out[(sigma_low, sigma_high)] = {
            "mean": {k: float(np.mean([t[k] for t in trials])) for k in keys},
            "std": {k: float(np.std([t[k] for t in trials])) for k in keys},
        }
    return out


if __name__ == "__main__":
    dag, verified = run_adaptive_simulation(sigma="adaptive", seed=0)
    print(evaluate_adaptive(dag, verified))

    adaptive_sigmas_seen: list[float] = []

    def _collect_sigma(event):
        adaptive_sigmas_seen.append(event["sigma"])

    dag_adaptive, verified_adaptive = run_adaptive_simulation(
        sigma="adaptive", seed=0, on_step=_collect_sigma
    )
    print("Adaptive sigma:", evaluate_adaptive(dag_adaptive, verified_adaptive))
    print("Damages tracked:", len(adaptive_sigmas_seen))
    print("Sample current_sigma values:", adaptive_sigmas_seen[:10])