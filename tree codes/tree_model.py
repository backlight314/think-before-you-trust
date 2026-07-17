"""
    Hallucination Propagation Simulation
    Models AI reasoning as a claim tree where errors cascade through dependencies.
    Compares verification strategies under a fixed budget.

    - DP optimal strategy (tree knapsack, O(N * k^2)) treats confidence scores
    as P(false) to find the best verification set given the AI's own beliefs
    - Heuristic gap plot shows how far each strategy falls from that optimum
    - Adaptive simulation supports a fixed sigma threshold or sigma= "adaptive",
    which tracks observed expected_damage values and sets the verification
    threshold to whatever quantile matches the current budget-spend rate
    - Adaptive sigma works with any frontier selection policy (greedy,
    epsilon-greedy, UCB) and any descendant estimator
"""

import numpy as np
import random
import os
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional

# Single source of truth for the damage/propagation math (see damage_core
# docstring). p_union is re-exported from here so existing
# `from tree_model import p_union` imports keep working.
from damage_core import geometric_descendants, expected_damage_score, p_union

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. DATA STRUCTURES

@dataclass
class Node:
    id: int
    depth: int
    parent_id: Optional[int]
    truth: int  # hidden ground truth: 0=false, 1=true
    confidence: float  # AI's self-reported P(true) — miscalibrated, noisy
    # For OFFLINE strategies, p_error is not stored — it is dynamic (depends
    # on verified_set) and computed by compute_predicted_error() at query time.
    # For the ADAPTIVE/online simulation, p_false below IS the live estimate
    # and is mutated in place as verification results propagate.
    children: list = field(default_factory=list)
    is_verified: bool = False
    is_corrupted: bool = False

    #Adaptive/online simulation state
    p_false: float = 0.0  # current estimated P(node is false)
    status: str = "active"  # active | expanded | verified_true |
    # verified_false | quarantined | terminal
    expanded: bool = False  # whether expand_node() has been called on this node

    #Exploration / exploitation tracking
    visits: int = 0           # times this node was selected from the frontier
    total_value: float = 0.0  # cumulative expected_damage seen at selection
    value: float = 0.0        # current expected_damage (pre-computed each step)

    @property
    def num_descendants(self):
        return self._num_descendants

    @num_descendants.setter
    def num_descendants(self, v):
        self._num_descendants = v

    def __post_init__(self):
        self._num_descendants = 0

# 2. TREE GENERATION

class ClaimTree:
    def __init__(
        self,
        max_nodes: int = 80,
        #Controlled variables (fixed across trials)
        base_error_rate: float = 0.15,  # P(node wrong | parent correct)
        propagation_rate: float = 0.85,  # P(child wrong | parent wrong)
        #Topology variables (randomized per trial)
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
        self.overconfident = overconfident

        #Randomize topology per trial if not provided
        self.max_depth = (
            max_depth if max_depth is not None else int(self.rng.integers(4, 8))
        ) 
        self.branching_lambda = (
            branching_lambda
            if branching_lambda is not None
            else float(self.rng.uniform(1.5, 3.0))
        )  #Sparse→bushy

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
        if truth == 1:
            return float(np.clip(self.rng.beta(5, 2), 0.40, 0.99))
        else:
            if getattr(self, "overconfident", False):
                #Matches real LLM behavior: false claims at 0.82–0.97 confidence
                return float(np.clip(self.rng.beta(8, 2), 0.55, 0.99))
            else:
                #Default: model is uncertain about false claims (mean ≈ 0.28)
                return float(np.clip(self.rng.beta(2, 5), 0.05, 0.80))
        
    def _compute_confidence(
        self,
        node_id: int,
        _truth: int,
        intrinsic: float,
    ) -> float:
        """
        Confidence = how certain we are in the predicted error estimate for this node.

        Driven by two structural factors:
        coverage    = fraction of ancestors that are verified
                        (more verified ancestors → more certain about propagation chain)
        consistency = 1 - std(truth values of verified ancestors)
                        (all-true or all-false chain → certain;
                        mixed true/false ancestors → uncertain about propagation)

        Final blend:
        confidence = coverage * (coverage * consistency + (1 - coverage) * intrinsic)

        Limiting cases:
        - No verified ancestors:    confidence ≈ intrinsic  (pure self-report)
        - All verified, clean chain: confidence ≈ 1.0
        - All verified, mixed chain: confidence ≈ coverage * consistency (penalized)
        """
        #Collect ancestor chain (parent → grandparent → ... → root)
        ancestors = []
        cur = self.nodes[node_id].parent_id
        while cur is not None:
            ancestors.append(cur)
            cur = self.nodes[cur].parent_id

        if not ancestors:
            return intrinsic  #root: no ancestors, pure intrinsic

        #Coverage: fraction of ancestors that are verified at BUILD time.
        #At build time nothing is verified yet, so this is 0 — but we
        #Compute it here so the method is reusable post-verification.
        #During tree construction, confidence = intrinsic (correct behavior:
        #the tree is built before any verification happens).
        verified_ancestors = [a for a in ancestors if self.nodes[a].is_verified]
        coverage = len(verified_ancestors) / len(ancestors)

        if coverage == 0.0:
            return intrinsic

        #std of Bernoulli(p) is sqrt(p*(1-p)), max at p=0.5 → std=0.5
        #Normalize: consistency = 1 - 2*std  so 0.5 → 0, 0 or 1 → 1
        truth_vals = [self.nodes[a].truth for a in verified_ancestors]
        p = float(np.mean(truth_vals))
        std = float(np.sqrt(p * (1.0 - p)))
        consistency = 1.0 - 2.0 * std  # ∈ [0, 1]

        # Blend structural certainty with intrinsic self-report
        structural = coverage * consistency + (1.0 - coverage) * intrinsic
        confidence = coverage * structural + (1.0 - coverage) * intrinsic

        #Simplifies to:
        #confidence = coverage^2 * consistency + coverage*(1-coverage)*intrinsic
        #            + (1-coverage)*intrinsic
        #= coverage^2 * consistency + (1 - coverage^2) * intrinsic

        return float(np.clip(confidence, 0.01, 0.99))

    def _true_error_prob(self, parent_id: Optional[int]) -> float:
        """
        The probability this node is false, given the true (hidden)
        states of all ancestors. Used only during tree construction to sample
        ground truth 
        The true error probability is not exposed to strategies.

        Rule: if the direct parent is false, the contamination propagates at
        propagation_rate. If the parent is true, the base rate applies.
        A single false parental node causes a chain reaction, 
        propagation_rate compounds through the ancestor chain via the parent's own truth value.
        """
        if parent_id is None:
            return 0.0  #Root is always true
        parent = self.nodes[parent_id]
        if parent.truth == 0:
            return self.propagation_rate
        else:
            return self.base_error_rate

    def _initialize_root(self):
        """Create only the root node, the tree is NOT fully built here."""
        root_intrinsic = self._make_intrinsic_confidence(1)
        root = Node(
            id=self._new_id(),
            depth=0,
            parent_id=None,
            truth=1,
            confidence=root_intrinsic,
        )
        root.p_false = float(np.clip(1.0 - root.confidence, 0.0, 1.0))
        self.nodes[root.id] = root

    def expand_node(
        self, node_id: int, max_new_children: Optional[int] = None
    ) -> list[int]:
        """
        Generate (reveal) the children of `node_id`, if it is eligible to expand.

        Returns the list of newly created child ids (empty if the node could
        not be expanded, e.g. quarantined, depth limit reached, or the node
        budget is exhausted).
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
            #ground truth: sampled from actual parent truth value
            if node.truth == 0:
                child_truth = int(self.rng.random() > self.propagation_rate)
                corrupted = child_truth == 0
            else:
                child_truth = int(self.rng.random() > self.base_error_rate)
                corrupted = False

            #confidence: intrinsic self-report at generation time ---
            intrinsic = self._make_intrinsic_confidence(child_truth)

            # Allocate node first so it exists in self.nodes when
            # _compute_confidence walks the ancestor chain
            child = Node(
                id=self._new_id(),
                depth=node.depth + 1,
                parent_id=node_id,
                truth=child_truth,
                confidence=intrinsic,  # placeholder
                is_corrupted=corrupted,
            )
            self.nodes[child.id] = child
            node.children.append(child.id)
            new_children.append(child.id)

            #Overwrite placeholder with full structural confidence, which
            #accounts for any ancestors verified so far.
            child.confidence = self._compute_confidence(
                child.id, child_truth, intrinsic
            )
            #Initial p_false estimate: with no verified ancestors this 
            #reduces to 1 - confidence.
            child.p_false = float(np.clip(1.0 - child.confidence, 0.0, 1.0))

        node.expanded = True
        if node.status == "active":
            node.status = "expanded" if new_children else "terminal"

        return new_children

    def _build_full(self):
        """Expand the whole tree via BFS (offline/legacy behavior)."""
        queue = deque([0])
        while queue and len(self.nodes) < self.max_nodes:
            pid = queue.popleft()
            new_children = self.expand_node(pid)
            queue.extend(new_children)

        self._compute_descendants()

    def _compute_descendants(self):
        for nid in self._postorder(0):
            node = self.nodes[nid]
            node.num_descendants = sum(
                1 + self.nodes[c].num_descendants for c in node.children
            )

    def _postorder(self, nid):
        for c in self.nodes[nid].children:
            yield from self._postorder(c)
        yield nid

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

    def subtree_ids(self, nid) -> list[int]:
        result, stack = [], [nid]
        while stack:
            cur = stack.pop()
            result.append(cur)
            stack.extend(self.nodes[cur].children)
        return result

# 2b. PREDICTED ERROR MODEL

def compute_predicted_error(
    node_id: int,
    tree: "ClaimTree",
    verified_set: set[int],
    p_base: float = None,
    delta_unc: float = None,
    delta_cont: float = None,
) -> float:
    """
    The verification model's estimate of P(node is false), given the
    current verification state of all ancestors. 
    (strategy observed value, not ground truth)

    Three-state propagation:
      p_error(v) = p_base                    if parent is verified-TRUE
                 = p_base + delta_unc        if parent is unchecked
                 = p_base + delta_cont       if parent is verified-FALSE

    delta_unc = delta_cont * pi_hat
    pi_hat = empirical false-node rate = E[P(any unchecked node is false)]
    approx. (1 - mean(confidence)) across unverified ancestors.

    For nodes with no verified ancestors this degrades gracefully to
    p_base + delta_unc (fully uncertain), which is higher than p_base
    but lower than the full contamination penalty.

    The AI's self-reported confidence is folded in as a Bayesian update:
    the final estimate blends the ancestor-propagation estimate with
    the node's own confidence score, weighted by the number of verified ancestors
    (more = trust propagation model more;
    fewer = trust the AI's own confidence more).
    """
    node = tree.nodes[node_id]

    # Use tree's calibrated parameters if not overridden
    if p_base is None:
        p_base = tree.base_error_rate
    if delta_cont is None:
        delta_cont = tree.propagation_rate - p_base
        delta_cont = max(delta_cont, 0.0)

    # Walk ancestor chain to find verification state of each ancestor
    # Collect all ancestors in order from parent up to root
    ancestors = []
    cur = node.parent_id
    while cur is not None:
        ancestors.append(cur)
        cur = tree.nodes[cur].parent_id

    if not ancestors:
        # Root node: no ancestors, return node's own miscalibration
        return 1.0 - node.confidence

    #Compute pi_hat: estimated false-node rate among unverified ancestors
    unverified_ancestors = [a for a in ancestors if a not in verified_set]
    if unverified_ancestors:
        pi_hat = float(
            np.mean([1.0 - tree.nodes[a].confidence for a in unverified_ancestors])
        )
    else:
        pi_hat = 0.0

    if delta_unc is None:
        delta_unc = delta_cont * pi_hat

    #Find the DIRECT PARENT's verification state
    parent_id = node.parent_id
    parent_verified = parent_id in verified_set

    if parent_verified:
        parent_node = tree.nodes[parent_id]
        # If parent is verified-true: clean propagation
        # If parent is verified-false: full contamination penalty
        # Note: in evaluate() --> quarantine verified false nodes' subtree,
        # but during strategy selection --> still want to score the node.
        if parent_node.truth == 1:
            p_prop = p_base
        else:
            p_prop = p_base + delta_cont
    else:
        # Parent is unchecked: use uncertainty penalty
        p_prop = p_base + delta_unc

    #Blend propagation estimate with AI's self-reported confidence
    # Weight toward propagation model when more ancestors are verified
    # (we have more structural information), toward AI confidence when few are.
    n_verified = len(ancestors) - len(unverified_ancestors)
    n_total = len(ancestors)
    # alpha: fraction of ancestors verified
    alpha = n_verified / n_total  # 0 = none verified, 1 = all verified

    #P(false) stated by AI
    p_ai = 1.0 - node.confidence

    # Blended estimate
    p_predicted = alpha * p_prop + (1.0 - alpha) * p_ai

    return float(np.clip(p_predicted, 0.0, 1.0))

# 2c. ADAPTIVE ONLINE VERIFICATION

def estimate_future_descendants(
    node: Node, tree: "ClaimTree", remaining_node_budget: int
) -> float:
    """
    Estimates how many more descendants node could produce if expansion 
    keeps going to tree.max_depth, bounded by whatever node budget remains.
    Just delegates to damage_core.geometric_descendants, the single shared 
    function both models use for branching/expansion math.
    """
    return geometric_descendants(
        tree.branching_lambda,
        tree.max_depth - node.depth,
        cap=remaining_node_budget,
    )


def estimate_branching_factor(tree: "ClaimTree") -> float:
    """
    Empirical global branching factor of a (typically fully-built) tree:
        b_hat = total_children / total_expandable_nodes
    where "expandable" nodes are those with depth < tree.max_depth.
    """
    expandable = [n for n in tree.nodes.values() if n.depth < tree.max_depth]
    if not expandable:
        return 0.0
    total_children = sum(len(n.children) for n in expandable)
    return total_children / len(expandable)

def calibrate_branching_from_simulations(
    n_trials: int = 200,
    max_nodes: int = 80,
    max_depth: int = 8,
    branching_lambda: float = 1.7,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
) -> dict:
    """
    Build `n_trials` full offline ClaimTrees and empirically calibrate the
    branching factor used by StructuralDescendantEstimator.

    Returns:
        {
            "global_mean": mean of per-tree estimate_branching_factor(),
            "global_std": std of the same,
            "ci95_low": 95% CI lower bound for the mean (normal approx.),
            "ci95_high": 95% CI upper bound for the mean,
            "depth_means": {depth: avg_children_at_depth}, pooled over
                           all trials,
        }
    """
    global_bhats = []
    children_by_depth: dict[int, list[int]] = defaultdict(list)

    for trial in range(n_trials):
        tree = ClaimTree(
            max_nodes=max_nodes,
            base_error_rate=base_error_rate,
            propagation_rate=base_error_rate + rho,
            max_depth=max_depth,
            branching_lambda=branching_lambda,
            seed=trial,
            build_full=True,
        )

        global_bhats.append(estimate_branching_factor(tree))
        for n in tree.nodes.values():
            if n.depth < tree.max_depth:
                children_by_depth[n.depth].append(len(n.children))

    global_bhats = np.array(global_bhats)
    global_mean = float(global_bhats.mean())
    global_std = float(global_bhats.std())
    se = global_std / np.sqrt(len(global_bhats))

    depth_means = {
        depth: float(np.mean(counts))
        for depth, counts in sorted(children_by_depth.items())
    }

    return {
        "global_mean": global_mean,
        "global_std": global_std,
        "ci95_low": global_mean - 1.96 * se,
        "ci95_high": global_mean + 1.96 * se,
        "depth_means": depth_means,
    }


DESCENDANT_FEATURE_NAMES = [
    "depth",
    "remaining_depth",
    "confidence",
    "p_false",
    "remaining_node_budget",
    "avg_branching_so_far",
    "expanded_nodes_so_far",
]

DESCENDANT_FEATURE_NAMES_CLEAN = [
    "depth",
    "remaining_depth",
    "confidence",
    "static_p_false",      # p_false at creation time; never propagation-updated
    "parent_n_children",   # number of siblings (parent already expanded)
    "parent_confidence",   # parent's own confidence (available: parent pre-exists)
]


def extract_descendant_features(
    tree: "ClaimTree",
    node_id: int,
    remaining_node_budget: int,
    avg_branching_so_far: float = 0.0,
    expanded_nodes_so_far: int = 0,
) -> list[float]:
    """
    Builds the feature vector for node's position in the tree, used by the
    learned descendant estimator. Order matches DESCENDANT_FEATURE_NAMES:
    [depth, remaining_depth, confidence, p_false, remaining_node_budget,
    avg_branching_so_far, expanded_nodes_so_far]
    The caller passes in avg_branching_so_far and expanded_nodes_so_far,
    and reflect only what's been revealed/expanded prior to node expansion.
    This function does not touch node.children or node.num_descendants
    """
    node = tree.nodes[node_id]

    return [
        float(node.depth),
        float(tree.max_depth - node.depth),
        float(node.confidence),
        float(node.p_false),
        float(remaining_node_budget),
        float(avg_branching_so_far),
        float(expanded_nodes_so_far),
    ]


def extract_descendant_features_clean(
    tree: "ClaimTree",
    node_id: int,
) -> list[float]:
    """
    Vector with no traversal state, only static node-time properties.
    Order matches DESCENDANT_FEATURE_NAMES_CLEAN.

    All features here is fixed as the node enters the tree (no
    global BFS counters, no policy budget)

    static_p_false = 1 - confidence instead of read from node.p_false,
    since node.p_false is mutable: verify_node() calls
    update_descendant_probabilities(), which overwrites p_false in place for
    any descendant of a verified node.
    Verified node is not called during offline training, however, a frontier node
    can already have a verified ancestor by the time it's scored, so reading node.p_false 
    life would drift from what the estimator actually trained on.  
    """
    node = tree.nodes[node_id]
    if node.parent_id is not None and node.parent_id in tree.nodes:
        parent_n_children = float(len(tree.nodes[node.parent_id].children))
        parent_confidence = float(tree.nodes[node.parent_id].confidence)
    else:
        parent_n_children = 0.0  # root: no parent
        parent_confidence = float(node.confidence)  # root: no parent, use own
    return [
        float(node.depth),
        float(tree.max_depth - node.depth),
        float(node.confidence),
        float(1.0 - node.confidence),  # static_p_false — see docstring
        parent_n_children,
        parent_confidence,
    ]


def collect_descendant_training_data_clean(
    n_trials: int = 200,
    max_nodes: int = 80,
    max_depth: int = 8,
    branching_lambda: float = 1.7,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Clean training data for a learned descendant-count estimator.
    Uses descendant_feature_names_clean (no BFS traversal state) 
    Nodes are collected in BFS order from fully-built trees so that a
    sequential slice gives a valid temporal-style split:
        X[:n_train], y[:n_train]  → training set
        X[n_train:], y[n_train:]  → held-out test set

    Target: y = node.num_descendants (unchanged from original).
    """
    X, y = [], []
    for trial in range(n_trials):
        full_tree = ClaimTree(
            max_nodes=max_nodes,
            base_error_rate=base_error_rate,
            propagation_rate=base_error_rate + rho,
            max_depth=max_depth,
            branching_lambda=branching_lambda,
            seed=trial,
            build_full=True,
        )
        # BFS order preserves parent-before-child ordering (creation order)
        queue = deque([0])
        visited: set[int] = set()
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            X.append(extract_descendant_features_clean(full_tree, nid))
            y.append(float(full_tree.nodes[nid].num_descendants))
            for child in full_tree.nodes[nid].children:
                queue.append(child)
    return np.array(X, dtype=np.float32), np.array(y), list(DESCENDANT_FEATURE_NAMES_CLEAN)


def collect_descendant_training_data(
    n_trials: int = 200,
    max_nodes: int = 80,
    max_depth: int = 8,
    branching_lambda: float = 1.7,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build `n_trials` full offline ClaimTrees (to learn the eventual
    num_descendants targets), then replay each tree as an online BFS reveal
    process and snapshot extract_descendant_features() right *before* each
    node is expanded — matching exactly what run_adaptive_simulation() sees
    when it scores a frontier node.

    avg_branching_so_far and expanded_nodes_so_far are tracked from the
    reveal walk itself (not read off the finished tree), so features never
    leak a node's own (future) children.

    Returns (X, y, feature_names) where feature_names == DESCENDANT_FEATURE_NAMES.
    """
    X, y = [], []

    for trial in range(n_trials):
        full_tree = ClaimTree(
            max_nodes=max_nodes,
            base_error_rate=base_error_rate,
            propagation_rate=base_error_rate + rho,
            max_depth=max_depth,
            branching_lambda=branching_lambda,
            seed=trial,
            build_full=True,
        )

        revealed = {0}
        frontier = deque([0])
        expanded: set[int] = set()
        edges_revealed = 0

        while frontier:
            nid = frontier.popleft()
            node = full_tree.nodes[nid]

            remaining_node_budget = max_nodes - len(revealed)
            avg_branching_so_far = edges_revealed / len(expanded) if expanded else 0.0

            X.append(
                extract_descendant_features(
                    full_tree,
                    nid,
                    remaining_node_budget,
                    avg_branching_so_far=avg_branching_so_far,
                    expanded_nodes_so_far=len(expanded),
                )
            )
            y.append(float(node.num_descendants))

            expanded.add(nid)
            edges_revealed += len(node.children)
            for child in node.children:
                revealed.add(child)
                frontier.append(child)

    return np.array(X), np.array(y), list(DESCENDANT_FEATURE_NAMES)


class StructuralDescendantEstimator:
    """
    Hand-crafted estimator for D_hat(v) = E[future descendants of v],
    capped by the remaining node budget.

    branching_factor=None, depth_branching=None (default):
        delegates to estimate_future_descendants(), i.e.
        D_hat = sum_{i=1}^{r} b^i  with b = tree.branching_lambda.

    branching_factor=b_hat (a calibrated scalar):
        D_hat = sum_{i=1}^{r} b_hat^i

    depth_branching={depth: b_d} (a calibrated per-depth branching factor):
        D_hat(d) = b_d + b_d*b_{d+1} + b_d*b_{d+1}*b_{d+2} + ...
        i.e. the product of branching factors accumulates one depth at a
        time. Depths missing from depth_branching fall back to
        branching_factor (if given) or tree.branching_lambda.

    r = remaining_depth = tree.max_depth - node.depth.
    """

    def __init__(self, branching_factor: float = None, depth_branching: dict = None):
        self.branching_factor = branching_factor
        self.depth_branching = depth_branching

    def predict(
        self, tree: "ClaimTree", node_id: int, remaining_node_budget: int, **_
    ) -> float:
        node = tree.nodes[node_id]

        if self.branching_factor is None and self.depth_branching is None:
            return estimate_future_descendants(node, tree, remaining_node_budget)

        remaining_depth = tree.max_depth - node.depth
        if remaining_depth <= 0:
            return 0.0

        fallback_b = (
            self.branching_factor
            if self.branching_factor is not None
            else tree.branching_lambda
        )

        if self.depth_branching is not None:
            structural = 0.0
            product = 1.0
            for j in range(remaining_depth):
                b_d = self.depth_branching.get(node.depth + j, fallback_b)
                product *= b_d
                structural += product
        else:
            structural = geometric_descendants(self.branching_factor, remaining_depth)

        return min(structural, remaining_node_budget)


def expected_damage(
    node_id: int,
    tree: "ClaimTree",
    _rho: float,
    remaining_node_budget: int,
    descendant_estimator=None,
) -> float:
    """
    ExpectedDamage(v) = p_error(v) * E[future descendants of v]

    Under the union error model, rho is already embedded in p_false through
    the recursive propagation chain, so it is not an additional multiplier here.
    The rho parameter is retained in the signature for backward compatibility
    but is no longer used in the computation.

    descendant_estimator supplies E[future descendants] (an object with a
    .predict(tree, node_id, remaining_node_budget, avg_branching_so_far=...,
    expanded_nodes_so_far=...) method); defaults to
    StructuralDescendantEstimator() if not given.
    """
    if descendant_estimator is None:
        descendant_estimator = StructuralDescendantEstimator()

    node = tree.nodes[node_id]

    expanded_nodes = [n for n in tree.nodes.values() if n.expanded]
    edges_revealed = sum(len(n.children) for n in expanded_nodes)
    avg_branching_so_far = (
        edges_revealed / len(expanded_nodes) if expanded_nodes else 0.0
    )

    d_hat = descendant_estimator.predict(
        tree,
        node_id,
        remaining_node_budget,
        avg_branching_so_far=avg_branching_so_far,
        expanded_nodes_so_far=len(expanded_nodes),
    )
    d_hat = max(0.0, min(d_hat, remaining_node_budget))
    # UNION model: rho already embedded in p_false, so rho=1.0 here
    # (see damage_core.expected_damage_score docstring).
    return expected_damage_score(node.p_false, d_hat)


# p_union now lives in damage_core (imported at top of this file) — the
# core error model used throughout propagation. It replaces the additive
# approximation clip(epsilon + rho * p_parent, 0, 1).


#Exploration / exploitation selection policies


def ucb_score(node_id: int, tree: "ClaimTree", total_visits: int, c: float = 1.5) -> float:
    """
    UCB(i) = Q(i) + c * sqrt(ln(N + 1) / (n_i + 1))

    Q(i)  = node.value  (expected_damage, pre-computed for the whole frontier)
    N     = total selections so far (global visit count)
    n_i   = times node i has been selected from the frontier
    c     = exploration constant (higher → more exploration; 1.5 is a good start)

    Call after pre-computing node.value for all frontier nodes; the loop in
    run_adaptive_simulation does this automatically each step.
    """
    node = tree.nodes[node_id]
    return node.value + c * np.sqrt(np.log(total_visits + 1) / (node.visits + 1))

def select_node_ucb(
    frontier: set, tree: "ClaimTree", total_visits: int, c: float = 1.5
) -> int:
    """Return the frontier node with the highest UCB score (uses node.value)."""
    return max(frontier, key=lambda nid: ucb_score(nid, tree, total_visits, c))


def select_node_epsilon(
    frontier: set, tree: "ClaimTree", epsilon: float = 0.15
) -> int:
    """
    ε-greedy selection (uses node.value — must be pre-computed):
        with prob ε → explore: pick the least-visited frontier node
        otherwise  → exploit: pick argmax node.value
    """
    if random.random() < epsilon:
        return min(frontier, key=lambda nid: tree.nodes[nid].visits)
    return max(frontier, key=lambda nid: tree.nodes[nid].value)


def update_descendant_probabilities(
    tree: "ClaimTree", verified_node_id: int, epsilon: float, rho: float
) -> None:
    """
    After `verified_node_id` has been verified (its p_false already set to
    0.0 or 1.0), propagate the update to its EXISTING descendants via BFS:

        p_prop = rho * parent.p_false
        child.p_false = 1 - (1 - epsilon)(1 - p_prop)

    Only descendants that already exist in the tree are touched — nodes
    that have not been generated yet simply inherit the updated p_false
    of their parent when they are eventually created by expand_node().

    The BFS prunes a branch once the update no longer changes p_false by
    more than 1e-4, since deeper descendants would not change either.
    """
    queue = deque(tree.nodes[verified_node_id].children)

    while queue:
        child_id = queue.popleft()
        child = tree.nodes[child_id]
        parent = tree.nodes[child.parent_id]

        old = child.p_false
        child.p_false = p_union(epsilon, rho * parent.p_false)

        if abs(child.p_false - old) > 1e-4:
            queue.extend(child.children)


def verify_node(
    tree: "ClaimTree",
    node_id: int,
    verified_set: set[int],
    epsilon: Optional[float] = None,
    rho: Optional[float] = None,
) -> int:
    """
    Reveal the ground truth of `node_id`, lock in its p_false (0 or 1), and
    propagate the update to its existing descendants. Returns node.truth.
    """
    if epsilon is None:
        epsilon = tree.base_error_rate
    if rho is None:
        rho = max(tree.propagation_rate - tree.base_error_rate, 0.0)

    node = tree.nodes[node_id]
    verified_set.add(node_id)
    node.is_verified = True

    if node.truth == 1:
        node.status = "verified_true"
        node.p_false = 0.0
    else:
        node.status = "verified_false"
        node.p_false = 1.0

    update_descendant_probabilities(tree, node_id, epsilon, rho)

    return node.truth


def quarantine_node(tree: "ClaimTree", node_id: int, frontier: set[int]) -> None:
    """
    Mark a verified-false node as quarantined: it will never be expanded
    again, and is removed from the frontier. Its (nonexistent) descendants
    are never generated, so there is nothing to delete.
    """
    node = tree.nodes[node_id]
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

# 2d. SINGLE-BRANCH DFS SIMULATION

def expand_and_propagate(
    tree: "ClaimTree", node_id: int, epsilon: float, rho: float
) -> list[int]:
    """
    Generate children of node_id and set each child's p_false via the union error model:
        p_prop = rho * parent.p_false
        child.p_false = 1 - (1 - epsilon)(1 - p_prop)

    Returns [] if the node was already expanded (idempotent guard).
    """
    parent = tree.nodes[node_id]
    if parent.expanded:
        return []
    children = tree.expand_node(node_id)
    for cid in children:
        child = tree.nodes[cid]
        child.p_false = p_union(epsilon, rho * parent.p_false)
        child.error_propagation = child.p_false
    return children


def run_dfs_simulation(
    max_nodes: int = 80,
    verify_budget: int = 8,
    sigma: float = 0.50,
    seed: int = 0,
    branching_lambda: float = 1.7,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
) -> tuple:
    """
    Single-branch live-propagating DFS simulation.

    Mirrors tree_simulation.dfs_with_log exactly, without the
    logging/frame-building overhead:

      1. DFS follows ONE branch at a time; all siblings go to a LIFO frontier.
      2. Each child's p_false is set via the Markov formula on expansion:
             child.p_false = clip(epsilon + rho * parent.p_false, 0, 1)
      3. When p_false >= sigma (THRESHOLD_HIT), the direct parent is verified.
         - TRUE  → p_false resets to 0; child cleared to eps; continue deeper.
         - FALSE → quarantine parent's subtree; pop next frontier branch.
      4. Backtrack on LEAF (no children) or MAX_DEPTH.

    sigma is a p_false threshold in [0, 1], not a risk threshold.

    Returns (tree, candidates, explored, verified_set, quarantined).
    """
    tree = ClaimTree(
        max_nodes=max_nodes,
        base_error_rate=base_error_rate,
        propagation_rate=base_error_rate + rho,
        branching_lambda=branching_lambda,
        seed=seed,
        build_full=False,
    )

    epsilon = base_error_rate
    rho_prop = rho

    path: list = []
    frontier: list = []
    candidates: set = set()
    explored: set = set()
    verified_set: set = set()
    quarantined: set = set()
    cleared: set = set()

    def _d_hat(nid: int) -> float:
        node = tree.nodes[nid]
        remaining = max(0, max_nodes - len(tree.nodes))
        rem_d = max(0, tree.max_depth - node.depth)
        lam = tree.branching_lambda
        if lam > 1 and rem_d > 0:
            est = geometric_descendants(lam, rem_d)
        else:
            est = float(rem_d)
        return float(min(remaining, max(0.0, est)))

    def _annotate(nid: int) -> None:
        node = tree.nodes[nid]
        d_hat = _d_hat(nid)
        node.d_hat = d_hat
        node.error_propagation = node.p_false
        node.risk = node.p_false * d_hat

    def _do_verify(nid: int):
        if len(verified_set) >= verify_budget or nid in verified_set:
            return None
        truth = verify_node(tree, nid, verified_set, epsilon=epsilon, rho=rho_prop)
        for pid in path:
            if pid in tree.nodes:
                _annotate(pid)
        return truth

    def step_down(nid: int) -> tuple:
        node = tree.nodes[nid]
        _annotate(nid)

        if node.error_propagation >= sigma:
            return "THRESHOLD_HIT", nid
        if node.depth >= tree.max_depth:
            return "MAX_DEPTH", nid

        children = expand_and_propagate(tree, nid, epsilon, rho_prop)
        explored.add(nid)

        if not children:
            return "LEAF", nid

        for cid in children:
            _annotate(cid)

        for cid in children:
            if tree.nodes[cid].p_false >= sigma:
                candidates.add(cid)

        by_ep = sorted(children, key=lambda c: tree.nodes[c].error_propagation)
        next_child = by_ep[0]
        for sib in by_ep[1:]:
            frontier.append(sib)

        return "CONTINUE", next_child

    #Initialise root
    root = tree.nodes[0]
    root.p_false = float(np.clip(1.0 - root.confidence, 0.0, 1.0))
    _annotate(0)
    current = 0
    path = [0]

    guard = 0
    while (path or frontier) and len(verified_set) < verify_budget:
        guard += 1
        if guard > 20_000:
            break

        if not path:
            if not frontier:
                break
            current = frontier.pop()
            _annotate(current)
            if tree.nodes[current].error_propagation >= sigma:
                candidates.add(current)
                _do_verify(current)
            else:
                path = [current]
            continue

        _annotate(current)
        status, result = step_down(current)

        if status == "CONTINUE":
            current = result
            path.append(current)

        elif status == "THRESHOLD_HIT":
            candidates.add(current)
            ancestor = path[-2] if len(path) >= 2 else path[-1]
            truth = _do_verify(ancestor)

            for pid in path:
                if pid in tree.nodes:
                    _annotate(pid)
                    if pid in candidates and tree.nodes[pid].p_false < sigma:
                        cleared.add(pid)
                        candidates.discard(pid)

            if truth is None:
                path = []
            elif truth == 1:
                pass  # current stays in path; next iter calls step_down(current)
            else:
                qsub = set(tree.subtree_ids(ancestor))
                for nid in qsub:
                    tree.nodes[nid].status = "quarantined"
                frontier[:] = [f for f in frontier if f not in qsub]
                quarantined.add(ancestor)
                path = []

        else:  # LEAF or MAX_DEPTH
            path.pop()
            if path:
                current = path[-1]

    return tree, candidates, explored, verified_set, quarantined


# 3. VERIFICATION STRATEGIES

def strategy_oracle(tree: ClaimTree, _budget: int, **_) -> set[int]:
    """Check everything — upper bound baseline."""
    return set(tree.all_ids)


def strategy_random(tree: ClaimTree, budget: int, **_) -> set[int]:
    pool = tree.all_ids
    return set(random.sample(pool, min(budget, len(pool))))


def strategy_level(tree: ClaimTree, budget: int, **_) -> set[int]:
    by_depth = defaultdict(list)
    for nid, node in tree.nodes.items():
        by_depth[node.depth].append(nid)
    depths = sorted(by_depth.keys())
    n_levels = max(1, budget // max(len(v) for v in by_depth.values()))
    step = max(1, len(depths) // n_levels)
    selected_depths = set(depths[::step])
    candidates = [nid for d in selected_depths for nid in by_depth[d]]
    return set(random.sample(candidates, min(budget, len(candidates))))


def strategy_recent(tree: ClaimTree, budget: int, **_) -> set[int]:
    sorted_nodes = sorted(tree.nodes.items(), key=lambda x: -x[1].depth)
    return {nid for nid, _ in sorted_nodes[:budget]}


def strategy_uncertainty(tree: ClaimTree, budget: int, **_) -> set[int]:
    """Greedy on predicted error probability given current verification state."""
    # At selection time, no nodes are pre-verified, so this uses the
    # prior predicted error (blended confidence + propagation prior).
    scored = sorted(
        tree.nodes.items(),
        key=lambda x: compute_predicted_error(x[0], tree, set()),
        reverse = True,
    )
    return {nid for nid, _ in scored[:budget]}


def strategy_dependency_aware(
    tree: ClaimTree,
    budget: int,
    w_conf: float = 0.57,
    w_depth: float = -0.22,
    w_desc: float = 0.65,
    w_parent: float = 0.0,
) -> set[int]:
    """
    Composite score using predicted error (not raw confidence).

    score(v) = w_conf  * predicted_p_error(v)
             + w_depth * (depth / max_depth)
             + w_desc  * (descendants / max_descendants)
             + w_parent* predicted_p_error(parent(v))

    predicted_p_error accounts for ancestor verification state,
    folding in delta_unc from the unchecked ancestor chain.

    Defaults fit via grid search maximizing mean reliability across budgets
    10-40 (see weight_fitting.py, results/weight_fitting_notes.md). Holdout
    (50 trees never used in fitting): mean reliability 0.967 vs. 0.918
    hand-picked, paired Delta=+0.0495, 95% CI [0.036, 0.063]. The 'Conf +
    Desc' ablation (w_conf=0.5, w_desc=0.5, no depth/parent term at all)
    scores 0.964 on the same holdout set, corroborating that depth/parent
    add little once predicted-error and descendant-count are in the score.
    w_depth is negative (not the in-sample-optimal 0.0 from an earlier,
    boundary-clamped search) because reliability keeps improving as deeper
    nodes are mildly deprioritized beyond what lower descendant-count
    already implies; this was confirmed to be a true interior optimum by
    widening the search until it stopped touching its own bounds.
    """
    max_depth = tree.max_depth_actual or 1
    max_desc = tree.max_descendants
    empty_verified: set[int] = set()  # no verifications yet at scoring time

    scores = {}
    for nid, node in tree.nodes.items():
        s_conf = compute_predicted_error(nid, tree, empty_verified)
        s_depth = node.depth / max_depth
        s_desc = node.num_descendants / max_desc
        s_parent = (
            compute_predicted_error(node.parent_id, tree, empty_verified)
            if node.parent_id is not None
            else 0.0
        )
        scores[nid] = (
            w_conf * s_conf + w_depth * s_depth + w_desc * s_desc + w_parent * s_parent
        )
    top = sorted(scores, key=scores.__getitem__, reverse=True)
    return set(top[:budget])

# DP OPTIMAL STRATEGY


def strategy_dp_optimal(tree: ClaimTree, budget: int, **_) -> set[int]:
    """
    Theoretically optimal verification set under budget k.

    Tree DP (knapsack on a tree):
      V(v, b) = expected number of false nodes caught or quarantined in
                subtree(v) when optimally spending b checks.

    This directly maximises reliability = 1 − undetected_false / total.
    Verifying a TRUE node contributes 0 to reducing undetected_false, so
    it receives no direct credit in the value function — only the budget
    freed for its children matters.

    Recurrence:
      V(v, b) = max(
          BestSplit(children, b),                           <- skip v
          pf·sub(v) + (1−pf)·BestSplit(children, b−1)     <- verify v
      )
      where pf = compute_predicted_error(v), sub(v) = num_descendants + 1

    Base case: V(leaf, 0) = 0; V(leaf, b≥1) = pf
    (leaf verification catches it only when it is actually false)

    Note: pf·sub(v) approximates the expected false nodes quarantined when v
    is false, treating pf as the uniform contamination rate across the subtree.
    This is a static-risk relaxation (p_false fixed at pre-verification values).

    Budget capped at DP_BUDGET_CAP to stay fast.
    """
    DP_BUDGET_CAP = 50  # cap so O(N*k^2) stays tractable
    budget = min(budget, DP_BUDGET_CAP)

    nodes = tree.nodes
    p_false = {nid: compute_predicted_error(nid, tree, set()) for nid in nodes}
    max_b = budget

    val = {}
    choice = {}
    # Store skip-alloc and verify-alloc separately so backtrack picks the right one
    skip_alloc = {}  # child budget split when we do NOT verify nid
    verify_alloc = {}  # child budget split when we DO verify nid (children get b-1)

    def _best_child_split(children, b):
        """Optimal budget allocation across children given total budget b."""
        if not children or b == 0:
            return 0.0, {c: 0 for c in children}
        n = len(children)
        dp2 = [[0.0] * (b + 1) for _ in range(n + 1)]
        for i, child in enumerate(children):
            cv = val[child]
            for j in range(b + 1):
                dp2[i + 1][j] = dp2[i][j]
                for k in range(1, j + 1):
                    cand = dp2[i][j - k] + cv[min(k, max_b)]
                    if cand > dp2[i + 1][j]:
                        dp2[i + 1][j] = cand
        alloc = {}
        rem = b
        for i in range(n - 1, -1, -1):
            child = children[i]
            cv = val[child]
            best_k, best_v = 0, dp2[i][rem]
            for k in range(1, rem + 1):
                cand = dp2[i][rem - k] + cv[min(k, max_b)]
                if cand > best_v:
                    best_v, best_k = cand, k
            alloc[child] = best_k
            rem -= best_k
        for c in children:
            alloc.setdefault(c, 0)
        return dp2[n][b], alloc

    def dp(nid):
        node = nodes[nid]
        children = node.children
        sub_size = node.num_descendants + 1  # sub(v) = subtree size including v

        for c in children:
            if c not in val:
                dp(c)

        v = [0.0] * (max_b + 1)
        ch = [False] * (max_b + 1)
        sa = [None] * (max_b + 1)  # skip allocations
        va = [None] * (max_b + 1)  # verify allocations

        for b in range(max_b + 1):
            # Option A: skip v, give all b budget to children
            a_val, a_alloc = _best_child_split(children, b)

            # Option B: verify v (costs 1), give b-1 to children
            # pf·sub(v):      if v is false, quarantine entire subtree
            # (1−pf)·c_val:   if v is true, no direct gain, only budget
            #                  freed for children matters (no +1 credit:
            #                  verifying a true node does not reduce
            #                  undetected_false, so it has zero value here)
            if b >= 1:
                c_val, c_alloc = _best_child_split(children, b - 1)
                b_val = p_false[nid] * sub_size + (1.0 - p_false[nid]) * c_val
            else:
                b_val, c_alloc = -1.0, {}

            if b >= 1 and b_val > a_val:
                v[b], ch[b], sa[b], va[b] = b_val, True, a_alloc, c_alloc
            else:
                v[b], ch[b], sa[b], va[b] = a_val, False, a_alloc, c_alloc

        val[nid] = v
        choice[nid] = ch
        skip_alloc[nid] = sa
        verify_alloc[nid] = va

    import sys

    sys.setrecursionlimit(10000)
    dp(0)

    selected = set()

    def backtrack(nid, b):
        if b <= 0:
            return
        if choice[nid][b]:
            # Verify branch: nid selected, children share b-1
            selected.add(nid)
            alloc = verify_alloc[nid][b] or {}
        else:
            # Skip branch: children share full b
            alloc = skip_alloc[nid][b] or {}
        for c in nodes[nid].children:
            backtrack(c, alloc.get(c, 0))

    backtrack(0, budget)
    return selected


#Ablation variants of dependency-aware


def _make_ablation(w_conf, w_depth, w_desc, w_parent, name):
    def fn(tree, budget, **_):
        return strategy_dependency_aware(
            tree,
            budget,
            w_conf=w_conf,
            w_depth=w_depth,
            w_desc=w_desc,
            w_parent=w_parent,
        )

    fn.__name__ = name
    return fn

def strategy_dp_oracle(tree: ClaimTree, budget: int, **_) -> set[int]:
    """
    DP-optimal strategy under perfect knowledge of ground truth.
    Same recurrence as strategy_dp_optimal, but uses 1 - node.truth
    as p_false instead of compute_predicted_error().
    This is the theoretical ceiling for any DP-style policy.
    """
    import sys
    sys.setrecursionlimit(10000)

    DP_BUDGET_CAP = 50
    budget = min(budget, DP_BUDGET_CAP)

    nodes = tree.nodes
    p_false = {nid: float(1 - node.truth) for nid, node in nodes.items()}
    max_b = budget

    val = {}
    choice = {}
    skip_alloc = {}
    verify_alloc = {}

    def _best_child_split(children, b):
        if not children or b == 0:
            return 0.0, {c: 0 for c in children}
        n = len(children)
        dp2 = [[0.0] * (b + 1) for _ in range(n + 1)]
        for i, child in enumerate(children):
            cv = val[child]
            for j in range(b + 1):
                dp2[i + 1][j] = dp2[i][j]
                for k in range(1, j + 1):
                    cand = dp2[i][j - k] + cv[min(k, max_b)]
                    if cand > dp2[i + 1][j]:
                        dp2[i + 1][j] = cand
        alloc = {}
        rem = b
        for i in range(n - 1, -1, -1):
            child = children[i]
            cv = val[child]
            best_k, best_v = 0, dp2[i][rem]
            for k in range(1, rem + 1):
                cand = dp2[i][rem - k] + cv[min(k, max_b)]
                if cand > best_v:
                    best_v, best_k = cand, k
            alloc[child] = best_k
            rem -= best_k
        for c in children:
            alloc.setdefault(c, 0)
        return dp2[n][b], alloc

    def dp(nid):
        node = nodes[nid]
        children = node.children
        sub_size = node.num_descendants + 1

        for c in children:
            if c not in val:
                dp(c)

        v = [0.0] * (max_b + 1)
        ch = [False] * (max_b + 1)
        sa = [None] * (max_b + 1)
        va = [None] * (max_b + 1)

        for b in range(max_b + 1):
            a_val, a_alloc = _best_child_split(children, b)
            if b >= 1:
                c_val, c_alloc = _best_child_split(children, b - 1)
                b_val = p_false[nid] * sub_size + (1.0 - p_false[nid]) * (c_val)
            else:
                b_val, c_alloc = -1.0, {}
            if b >= 1 and b_val > a_val:
                v[b], ch[b], sa[b], va[b] = b_val, True, a_alloc, c_alloc
            else:
                v[b], ch[b], sa[b], va[b] = a_val, False, a_alloc, c_alloc

        val[nid] = v
        choice[nid] = ch
        skip_alloc[nid] = sa
        verify_alloc[nid] = va

    dp(0)
    selected = set()

    def backtrack(nid, b):
        if b <= 0:
            return
        if choice[nid][b]:
            selected.add(nid)
            alloc = verify_alloc[nid][b] or {}
        else:
            alloc = skip_alloc[nid][b] or {}
        for c in nodes[nid].children:
            backtrack(c, alloc.get(c, 0))

    backtrack(0, budget)
    return selected

ABLATIONS = {
    "Conf only": _make_ablation(1.0, 0.0, 0.0, 0.0, "Conf only"),
    "Descendants only": _make_ablation(0.0, 0.0, 1.0, 0.0, "Descendants only"),
    "Depth only": _make_ablation(0.0, 1.0, 0.0, 0.0, "Depth only"),
    "Conf + Desc": _make_ablation(0.5, 0.0, 0.5, 0.0, "Conf + Desc"),
    "Full composite": _make_ablation(0.35, 0.20, 0.30, 0.15, "Full composite"),
}

STRATEGIES = {
    "Random": strategy_random,
    "Level-sampling": strategy_level,
    "Recent (deep)": strategy_recent,
    "Uncertainty": strategy_uncertainty,
    "Dependency-aware": strategy_dependency_aware,
    "DP Optimal": strategy_dp_optimal,
    "DP Oracle": strategy_dp_oracle, 
}

# 4. EVALUATION

def evaluate(tree: ClaimTree, verify_set: set[int]) -> dict:
    false_ids = set(tree.false_ids)
    total = len(tree.nodes)

    TP = verify_set & false_ids                      # false nodes verified directly

    # Quarantine: a verified-false node invalidates its entire downstream
    # subtree (inclusive of itself). subtree_ids() includes nid.
    quarantined = set()
    for nid in TP:
        quarantined |= set(tree.subtree_ids(nid))

    # A false node is "caught" if it was verified directly OR neutralized by
    # quarantine under a verified-false ancestor. Credit the cascade, not just
    # direct hits — otherwise strategies that verify few-but-high (DP, oracle)
    # look artificially weak.
    caught_false = (TP | quarantined) & false_ids

    recall = len(caught_false) / len(false_ids) if false_ids else 1.0
    # precision = false nodes neutralized per check spent (can exceed-look
    # efficient because one check can cascade — capped at 1.0 for sanity)
    precision = min(1.0, len(caught_false) / len(verify_set)) if verify_set else 0.0

    undetected_false = false_ids - verify_set - quarantined
    reliability = 1.0 - len(undetected_false) / total
    cascade_prevented = len(quarantined - TP)

    return {
        "precision": precision,
        "recall": recall,
        "reliability": reliability,
        "cascade_prevented": cascade_prevented,
        "true_positives": len(TP),
        "false_nodes": len(false_ids),
        "undetected_false": len(undetected_false),
        "verified_false": len(TP),       # false nodes directly caught
        "verified_true": len(verify_set - false_ids),  # true nodes directly verified
    }

# 5. EXPERIMENTS

def run_experiment(
    budgets: list[int],
    n_trials: int = 50,
    max_nodes: int = 80,
    base_error_rate: float = 0.21,
    propagation_rate: float = 0.85,
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
        tree = ClaimTree(
            max_nodes=max_nodes,
            base_error_rate=base_error_rate,
            propagation_rate=propagation_rate,
            seed=trial,  # topology randomized via seed
        )
        for budget in budgets:
            for name, fn in strategy_set.items():
                vset = fn(tree, budget)
                metrics = evaluate(tree, vset)
                raw[name][budget].append(metrics)

    # Compute mean + std per metric
    out = {}
    for strat in strategy_set:
        out[strat] = {}
        for budget in budgets:
            trials = raw[strat][budget]
            keys = trials[0].keys()
            out[strat][budget] = {
                "mean": {k: np.mean([t[k] for t in trials]) for k in keys},
                "std": {k: np.std([t[k] for t in trials]) for k in keys},
            }
    return out


def run_sensitivity(budgets, n_trials=50, max_nodes=80):
    """Run experiment across 3 error regimes."""
    regimes = {
        "Low  (ε=0.05, ρ=0.40)": {"base_error_rate": 0.05, "propagation_rate": 0.40},
        "Med  (ε=0.15, ρ=0.70)": {"base_error_rate": 0.15, "propagation_rate": 0.70},
        "High (ε=0.30, ρ=0.90)": {"base_error_rate": 0.30, "propagation_rate": 0.90},
    }
    return {
        label: run_experiment(budgets, n_trials, max_nodes, **params)
        for label, params in regimes.items()
    }

# 6. ADAPTIVE ONLINE SIMULATION

"""
Offline (above): build the full tree, then pick a fixed verify_set.

Adaptive (below): generate the tree incrementally 
    at each step, pick the frontier node with the highest expected_damage. 
    If damage > sigma and verify budget remains, verify (and possibly quarantine) it;
    otherwise expand it. 
    
Repeat until the node budget or frontier is exhausted.
"""

def run_adaptive_simulation(
    max_nodes: int = 80,
    verify_budget: int = 20,
    sigma = 5.0,
    seed: int = 0,
    max_depth: int = 8,
    branching_lambda: float = 1.7,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
    on_step=None,
    descendant_estimator=None,
    policy: str = "greedy",
    epsilon: float = 0.15,
    epsilon_decay: float = 0.0,
    ucb_c: float = 1.5,
) -> tuple["ClaimTree", set[int]]:
    """
    Generate a claim tree online, interleaving expansion, verification, and
    quarantine decisions based on expected_damage(v) vs. sigma.

    propagation_rate is derived as base_error_rate + rho so that
    compute_predicted_error's delta_cont matches rho exactly.

    policy controls frontier node selection:
        "greedy"         — always pick argmax expected_damage (default, original)
        "epsilon_greedy" — with prob epsilon pick least-visited node, else greedy
        "ucb"            — UCB1: Q(i) + c * sqrt(ln(N+1) / (n_i+1))

    epsilon_decay (float >= 0): if > 0, epsilon decays as max(0.05, epsilon * exp(-decay * step))
    ucb_c: exploration constant for UCB (higher → more exploration)

    on_step (optional): callback invoked after each iteration.
    descendant_estimator (optional): defaults to StructuralDescendantEstimator().

    Returns (tree, verified_set).
    """
    if descendant_estimator is None:
        descendant_estimator = StructuralDescendantEstimator()

    tree = ClaimTree(
        max_nodes=max_nodes,
        base_error_rate=base_error_rate,
        propagation_rate=base_error_rate + rho,
        max_depth=max_depth,
        branching_lambda=branching_lambda,
        seed=seed,
        build_full=False,
    )

    frontier = {0}
    verified_set: set[int] = set()
    verify_budget_remaining = verify_budget
    total_visits = 0  # global selection counter for UCB / visit tracking
    step = 0

    if sigma == "adaptive":
        sigma_policy = AdaptiveSigma(verify_budget, max_nodes)
    else:
        sigma_policy = FixedSigma(float(sigma))

    while frontier and len(tree.nodes) < max_nodes:
        remaining_node_budget = max_nodes - len(tree.nodes)
        step += 1

        # Step 1: pre-compute value for every frontier node
        # Separates value estimation from the selection policy so both can be evaluated independently.
        for fid in frontier:
            tree.nodes[fid].value = expected_damage(
                fid, tree, rho, remaining_node_budget, descendant_estimator
            )

        # Step 2: select using policy
        if policy == "ucb":
            node_id = select_node_ucb(frontier, tree, total_visits, ucb_c)
        elif policy == "epsilon_greedy":
            current_epsilon = (
                max(0.05, epsilon * float(np.exp(-epsilon_decay * step)))
                if epsilon_decay > 0
                else epsilon
            )
            node_id = select_node_epsilon(frontier, tree, current_epsilon)
        else:  
            node_id = max(frontier, key=lambda nid: tree.nodes[nid].value)

        node_obj = tree.nodes[node_id]
        damage = node_obj.value  # already computed above

        # Step 3: compute UCB breakdown for logging (pre-increment) ─
        if policy == "ucb":
            explore_bonus = ucb_c * float(
                np.sqrt(np.log(total_visits + 1) / (node_obj.visits + 1))
            )
            q_term, ucb_val = damage, damage + explore_bonus
        else:
            q_term, explore_bonus, ucb_val = damage, 0.0, damage

        #Step 4: update visit statistics
        node_obj.visits += 1
        node_obj.total_value += damage
        total_visits += 1

        sigma_policy.update(damage)
        current_sigma = sigma_policy.get(verify_budget_remaining, len(tree.nodes))

        p_false_before = node_obj.p_false
        base_event = {
            "node_id": node_id,
            "damage": damage,
            "sigma": current_sigma,
            "p_false_before": p_false_before,
            "frontier_before": set(frontier),
            "visits": node_obj.visits,
            "total_visits": total_visits,
            "policy": policy,
            "q_term": q_term,
            "explore_bonus": explore_bonus,
            "ucb_score": ucb_val,
        }

        if damage > current_sigma and verify_budget_remaining > 0:
            truth = verify_node(
                tree, node_id, verified_set, epsilon=base_error_rate, rho=rho
            )
            verify_budget_remaining -= 1

            if truth == 0:
                quarantine_node(tree, node_id, frontier)
                if on_step:
                    p_false_after = {nid: n.p_false for nid, n in tree.nodes.items()}
                    on_step(
                        {
                            **base_event,
                            "action": "verify_quarantine",
                            "truth": truth,
                            "new_children": [],
                            "p_false_after": p_false_after,
                        }
                    )
                continue
            else:
                # Verified true --> expand
                new_children = tree.expand_node(node_id)
                frontier.discard(node_id)
                frontier.update(new_children)
                if on_step:
                    p_false_after = {nid: n.p_false for nid, n in tree.nodes.items()}
                    on_step(
                        {
                            **base_event,
                            "action": "verify_expand",
                            "truth": truth,
                            "new_children": new_children,
                            "p_false_after": p_false_after,
                        }
                    )
        else:
            # Risk < threshold, expand without verification.
            new_children = tree.expand_node(node_id)
            frontier.discard(node_id)
            frontier.update(new_children)
            if on_step:
                p_false_after = {nid: n.p_false for nid, n in tree.nodes.items()}
                on_step(
                    {
                        **base_event,
                        "action": "expand",
                        "truth": None,
                        "new_children": new_children,
                        "p_false_after": p_false_after,
                    }
                )

    return tree, verified_set


def evaluate_adaptive(tree: "ClaimTree", verified_set: set[int]) -> dict:
    """
    Metrics for an adaptively-generated tree (compare to evaluate() for the
    offline/fixed-budget case).
    """
    false_ids = set(tree.false_ids)
    total = len(tree.nodes)

    verified_false = {nid for nid in verified_set if tree.nodes[nid].truth == 0}
    verified_true = {nid for nid in verified_set if tree.nodes[nid].truth == 1}

    quarantined = {nid for nid, n in tree.nodes.items() if n.status == "quarantined"}
    true_branches_blocked = {nid for nid in quarantined if tree.nodes[nid].truth == 1}
    false_branches_blocked = {nid for nid in quarantined if tree.nodes[nid].truth == 0}

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
    max_nodes: int = 80,
    verify_budget: int = 20,
    max_depth: int = 8,
    branching_lambda: float = 1.7,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
    estimator_type: str = "structural",
    branching_calibration_trials: int = 200,
    branching_calibration: dict = None,
) -> dict:
    """
    Sigma sweep: for each sigma, run n_trials adaptive simulations and
    record mean/std of evaluate_adaptive()'s metrics.

    estimator_type selects the future-descendant estimator used inside
    expected_damage():
      - "structural":       StructuralDescendantEstimator (hand-crafted,
                             uses tree.branching_lambda)
      - "depth_structural": StructuralDescendantEstimator calibrated with
                             a per-depth branching factor (see
                             calibrate_branching_from_simulations()); if
                             branching_calibration is not given, it is
                             computed from branching_calibration_trials
                             offline trees before the sweep

    Returns nested dict: results[sigma] = {"mean": {...}, "std": {...}}
    """
    if estimator_type == "structural":
        descendant_estimator = StructuralDescendantEstimator()
    elif estimator_type == "depth_structural":
        if branching_calibration is None:
            branching_calibration = calibrate_branching_from_simulations(
                n_trials=branching_calibration_trials,
                max_nodes=max_nodes,
                max_depth=max_depth,
                branching_lambda=branching_lambda,
                base_error_rate=base_error_rate,
                rho=rho,
            )
        descendant_estimator = StructuralDescendantEstimator(
            branching_factor=branching_calibration["global_mean"],
            depth_branching=branching_calibration["depth_means"],
        )
    else:
        raise ValueError(f"Unknown estimator_type: {estimator_type!r}")

    raw = {sigma: [] for sigma in sigmas}

    for sigma in sigmas:
        for trial in range(n_trials):
            tree, verified_set = run_adaptive_simulation(
                max_nodes=max_nodes,
                verify_budget=verify_budget,
                sigma=sigma,
                seed=trial,
                max_depth=max_depth,
                branching_lambda=branching_lambda,
                base_error_rate=base_error_rate,
                rho=rho,
                descendant_estimator=descendant_estimator,
            )
            raw[sigma].append(evaluate_adaptive(tree, verified_set))

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
    max_nodes: int = 80,
    verify_budget: int = 20,
    max_depth: int = 8,
    branching_lambda: float = 1.7,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
) -> dict:
    """
    Runs n_trials for each fixed sigma and for sigma = "adaptive", using the
    default StructuralDescendantEstimator (no exploration policy, kept
    minimal so it's directly comparable to the DAG version's sweep).

    Returns results[key] = {"mean": {...}, "std": {...}}
    where key is the float sigma or the string "adaptive".
    """
    all_sigmas = list(fixed_sigmas) + ["adaptive"]
    raw = {s: [] for s in all_sigmas}

    for s in all_sigmas:
        for trial in range(n_trials):
            tree, verified_set = run_adaptive_simulation(
                max_nodes=max_nodes,
                verify_budget=verify_budget,
                sigma=s,
                seed=trial,
                max_depth=max_depth,
                branching_lambda=branching_lambda,
                base_error_rate=base_error_rate,
                rho=rho,
            )
            raw[s].append(evaluate_adaptive(tree, verified_set))

    out = {}
    for s in all_sigmas:
        trials = raw[s]
        keys = trials[0].keys()
        out[s] = {
            "mean": {k: float(np.mean([t[k] for t in trials])) for k in keys},
            "std": {k: float(np.std([t[k] for t in trials])) for k in keys},
        }
    return out


def run_estimator_comparison(
    sigmas: list[float] = (0.5, 1, 1.5, 2, 3, 5),
    estimator_types: list[str] = ("structural", "depth_structural"),
    n_trials: int = 50,
    max_nodes: int = 80,
    verify_budget: int = 20,
    max_depth: int = 8,
    branching_lambda: float = 1.7,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
    branching_calibration_trials: int = 200,
) -> dict:
    """
    Compare ExpectedDamage(v) = p_false(v) * rho * E[future descendants]
    across future-descendant estimators, over a sigma sweep.

    For each estimator_type, runs run_adaptive_experiment() over `sigmas`.
    A single branching calibration (see calibrate_branching_from_simulations)
    is computed once up front and shared by any "depth_structural" run.

    Returns nested dict:
        results[estimator_type][sigma] = {"mean": {...}, "std": {...}}
    with the metrics produced by evaluate_adaptive() (reliability,
    contamination_rate, total_nodes, false_nodes, verified_false,
    verified_true, quarantined, verify_budget_used, ...).
    """
    branching_calibration = None
    if "depth_structural" in estimator_types:
        branching_calibration = calibrate_branching_from_simulations(
            n_trials=branching_calibration_trials,
            max_nodes=max_nodes,
            max_depth=max_depth,
            branching_lambda=branching_lambda,
            base_error_rate=base_error_rate,
            rho=rho,
        )

    results = {}
    for estimator_type in estimator_types:
        results[estimator_type] = run_adaptive_experiment(
            sigmas=sigmas,
            n_trials=n_trials,
            max_nodes=max_nodes,
            verify_budget=verify_budget,
            max_depth=max_depth,
            branching_lambda=branching_lambda,
            base_error_rate=base_error_rate,
            rho=rho,
            estimator_type=estimator_type,
            branching_calibration_trials=branching_calibration_trials,
            branching_calibration=branching_calibration,
        )
    return results

# 7. WARM-UP + SUBTREE-CHECKPOINT ADAPTIVE POLICY
# Variant of the simple adaptive loop above. Instead of scoring every
# frontier node in isolation and deciding immediately, this policy:
#   1. expands the first `warmup_depth` levels unconditionally (no checks),
#   2. picks the highest-ExpectedDamage frontier node as a "subtree root",
#   3. lets that subtree grow for a few more levels before checking it,
#   4. verifies the subtree root as a single checkpoint, and
#   5. either quarantines the whole (partially-grown) subtree, or keeps
#      growing inside it.
# The idea: let enough local structure emerge to estimate a subtree's risk
# accurately, while still catching it before it grows unbounded.


def expand_to_depth(tree: "ClaimTree", frontier: set[int], target_depth: int) -> None:
    """
    Warm-up: expand active frontier nodes, shallowest first, until no
    active frontier node remains with depth < target_depth. No
    verification happens here.
    """
    while len(tree.nodes) < tree.max_nodes:
        shallow = [
            nid
            for nid in frontier
            if tree.nodes[nid].depth < target_depth
            and tree.nodes[nid].status == "active"
        ]
        if not shallow:
            break
        node_id = min(shallow, key=lambda nid: (tree.nodes[nid].depth, nid))
        new_children = tree.expand_node(node_id)
        frontier.discard(node_id)
        frontier.update(new_children)


def select_subtree_root(
    tree: "ClaimTree", frontier: set[int], rho: float, remaining_node_budget: int
) -> int:
    """Pick the frontier node with the highest expected_damage()."""
    return max(
        frontier,
        key=lambda nid: expected_damage(nid, tree, rho, remaining_node_budget),
    )


def expand_subtree_for_k_levels(
    tree: "ClaimTree", root_id: int, k: int, frontier: set[int]
) -> list[int]:
    """
    BFS-expand only the subtree rooted at `root_id`, for up to `k`
    additional levels beyond root_id's depth. Stops early on a quarantined
    node or once the node budget is exhausted. Returns the ids of the
    nodes that were expanded.
    """
    start_depth = tree.nodes[root_id].depth
    local_frontier = deque([root_id])
    expanded: list[int] = []

    while local_frontier:
        node_id = local_frontier.popleft()
        node = tree.nodes[node_id]

        if node.depth >= start_depth + k:
            continue
        if node.status == "quarantined":
            continue
        if len(tree.nodes) >= tree.max_nodes:
            break

        new_children = tree.expand_node(node_id)
        frontier.discard(node_id)
        frontier.update(new_children)
        expanded.append(node_id)
        local_frontier.extend(new_children)

    return expanded


def quarantine_subtree(tree: "ClaimTree", root_id: int, frontier: set[int]) -> None:
    """
    Quarantine `root_id` AND every node already generated in its subtree
    (unlike quarantine_node(), which only marks a single node since, in the
    simple adaptive policy, descendants don't exist yet).
    """
    for nid in tree.subtree_ids(root_id):
        node = tree.nodes[nid]
        node.status = "quarantined"
        node.expanded = True
        frontier.discard(nid)


def run_warmup_checkpoint_simulation(
    max_nodes: int = 80,
    verify_budget: int = 20,
    sigma_low: float = 3.0,
    sigma_high: float = 8.0,
    warmup_depth: int = 2,
    subtree_expand_depth: int = 2,
    seed: int = 0,
    max_depth: int = 8,
    branching_lambda: float = 1.7,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
    on_step=None,
) -> tuple["ClaimTree", set[int]]:
    """
    Warm-up + subtree-checkpoint adaptive policy.

      damage > sigma_high   (high risk): verify the root immediately;
          quarantine its (still-tiny) subtree if false, else expand it one
          more level.
      sigma_low < damage <= sigma_high (medium risk): let the subtree grow
          for `subtree_expand_depth` levels, then verify the root as a
          checkpoint; quarantine the whole grown subtree if false.
      damage <= sigma_low   (low risk): expand the node normally, no
          verification.

    If the verification budget is exhausted, a node that would have been
    verified (damage > sigma_high) is quarantined instead of being expanded
    further.

    Returns (tree, verified_set).
    """
    tree = ClaimTree(
        max_nodes=max_nodes,
        base_error_rate=base_error_rate,
        propagation_rate=base_error_rate + rho,
        max_depth=max_depth,
        branching_lambda=branching_lambda,
        seed=seed,
        build_full=False,
    )

    frontier = {0}
    verified_set: set[int] = set()
    verify_budget_remaining = verify_budget

    # 1. Warm-up — expand the first `warmup_depth` levels, no verification.
    expand_to_depth(tree, frontier, warmup_depth)

    # 2-8. Subtree-checkpoint loop.
    while frontier and len(tree.nodes) < max_nodes:
        remaining_node_budget = max_nodes - len(tree.nodes)
        root_id = select_subtree_root(tree, frontier, rho, remaining_node_budget)
        damage = expected_damage(root_id, tree, rho, remaining_node_budget)
        base_event = {
            "node_id": root_id,
            "damage": damage,
            "sigma_low": sigma_low,
            "sigma_high": sigma_high,
            "p_false_before": tree.nodes[root_id].p_false,
            "frontier_before": set(frontier),
        }

        if damage > sigma_high:
            if verify_budget_remaining > 0:
                truth = verify_node(
                    tree, root_id, verified_set, epsilon=base_error_rate, rho=rho
                )
                verify_budget_remaining -= 1
                if truth == 0:
                    quarantine_subtree(tree, root_id, frontier)
                    action = "checkpoint_quarantine"
                else:
                    expand_subtree_for_k_levels(tree, root_id, 1, frontier)
                    action = "checkpoint_verify_expand"
            else:
                # No budget left for a high-risk node — quarantine rather
                # than risk expanding it further.
                quarantine_subtree(tree, root_id, frontier)
                action = "budget_exhausted_quarantine"

        elif damage > sigma_low:
            expand_subtree_for_k_levels(tree, root_id, subtree_expand_depth, frontier)
            if verify_budget_remaining > 0:
                truth = verify_node(
                    tree, root_id, verified_set, epsilon=base_error_rate, rho=rho
                )
                verify_budget_remaining -= 1
                if truth == 0:
                    quarantine_subtree(tree, root_id, frontier)
                    action = "subtree_checkpoint_quarantine"
                else:
                    action = "subtree_checkpoint_verify"
            else:
                action = "subtree_expand_no_budget"

        else:
            new_children = tree.expand_node(root_id)
            frontier.discard(root_id)
            frontier.update(new_children)
            action = "expand"

        # Safety net: every branch above is expected to remove root_id from
        # the frontier (directly or via quarantine_subtree /
        # expand_subtree_for_k_levels), but discard defensively to avoid an
        # infinite loop on degenerate parameters (e.g. subtree_expand_depth=0).
        frontier.discard(root_id)

        if on_step:
            p_false_after = {nid: n.p_false for nid, n in tree.nodes.items()}
            on_step({**base_event, "action": action, "p_false_after": p_false_after})

    return tree, verified_set


def run_warmup_checkpoint_experiment(
    sigma_lows: list[float] = (1, 2, 3, 5),
    sigma_highs: list[float] = (5, 8, 10, 15),
    n_trials: int = 50,
    max_nodes: int = 80,
    verify_budget: int = 20,
    warmup_depth: int = 2,
    subtree_expand_depth: int = 2,
    max_depth: int = 8,
    branching_lambda: float = 1.7,
    base_error_rate: float = 0.1758,
    rho: float = 0.7504,
) -> dict:
    """
    Sweep (sigma_low, sigma_high) pairs with sigma_high > sigma_low. For
    each pair, run n_trials warm-up + checkpoint simulations and record
    mean/std of evaluate_adaptive()'s metrics.

    Returns results[(sigma_low, sigma_high)] = {"mean": {...}, "std": {...}}
    """
    pairs = [(lo, hi) for lo in sigma_lows for hi in sigma_highs if hi > lo]
    out = {}

    for sigma_low, sigma_high in pairs:
        trials = []
        for trial in range(n_trials):
            tree, verified_set = run_warmup_checkpoint_simulation(
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
            )
            trials.append(evaluate_adaptive(tree, verified_set))

        keys = trials[0].keys()
        out[(sigma_low, sigma_high)] = {
            "mean": {k: float(np.mean([t[k] for t in trials])) for k in keys},
            "std": {k: float(np.std([t[k] for t in trials])) for k in keys},
        }
    return out

# DIAGNOSTIC: evaluate strategies on DP's own objective

def evaluate_belief(tree: ClaimTree, verify_set: set[int]) -> float:
    """
    Score a verify_set on the SAME nested objective that strategy_dp_optimal
    maximizes, by evaluating the DP recurrence for the FIXED set:

        score(v) = pf(v)*sub(v) + (1-pf(v)) * Σ_c score(c)   if v ∈ verify_set
                 = Σ_c score(c)                               otherwise

    where pf = compute_predicted_error(v, tree, {}) and sub(v) = num_descendants+1.

    A verified node's subtree (pf*sub) is only credited when the node is FALSE;
    in the (1-pf) TRUE branch we recurse into children. This means a child's
    contribution is discounted by the probability its parent is true, so
    verifying a parent AND a descendant does NOT double-count the overlapping
    subtree.

    The previous version summed pf*sub independently over every verified node,
    which double-counted nested selections and let greedy heuristics
    (e.g. Dependency-aware) out-score the genuinely optimal DP — tripping
    assert_dp_dominates. Following the recurrence makes DP provably dominant,
    so the dominance check is a real correctness test, not a metric artifact.
    """
    empty: set[int] = set()

    def score(nid: int) -> float:
        node = tree.nodes[nid]
        children_score = 0.0
        for c in node.children:
            children_score += score(c)
        if nid in verify_set:
            pf = compute_predicted_error(nid, tree, empty)
            sub = node.num_descendants + 1
            # False (prob pf): whole subtree quarantined -> sub nodes safe.
            # True  (prob 1-pf): recurse; children still verifiable.
            return pf * sub + (1.0 - pf) * children_score
        return children_score

    import sys
    sys.setrecursionlimit(10000)
    return score(0)

def run_calibration_comparison(
    budgets: list = None,
    n_trials: int = 50,
    max_nodes: int = 80,
    base_error_rate: float = 0.21,
    propagation_rate: float = 0.85,
) -> dict:
    """
    Run the main experiment under both regimes.
    Returns {"default": results, "overconfident": results}
    where the gap between DP Oracle and DP Optimal (belief) on reliability
    is the key calibration-cost metric.
    """
    if budgets is None:
        budgets = [5, 10, 15, 20, 25, 30, 40, 50]

    strategy_set = {
        "Random": strategy_random,
        "Uncertainty": strategy_uncertainty,
        "Dependency-aware": strategy_dependency_aware,
        "DP Optimal": strategy_dp_optimal,
        "DP Oracle": strategy_dp_oracle,
    }

    results = {}
    for overconfident in (False, True):
        regime = "overconfident" if overconfident else "default"
        raw = {s: {b: [] for b in budgets} for s in strategy_set}

        for trial in range(n_trials):
            tree = ClaimTree(
                max_nodes=max_nodes,
                base_error_rate=base_error_rate,
                propagation_rate=propagation_rate,
                seed=trial,
                overconfident=overconfident,
            )
            for budget in budgets:
                for name, fn in strategy_set.items():
                    vset = fn(tree, budget)
                    raw[name][budget].append(evaluate(tree, vset))

        out = {}
        for strat in strategy_set:
            out[strat] = {}
            for budget in budgets:
                trials_list = raw[strat][budget]
                keys = trials_list[0].keys()
                out[strat][budget] = {
                    "mean": {k: float(np.mean([t[k] for t in trials_list])) for k in keys},
                    "std":  {k: float(np.std( [t[k] for t in trials_list])) for k in keys},
                }
        results[regime] = out

    return results


def print_calibration_table(results: dict, budgets: list, key: str = "reliability"):
    print(f"\n── Calibration regime comparison  (metric: {key}) ──")
    strats = ["Random", "Uncertainty", "Dependency-aware", "DP Optimal", "DP Oracle"]

    for regime in ("default", "overconfident"):
        print(f"\n  Regime: {regime}")
        header = f"  {'Budget':>7}" + "".join(f"  {s:>16}" for s in strats)
        print(header)
        print("  " + "-" * (9 + 18 * len(strats)))
        for b in budgets:
            row = f"  {b:>7}"
            for s in strats:
                v = results[regime][s][b]["mean"][key]
                row += f"  {v:>16.3f}"
            print(row)

        mid = budgets[len(budgets) // 2]
        gap = (
            results[regime]["DP Oracle"][mid]["mean"][key]
            - results[regime]["DP Optimal"][mid]["mean"][key]
        )
        print(f"\n  Oracle-DP vs Belief-DP gap at budget={mid}: {gap:+.4f}")

def assert_dp_dominates(n_trees: int = 30, budget: int = 20, seed_offset: int = 0):
    """
    On n_trees random trees, assert that DP's belief-score >= every other strategy.
    Prints a summary and raises AssertionError if DP ever loses.

    This is the core diagnostic: if DP loses here, the recurrence or backtrack
    has a bug. If DP wins here but loses on evaluate() (ground truth), the gap
    is purely metric mismatch not a code error.
    """
    print(f"\n── DP dominance check ({n_trees} trees, budget={budget}) ──")
    violations = []

    for i in range(n_trees):
        tree = ClaimTree(max_nodes=80, seed=seed_offset + i, max_depth=8, branching_lambda=1.7)

        dp_set = strategy_dp_optimal(tree, budget)
        dp_score = evaluate_belief(tree, dp_set)

        for name, fn in STRATEGIES.items():
            if name == "DP Optimal":
                continue
            vset = fn(tree, budget)
            score = evaluate_belief(tree, vset)
            if score > dp_score + 1e-6:  # tolerance for float noise
                violations.append({
                    "tree": i,
                    "strategy": name,
                    "their_score": score,
                    "dp_score": dp_score,
                    "gap": score - dp_score,
                })

    if violations:
        print(f"  FAILED: DP beaten on {len(violations)} occasions")
        for v in violations[:5]:  # show first 5
            print(f"    tree={v['tree']}  {v['strategy']} scored {v['their_score']:.4f} vs DP {v['dp_score']:.4f}  (gap={v['gap']:.4f})")
        raise AssertionError(f"DP is not belief-optimal on {len(violations)} tree-strategy pairs")
    else:
        print(f"  PASSED: DP dominates all strategies on belief objective across {n_trees} trees")
        return True