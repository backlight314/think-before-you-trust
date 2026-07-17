"""
Animated walkthrough of the single-branch, live-propagating adaptive DFS.

The claim tree grows incrementally (build_full=False). DFS follows one
branch at a time, going as deep as it can before backtracking.

At each node:
  1. Compute error_propagation = p_false, D_hat (RF), risk = p_false × D_hat.
  2. If p_false < sigma and the node hasn't been expanded yet:
       Generate children on the fly via expand_node().
       Pick the one eligible child with the lowest error_propagation.
       Push the rest of the eligible children onto a frontier for later.
       Ineligible children (p_false >= sigma) get marked as candidates.
  3. If p_false >= sigma, we've hit a risky node:
       Backtrack to PATH[-2], the parent of the risky node, and verify it.
         Parent TRUE  → p_false = 0, propagate downward; risky child clears
                        (its p_false drops to ε), we go deeper on the same branch.
         Parent FALSE → quarantine the parent's subtree, clear the DFS path,
                        pop the next frontier branch (a sideways move).
  4. Once the branch runs out (no eligible children, budget spent, or a leaf),
     pop the next node off the frontier and start a new branch.

What you should see: max_path_depth stays high since a branch gets pushed
deep before backtracking, sideways_moves stays low (only kicks in after a
quarantine or a branch running dry), probabilities update live, and a color
wave runs down the path each time an ancestor gets verified.

  Fill (heat map)  : p_false, updated live; orange glow = node just changed
  Cyan ring        : current DFS path
  Gold ring        : candidate (risky, p_false >= sigma)
  White ring       : verified TRUE
  Red + gold ring  : verified FALSE (quarantine root)
  Green glow       : recently cleared (was risky, parent verified TRUE)
  Blue dashed ring : nodes in frontier waiting to be explored

REVEAL_SUBTREE_FOR_VIS = False, so the tree only grows from DFS exploration.

Output: results_tree/simulation_walkthrough.gif
"""

import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.ensemble import RandomForestRegressor

from tree_model import (
    ClaimTree,
    collect_descendant_training_data,
    extract_descendant_features,
    verify_node,
)

OUT = "results_tree"
os.makedirs(OUT, exist_ok=True)

# Parameters
MAX_NODES = 80
BRANCHING_LAMBDA = 1.7
BASE_ERROR_RATE = 0.1758
RHO = 0.7504
VERIFY_BUDGET = 8
SIGMA = 0.30
FPS = 0.8
N_RF_TRAINING = 80
RANDOM_STATE = 42
MIN_TREE_NODES = 70
MIN_FALSE_NODES = 10
REVEAL_SUBTREE_FOR_VIS = False  # True: expand candidate subtrees for visualization


# RF training


def train_rf() -> RandomForestRegressor:
    X, y, _ = collect_descendant_training_data(
        n_trials=N_RF_TRAINING,
        max_nodes=MAX_NODES,
        branching_lambda=BRANCHING_LAMBDA,
        base_error_rate=BASE_ERROR_RATE,
        rho=RHO,
    )
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE
    )
    rf.fit(X, y)
    print(f"  RF: R²={rf.score(X, y):.3f}  ({len(y)} samples)")
    return rf


def _compute_d_hat(
    nid: int, tree: ClaimTree, rf, explored: set, n_edges: list
) -> float:
    """RF estimate of remaining descendants; geometric series fallback."""
    node = tree.nodes[nid]
    remaining = max(0, tree.max_nodes - len(tree.nodes))
    n_exp = len(explored)
    avg_b = n_edges[0] / n_exp if n_exp > 0 else 0.0

    if rf is not None:
        feats = extract_descendant_features(
            tree,
            nid,
            remaining,
            avg_branching_so_far=avg_b,
            expanded_nodes_so_far=n_exp,
        )
        return float(max(0.0, rf.predict([feats])[0]))

    rem_d = max(0, tree.max_depth - node.depth)
    lam = BRANCHING_LAMBDA
    if lam > 1 and rem_d > 0:
        est = lam * (lam**rem_d - 1) / (lam - 1)
    else:
        est = float(rem_d)
    return float(min(remaining, max(0.0, est)))


def expand_and_propagate(
    tree: ClaimTree, node_id: int, epsilon: float, rho: float
) -> list:
    """
    Generate children of node_id and set each child's p_false via the union
    error model:
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
        child.p_false = float(1.0 - (1.0 - epsilon) * (1.0 - rho * parent.p_false))
        child.error_propagation = child.p_false
    return children


# Single-branch DFS with live propagation


def dfs_with_log(tree: ClaimTree, rf, sigma: float, verify_budget: int) -> tuple:
    """
    Strict single-branch DFS with live propagation.

    At each step exactly one child is chosen (lowest error_propagation);
    all other children go to the frontier (LIFO) unexplored.
    When the current node crosses sigma, DFS stops and verifies the
    direct parent (path[-2]).  No verification happens during expansion.

    Returns (candidates, explored, verified_set, quarantined, log).
    """
    epsilon = tree.base_error_rate
    rho_prop = max(tree.propagation_rate - tree.base_error_rate, 0.0)

    path: list = []
    frontier: list = []
    candidates: set = set()
    explored: set = set()
    verified_set: set = set()
    quarantined: set = set()
    cleared: set = set()
    log: list = []
    n_edges = [0]

    max_path_depth = 0
    sideways_moves = 0

    def _annotate(nid: int) -> None:
        node = tree.nodes[nid]
        d_hat = _compute_d_hat(nid, tree, rf, explored, n_edges)
        node.d_hat = d_hat
        node.error_propagation = node.p_false
        node.risk = node.p_false * d_hat

    def _snap() -> dict:
        return {
            "p_map": {n: tree.nodes[n].p_false for n in tree.nodes},
            "candidates": set(candidates),
            "verified": set(verified_set),
            "quarantined": set(quarantined),
            "cleared": set(cleared),
            "path": list(path),
            "frontier": list(frontier),
        }

    def _do_verify(nid: int):
        if len(verified_set) >= verify_budget or nid in verified_set:
            return None
        node = tree.nodes[nid]
        p_before = {n: tree.nodes[n].p_false for n in tree.nodes}

        log.append(
            {
                "type": "verify_start",
                "node_id": nid,
                "p_false_now": node.p_false,
                **_snap(),
            }
        )

        truth = verify_node(tree, nid, verified_set, epsilon=epsilon, rho=rho_prop)

        p_after = {n: tree.nodes[n].p_false for n in tree.nodes}
        changed = {
            n
            for n in tree.nodes
            if abs(p_after[n] - p_before.get(n, p_after[n])) > 1e-4
        }

        for pid in path:
            if pid in tree.nodes:
                _annotate(pid)

        log.append(
            {
                "type": "verify_result",
                "node_id": nid,
                "truth": truth,
                "p_false_after": p_after,
                "changed_nodes": changed,
                **_snap(),
            }
        )
        return truth

    # Core step function

    def step_down(nid: int) -> tuple:
        """
        Expansion only, doesn't verify anything.

        Checks the sigma threshold first. If the node is eligible, expands
        it one level using the propagation formula, picks the one child
        with the lowest error_propagation to continue, and pushes the rest
        to the frontier.

        Returns one of:
          ("THRESHOLD_HIT", nid)   : node p_false >= sigma, needs verifying
          ("MAX_DEPTH",     nid)   : hit the tree depth limit, backtrack
          ("LEAF",          nid)   : no children came out (tree full or empty)
          ("CONTINUE",  next_nid)  : one child chosen, rest went to frontier
        """
        node = tree.nodes[nid]
        _annotate(nid)

        if node.error_propagation >= sigma:
            return "THRESHOLD_HIT", nid

        if node.depth >= tree.max_depth:
            return "MAX_DEPTH", nid

        children = expand_and_propagate(tree, nid, epsilon, rho_prop)
        explored.add(nid)
        n_edges[0] += len(children)

        if not children:
            return "LEAF", nid

        for cid in children:
            _annotate(cid)

        risky = [c for c in children if tree.nodes[c].p_false >= sigma]
        eligible = [c for c in children if tree.nodes[c].p_false < sigma]
        for cid in risky:
            candidates.add(cid)

        log.append(
            {
                "type": "dfs_expand",
                "node_id": nid,
                "children": list(children),
                "eligible": eligible,
                "ineligible": risky,
                "sigma": sigma,
                **_snap(),
            }
        )

        # Single-branch: one child chosen, all others go to frontier
        by_ep = sorted(children, key=lambda c: tree.nodes[c].error_propagation)
        next_child = by_ep[0]
        for sib in by_ep[1:]:
            frontier.append(sib)

        return "CONTINUE", next_child

    # Initialise root
    root = tree.nodes[0]
    root.p_false = float(np.clip(1.0 - root.confidence, 0.0, 1.0))
    _annotate(0)

    current = 0
    path = [0]

    # Main loop
    guard = 0
    while (path or frontier) and len(verified_set) < verify_budget:
        guard += 1
        if guard > 20_000:
            break

        max_path_depth = max(max_path_depth, len(path))

        # Pop from frontier when path is exhausted
        if not path:
            if not frontier:
                break
            current = frontier.pop()
            sideways_moves += 1
            _annotate(current)

            log.append(
                {
                    "type": "dfs_sideways",
                    "node_id": current,
                    "p_false": tree.nodes[current].p_false,
                    "frontier_remaining": len(frontier),
                    "reason": "branch exhausted or quarantined",
                    **_snap(),
                }
            )

            if tree.nodes[current].error_propagation >= sigma:
                candidates.add(current)
                log.append(
                    {
                        "type": "dfs_candidate",
                        "node_id": current,
                        "risk": tree.nodes[current].risk,
                        "reason": "frontier node risky after propagation",
                        **_snap(),
                    }
                )
                _do_verify(current)
            else:
                path = [current]
            continue

        # Visit current path tip
        _annotate(current)
        node = tree.nodes[current]

        log.append(
            {
                "type": "dfs_visit",
                "node_id": current,
                "p_false": node.p_false,
                "d_hat": node.d_hat,
                "risk": node.risk,
                **_snap(),
            }
        )

        status, result = step_down(current)

        # CONTINUE: one child chosen
        if status == "CONTINUE":
            current = result
            path.append(current)

        # THRESHOLD HIT: verify direct parent, then decide
        elif status == "THRESHOLD_HIT":
            candidates.add(current)
            log.append(
                {
                    "type": "dfs_candidate",
                    "node_id": current,
                    "risk": node.risk,
                    "reason": f"p_false={node.p_false:.3f}≥σ={sigma}",
                    **_snap(),
                }
            )

            # Verify direct parent (path[-2]); if path has only root, verify root.
            ancestor = path[-2] if len(path) >= 2 else path[-1]
            truth = _do_verify(ancestor)

            # Re-annotate entire path after propagation
            for pid in path:
                if pid in tree.nodes:
                    _annotate(pid)
                    if pid in candidates and tree.nodes[pid].p_false < sigma:
                        cleared.add(pid)
                        candidates.discard(pid)

            if truth is None:
                # Budget full or ancestor already verified → abandon branch
                path = []

            elif truth == 1:
                # Ancestor TRUE: current's p_false dropped, retry step_down next iter
                log.append(
                    {
                        "type": "dfs_branch_cleared",
                        "node_id": ancestor,
                        "p_false_new": tree.nodes[ancestor].p_false,
                        **_snap(),
                    }
                )
                # current stays in path; next iteration calls step_down(current)
                # which will see p_false < sigma and expand it

            else:
                # Ancestor FALSE: quarantine subtree, clear path
                quarantined.add(ancestor)
                qsub = set(tree.subtree_ids(ancestor))
                frontier[:] = [f for f in frontier if f not in qsub]
                log.append(
                    {
                        "type": "dfs_quarantine",
                        "node_id": ancestor,
                        "reason": "verified FALSE → subtree abandoned",
                        **_snap(),
                    }
                )
                path = []
                sideways_moves += 1

        # LEAF: backtrack (terminal node or already explored)
        elif status == "LEAF":
            node = tree.nodes[current]
            reason = (
                "terminal leaf — Poisson drew 0 children"
                if node.status == "terminal"
                else "already explored"
            )
            log.append(
                {"type": "dfs_leaf", "node_id": current, "reason": reason, **_snap()}
            )
            path.pop()
            if path:
                current = path[-1]

        # MAX_DEPTH: backtrack (tree depth cap reached)
        elif status == "MAX_DEPTH":
            node = tree.nodes[current]
            log.append(
                {
                    "type": "dfs_max_depth",
                    "node_id": current,
                    "depth": node.depth,
                    "p_false": node.p_false,
                    **_snap(),
                }
            )
            path.pop()
            if path:
                current = path[-1]

    false_v = sum(1 for n in verified_set if tree.nodes[n].truth == 0)
    true_v = sum(1 for n in verified_set if tree.nodes[n].truth == 1)

    log.append(
        {
            "type": "dfs_done",
            "candidates": set(candidates),
            "explored": set(explored),
            "verified": set(verified_set),
            "quarantined": set(quarantined),
            "cleared": set(cleared),
            "sideways_moves": sideways_moves,
            "max_path_depth": max_path_depth,
            "num_false_verified": false_v,
            "num_true_verified": true_v,
            **_snap(),
        }
    )

    return candidates, explored, verified_set, quarantined, log

def compute_layout(tree: ClaimTree) -> dict:
    by_depth: dict = defaultdict(list)
    for nid, n in tree.nodes.items():
        by_depth[n.depth].append(nid)
    max_d = max(by_depth) + 1 if by_depth else 1
    pos = {}
    for d, ids in by_depth.items():
        for i, nid in enumerate(sorted(ids)):
            pos[nid] = ((i + 1) / (len(ids) + 1), 1.0 - d / max_d)
    return pos

def pf_color(p: float):
    return plt.cm.RdYlGn_r(0.1 + 0.8 * float(np.clip(p, 0, 1)))

def draw_frame(
    ax,
    fig,
    tree: ClaimTree,
    pos: dict,
    p_map: dict,
    title: str,
    dfs_path=None,
    candidates=None,
    eligible_set=None,
    ineligible_set=None,
    verified_set=None,
    quarantined=None,
    changed_nodes=None,
    cleared_nodes=None,
    frontier=None,
    truth_mode: bool = False,
) -> None:

    ax.clear()
    ax.set_facecolor("#0f0f14")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)

    visible = set(p_map.keys()) if not truth_mode else set(tree.nodes.keys())
    path_s = set(dfs_path or [])
    cands = set(candidates or [])
    elig = set(eligible_set or [])
    inelig = set(ineligible_set or [])
    vset = set(verified_set or [])
    quar_s = set(quarantined or [])
    changed = set(changed_nodes or [])
    cleared_s = set(cleared_nodes or [])
    front_s = set(frontier or [])
    vf = {n for n in vset if tree.nodes[n].truth == 0}

    # Edges
    for nid in visible:
        n = tree.nodes[nid]
        if n.parent_id is not None and n.parent_id in visible and nid in pos:
            x0, y0 = pos[n.parent_id]
            x1, y1 = pos[nid]
            col = "#2a2a33" if nid in inelig else "#555566"
            ax.plot([x0, x1], [y0, y1], color=col, lw=0.8, zorder=1)

    for nid in visible:
        if nid not in pos:
            continue
        n = tree.nodes[nid]
        x, y = pos[nid]

        # Outer glows
        if nid in changed:
            ax.scatter(
                x,
                y,
                s=640,
                c="none",
                edgecolors="#f39c12",
                lw=0.9,
                alpha=0.30,
                zorder=2,
            )
        if nid in cleared_s:
            ax.scatter(
                x,
                y,
                s=640,
                c="none",
                edgecolors="#2ecc71",
                lw=1.0,
                alpha=0.35,
                zorder=2,
            )

        # Status rings (back to front) 
        if nid in front_s:
            ax.scatter(
                x,
                y,
                s=440,
                c="none",
                edgecolors="#4fc3f7",
                lw=1.2,
                linestyle="--",
                alpha=0.55,
                zorder=3,
            )
        if nid in inelig:
            ax.scatter(
                x,
                y,
                s=430,
                c="none",
                edgecolors="#444455",
                lw=1.2,
                linestyle=":",
                alpha=0.55,
                zorder=3,
            )
        if nid in elig:
            ax.scatter(
                x,
                y,
                s=430,
                c="none",
                edgecolors="#4fc3f7",
                lw=1.5,
                linestyle="--",
                alpha=0.8,
                zorder=3,
            )
        if nid in cands and nid not in vset and nid not in cleared_s:
            ax.scatter(
                x,
                y,
                s=520,
                c="none",
                edgecolors="#f39c12",
                lw=2.2,
                alpha=0.95,
                zorder=4,
            )
        if nid in path_s and nid not in cands and nid not in vset:
            ax.scatter(
                x, y, s=480, c="none", edgecolors="#00e5ff", lw=2.0, alpha=0.9, zorder=4
            )
        if nid in vf:
            ax.scatter(
                x,
                y,
                s=600,
                c="none",
                edgecolors="#f1c40f",
                lw=2.0,
                alpha=0.85,
                zorder=5,
            )
            ax.scatter(
                x,
                y,
                s=500,
                c="none",
                edgecolors="#e74c3c",
                lw=2.5,
                alpha=0.95,
                zorder=5,
            )
        if nid in vset and nid not in vf:
            ax.scatter(
                x,
                y,
                s=490,
                c="none",
                edgecolors="#ffffff",
                lw=2.2,
                alpha=0.95,
                zorder=5,
            )

        if truth_mode:
            fill = "#2ecc71" if n.truth == 1 else "#e74c3c"
        else:
            fill = pf_color(p_map.get(nid, n.p_false))

        ax.scatter(
            x, y, s=190, c=[fill], edgecolors="#1a1a22", linewidths=0.5, zorder=6
        )

        ax.text(
            x,
            y - 0.030,
            f"#{nid}",
            ha="center",
            va="top",
            fontsize=5.5,
            color="#666677",
        )
        if truth_mode:
            ax.text(
                x,
                y + 0.036,
                "T" if n.truth == 1 else "F",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#cccccc",
            )
        else:
            pf_val = p_map.get(nid, n.p_false)
            ax.text(
                x,
                y + 0.036,
                f"{pf_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="#aaaaaa",
            )

    fig.suptitle(title, color="#eeeeee", fontsize=9.5, y=0.97, fontweight="bold")


def build_frames(log: list, tree: ClaimTree, sigma: float) -> list:
    frames = []

    def _base(ev: dict) -> dict:
        return {
            "p_map": dict(ev["p_map"]),
            "candidates": set(ev["candidates"]),
            "verified": set(ev["verified"]),
            "quarantined": set(ev["quarantined"]),
            "cleared": set(ev["cleared"]),
            "path": list(ev["path"]),
            "frontier": list(ev["frontier"]),
        }

    for ev in log:
        etype = ev["type"]
        base = _base(ev)

        if etype == "dfs_visit":
            nid = ev["node_id"]
            node = tree.nodes[nid]
            depth_str = f"depth={node.depth}"
            path_str = "→".join(f"#{p}" for p in ev["path"])
            frames.append(
                {
                    **base,
                    "title": (
                        f"DFS  #{nid}  {depth_str}  |  "
                        f"p_false={ev['p_false']:.3f}  D̂={ev['d_hat']:.1f}  risk={ev['risk']:.2f}  |  "
                        f"path: {path_str or '(root)'}  |  frontier: {len(ev['frontier'])} branches"
                    ),
                }
            )

        elif etype == "dfs_candidate":
            nid = ev["node_id"]
            node = tree.nodes[nid]
            reason = ev.get("reason", "")
            frames.append(
                {
                    **base,
                    "title": (
                        f"RISKY  #{nid}  ({reason})  |  "
                        f"p_false={node.p_false:.3f}  risk={ev['risk']:.2f}  |  "
                        f"{len(ev['candidates'])} candidate(s)  |  backtracking to parent…"
                    ),
                }
            )

        elif etype == "verify_start":
            nid = ev["node_id"]
            frames.append(
                {
                    **base,
                    "title": (
                        f"VERIFYING PARENT  #{nid}  |  "
                        f"p_false={ev['p_false_now']:.3f}  |  "
                        f"revealing ground truth…"
                    ),
                }
            )

        elif etype == "verify_result":
            nid = ev["node_id"]
            truth = ev["truth"]
            label = (
                "TRUE → p_false=0, branch cleared, propagating downward"
                if truth == 1
                else "FALSE → p_false=1, descendants inherit error, subtree quarantined"
            )
            frames.append(
                {
                    "p_map": dict(ev["p_false_after"]),
                    "candidates": set(ev["candidates"]),
                    "verified": set(ev["verified"]),
                    "quarantined": set(ev["quarantined"]),
                    "cleared": set(ev["cleared"]),
                    "path": list(ev["path"]),
                    "frontier": list(ev["frontier"]),
                    "changed_nodes": set(ev["changed_nodes"]),
                    "title": (
                        f"VERIFIED  #{nid}: {label}  |  "
                        f"{len(ev['changed_nodes'])} nodes updated"
                    ),
                }
            )

        elif etype == "dfs_branch_cleared":
            nid = ev["node_id"]
            node = tree.nodes[nid]
            frames.append(
                {
                    **base,
                    "title": (
                        f"CLEARED  #{nid}  |  "
                        f"p_false dropped to {ev['p_false_new']:.3f} < σ={sigma}  |  "
                        f"DFS continues deeper on same branch"
                    ),
                }
            )

        elif etype == "dfs_quarantine":
            nid = ev["node_id"]
            frames.append(
                {
                    **base,
                    "title": (
                        f"QUARANTINE  #{nid}  |  "
                        f"subtree abandoned  |  "
                        f"moving sideways to frontier  ({len(ev['frontier'])} branches remaining)"
                    ),
                }
            )

        elif etype == "dfs_sideways":
            nid = ev["node_id"]
            node = tree.nodes[nid]
            frames.append(
                {
                    **base,
                    "title": (
                        f"SIDEWAYS → frontier  #{nid}  |  "
                        f"p_false={ev['p_false']:.3f}  |  "
                        f"{ev['frontier_remaining']} branches still in frontier  |  "
                        f"reason: {ev['reason']}"
                    ),
                }
            )

        elif etype == "dfs_expand":
            nid = ev["node_id"]
            elig = ev["eligible"]
            inelig = ev["ineligible"]
            e_str = ", ".join(
                f"#{c}(p={tree.nodes[c].p_false:.2f})" for c in sorted(elig)
            )
            i_str = ", ".join(
                f"#{c}(p={tree.nodes[c].p_false:.2f})" for c in sorted(inelig)
            )
            frames.append(
                {
                    **base,
                    "eligible": set(elig),
                    "ineligible": set(inelig),
                    "title": (
                        f"EXPAND  #{nid}  |  "
                        f"eligible (p<σ={sigma}): [{e_str or 'none'}]  |  "
                        f"risky candidates (p≥σ): [{i_str or 'none'}]  |  "
                        f"descending into min-error child"
                    ),
                }
            )

        elif etype == "dfs_leaf":
            nid = ev["node_id"]
            node = tree.nodes[nid]
            reason = ev.get("reason", "terminal leaf")
            frames.append(
                {
                    **base,
                    "title": (
                        f"LEAF  #{nid}  depth={node.depth}  |  "
                        f"p_false={node.p_false:.3f}  |  "
                        f"{reason}  |  backtrack"
                    ),
                }
            )

        elif etype == "dfs_max_depth":
            nid = ev["node_id"]
            node = tree.nodes[nid]
            frames.append(
                {
                    **base,
                    "title": (
                        f"MAX DEPTH  #{nid}  depth={node.depth}/{tree.max_depth}  |  "
                        f"p_false={ev['p_false']:.3f} < σ={sigma}  |  "
                        f"tree depth cap — cannot expand further  |  backtrack"
                    ),
                }
            )

        elif etype == "dfs_no_elig":
            nid = ev["node_id"]
            frames.append(
                {
                    **base,
                    "title": (
                        f"DEAD END  #{nid}  |  "
                        f"all children p_false≥σ={sigma}  |  backtrack"
                    ),
                }
            )

        elif etype == "dfs_done":
            n_gen = len(ev["p_map"])
            n_exp = len(ev["explored"])
            n_cand = len(ev["candidates"])
            n_ver = len(ev["verified"])
            n_quar = len(ev["quarantined"])
            false_v = ev.get("num_false_verified", 0)
            true_v = ev.get("num_true_verified", 0)
            depth = ev.get("max_path_depth", 0)
            sw = ev.get("sideways_moves", 0)
            frames.append(
                {
                    **base,
                    "title": (
                        f"DONE  |  {n_gen} nodes generated  {n_exp} explored  |  "
                        f"{n_cand} risky  {n_ver} verified (↓{false_v} false, ↑{true_v} true)  "
                        f"{n_quar} quarantined  |  "
                        f"max depth={depth}  sideways={sw}"
                    ),
                }
            )

    # Final truth reveal
    last = log[-1]
    visible = set(tree.nodes.keys())
    false_ids = set(tree.false_ids) & visible
    vset = set(last["verified"])
    vf = {n for n in vset if tree.nodes[n].truth == 0}
    quar: set = set()
    for nid in vf:
        quar.update(tree.subtree_ids(nid))
    quar &= visible
    undetected = false_ids - vset - quar
    reliability = 1.0 - len(undetected) / len(visible) if visible else 1.0
    recall = len(vf | (quar & false_ids)) / len(false_ids) if false_ids else 1.0

    frames.append(
        {
            "p_map": {n: tree.nodes[n].p_false for n in tree.nodes},
            "candidates": set(last["candidates"]),
            "verified": set(last["verified"]),
            "quarantined": set(last["quarantined"]),
            "cleared": set(last["cleared"]),
            "path": [],
            "frontier": [],
            "truth_mode": True,
            "title": (
                f"FINAL — ground truth  |  "
                f"{len(visible)} nodes generated  {len(false_ids)} false  |  "
                f"reliability={reliability:.2f}  recall={recall:.2f}  "
                f"undetected={len(undetected)}"
            ),
        }
    )

    return frames


def _dfs_legend() -> list:
    return [
        mpatches.Patch(color=pf_color(0.1), label="p_false low"),
        mpatches.Patch(color=pf_color(0.9), label="p_false high"),
        mpatches.Patch(facecolor="none", edgecolor="#00e5ff", lw=2, label="DFS path"),
        mpatches.Patch(
            facecolor="none", edgecolor="#f39c12", lw=2.2, label="risky candidate"
        ),
        mpatches.Patch(
            facecolor="none",
            edgecolor="#4fc3f7",
            lw=1.5,
            linestyle="--",
            label="frontier / eligible",
        ),
        mpatches.Patch(
            facecolor="none", edgecolor="white", lw=2.2, label="verified true"
        ),
        mpatches.Patch(
            facecolor="none", edgecolor="#e74c3c", lw=2.5, label="verified false"
        ),
        mpatches.Patch(
            facecolor="none", edgecolor="#f1c40f", lw=2.0, label="quarantined"
        ),
        mpatches.Patch(facecolor="#2ecc71", alpha=0.35, label="cleared glow"),
    ]


def _final_legend() -> list:
    return [
        mpatches.Patch(color="#2ecc71", label="truth = true"),
        mpatches.Patch(color="#e74c3c", label="truth = false"),
        mpatches.Patch(
            facecolor="none", edgecolor="#f39c12", lw=2.2, label="candidate"
        ),
        mpatches.Patch(
            facecolor="none", edgecolor="white", lw=2.2, label="verified true"
        ),
        mpatches.Patch(
            facecolor="none", edgecolor="#e74c3c", lw=2.5, label="verified false"
        ),
        mpatches.Patch(
            facecolor="none", edgecolor="#f1c40f", lw=2, label="quarantined"
        ),
    ]


#  Animation

def animate(
    tree: ClaimTree,
    frames: list,
    sigma: float,
    fps: int = FPS,
    out_name: str = "tree_live_simulation.gif",
) -> None:
    pos = compute_layout(tree)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#0f0f14")
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.02, right=0.98)

    rule = (
        f"single-branch DFS  |  "
        f"risky node (p≥σ={sigma}) → verify parent  |  "
        f"parent TRUE → continue deeper  |  "
        f"parent FALSE → quarantine, move to frontier"
    )
    fig.text(0.5, 0.01, rule, ha="center", va="bottom", fontsize=7.5, color="#888899")

    def update(idx: int) -> None:
        f = frames[idx]
        truth_mode = f.get("truth_mode", False)
        draw_frame(
            ax,
            fig,
            tree,
            pos,
            p_map=f["p_map"],
            title=f["title"],
            dfs_path=f.get("path"),
            candidates=f.get("candidates"),
            eligible_set=f.get("eligible"),
            ineligible_set=f.get("ineligible"),
            verified_set=f.get("verified"),
            quarantined=f.get("quarantined"),
            changed_nodes=f.get("changed_nodes"),
            cleared_nodes=f.get("cleared"),
            frontier=f.get("frontier"),
            truth_mode=truth_mode,
        )
        for leg in fig.legends:
            leg.remove()
        handles = _final_legend() if truth_mode else _dfs_legend()
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=5,
            fontsize=7,
            labelcolor="#dddddd",
            framealpha=0.15,
            bbox_to_anchor=(0.5, 0.052),
        )

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, repeat=False
    )
    out_path = os.path.join(OUT, out_name)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close()
    print(f"Saved {out_path}  ({len(frames)} frames, ~{len(frames) / fps:.0f}s)")


if __name__ == "__main__":
    print("Training RF descendant estimator …")
    rf = train_rf()

    rf_path = os.path.join(OUT, "descendant_rf.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"  Saved trained RF to {rf_path}")

    # Find a seed with a rich full tree.  DFS uses build_full=False on same seed.
    print("Searching for seed …")
    seed = None
    for trial in range(200):
        t = ClaimTree(
            max_nodes=MAX_NODES,
            base_error_rate=BASE_ERROR_RATE,
            propagation_rate=BASE_ERROR_RATE + RHO,
            branching_lambda=BRANCHING_LAMBDA,
            seed=trial,
            build_full=True,
        )
        if len(t.nodes) >= MIN_TREE_NODES and len(t.false_ids) >= MIN_FALSE_NODES:
            seed = trial
            break

    if seed is None:
        seed = 0
    print(
        f"  seed={seed}  max_depth={t.max_depth}  "
        f"full_nodes={len(t.nodes)}  false={len(t.false_ids)}"
    )

    # DFS tree built incrementally
    tree = ClaimTree(
        max_nodes=MAX_NODES,
        base_error_rate=BASE_ERROR_RATE,
        propagation_rate=BASE_ERROR_RATE + RHO,
        branching_lambda=BRANCHING_LAMBDA,
        seed=seed,
        build_full=False,
    )

    print(f"Running single-branch DFS (sigma={SIGMA}, budget={VERIFY_BUDGET}) …")
    candidates, explored, verified_set, quarantined, log = dfs_with_log(
        tree,
        rf,
        sigma=SIGMA,
        verify_budget=VERIFY_BUDGET,
    )

    false_v = sum(1 for n in verified_set if tree.nodes[n].truth == 0)
    true_v = sum(1 for n in verified_set if tree.nodes[n].truth == 1)
    last = log[-1]

    print(
        f"  Generated: {len(tree.nodes)} nodes  |  "
        f"Explored: {len(explored)}  |  "
        f"Candidates (risky): {len(candidates)}  |  "
        f"Verified: {len(verified_set)}  (↓{false_v} false, ↑{true_v} true)  |  "
        f"Quarantined roots: {len(quarantined)}"
    )
    print(
        f"  Max path depth: {last.get('max_path_depth', '?')}  |  "
        f"Sideways moves: {last.get('sideways_moves', '?')}"
    )

    print("Building animation frames …")
    frames = build_frames(log, tree, sigma=SIGMA)
    print(f"  {len(frames)} frames")

    animate(tree, frames, sigma=SIGMA)
