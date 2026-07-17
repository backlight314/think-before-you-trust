"""
Measure the graph geometry of real LLM claim-dependency graphs.

Computes the structural metrics that map onto the generator's parameters,
following Xiong et al. 2505.13890 (branching / convergence) and
Minegishi et al. 2506.05744 (diameter / small-world), on your real claim
trace JSONs. Then compares to the simulator's current settings so you can
see whether generation is realistic or needs retuning.

Metrics
-------
branching_ratio (gamma_B) : mean out-degree over non-leaf nodes
                            -> corresponds to simulator branching_lambda
convergence_ratio (gamma_C): fraction of nodes with >1 parent
                            -> corresponds to simulator extra_edge_prob
mean_in_degree            : average parents per non-root node
diameter                  : longest shortest-path in the undirected graph
avg_clustering            : mean clustering coefficient (undirected)
avg_path_length           : mean shortest-path length (largest component)
small_world_index (S)     : (C/C_rand) / (L/L_rand), Humphries-Gurney
                            S > 1 => small-world

Self-loops (a node listing itself as its own parent) are dropped at load and
counted separately. They are a tail-degeneration artifact in some models and
would otherwise inflate in-degree, and therefore convergence_ratio, for
exactly the models whose graphs are malformed.

Usage: python3 measure_graph_geometry.py <folder or files>
"""

import sys
import os
import json
import math
import numpy as np
import networkx as nx


def load_graph(path):
    raw = open(path, encoding="utf-8-sig").read()
    data = json.loads(raw)
    G = nx.DiGraph()
    for node in data:
        G.add_node(node["node_id"])
    n_self = 0
    for node in data:
        vid = node["node_id"]
        for p in node.get("parents", []):
            if p == vid:                 # self-loop: not a real dependency
                n_self += 1
                continue
            G.add_edge(p, vid)           # parent -> child
    return G, data, n_self


def measure(G, n_self=0):
    n = G.number_of_nodes()
    m = G.number_of_edges()

    out_deg = dict(G.out_degree())
    in_deg = dict(G.in_degree())

    # branching ratio: mean out-degree over nodes that branch (non-leaves)
    non_leaf = [v for v in G if out_deg[v] > 0]
    gamma_B = float(np.mean([out_deg[v] for v in non_leaf])) if non_leaf else 0.0

    # convergence ratio: fraction of nodes with >1 parent
    multi_parent = [v for v in G if in_deg[v] > 1]
    gamma_C = len(multi_parent) / n if n else 0.0

    non_root = [v for v in G if in_deg[v] > 0]
    mean_in = float(np.mean([in_deg[v] for v in non_root])) if non_root else 0.0

    # undirected for distance / clustering / small-world
    UG = G.to_undirected()

    # diameter + avg path length on the largest connected component
    if UG.number_of_nodes() > 1:
        comps = list(nx.connected_components(UG))
        largest = max(comps, key=len)
        H = UG.subgraph(largest)
        diameter = nx.diameter(H) if H.number_of_nodes() > 1 else 0
        avg_path = nx.average_shortest_path_length(H) if H.number_of_nodes() > 1 else 0.0
    else:
        diameter, avg_path = 0, 0.0

    avg_clustering = nx.average_clustering(UG)

    # small-world index (Humphries-Gurney): compare C and L to an
    # Erdos-Renyi random graph with same n and mean degree K.
    K = (2.0 * UG.number_of_edges()) / n if n else 0.0
    if K > 1 and n > 1 and avg_path > 0 and avg_clustering > 0:
        C_rand = K / (n - 1)
        L_rand = math.log(n) / math.log(K) if K > 1 else float("nan")
        if C_rand > 0 and L_rand and not math.isnan(L_rand) and L_rand > 0:
            S = (avg_clustering / C_rand) / (avg_path / L_rand)
        else:
            S = float("nan")
    else:
        S = float("nan")

    return {
        "n": n, "edges": m,
        "branching_ratio": gamma_B,
        "convergence_ratio": gamma_C,
        "mean_in_degree": mean_in,
        "n_multi_parent": len(multi_parent),
        "n_self_loops": n_self,
        "is_dag": nx.is_directed_acyclic_graph(G),
        "diameter": diameter,
        "avg_path_length": avg_path,
        "avg_clustering": avg_clustering,
        "mean_degree_K": K,
        "small_world_index": S,
    }


def print_report(path, mets):
    print(f"\n{'='*60}\n  {os.path.basename(path)}   [{mets['_model']}]\n{'='*60}")
    print(f"  nodes / edges          : {mets['n']} / {mets['edges']}")
    print(f"  branching ratio  (γ_B) : {mets['branching_ratio']:.3f}   "
          f"[sim branching_lambda = 2.2]")
    print(f"  convergence ratio(γ_C) : {mets['convergence_ratio']:.3f}   "
          f"[sim extra_edge_prob  = 0.20]")
    print(f"  mean in-degree         : {mets['mean_in_degree']:.3f}")
    print(f"  multi-parent nodes     : {mets['n_multi_parent']}")
    if mets["n_self_loops"]:
        print(f"  self-loops dropped     : {mets['n_self_loops']}  "
              f"<-- malformed: node listed itself as parent")
    if not mets["is_dag"]:
        print(f"  acyclic                : NO  <-- graph contains a cycle")
    print(f"  diameter               : {mets['diameter']}")
    print(f"  avg path length        : {mets['avg_path_length']:.3f}")
    print(f"  avg clustering         : {mets['avg_clustering']:.3f}")
    print(f"  mean degree K          : {mets['mean_degree_K']:.3f}")
    sw = mets['small_world_index']
    sw_str = f"{sw:.3f}" if not math.isnan(sw) else "n/a (no triangles)"
    print(f"  small-world index (S)  : {sw_str}   [S>1 => small-world]")


# Size-qualified tags, longest first, so 'llama-70b' never collapses into a
# generic 'llama' bucket and 'gptoss-20b' never falls through to 'chatgpt'.
# NOTE: real filenames use 'gptoss-20b' (no hyphen after gpt); the hyphenated
# spellings are kept as aliases in case older files use them.
SPECIFIC_MODELS = (
    "gpt-oss-120b", "gpt-oss-20b",
    "gptoss-120b", "gptoss-20b",
    "llama-3.3-70b", "llama-3.1-8b",
    "llama-70b", "llama-8b",
    "qwen-80b", "qwen-72b", "qwen-32b", "qwen-7b",
    "deepseek-70b", "deepseek-7b",
)

GENERIC_MODELS = ("chatgpt", "gpt", "chat", "claude", "gemini",
                  "llama", "qwen", "deepseek")


def model_of(path):
    """Infer model name from filename, keeping size (8b vs 70b) distinct."""
    base = os.path.basename(path).lower()
    for m in sorted(SPECIFIC_MODELS, key=len, reverse=True):
        if m in base:
            return m.replace("gpt-oss", "gptoss")     # normalise aliases
    for m in GENERIC_MODELS:
        if (base.startswith(m) or f"_{m}" in base or f"-{m}" in base
                or m in base.split("_")[0]):
            return "chatgpt" if m in ("chat", "gpt") else m
    return "unknown"


def aggregate(mets_list, label):
    print(f"\n{'='*60}\n  {label}  ({len(mets_list)} graphs)\n{'='*60}")
    for key in ["n", "branching_ratio", "convergence_ratio", "mean_in_degree",
                "n_multi_parent", "n_self_loops", "diameter", "avg_path_length",
                "avg_clustering", "small_world_index"]:
        vals = [m[key] for m in mets_list
                if not (isinstance(m[key], float) and math.isnan(m[key]))]
        if vals:
            print(f"  {key:<20}: mean {np.mean(vals):7.3f}  "
                  f"(min {np.min(vals):6.3f}, max {np.max(vals):7.3f}, n={len(vals)})")


if __name__ == "__main__":
    import glob
    args = sys.argv[1:]
    if not args:
        print("usage: python measure_graph_geometry.py <folder or files>")
        sys.exit(1)

    paths = []
    for a in args:
        if os.path.isdir(a):
            paths.extend(sorted(glob.glob(os.path.join(a, "*.json"))))
        else:
            expanded = glob.glob(a)
            paths.extend(sorted(expanded) if expanded else [a])

    # skip derived files: only raw run traces are claim graphs
    paths = [p for p in paths
             if not (p.endswith("_verified.json") or p.endswith("_pooled.json"))]

    by_model = {}
    all_mets = []
    for path in paths:
        try:
            G, data, n_self = load_graph(path)
        except Exception as e:
            print(f"  SKIP {os.path.basename(path)}: {e}")
            continue
        mets = measure(G, n_self)
        mets["_model"] = model_of(path)
        all_mets.append(mets)
        by_model.setdefault(mets["_model"], []).append(mets)
        print_report(path, mets)

    # per-model aggregates
    for model in sorted(by_model):
        aggregate(by_model[model], f"MODEL: {model}")

    # overall
    if len(all_mets) > 1:
        aggregate(all_mets, "ALL MODELS COMBINED")

    # compact comparison table across models
    if len(by_model) > 1:
        print(f"\n{'='*60}\n  BY-MODEL COMPARISON (means)\n{'='*60}")
        print(f"  {'model':<14}{'runs':>6}{'γ_B':>8}{'γ_C':>8}{'nodes':>8}"
              f"{'diam':>8}{'clust':>8}{'S':>8}{'selfloop':>10}")
        for model in sorted(by_model):
            ms = by_model[model]

            def mean_of(k, _ms=ms):
                v = [m[k] for m in _ms
                     if not (isinstance(m[k], float) and math.isnan(m[k]))]
                return np.mean(v) if v else float("nan")

            print(f"  {model:<14}{len(ms):>6}{mean_of('branching_ratio'):>8.2f}"
                  f"{mean_of('convergence_ratio'):>8.3f}{mean_of('n'):>8.0f}"
                  f"{mean_of('diameter'):>8.1f}{mean_of('avg_clustering'):>8.3f}"
                  f"{mean_of('small_world_index'):>8.2f}"
                  f"{mean_of('n_self_loops'):>10.2f}")