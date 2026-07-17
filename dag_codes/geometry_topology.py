"""
  PART 1  Topology/geometry invariants, real vs synthetic
          branching factor, out-degree dispersion (Poisson => 1.0),
          multi-parent fraction, depth profile, ball-expansion ratio
          |B2|/|B1|, betweenness mass — for every real trace and for
          size-matched ClaimDAG samples. KS test on out-degree
          distributions. Per-model lambda_hat you could plug back into
          the generator.

  PART 2  Betweenness as a verification signal, validated on REAL data
          Does betweenness correlate with descendant count (the damage
          proxy) in real traces? Do FALSE claims sit at different
          betweenness than true ones? (Mann-Whitney, per model.)

  PART 3  strategy_betweenness on the synthetic harness
          Top-k betweenness (pure structure, confidence-free) and a
          betweenness x predicted-error hybrid, evaluated with the
          frozen evaluate() harness against Random/Uncertainty/
          Dep-aware/Greedy MC in both confidence regimes. Prediction:
          structure-only betweenness inherits Greedy MC's regime
          immunity at a fraction of its compute cost.

Outputs: results_geometry/*.csv + printed report.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import networkx as nx

try:
    from scipy import stats as sps
except ImportError:
    sps = None

from constants import (
    BASE_ERROR_RATE,
    RHO,
    BRANCHING_LAMBDA_DAG,
    MAX_DEPTH,
    EXTRA_EDGE_PROB,
)
from dag_model import (
    ClaimDAG,
    STRATEGIES as DAG_STRATEGIES,
    evaluate as dag_evaluate,
    compute_predicted_error,
)

OUT_DIR = "results_geometry"
os.makedirs(OUT_DIR, exist_ok=True)

RUNS_GLOB = os.path.join("data", "runs", "*", "*_run1.json")
VERIFIED_GLOB = os.path.join("data", "runs", "*", "*_verified.json")


# loading real traces

def _load_json(path):
    raw = open(path, encoding="utf-8-sig").read().strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def load_real_graphs():
    """List of {model, name, G (nx.DiGraph), has_truth} dicts.

    Verified files carry a `truth` label so we prefer those; raw runs
    are the fallback. Node attrs are confidence and truth (True/False/None).
    """
    graphs = []
    seen = set()

    def add(path, prefer=False):
        data = _load_json(path)
        if not data or not isinstance(data, list):
            return
        name = os.path.basename(path).replace(".json", "")
        model = name.split("_")[0].rstrip("0123456789.")  # claude/chat/gemini...
        key = name.replace("_verified", "")
        if key in seen and not prefer:
            return
        G = nx.DiGraph()
        for nd in data:
            G.add_node(nd["node_id"], confidence=nd.get("confidence"),
                       truth=nd.get("truth", None))
        for nd in data:
            for p in nd.get("parents", []):
                G.add_edge(p, nd["node_id"])
        if len(G) < 5:
            return
        if prefer and key in seen:
            graphs[:] = [g for g in graphs if g["key"] != key]
        seen.add(key)
        has_truth = any(d.get("truth") is not None for d in data)
        graphs.append({"model": model, "name": name, "key": key,
                       "G": G, "has_truth": has_truth})

    for path in sorted(glob.glob(VERIFIED_GLOB)):
        add(path, prefer=True)
    for path in sorted(glob.glob(RUNS_GLOB)):
        add(path, prefer=False)
    return graphs


def claimdag_to_nx(dag: ClaimDAG) -> nx.DiGraph:
    G = nx.DiGraph()
    for nid, node in dag.nodes.items():
        G.add_node(nid, confidence=node.confidence,
                   truth=(node.truth == 1))
    for nid, node in dag.nodes.items():
        for c in node.children:
            G.add_edge(nid, c)
    return G


# PART 1: invariants


def invariants(G: nx.DiGraph) -> dict:
    n = len(G)
    roots = [v for v in G if G.in_degree(v) == 0]
    # depth = shortest directed distance from (first) root
    depth = {}
    for r in roots:
        for v, d in nx.single_source_shortest_path_length(G, r).items():
            depth[v] = min(depth.get(v, 10 ** 9), d)
    max_depth = max(depth.values()) if depth else 0

    # branching: mean out-degree of internal (non-leaf) nodes
    out_deg = [G.out_degree(v) for v in G]
    internal = [d for d in out_deg if d > 0]
    branching = float(np.mean(internal)) if internal else 0.0
    # dispersion index of internal out-degree (Poisson => 1.0)
    dispersion = (float(np.var(internal) / np.mean(internal))
                  if internal and np.mean(internal) > 0 else 0.0)

    multi_parent = sum(1 for v in G if G.in_degree(v) > 1) / n

    bc = nx.betweenness_centrality(G, normalized=True)
    bc_vals = np.array(list(bc.values()))

    # ball expansion |B2|/|B1| on the UNDIRECTED graph (geometry proxy:
    # how fast do balls grow around a typical node)
    U = G.to_undirected()
    ratios = []
    for v in U:
        b1 = set(nx.single_source_shortest_path_length(U, v, cutoff=1))
        b2 = set(nx.single_source_shortest_path_length(U, v, cutoff=2))
        if len(b1) > 1:
            ratios.append(len(b2) / len(b1))
    expansion = float(np.mean(ratios)) if ratios else 0.0

    return {
        "n": n,
        "branching": branching,
        "dispersion": dispersion,
        "multi_parent_frac": float(multi_parent),
        "max_depth": max_depth,
        "bc_mean": float(bc_vals.mean()),
        "bc_max": float(bc_vals.max()),
        "expansion_b2_b1": expansion,
        "out_degrees": internal,       # kept for KS tests, dropped in CSV
    }


def synthetic_invariants(n_target: int, n_samples: int = 8,
                         seed0: int = 1000) -> list:
    rows = []
    for i in range(n_samples):
        dag = ClaimDAG(max_nodes=n_target,
                       base_error_rate=BASE_ERROR_RATE,
                       propagation_rate=BASE_ERROR_RATE + RHO,
                       extra_edge_prob=EXTRA_EDGE_PROB,
                       max_depth=MAX_DEPTH,
                       branching_lambda=BRANCHING_LAMBDA_DAG,
                       seed=seed0 + i)
        rows.append(invariants(claimdag_to_nx(dag)))
    return rows


def part1_report(graphs, out_csv):
    print("\n" + "=" * 76)
    print("  PART 1 — topology/geometry: real LLM traces vs lambda-random ClaimDAG")
    print("=" * 76)

    by_model = defaultdict(list)
    for g in graphs:
        by_model[g["model"]].append(g)

    cols = ["branching", "dispersion", "multi_parent_frac",
            "max_depth", "bc_mean", "expansion_b2_b1"]
    rows_out = []

    print(f"\n{'graph set':<22}{'n':>6}" + "".join(f"{c:>14}" for c in cols))
    print("-" * (28 + 14 * len(cols)))

    real_degrees = {}
    for model, gs in sorted(by_model.items()):
        invs = [invariants(g["G"]) for g in gs]
        real_degrees[model] = [d for inv in invs for d in inv["out_degrees"]]
        mean_n = float(np.mean([inv["n"] for inv in invs]))
        agg = {c: float(np.mean([inv[c] for inv in invs])) for c in cols}
        print(f"{model + ' (real)':<22}{mean_n:>6.0f}"
              + "".join(f"{agg[c]:>14.3f}" for c in cols))
        rows_out.append([model, "real", len(gs), mean_n] + [agg[c] for c in cols])

    # synthetic reference at the pooled real scale
    mean_real_n = int(np.mean([len(g["G"]) for g in graphs]))
    syn = synthetic_invariants(mean_real_n)
    syn_degrees = [d for inv in syn for d in inv["out_degrees"]]
    agg = {c: float(np.mean([inv[c] for inv in syn])) for c in cols}
    print(f"{'ClaimDAG (synthetic)':<22}{np.mean([i['n'] for i in syn]):>6.0f}"
          + "".join(f"{agg[c]:>14.3f}" for c in cols))
    rows_out.append(["ClaimDAG", "synthetic", len(syn), mean_real_n]
                    + [agg[c] for c in cols])

    # KS tests: real out-degree distribution vs synthetic
    print("\nOut-degree distribution vs synthetic (KS test):")
    for model, degs in sorted(real_degrees.items()):
        if sps is not None and degs and syn_degrees:
            ks = sps.ks_2samp(degs, syn_degrees)
            lam_hat = float(np.mean(degs))
            print(f"  {model:<8} lambda_hat={lam_hat:5.2f}   "
                  f"KS stat={ks.statistic:.3f}  p={ks.pvalue:.2e}  "
                  f"{'DIFFERENT' if ks.pvalue < 0.05 else 'compatible'}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "kind", "n_graphs", "mean_nodes"] + cols)
        w.writerows(rows_out)
    print(f"\nsaved {out_csv}")


# PART 2: betweenness vs damage/truth on REAL data


def part2_report(graphs, out_csv):
    print("\n" + "=" * 76)
    print("  PART 2 — is betweenness a real verification signal? (real traces)")
    print("=" * 76)
    print(f"\n{'trace':<34}{'rho(bc,desc)':>13}{'p':>10}"
          f"{'bc false vs true':>18}{'p':>10}")
    print("-" * 86)

    rows = []
    by_model_bc = defaultdict(lambda: {"false": [], "true": []})

    for g in graphs:
        G = g["G"]
        bc = nx.betweenness_centrality(G, normalized=True)
        desc = {v: len(nx.descendants(G, v)) for v in G}
        vs = list(G)
        rho_s, p_s = (sps.spearmanr([bc[v] for v in vs],
                                    [desc[v] for v in vs])
                      if sps is not None else (float("nan"),) * 2)

        line = f"{g['name'][:33]:<34}{rho_s:>13.3f}{p_s:>10.4f}"
        mw_stat = mw_p = float("nan")
        if g["has_truth"]:
            f_bc = [bc[v] for v in vs if G.nodes[v]["truth"] is False]
            t_bc = [bc[v] for v in vs if G.nodes[v]["truth"] is True]
            by_model_bc[g["model"]]["false"].extend(f_bc)
            by_model_bc[g["model"]]["true"].extend(t_bc)
            if sps is not None and len(f_bc) >= 3 and len(t_bc) >= 3:
                mw = sps.mannwhitneyu(f_bc, t_bc, alternative="two-sided")
                mw_stat, mw_p = float(np.mean(f_bc) - np.mean(t_bc)), mw.pvalue
                line += f"{mw_stat:>18.5f}{mw_p:>10.4f}"
        print(line)
        rows.append([g["name"], g["model"], rho_s, p_s, mw_stat, mw_p])

    print("\nPooled false-vs-true betweenness by model (Mann-Whitney):")
    for model, d in sorted(by_model_bc.items()):
        if sps is not None and len(d["false"]) >= 5 and len(d["true"]) >= 5:
            mw = sps.mannwhitneyu(d["false"], d["true"], alternative="two-sided")
            print(f"  {model:<8} n_false={len(d['false']):<4} "
                  f"n_true={len(d['true']):<5} "
                  f"mean bc false={np.mean(d['false']):.5f} "
                  f"true={np.mean(d['true']):.5f}  p={mw.pvalue:.4f}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace", "model", "spearman_bc_desc", "p_spearman",
                    "bc_false_minus_true", "p_mannwhitney"])
        w.writerows(rows)
    print(f"saved {out_csv}")


# PART 3: betweenness as a strategy on the synthetic harness


def strategy_betweenness(dag: ClaimDAG, budget: int, **_) -> set:
    """Verify the top-k betweenness nodes. No confidence signal at all,
    so it should get Greedy MC's regime immunity for free."""
    G = claimdag_to_nx(dag)
    bc = nx.betweenness_centrality(G, normalized=True)
    top = sorted(bc, key=bc.__getitem__, reverse=True)
    return set(top[:budget])


def strategy_betweenness_risk(dag: ClaimDAG, budget: int, **_) -> set:
    """betweenness x predicted error - structure weighted by risk."""
    G = claimdag_to_nx(dag)
    bc = nx.betweenness_centrality(G, normalized=True)
    empty: set = set()
    score = {v: bc[v] * compute_predicted_error(v, dag, empty) for v in bc}
    top = sorted(score, key=score.__getitem__, reverse=True)
    return set(top[:budget])


def part3_report(n_trials, budget, out_csv):
    print("\n" + "=" * 76)
    print(f"  PART 3 — betweenness strategies on the harness "
          f"({n_trials} trials, budget {budget})")
    print("=" * 76)

    strategies = {
        "Random": DAG_STRATEGIES["Random"],
        "Uncertainty": DAG_STRATEGIES["Uncertainty"],
        "Dependency-aware": DAG_STRATEGIES["Dependency-aware"],
        "Greedy MC": DAG_STRATEGIES["Greedy MC"],
        "Betweenness": strategy_betweenness,
        "Betweenness x risk": strategy_betweenness_risk,
    }

    rel = {(reg, s): [] for reg in ("default", "overconf") for s in strategies}
    for trial in range(n_trials):
        for reg, oc in (("default", False), ("overconf", True)):
            d = ClaimDAG(max_nodes=80, seed=trial, max_depth=MAX_DEPTH,
                         branching_lambda=BRANCHING_LAMBDA_DAG,
                         overconfident=oc)
            for s, fn in strategies.items():
                rel[(reg, s)].append(
                    dag_evaluate(d, fn(d, budget))["reliability"])

    print(f"\n{'strategy':<22}{'default':>10}{'overconf':>10}{'delta':>10}")
    print("-" * 52)
    rows = []
    for s in strategies:
        dm = float(np.mean(rel[("default", s)]))
        om = float(np.mean(rel[("overconf", s)]))
        print(f"{s:<22}{dm:>10.4f}{om:>10.4f}{om - dm:>+10.4f}")
        rows.append([s, dm, om, om - dm])

    from stats_significance import paired_bootstrap
    print("\nPaired bootstrap (overconfident regime):")
    a = lambda k: np.array(rel[k])
    for label, diffs in [
        ("Betweenness - Uncertainty",
         a(("overconf", "Betweenness")) - a(("overconf", "Uncertainty"))),
        ("Betweenness - Dep-aware",
         a(("overconf", "Betweenness")) - a(("overconf", "Dependency-aware"))),
        ("Betweenness - Greedy MC",
         a(("overconf", "Betweenness")) - a(("overconf", "Greedy MC"))),
        ("Betweenness regime delta (immunity check)",
         a(("overconf", "Betweenness")) - a(("default", "Betweenness"))),
    ]:
        r = paired_bootstrap(diffs)
        sig = "SIG" if r["significant_95"] else "ns"
        print(f"  {label:<44} {r['mean_diff']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p={r['p_boot']:.4f} {sig}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "reliability_default",
                    "reliability_overconf", "delta"])
        w.writerows(rows)
    print(f"saved {out_csv}")


# ── main ─────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_trials = 10 if args.quick else args.trials

    graphs = load_real_graphs()
    print(f"loaded {len(graphs)} real LLM reasoning graphs "
          f"({sum(g['has_truth'] for g in graphs)} with truth labels)")

    part1_report(graphs, os.path.join(OUT_DIR, "invariants_real_vs_synthetic.csv"))
    part2_report(graphs, os.path.join(OUT_DIR, "betweenness_real_validation.csv"))
    part3_report(n_trials, args.budget,
                 os.path.join(OUT_DIR, "betweenness_strategies.csv"))

    print("\nDone.")
