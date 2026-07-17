
"""
stats_significance.py

Paired significance tests for the headline reliability comparisons across
the tree, DAG, and adaptive experiments.

Everything here runs on matched seeds, so each comparison is a paired
difference (same tree topology, same trial, two strategies or two regimes)
rather than an unpaired comparison across separate samples. paired_bootstrap
resamples those per-trial differences with replacement, builds a 95% CI on
the mean difference, and reports a two-sided bootstrap p-value alongside a
paired t-test as a parametric check.

Three blocks of comparisons:

tree_comparisons runs the static ClaimTree strategies under both the
default and overconfident regimes and checks whether Dependency-aware and
Uncertainty actually beat Random, and whether the overconfident regime
hurts each strategy differently.

dag_comparisons does the same on ClaimDAG, with Greedy MC added, to check
whether Greedy MC holds up under the overconfident regime while the
confidence-based strategies degrade (the regime immunity claim).

adaptive_comparisons compares the online adaptive policy at a couple of
fixed sigma values against a random-baseline adaptive policy, to see if
sigma-based verification actually beats verifying at random.

Run directly to print all three reports and save them as CSVs under
results_stats/. Use --quick for a fast 12-trial run, or --trials to set an
exact count.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tree_codes"))

try:
    from scipy import stats as sps
except ImportError:
    sps = None

OUT_DIR = "results_stats"
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOT = 10_000


# core paired-bootstrap machinery


def paired_bootstrap(diffs, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """
    Bootstrap the mean of per-trial paired differences.

    Returns mean_diff, 95% CI, two-sided bootstrap p-value (probability
    that a resampled mean lands on the other side of zero, doubled and
    clipped), and the paired t-test p as a parametric cross-check.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)

    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # two-sided bootstrap p: how often does the resampled mean cross zero?
    p_boot = 2.0 * min(
        float((boot_means <= 0).mean()),
        float((boot_means >= 0).mean()),
    )
    p_boot = min(1.0, max(p_boot, 1.0 / n_boot))  # can't resolve below 1/n_boot

    p_t = float("nan")
    if sps is not None and np.std(diffs) > 0:
        p_t = float(sps.ttest_rel(diffs, np.zeros_like(diffs)).pvalue)

    return {
        "n": n,
        "mean_diff": mean_diff,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_boot": p_boot,
        "p_ttest": p_t,
        "significant_95": bool(ci_lo > 0 or ci_hi < 0),
    }


def _report(rows, title, path):
    print(f"\n== {title} ==")
    header = f"{'comparison':<46}{'mean diff':>10}{'95% CI':>22}{'p(boot)':>9}  sig?"
    print(header)
    print("-" * len(header))
    for label, r in rows:
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        star = "YES" if r["significant_95"] else "no"
        print(f"{label:<46}{r['mean_diff']:>+10.4f}{ci:>22}{r['p_boot']:>9.4f}  {star}")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "n", "mean_diff", "ci_lo", "ci_hi",
                    "p_boot", "p_ttest", "significant_95"])
        for label, r in rows:
            w.writerow([label, r["n"], r["mean_diff"], r["ci_lo"], r["ci_hi"],
                        r["p_boot"], r["p_ttest"], r["significant_95"]])
    print(f"saved {path}")


# 1. static tree regime flip


def tree_comparisons(n_trials: int, max_nodes: int = 80, budget: int = 20) -> list:
    from tree_model import ClaimTree, STRATEGIES, evaluate

    strat_names = ["Random", "Uncertainty", "Dependency-aware", "DP Optimal"]
    rel = {(reg, s): [] for reg in ("default", "overconf") for s in strat_names}

    for trial in range(n_trials):
        for reg, oc in (("default", False), ("overconf", True)):
            t = ClaimTree(max_nodes=max_nodes, seed=trial, max_depth=6,
                          branching_lambda=1.7, base_error_rate=0.1758,
                          propagation_rate=0.1758 + 0.7504, overconfident=oc)
            for s in strat_names:
                m = evaluate(t, STRATEGIES[s](t, budget))
                rel[(reg, s)].append(m["reliability"])

    rows = []
    a = lambda k: np.array(rel[k])
    # within-regime strategy gaps
    rows.append(("default: Dep-aware - Uncertainty",
                 paired_bootstrap(a(("default", "Dependency-aware")) - a(("default", "Uncertainty")))))
    rows.append(("default: Uncertainty - Random",
                 paired_bootstrap(a(("default", "Uncertainty")) - a(("default", "Random")))))
    rows.append(("overconf: Dep-aware - Uncertainty",
                 paired_bootstrap(a(("overconf", "Dependency-aware")) - a(("overconf", "Uncertainty")))))
    rows.append(("overconf: Uncertainty - Random",
                 paired_bootstrap(a(("overconf", "Uncertainty")) - a(("overconf", "Random")))))
    rows.append(("overconf: DP Optimal - Dep-aware",
                 paired_bootstrap(a(("overconf", "DP Optimal")) - a(("overconf", "Dependency-aware")))))
    # the regime effect itself (same seeds, same tree structure)
    rows.append(("Uncertainty: overconf - default (regime hit)",
                 paired_bootstrap(a(("overconf", "Uncertainty")) - a(("default", "Uncertainty")))))
    rows.append(("Dep-aware: overconf - default (regime hit)",
                 paired_bootstrap(a(("overconf", "Dependency-aware")) - a(("default", "Dependency-aware")))))
    return rows


# 2. DAG regime flip


def dag_comparisons(n_trials: int, max_nodes: int = 80, budget: int = 20) -> list:
    from dag_model import ClaimDAG, STRATEGIES, evaluate

    strat_names = ["Random", "Uncertainty", "Dependency-aware", "Greedy MC"]
    rel = {(reg, s): [] for reg in ("default", "overconf") for s in strat_names}

    for trial in range(n_trials):
        for reg, oc in (("default", False), ("overconf", True)):
            d = ClaimDAG(max_nodes=max_nodes, seed=trial, max_depth=6,
                         branching_lambda=2.2, overconfident=oc)
            for s in strat_names:
                m = evaluate(d, STRATEGIES[s](d, budget))
                rel[(reg, s)].append(m["reliability"])

    rows = []
    a = lambda k: np.array(rel[k])
    rows.append(("overconf: Greedy MC - Uncertainty",
                 paired_bootstrap(a(("overconf", "Greedy MC")) - a(("overconf", "Uncertainty")))))
    rows.append(("overconf: Greedy MC - Dep-aware",
                 paired_bootstrap(a(("overconf", "Greedy MC")) - a(("overconf", "Dependency-aware")))))
    rows.append(("default: Greedy MC - Uncertainty",
                 paired_bootstrap(a(("default", "Greedy MC")) - a(("default", "Uncertainty")))))
    rows.append(("Uncertainty: overconf - default (regime hit)",
                 paired_bootstrap(a(("overconf", "Uncertainty")) - a(("default", "Uncertainty")))))
    rows.append(("Greedy MC: overconf - default (regime immunity)",
                 paired_bootstrap(a(("overconf", "Greedy MC")) - a(("default", "Greedy MC")))))
    return rows


# 3. adaptive vs random baseline


def adaptive_comparisons(n_trials: int, max_nodes: int = 80,
                         verify_budget: int = 10) -> list:
    from dag_model import ClaimDAG
    from adaptive_dag_model import (
        run_adaptive, run_adaptive_random, compare_to_static,
    )

    sigmas = (1.0, 2.0)
    rel = {f"sigma={s}": [] for s in sigmas}
    rel["baseline"] = []

    for trial in range(n_trials):
        true_dag = ClaimDAG(max_nodes=max_nodes, seed=trial, branching_lambda=2.2)
        b = run_adaptive_random(true_dag, verify_budget=verify_budget, rng_seed=trial)
        rel["baseline"].append(compare_to_static(b, true_dag)["reliability"])
        for s in sigmas:
            st = run_adaptive(true_dag, sigma=s, verify_budget=verify_budget,
                              rng_seed=trial)
            rel[f"sigma={s}"].append(compare_to_static(st, true_dag)["reliability"])

    rows = []
    base = np.array(rel["baseline"])
    for s in sigmas:
        rows.append((f"adaptive sigma={s} - random baseline",
                     paired_bootstrap(np.array(rel[f"sigma={s}"]) - base)))
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="12 trials instead of 50")
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()

    n = args.trials if args.trials else (12 if args.quick else 50)

    print("=" * 72)
    print(f"  Paired significance tests for headline comparisons  (n={n} trials)")
    print("=" * 72)

    _report(tree_comparisons(n), "STATIC TREE (reliability @ budget 20)",
            os.path.join(OUT_DIR, "significance_tree.csv"))

    _report(adaptive_comparisons(n), "ADAPTIVE vs RANDOM BASELINE (reliability)",
            os.path.join(OUT_DIR, "significance_adaptive.csv"))

    print("\n(DAG block last — Greedy MC makes it the slow one)")
    _report(dag_comparisons(n), "DAG (reliability @ budget 20)",
            os.path.join(OUT_DIR, "significance_dag.csv"))

    print("\nDone.")
