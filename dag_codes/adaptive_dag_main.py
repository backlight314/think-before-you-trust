"""
Adaptive DAG entry script.

This script reuses the existing `dag_model` and `dag_visualization` modules
to run experiments and save visualization outputs to `results_adaptive_dag`.
"""

import os
import argparse

from dag_model import (
    ClaimDAG, STRATEGIES, ABLATIONS,
    run_experiment, run_sensitivity, strategy_oracle, evaluate,
)
import dag_visualization as dv


def main():
    parser = argparse.ArgumentParser(description="Run adaptive DAG-style experiments")
    parser.add_argument("--max-nodes", type=int, default=80, help="Max nodes in DAG")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials per experiment")
    parser.add_argument(
        "--budgets",
        type=str,
        default="5,10,15,20,25,30,40,50",
        help="Comma-separated list of budgets (checks) to evaluate",
    )
    parser.add_argument("--output-dir", type=str, default="results_adaptive_dag", help="Directory for plot outputs")
    parser.add_argument("--seed", type=int, default=2, help="Random seed for reference DAG")

    args = parser.parse_args()

    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    # Ensure dag_visualization writes into the adaptive results folder
    dv.OUT_DIR = OUT_DIR

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]

    print("=" * 65)
    print("  Adaptive-style DAG Experiment — using existing DAG modules")
    print("=" * 65)

    dag = ClaimDAG(max_nodes=args.max_nodes, seed=args.seed)
    n = len(dag.nodes)
    nf = len(dag.false_ids)

    print(f"Reference DAG: {n} nodes, {nf} false ({nf / n * 100:.1f}%), max depth {dag.max_depth_actual}")

    budget_20 = budgets[3] if len(budgets) > 3 else (budgets[-1] if budgets else 20)

    # Single-DAG table
    print(f"\n{'Strategy':<22}  {'Prec':>6}  {'Recall':>7}  {'Rel':>7}  {'Casc.Prev':>10}")
    print("-" * 62)
    da_vset = None
    for name, fn in STRATEGIES.items():
        vset = fn(dag, budget_20)
        m = evaluate(dag, vset)
        if name == "Dependency-aware":
            da_vset = vset
        print(f"{name:<22}  {m['precision']:>6.3f}  {m['recall']:>7.3f}  {m['reliability']:>7.3f}  {m['cascade_prevented']:>10.0f}")

    dv.plot_dag_snapshot(dag, da_vset, title="Adaptive-style DAG — Dependency-aware (snapshot)")
    dv.plot_all_strategy_dags(dag, budget_20)

    print(f"\nRunning main experiment ({args.trials} trials) ...")
    # Wrap/adjust heavy strategies for quicker adaptive runs (lower MC samples)
    strategy_set = {}
    for name, fn in STRATEGIES.items():
        if name == "Greedy MC":
            # reduce Monte-Carlo simulations to speed up runs
            strategy_set[name] = (lambda fn: (lambda dag, budget: fn(dag, budget, n_simulations=30)))(fn)
        else:
            strategy_set[name] = fn

    results = run_experiment(budgets, n_trials=args.trials, max_nodes=args.max_nodes, strategy_set=strategy_set)

    oracle_res = run_experiment(budgets, n_trials=args.trials, max_nodes=args.max_nodes, strategy_set={"Oracle": strategy_oracle})

    dv.plot_results(results, budgets, total_nodes=n, oracle_results=oracle_res)
    dv.plot_heuristic_gap(results, budgets, total_nodes=n)

    print(f"\nRunning sensitivity analysis ({args.trials} trials x 3 regimes) ...")
    sensitivity = run_sensitivity(budgets, n_trials=args.trials, max_nodes=args.max_nodes)
    dv.plot_sensitivity(sensitivity, budgets, total_nodes=n)

    print(f"\nRunning ablation study ({args.trials} trials) ...")
    ablation_results = run_experiment(budgets, n_trials=args.trials, max_nodes=args.max_nodes, strategy_set=ABLATIONS)
    dv.plot_ablation(ablation_results, budgets, total_nodes=n)

    print(f"\nDone. Outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
