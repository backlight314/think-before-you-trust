"""
Entry-point to run the DAG experiments and visualizations.

This mirrors the structure of `tree_main.py` but runs the DAG model and
visualizations from `dag_model.py` and `dag_visualization.py`.
"""

from dag_model import (
    ClaimDAG, STRATEGIES, ABLATIONS,
    run_experiment, run_sensitivity,
    strategy_oracle, evaluate,
    run_adaptive_sigma_experiment,
)
from dag_visualization import (
    plot_results, plot_sensitivity, plot_ablation,
    plot_dag_snapshot, plot_all_strategy_dags,
    plot_heuristic_gap,
)


if __name__ == "__main__":
    print("=" * 65)
    print("  Hallucination Propagation Sim  — DAG version")
    print("=" * 65)

    MAX_NODES = 80
    N_TRIALS = 50

    #Reference DAG (fixed topology for snapshot)
    dag = ClaimDAG(max_nodes=MAX_NODES, seed=2, max_depth=6, branching_lambda=2.2)
    n = len(dag.nodes)
    nf = len(dag.false_ids)
    budget_20 = 20  #fixed absolute budget

    print(
        f"\nReference DAG: {n} nodes, {nf} false ({nf / n * 100:.1f}%), "
        f"max depth {dag.max_depth_actual}"
    )
    print(f"Budget (20): {budget_20} checks\n")

    #Single-DAG table — all strategies including Greedy MC
    print(
        f"{'Strategy':<22}  {'Prec':>6}  {'Recall':>7}  {'Rel':>7}  {'Casc.Prev':>10}"
    )
    print("-" * 62)
    da_vset = None
    for name, fn in STRATEGIES.items():
        vset = fn(dag, budget_20)
        m = evaluate(dag, vset)
        if name == "Dependency-aware":
            da_vset = vset
        print(
            f"{name:<22}  {m['precision']:>6.3f}  {m['recall']:>7.3f}  "
            f"{m['reliability']:>7.3f}  {m['cascade_prevented']:>10.0f}"
        )

    plot_dag_snapshot(
        dag, da_vset, title="Claim DAG — Dependency-aware verification (20 checks)"
    )

    #One subplot per strategy on the same DAG
    plot_all_strategy_dags(dag, budget_20)

    #Main experiments
    budgets = [5, 10, 15, 20, 25, 30, 40, 50]
    print(f"\nRunning main experiment ({N_TRIALS} trials) ...")
    results = run_experiment(budgets, n_trials=N_TRIALS, max_nodes=MAX_NODES)

    #Oracle baseline (100% budget)
    oracle_res = run_experiment(
        budgets,
        n_trials=N_TRIALS,
        max_nodes=MAX_NODES,
        strategy_set={"Oracle": strategy_oracle},
    )

    plot_results(results, budgets, total_nodes=n, oracle_results=oracle_res)

    #Crossover table
    print("\n── Strategy reliability vs Greedy MC Optimal vs Oracle ──")
    print(
        f"  {'Budget%':>7}  {'Random':>8}  {'Dep-aware':>10}  {'Greedy MC':>10}  {'Oracle':>8}"
    )
    for b in budgets:
        pct = b
        r = results["Random"][b]["mean"]["reliability"]
        d = results["Dependency-aware"][b]["mean"]["reliability"]
        gm = results["Greedy MC"][b]["mean"]["reliability"]
        o = oracle_res["Oracle"][b]["mean"]["reliability"]
        print(f"  {pct:>7}   {r:>8.3f}  {d:>10.3f}  {gm:>10.3f}  {o:>8.3f}")

    #Gap: dep-aware vs Greedy MC at 20
    b20 = 20
    gap = (
        results["Greedy MC"][b20]["mean"]["reliability"]
        - results["Dependency-aware"][b20]["mean"]["reliability"]
    )
    print(f"\n  Dep-aware gap vs Greedy MC at 20 checks: {gap:.4f}")

    #Heuristic gap plot
    plot_heuristic_gap(results, budgets, total_nodes=n)

    #Sensitivity analysis
    print(f"\nRunning sensitivity analysis ({N_TRIALS} trials x 3 regimes) ...")
    sensitivity = run_sensitivity(budgets, n_trials=N_TRIALS, max_nodes=MAX_NODES)
    plot_sensitivity(sensitivity, budgets, total_nodes=n)

    #Ablation study
    print(f"\nRunning ablation study ({N_TRIALS} trials) ...")
    ablation_results = run_experiment(
        budgets,
        n_trials=N_TRIALS,
        max_nodes=MAX_NODES,
        strategy_set=ABLATIONS,
    )
    plot_ablation(ablation_results, budgets, total_nodes=n)

    #Adaptive sigma (empirical-quantile policy)
    print(f"\nRunning adaptive sigma experiment ({N_TRIALS} trials) ...")
    adaptive_results = run_adaptive_sigma_experiment(
        fixed_sigmas=[],   # adaptive only, no fixed-sigma comparisons
        n_trials=N_TRIALS,
        max_nodes=MAX_NODES,
    )
    am = adaptive_results["adaptive"]["mean"]
    print(
        f"  reliability: {am['reliability']:.3f}  "
        f"contamination_rate: {am['contamination_rate']:.3f}  "
        f"verify_budget_used: {am['verify_budget_used']:.1f}  "
        f"quarantined: {am['quarantined']:.1f}"
    )

    print("\nDone. Outputs saved to ./results_dag/")
