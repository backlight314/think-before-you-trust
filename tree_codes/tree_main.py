# 7. MAIN
from tree_model import (
    ClaimTree,
    STRATEGIES,
    ABLATIONS,
    run_experiment,
    run_sensitivity,
    strategy_oracle,
    evaluate,
    run_calibration_comparison,
    print_calibration_table,
    run_adaptive_sigma_experiment,
)
from tree_results import (
    plot_results,
    plot_sensitivity,
    plot_ablation,
    plot_tree_snapshot,
    plot_all_strategy_trees,
    plot_heuristic_gap,
    plot_calibration_comparison,
    plot_fixed_vs_adaptive,
)


if __name__ == "__main__":
    print("=" * 65)
    print("  Hallucination Propagation Sim  v9  — fixed budget + DP")
    print("=" * 65)

    #DP correctness diagnostic (run before main experiment)
    from tree_model import assert_dp_dominates
    assert_dp_dominates(n_trees=30, budget=20)

    MAX_NODES = 80
    N_TRIALS = 50

    #Reference tree (fixed topology for snapshot)
    tree = ClaimTree(max_nodes=MAX_NODES, seed=2, max_depth=8, branching_lambda=1.7)
    n = len(tree.nodes)
    nf = len(tree.false_ids)
    budget_20 = 20  # fixed absolute budget

    print(
        f"\nReference tree: {n} nodes, {nf} false ({nf / n * 100:.1f}%), "
        f"max depth {tree.max_depth_actual}"
    )
    print(f"Budget (20%): {budget_20} checks\n")

    # Single-tree table — all strategies including DP
    print(
        f"{'Strategy':<22}  {'Prec':>6}  {'Recall':>7}  {'Rel':>7}  {'Casc.Prev':>10}"
    )
    print("-" * 62)
    da_vset = None
    for name, fn in STRATEGIES.items():
        vset = fn(tree, budget_20)
        m = evaluate(tree, vset)
        if name == "Dependency-aware":
            da_vset = vset
        print(
            f"{name:<22}  {m['precision']:>6.3f}  {m['recall']:>7.3f}  "
            f"{m['reliability']:>7.3f}  {m['cascade_prevented']:>10.0f}"
        )

    plot_tree_snapshot(
        tree, da_vset, title="Claim Tree — Dependency-aware verification (20% budget)"
    )

    # One subplot per strategy on the same tree
    plot_all_strategy_trees(tree, budget_20)

    #Main 50-trial experiment
    # Note: DP is capped at 20 checks internally (DP_BUDGET_CAP).
    # Budget sweep goes up to 50% for other strategies; DP results are
    # most meaningful in the low-budget region (5-17% of 120 nodes).
    budgets = [5, 10, 15, 20, 25, 30, 40, 50]
    print(f"\nRunning main experiment ({N_TRIALS} trials) ...")
    results = run_experiment(budgets, n_trials=N_TRIALS, max_nodes=MAX_NODES)

    # Oracle baseline (100% budget)
    oracle_res = run_experiment(
        budgets,
        n_trials=N_TRIALS,
        max_nodes=MAX_NODES,
        strategy_set={"Oracle": strategy_oracle},
    )

    plot_results(results, budgets, total_nodes=n, oracle_results=oracle_res)

    #Crossover table
    print("\n── Strategy reliability vs DP Optimal vs Oracle ──")
    print(f"  {'Budget%':>7}  {'Random':>8}  {'Uncertainty':>12}  {'Dep-aware':>10}  {'DP Opt':>8}  {'Oracle':>8}")
    for b in budgets:
        r  = results["Random"][b]["mean"]["reliability"]
        u  = results["Uncertainty"][b]["mean"]["reliability"]
        d  = results["Dependency-aware"][b]["mean"]["reliability"]
        dp = results["DP Optimal"][b]["mean"]["reliability"]
        o  = oracle_res["Oracle"][b]["mean"]["reliability"]
        print(f"  {b:>7}   {r:>8.3f}  {u:>12.3f}  {d:>10.3f}  {dp:>8.3f}  {o:>8.3f}")

    # Gap: dep-aware vs DP at 20%
    b20 = 20
    gap = (
        results["DP Optimal"][b20]["mean"]["reliability"]
        - results["Dependency-aware"][b20]["mean"]["reliability"]
    )
    print(f"\n  Dep-aware gap vs DP Optimal at 20% budget: {gap:.4f}")

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

    #Adaptive sigma vs fixed sigma comparison
    print(f"\nRunning adaptive sigma experiment ({N_TRIALS} trials) ...")
    FIXED_SIGMAS = [1, 2, 5, 10, 20]
    adaptive_results = run_adaptive_sigma_experiment(
        fixed_sigmas=FIXED_SIGMAS,
        n_trials=N_TRIALS,
        max_nodes=MAX_NODES,
    )
    am = adaptive_results["adaptive"]["mean"]
    print(
        f"  adaptive — reliability: {am['reliability']:.3f}  "
        f"contamination_rate: {am['contamination_rate']:.3f}  "
        f"verify_budget_used: {am['verify_budget_used']:.1f}  "
        f"quarantined: {am['quarantined']:.1f}  "
        f"false: {am['false_nodes']:.1f}  "
        f"undetected: {am['undetected_false']:.1f}"
    )

    for s in FIXED_SIGMAS:
        fm = adaptive_results[s]["mean"]
        print(
            f"  sigma={s:>5} — reliability: {fm['reliability']:.3f}  "
            f"contamination_rate: {fm['contamination_rate']:.3f}"
        )

    plot_fixed_vs_adaptive(adaptive_results, fixed_sigmas=FIXED_SIGMAS)

    print("\nDone. Outputs saved to ./results/")

    # Oracle-DP single-tree sanity check
    from tree_model import strategy_dp_oracle
    oracle_dp_vset = strategy_dp_oracle(tree, budget_20)
    belief_dp_vset = STRATEGIES["DP Optimal"](tree, budget_20)
    m_odp = evaluate(tree, oracle_dp_vset)
    m_bdp = evaluate(tree, belief_dp_vset)
    false_ids_set = set(tree.false_ids)
    overlap = oracle_dp_vset & belief_dp_vset

    print("\n── Oracle-DP vs Belief-DP (reference tree, budget=20) ──")
    print(f"  {'Strategy':<12}  {'Prec':>6}  {'Recall':>7}  {'Rel':>7}")
    print(f"  {'DP Oracle':<12}  {m_odp['precision']:>6.3f}  {m_odp['recall']:>7.3f}  {m_odp['reliability']:>7.3f}")
    print(f"  {'DP Belief':<12}  {m_bdp['precision']:>6.3f}  {m_bdp['recall']:>7.3f}  {m_bdp['reliability']:>7.3f}")
    print(f"  Overlap: {len(overlap)}  |  Oracle-only: {len(oracle_dp_vset - belief_dp_vset)} ({len((oracle_dp_vset - belief_dp_vset) & false_ids_set)} false)  |  Belief-only: {len(belief_dp_vset - oracle_dp_vset)} ({len((belief_dp_vset - oracle_dp_vset) & false_ids_set)} false)")

    #Calibration comparison (50 trials × 2 regimes)
    budgets_cal = [5, 10, 15, 20, 25, 30, 40, 50]
    print(f"\nRunning calibration comparison ({N_TRIALS} trials × 3 regimes) ...")
    cal_results = run_calibration_comparison(budgets=budgets_cal, n_trials=N_TRIALS, max_nodes=MAX_NODES)
    print_calibration_table(cal_results, budgets_cal, key="reliability")
    print_calibration_table(cal_results, budgets_cal, key="recall")
    print_calibration_table(cal_results, budgets_cal, key="precision")

    plot_calibration_comparison(cal_results, budgets_cal, total_nodes=MAX_NODES)