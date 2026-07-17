# 6. VISUALISATION

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import os
from tree_model import ClaimTree, STRATEGIES, ABLATIONS, evaluate

OUT_DIR = "results_tree"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "Random": "#888888",
    "Level-sampling": "#4e9af1",
    "Recent (deep)": "#f4a261",
    "Uncertainty": "#e76f51",
    "Dependency-aware": "#2ecc71",
    "DP Optimal": "#f1c40f",
    "DP Oracle": "#a29bfe",   
}

ABLATION_COLORS = {
    "Conf only": "#e74c3c",
    "Descendants only": "#3498db",
    "Depth only": "#9b59b6",
    "Conf + Desc": "#f39c12",
    "Full composite": "#2ecc71",
}


def _ax_style(ax):
    ax.set_facecolor("#16161e")
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.grid(alpha=0.15, color="#444455")


def plot_results(
    results: dict,
    budgets: list[int],
    total_nodes: int,
    oracle_results: dict = None,
    filename: str = "results.png",
):
    """Main 4-panel comparison plot with error bars and oracle baseline."""
    metrics_to_plot = [
        ("reliability", "Reliability  (↑ better)"),
        ("recall", "Recall of False Nodes  (↑ better)"),
        ("precision", "Precision  (↑ better)"),
        ("cascade_prevented", "Cascade Nodes Prevented  (↑ better)"),
    ]

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0f0f14")
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)
    budget_pct = budgets

    for idx, (metric, label) in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        _ax_style(ax)

        # Oracle flat line
        if oracle_results:
            oracle_means = [
                oracle_results["Oracle"][b]["mean"][metric] for b in budgets
            ]
            ax.plot(budget_pct, oracle_means, color="#ffffff", linewidth=1.0, linestyle="--", alpha=0.4, label="Oracle (100%)")

        for name, color in COLORS.items():
            means = [results[name][b]["mean"][metric] for b in budgets]
            stds = [results[name][b]["std"][metric] for b in budgets]
            ax.errorbar(
                budget_pct,
                means,
                yerr=stds,
                label=name,
                color=color,
                linewidth=2.0 if name == "Dependency-aware" else 1.3,
                marker="o",
                markersize=4,
                capsize=3,
                elinewidth=0.8,
                zorder=5 if name == "Dependency-aware" else 3,
            )

        ax.set_xlabel("Number of checks (budget)", color="#aaaaaa", fontsize=9)
        ax.set_ylabel(label, color="#aaaaaa", fontsize=9)
        ax.set_title(label, color="#dddddd", fontsize=10, pad=8)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        framealpha=0.15,
        labelcolor="#dddddd",
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Verification Strategy Comparison — 50-Trial Average ± 1 SD (fixed budget)",
        color="#eeeeee",
        fontsize=13,
        y=0.97,
        fontweight="bold",
    )

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def plot_sensitivity(sensitivity: dict, budgets: list[int], total_nodes: int):
    """One reliability subplot per regime, all strategies overlaid."""
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor("#0f0f14")
    gs = gridspec.GridSpec(1, 3, wspace=0.30)
    budget_pct = budgets

    for col, (regime_label, results) in enumerate(sensitivity.items()):
        ax = fig.add_subplot(gs[col])
        _ax_style(ax)
        for name, color in COLORS.items():
            means = [results[name][b]["mean"]["reliability"] for b in budgets]
            stds = [results[name][b]["std"]["reliability"] for b in budgets]
            ax.errorbar(
                budget_pct,
                means,
                yerr=stds,
                label=name,
                color=color,
                linewidth=2.0 if name == "Dependency-aware" else 1.2,
                marker="o",
                markersize=3,
                capsize=2,
                elinewidth=0.7,
            )
        ax.set_title(regime_label, color="#dddddd", fontsize=10)
        ax.set_xlabel("# checks", color="#aaaaaa", fontsize=9)
        ax.set_ylabel("Reliability" if col == 0 else "", color="#aaaaaa", fontsize=9)
        ax.set_ylim(0.6, 1.02)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        framealpha=0.15,
        labelcolor="#dddddd",
        fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        "Sensitivity Analysis — Reliability Across Error Regimes",
        color="#eeeeee",
        fontsize=12,
        y=1.02,
        fontweight="bold",
    )

    path = os.path.join(OUT_DIR, "sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def plot_ablation(ablation_results: dict, budgets: list[int], total_nodes: int):
    """Show which terms of the scoring function contribute most."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f14")
    budget_pct = budgets

    for ax, metric in zip(axes, ["reliability", "recall"]):
        _ax_style(ax)
        for name, color in ABLATION_COLORS.items():
            means = [ablation_results[name][b]["mean"][metric] for b in budgets]
            stds = [ablation_results[name][b]["std"][metric] for b in budgets]
            ax.errorbar(
                budget_pct,
                means,
                yerr=stds,
                label=name,
                color=color,
                linewidth=2.2 if name == "Full composite" else 1.2,
                marker="o",
                markersize=4,
                capsize=3,
                elinewidth=0.8,
            )
        ax.set_xlabel("# checks", color="#aaaaaa", fontsize=9)
        ax.set_ylabel(metric.capitalize(), color="#aaaaaa", fontsize=9)
        ax.set_title(
            f"Ablation — {metric.capitalize()}", color="#dddddd", fontsize=10, pad=8
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        framealpha=0.15,
        labelcolor="#dddddd",
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(
        "Score Function Ablation Study",
        color="#eeeeee",
        fontsize=12,
        y=1.02,
        fontweight="bold",
    )

    path = os.path.join(OUT_DIR, "ablation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def plot_tree_snapshot(
    tree: ClaimTree,
    verify_set: set[int] = None,
    title="Claim Tree",
    filename="tree_snapshot.png",
):
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0f0f14")
    ax.set_facecolor("#0f0f14")
    ax.axis("off")

    pos = {}
    by_depth = defaultdict(list)
    for nid, node in tree.nodes.items():
        by_depth[node.depth].append(nid)
    for depth, nids in by_depth.items():
        w = len(nids)
        for i, nid in enumerate(nids):
            pos[nid] = ((i + 1) / (w + 1), 1 - depth / (tree.max_depth_actual + 1))

    for nid, node in tree.nodes.items():
        if node.parent_id is not None:
            x0, y0 = pos[node.parent_id]
            x1, y1 = pos[nid]
            ax.plot([x0, x1], [y0, y1], color="#333344", linewidth=0.7, zorder=1)

    verify_set = verify_set or set()
    for nid, node in tree.nodes.items():
        x, y = pos[nid]
        color = "#2ecc71" if node.truth == 1 else "#e74c3c"
        edge = "#ffffff" if nid in verify_set else color
        lw = 2.5 if nid in verify_set else 0.5
        ax.scatter(x, y, s=80, c=color, edgecolors=edge, linewidths=lw, zorder=3)
        ax.text(
            x,
            y + 0.025,
            f"{node.confidence:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="#aaaaaa",
        )

    legend = [
        mpatches.Patch(color="#2ecc71", label="True claim"),
        mpatches.Patch(color="#e74c3c", label="False claim"),
        mpatches.Patch(
            facecolor="none", edgecolor="white", linewidth=2, label="Verified"
        ),
    ]
    ax.legend(
        handles=legend,
        loc="upper right",
        framealpha=0.2,
        labelcolor="#dddddd",
        fontsize=8,
    )
    ax.set_title(title, color="#eeeeee", fontsize=11)

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def plot_all_strategy_trees(tree: ClaimTree, budget: int):
    """Save one PNG per strategy."""
    import matplotlib.patches as mpatches

    # Fixed layout — same positions for all plots
    pos = {}
    by_depth = defaultdict(list)
    for nid, node in tree.nodes.items():
        by_depth[node.depth].append(nid)
    for depth, nids in by_depth.items():
        w = len(nids)
        for i, nid in enumerate(nids):
            pos[nid] = ((i + 1) / (w + 1), 1 - depth / (tree.max_depth_actual + 1))

    for name, fn in STRATEGIES.items():
        verify_set = fn(tree, budget)
        m = evaluate(tree, verify_set)

        fig, ax = plt.subplots(figsize=(13, 7))
        fig.patch.set_facecolor("#0f0f14")
        ax.set_facecolor("#0f0f14")
        ax.axis("off")

        # Edges
        for nid, node in tree.nodes.items():
            if node.parent_id is not None:
                x0, y0 = pos[node.parent_id]
                x1, y1 = pos[nid]
                ax.plot([x0, x1], [y0, y1], color="#333344", linewidth=0.7, zorder=1)

        # Nodes
        for nid, node in tree.nodes.items():
            x, y = pos[nid]
            color = "#2ecc71" if node.truth == 1 else "#e74c3c"
            edge = "#ffffff" if nid in verify_set else color
            lw = 2.5 if nid in verify_set else 0.4
            ax.scatter(x, y, s=80, c=color, edgecolors=edge, linewidths=lw, zorder=3)
            ax.text(
                x,
                y + 0.025,
                f"{node.confidence:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="#888888",
            )

        ax.set_title(
            f"{name}  —  budget={budget} checks\n"
            f"Recall={m['recall']:.2f}   Reliability={m['reliability']:.3f}   "
            f"Precision={m['precision']:.2f}",
            color="#eeeeee",
            fontsize=11,
            pad=10,
        )

        legend = [
            mpatches.Patch(color="#2ecc71", label="True claim"),
            mpatches.Patch(color="#e74c3c", label="False claim"),
            mpatches.Patch(
                facecolor="none", edgecolor="white", linewidth=2, label="Verified"
            ),
        ]
        ax.legend(
            handles=legend,
            loc="upper right",
            framealpha=0.2,
            labelcolor="#dddddd",
            fontsize=9,
        )

        filename = (
            "tree_"
            + name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            + ".png"
        )
        path = os.path.join(OUT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"Saved {path}")


def plot_heuristic_gap(results: dict, budgets: list[int], total_nodes: int):
    """
    For each strategy, plot the gap vs DP Optimal on reliability.
    gap(s, b) = reliability(DP, b) - reliability(s, b)
    Smaller gap = closer to optimal.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f14")
    _ax_style(ax)

    budget_pct = budgets
    dp_means = [results["DP Optimal"][b]["mean"]["reliability"] for b in budgets]

    gap_colors = {k: v for k, v in COLORS.items() if k != "DP Optimal"}

    for name, color in gap_colors.items():
        gaps = [
            results["DP Optimal"][b]["mean"]["reliability"]
            - results[name][b]["mean"]["reliability"]
            for b in budgets
        ]
        stds = [results[name][b]["std"]["reliability"] for b in budgets]
        ax.errorbar(
            budget_pct,
            gaps,
            yerr=stds,
            label=name,
            color=color,
            linewidth=2.0 if name == "Dependency-aware" else 1.2,
            marker="o",
            markersize=4,
            capsize=3,
            elinewidth=0.8,
        )

    ax.axhline(
        0,
        color="#f1c40f",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label="DP Optimal (gap = 0)",
    )
    ax.set_xlabel("Number of checks (budget)", color="#aaaaaa", fontsize=10)
    ax.set_ylabel(
        "Reliability gap vs DP Optimal  (↓ better)", color="#aaaaaa", fontsize=10
    )
    ax.set_title(
        "Heuristic Gap — How Far Each Strategy Is From DP Optimal",
        color="#dddddd",
        fontsize=11,
        pad=10,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, framealpha=0.15, labelcolor="#dddddd", fontsize=9)

    path = os.path.join(OUT_DIR, "heuristic_gap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def plot_adaptive_sigma_sweep(results: dict, sigmas: list[float], filename: str = "adaptive_sigma_sweep.png"):
    """
    Plot mean ± std of evaluate_adaptive() metrics across a sigma sweep
    (output of run_adaptive_experiment()).

    Panels: contamination rate, reliability, nodes generated, quarantined
    nodes, and verification budget used, all against sigma. Look for the
    sigma that keeps contamination down without over-quarantining, i.e.
    burning the whole verify budget on nodes that turn out to be true.
    """
    panels = [
        ("contamination_rate", "Contamination Rate  (↓ better)", "#e74c3c"),
        ("reliability", "Reliability  (↑ better)", "#2ecc71"),
        ("total_nodes", "Total Nodes Generated", "#4e9af1"),
        ("quarantined", "Quarantined Nodes", "#f39c12"),
        ("verify_budget_used", "Verification Budget Used", "#9b59b6"),
    ]

    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor("#0f0f14")
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3)

    for idx, (metric, label, color) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        _ax_style(ax)

        means = [results[s]["mean"][metric] for s in sigmas]
        stds = [results[s]["std"][metric] for s in sigmas]

        ax.errorbar(
            sigmas,
            means,
            yerr=stds,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            capsize=3,
            elinewidth=0.8,
        )
        ax.set_xlabel("sigma", color="#aaaaaa", fontsize=9)
        ax.set_title(label, color="#dddddd", fontsize=10, pad=8)

    fig.suptitle(
        "Adaptive Online Verification — Sigma Sweep",
        color="#dddddd",
        fontsize=13,
        y=1.0,
    )

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def plot_fixed_vs_adaptive(
    results: dict,
    fixed_sigmas: list[float],
    filename: str = "fixed_vs_adaptive.png",
):
    """
    Bar chart comparing fixed sigma values against the adaptive policy
    on reliability and contamination_rate.

    results: output of tree_model.run_adaptive_sigma_experiment(), keyed by
    each fixed sigma (float) plus the string "adaptive".
    """
    names = [str(s) for s in fixed_sigmas] + ["adaptive"]
    reliability = [results[s]["mean"]["reliability"] for s in fixed_sigmas] + [
        results["adaptive"]["mean"]["reliability"]
    ]
    contamination = [results[s]["mean"]["contamination_rate"] for s in fixed_sigmas] + [
        results["adaptive"]["mean"]["contamination_rate"]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f14")

    panels = [
        (axes[0], reliability, "Reliability  (↑ better)"),
        (axes[1], contamination, "Contamination Rate  (↓ better)"),
    ]

    for ax, values, label in panels:
        _ax_style(ax)
        colors_bar = ["#4e9af1"] * len(fixed_sigmas) + ["#f1c40f"]
        bars = ax.bar(names, values, color=colors_bar, alpha=0.9)
        ax.set_title(label, color="#dddddd", fontsize=11, pad=8)
        ax.tick_params(axis="x", labelsize=9)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", color="#dddddd", fontsize=8,
            )

    fig.suptitle(
        "Fixed Sigma vs Adaptive Sigma Policy",
        color="#dddddd", fontsize=13, y=1.02,
    )

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def plot_warmup_checkpoint_sweep(
    results: dict,
    sigma_lows: list[float],
    sigma_highs: list[float],
    filename: str = "warmup_checkpoint_sweep.png",
):
    """
    Heatmaps of evaluate_adaptive() metrics over the (sigma_low, sigma_high)
    grid (output of run_warmup_checkpoint_experiment()). Pairs with
    sigma_high <= sigma_low are not simulated and are left blank.

    Panels: reliability, contamination rate, quarantined nodes, and
    verification budget used.
    """
    panels = [
        ("reliability", "Reliability  (↑ better)", "viridis"),
        ("contamination_rate", "Contamination Rate  (↓ better)", "magma"),
        ("quarantined", "Quarantined Nodes", "magma"),
        ("verify_budget_used", "Verification Budget Used", "viridis"),
    ]

    n_low, n_high = len(sigma_lows), len(sigma_highs)

    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor("#0f0f14")
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    for idx, (metric, label, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        _ax_style(ax)

        grid = np.full((n_low, n_high), np.nan)
        for i, lo in enumerate(sigma_lows):
            for j, hi in enumerate(sigma_highs):
                if (lo, hi) in results:
                    grid[i, j] = results[(lo, hi)]["mean"][metric]

        im = ax.imshow(grid, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n_high))
        ax.set_xticklabels(sigma_highs)
        ax.set_yticks(range(n_low))
        ax.set_yticklabels(sigma_lows)
        ax.set_xlabel("sigma_high", color="#aaaaaa", fontsize=9)
        ax.set_ylabel("sigma_low", color="#aaaaaa", fontsize=9)
        ax.set_title(label, color="#dddddd", fontsize=10, pad=8)

        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
        span = vmax - vmin or 1.0
        for i in range(n_low):
            for j in range(n_high):
                if not np.isnan(grid[i, j]):
                    frac = (grid[i, j] - vmin) / span
                    txt_color = "#0f0f14" if frac > 0.5 else "#eeeeee"
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            color=txt_color, fontsize=8, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#aaaaaa", labelsize=8)

    fig.suptitle(
        "Warm-up + Subtree-Checkpoint Policy — (sigma_low, sigma_high) Sweep",
        color="#dddddd",
        fontsize=13,
        y=1.0,
    )

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def plot_policy_comparison(
    comparison: dict,
    filename: str = "policy_comparison.png",
):
    """
    Grouped bar chart comparing policies (offline strategies, simple
    adaptive, warm-up + checkpoint) on reliability and contamination rate.

    comparison: {policy_name: {"reliability": x, "contamination_rate": y}}
    """
    names = list(comparison.keys())
    reliability = [comparison[n]["reliability"] for n in names]
    contamination = [comparison[n]["contamination_rate"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f14")

    panels = [
        (axes[0], reliability, "Reliability  (↑ better)", "#2ecc71"),
        (axes[1], contamination, "Contamination Rate  (↓ better)", "#e74c3c"),
    ]

    for ax, values, label, color in panels:
        _ax_style(ax)
        bars = ax.bar(names, values, color=color, alpha=0.85)
        ax.set_title(label, color="#dddddd", fontsize=11, pad=8)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", color="#dddddd", fontsize=8)

    fig.suptitle(
        "Policy Comparison: Offline vs. Adaptive vs. Warm-up + Checkpoint",
        color="#dddddd",
        fontsize=13,
        y=1.02,
    )

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")

def plot_calibration_comparison(
    results: dict,
    budgets: list,
    total_nodes: int = 80,
    filename: str = "calibration_comparison.png",
):
    """
    4-panel figure comparing default vs overconfident miscalibration regimes.
      TL: reliability by regime + strategy
      TR: recall by regime + strategy
      BL: bar chart of oracle-DP minus belief-DP gap per budget, by regime
      BR: oracle-DP vs belief-DP vs Uncertainty on reliability, both regimes
    """
    budget_pcts = [b / total_nodes * 100 for b in budgets]
    strats_to_plot = [
        "Random", "Uncertainty", "Dependency-aware", "DP Optimal", "DP Oracle"
    ]
    regime_ls = {"default": "-", "overconfident": "--"}
    regime_colors_map = {"default": "#4C72B0", "overconfident": "#DD8452"}

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#0f0f14")
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30)

    # Top row: reliability and recall
    for panel_idx, metric in enumerate(("reliability", "recall")):
        ax = fig.add_subplot(gs[0, panel_idx])
        _ax_style(ax)
        for regime in ("default", "overconfident"):
            for strat in strats_to_plot:
                if strat not in results[regime]:
                    continue
                color = COLORS.get(strat, "#888888")
                ls = regime_ls[regime]
                means = [results[regime][strat][b]["mean"][metric] for b in budgets]
                stds  = [results[regime][strat][b]["std"][metric]  for b in budgets]
                label = f"{strat} ({'def' if regime == 'default' else 'over'})"
                ax.plot(budget_pcts, means, ls=ls, marker="o", markersize=4,
                        color=color, label=label, lw=1.8)
                ax.fill_between(budget_pcts,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                alpha=0.08, color=color)
        ax.set_xlabel("Budget (% of nodes)", color="#aaaaaa", fontsize=9)
        ax.set_ylabel(metric.capitalize(), color="#aaaaaa", fontsize=9)
        ax.set_title(
            f"{metric.capitalize()} by regime\n(solid=default, dashed=overconfident)",
            color="#dddddd", fontsize=10, pad=8
        )
        ax.set_ylim(0, 1.05)
        if panel_idx == 1:
            ax.legend(fontsize=6, ncol=2, loc="lower right",
                      framealpha=0.15, labelcolor="#dddddd")

    # Bottom-left: oracle-DP minus belief-DP gap, grouped bar
    ax_gap = fig.add_subplot(gs[1, 0])
    _ax_style(ax_gap)
    x = np.arange(len(budgets))
    width = 0.35
    for i, regime in enumerate(("default", "overconfident")):
        gaps = [
            results[regime]["DP Oracle"][b]["mean"]["reliability"]
            - results[regime]["DP Optimal"][b]["mean"]["reliability"]
            for b in budgets
        ]
        ax_gap.bar(x + i * width, gaps, width,
                   label=regime, color=regime_colors_map[regime], alpha=0.85)
    ax_gap.set_xticks(x + width / 2)
    ax_gap.set_xticklabels([str(b) for b in budgets], fontsize=8, color="#aaaaaa")
    ax_gap.set_xlabel("Budget (nodes)", color="#aaaaaa", fontsize=9)
    ax_gap.set_ylabel("DP Oracle − DP Belief reliability", color="#aaaaaa", fontsize=9)
    ax_gap.set_title("Miscalibration cost\n(how much perfect beliefs would help)",
                     color="#dddddd", fontsize=10, pad=8)
    ax_gap.axhline(0, color="#aaaaaa", lw=0.8)
    ax_gap.legend(framealpha=0.15, labelcolor="#dddddd", fontsize=9)

    # ── Bottom-right: oracle-DP / belief-DP / Uncertainty head-to-head ──
    ax_ceil = fig.add_subplot(gs[1, 1])
    _ax_style(ax_ceil)
    line_styles = {"DP Oracle": "-", "DP Optimal": "--", "Uncertainty": ":"}
    for regime in ("default", "overconfident"):
        for strat, ls in line_styles.items():
            if strat not in results[regime]:
                continue
            means = [results[regime][strat][b]["mean"]["reliability"] for b in budgets]
            color = regime_colors_map[regime]
            lw = 2.2 if strat == "DP Oracle" else 1.4
            ax_ceil.plot(budget_pcts, means, ls=ls, color=color,
                         label=f"{strat} ({regime[:3]})", lw=lw)
    ax_ceil.set_xlabel("Budget (% of nodes)", color="#aaaaaa", fontsize=9)
    ax_ceil.set_ylabel("Reliability", color="#aaaaaa", fontsize=9)
    ax_ceil.set_title(
        "Oracle vs Belief-DP vs Uncertainty\nacross both regimes",
        color="#dddddd", fontsize=10, pad=8
    )
    ax_ceil.set_ylim(0, 1.05)
    ax_ceil.legend(fontsize=8, loc="lower right", framealpha=0.15, labelcolor="#dddddd")

    fig.suptitle(
        "Oracle-DP vs Belief-DP — Default vs Overconfident Miscalibration",
        color="#eeeeee", fontsize=13, fontweight="bold", y=0.99,
    )

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")