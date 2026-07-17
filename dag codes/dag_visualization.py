"""
DAG Visualisation

Mirrors tree_results.py from the tree model exactly.
DAG-specific differences (cross edges shown in orange, 
descendants_ids instead of subtree_ids) are handled internally.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict
import os

from dag_model import ClaimDAG, STRATEGIES, ABLATIONS, evaluate

OUT_DIR = "results_dag"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "Random":            "#888888",
    "Level-sampling":    "#4e9af1",
    "Recent (deep)":     "#f4a261",
    "Uncertainty":       "#e76f51",
    "Dependency-aware":  "#2ecc71",
    "Greedy MC":         "#f1c40f",
}

ABLATION_COLORS = {
    "Conf only":        "#e74c3c",
    "Descendants only": "#3498db",
    "Depth only":       "#9b59b6",
    "Conf + Desc":      "#f39c12",
    "Full composite":   "#2ecc71",
}


def _ax_style(ax):
    ax.set_facecolor("#16161e")
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.grid(alpha=0.15, color="#444455")

def _dag_positions(dag: ClaimDAG) -> dict[int, tuple]:
    pos = {}
    by_depth = defaultdict(list)
    for nid, node in dag.nodes.items():
        by_depth[node.depth].append(nid)
    for depth, nids in by_depth.items():
        w = len(nids)
        for i, nid in enumerate(nids):
            pos[nid] = ((i + 1) / (w + 1), 1 - depth / (dag.max_depth_actual + 1))
    return pos


def _draw_dag(ax, dag: ClaimDAG, pos: dict, verify_set: set[int]):
    """Draw edges and nodes onto ax. Cross edges rendered in orange."""
    # Edges
    for nid, node in dag.nodes.items():
        for pidx, pid in enumerate(node.parents):
            x0, y0 = pos[pid]
            x1, y1 = pos[nid]
            color = "#e67e22" if pidx > 0 else "#333344"
            lw    = 1.2       if pidx > 0 else 0.7
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, zorder=1, alpha=0.8)

    # Nodes
    for nid, node in dag.nodes.items():
        x, y  = pos[nid]
        color = "#2ecc71" if node.truth == 1 else "#e74c3c"
        edge  = "#ffffff"  if nid in verify_set else color
        lw    = 2.5        if nid in verify_set else 0.4
        ax.scatter(x, y, s=80, c=color, edgecolors=edge, linewidths=lw, zorder=3)
        ax.text(x, y + 0.025, f"{node.confidence:.2f}",
                ha="center", va="bottom", fontsize=5.5, color="#888888")


def _dag_legend():
    return [
        mpatches.Patch(color="#2ecc71", label="True claim"),
        mpatches.Patch(color="#e74c3c", label="False claim"),
        mpatches.Patch(facecolor="none", edgecolor="white", linewidth=2, label="Verified"),
        mpatches.Patch(color="#e67e22", label="Cross edge (DAG)"),
    ]

#main plots

def plot_results(
    results: dict,
    budgets: list[int],
    total_nodes: int,
    oracle_results: dict = None,
    filename: str = "results.png",
):
    """4-panel comparison plot with error bars and oracle baseline"""
    metrics_to_plot = [
        ("reliability",       "Reliability  (↑ better)"),
        ("recall",            "Recall of False Nodes  (↑ better)"),
        ("precision",         "Precision  (↑ better)"),
        ("cascade_prevented", "Cascade Nodes Prevented  (↑ better)"),
    ]
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0f0f14")
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)

    for idx, (metric, label) in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        _ax_style(ax)

        if oracle_results:
            oracle_means = [oracle_results["Oracle"][b]["mean"][metric] for b in budgets]
            ax.plot(budgets, oracle_means, color="#ffffff", linewidth=1.0,
                    linestyle="--", alpha=0.4, label="Oracle (100%)")

        for name, color in COLORS.items():
            means = [results[name][b]["mean"][metric] for b in budgets]
            stds  = [results[name][b]["std"][metric]  for b in budgets]
            ax.errorbar(
                budgets, means, yerr=stds, label=name, color=color,
                linewidth=2.0 if name in ("Dependency-aware", "Greedy MC") else 1.3,
                marker="o", markersize=4, capsize=3, elinewidth=0.8,
                zorder=5 if name in ("Dependency-aware", "Greedy MC") else 3,
            )

        ax.set_xlabel("Number of checks (budget)", color="#aaaaaa", fontsize=9)
        ax.set_ylabel(label, color="#aaaaaa", fontsize=9)
        ax.set_title(label, color="#dddddd", fontsize=10, pad=8)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, framealpha=0.15,
               labelcolor="#dddddd", fontsize=8, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        f"DAG Verification Strategy Comparison — 50-Trial Average ± 1 SD  (extra_edge_prob=0.15)",
        color="#eeeeee", fontsize=13, y=0.97, fontweight="bold",
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

    for col, (regime_label, results) in enumerate(sensitivity.items()):
        ax = fig.add_subplot(gs[col])
        _ax_style(ax)
        for name, color in COLORS.items():
            means = [results[name][b]["mean"]["reliability"] for b in budgets]
            stds  = [results[name][b]["std"]["reliability"]  for b in budgets]
            ax.errorbar(
                budgets, means, yerr=stds, label=name, color=color,
                linewidth=2.0 if name in ("Dependency-aware", "Greedy MC") else 1.2,
                marker="o", markersize=3, capsize=2, elinewidth=0.7,
            )
        ax.set_title(regime_label, color="#dddddd", fontsize=10)
        ax.set_xlabel("# checks", color="#aaaaaa", fontsize=9)
        ax.set_ylabel("Reliability" if col == 0 else "", color="#aaaaaa", fontsize=9)
        ax.set_ylim(0.6, 1.02)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, framealpha=0.15,
               labelcolor="#dddddd", fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "DAG Sensitivity Analysis, Reliability Across Error Regimes",
        color="#eeeeee", fontsize=12, y=1.02, fontweight="bold",
    )
    path = os.path.join(OUT_DIR, "sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def plot_ablation(ablation_results: dict, budgets: list[int], total_nodes: int):
    """Show which terms of the scoring function contribute most."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f14")

    for ax, metric in zip(axes, ["reliability", "recall"]):
        _ax_style(ax)
        for name, color in ABLATION_COLORS.items():
            means = [ablation_results[name][b]["mean"][metric] for b in budgets]
            stds  = [ablation_results[name][b]["std"][metric]  for b in budgets]
            ax.errorbar(
                budgets, means, yerr=stds, label=name, color=color,
                linewidth=2.2 if name == "Full composite" else 1.2,
                marker="o", markersize=4, capsize=3, elinewidth=0.8,
            )
        ax.set_xlabel("# checks", color="#aaaaaa", fontsize=9)
        ax.set_ylabel(metric.capitalize(), color="#aaaaaa", fontsize=9)
        ax.set_title(f"Ablation — {metric.capitalize()}", color="#dddddd", fontsize=10, pad=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, framealpha=0.15,
               labelcolor="#dddddd", fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Score Function Ablation Study — DAG",
                 color="#eeeeee", fontsize=12, y=1.02, fontweight="bold")

    path = os.path.join(OUT_DIR, "ablation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def plot_heuristic_gap(results: dict, budgets: list[int], total_nodes: int):
    """
    For each strategy, plot reliability gap vs Greedy MC.
    gap(s, b) = reliability(Greedy MC, b) - reliability(s, b)
    Smaller gap means closer to the MC upper bound.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f14")
    _ax_style(ax)

    gap_colors = {k: v for k, v in COLORS.items() if k != "Greedy MC"}
    for name, color in gap_colors.items():
        gaps = [
            results["Greedy MC"][b]["mean"]["reliability"]
            - results[name][b]["mean"]["reliability"]
            for b in budgets
        ]
        stds = [results[name][b]["std"]["reliability"] for b in budgets]
        ax.errorbar(
            budgets, gaps, yerr=stds, label=name, color=color,
            linewidth=2.0 if name == "Dependency-aware" else 1.2,
            marker="o", markersize=4, capsize=3, elinewidth=0.8,
        )

    ax.axhline(0, color="#f1c40f", linewidth=1.5, linestyle="--",
               alpha=0.7, label="Greedy MC (gap = 0)")
    ax.set_xlabel("Number of checks (budget)", color="#aaaaaa", fontsize=10)
    ax.set_ylabel("Reliability gap vs Greedy MC  (↓ better)", color="#aaaaaa", fontsize=10)
    ax.set_title("Heuristic Gap — How Far Each Strategy Is From Greedy MC",
                 color="#dddddd", fontsize=11, pad=10)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, framealpha=0.15, labelcolor="#dddddd", fontsize=9)

    path = os.path.join(OUT_DIR, "heuristic_gap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def plot_adaptive_sigma_sweep(results: dict, sigmas: list[float], filename: str = "adaptive_sigma_sweep.png"):
    """
    Plot mean ± std of evaluate_adaptive() metrics across a sigma sweep
    (output of dag_model.run_adaptive_experiment()).

    Panels: contamination rate, reliability, nodes generated,
    quarantined nodes, and verification budget used — all vs sigma.
    Goal: find the sigma that minimizes contamination without
    over-quarantining (i.e. without burning the whole verify budget
    on nodes that turn out to be true).
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
            sigmas, means, yerr=stds, color=color,
            linewidth=2.0, marker="o", markersize=4, capsize=3, elinewidth=0.8,
        )
        ax.set_xlabel("sigma", color="#aaaaaa", fontsize=9)
        ax.set_title(label, color="#dddddd", fontsize=10, pad=8)

    fig.suptitle(
        "Adaptive Online Verification (DAG) — Sigma Sweep",
        color="#dddddd", fontsize=13, y=1.0,
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

    results: output of dag_model.run_adaptive_sigma_experiment()
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


def plot_dag_snapshot(
    dag: ClaimDAG,
    verify_set: set[int] = None,
    title: str = "Claim DAG",
    filename: str = "dag_snapshot.png",
):
    """White snapshot rendered identically to calibration_harness.draw()."""
    verify_set = verify_set or set()
    parents = {nid: node.parents for nid, node in dag.nodes.items()}
    truth   = {nid: node.truth for nid, node in dag.nodes.items()}
    conf    = {nid: node.confidence for nid, node in dag.nodes.items()}

    kids = {i: [] for i in parents}
    for i in parents:
        for p in parents[i]:
            if p in kids: kids[p].append(i)

    # depth = 1 + max(parent depth), like draw()
    depth = {}
    def _d(v):
        if v not in depth:
            ps = [p for p in parents[v] if p in parents]
            depth[v] = 0 if not ps else 1 + max(_d(p) for p in ps)
        return depth[v]
    for i in parents: _d(i)

    # barycenter ordering to reduce edge crossings (identical to draw())
    layers = {}
    for v, dp in depth.items(): layers.setdefault(dp, []).append(v)
    order = {dp: sorted(layers[dp]) for dp in layers}
    for _ in range(8):
        for dp in sorted(layers)[1:]:
            ab = {v: i for i, v in enumerate(order[dp - 1])}
            order[dp].sort(key=lambda v: (sum(ab[p] for p in parents[v] if p in ab) /
                                          max(1, len([p for p in parents[v] if p in ab]))))
        for dp in sorted(layers, reverse=True)[:-1]:
            bl = {v: i for i, v in enumerate(order[dp + 1])} if dp + 1 in order else {}
            if bl:
                order[dp].sort(key=lambda v: (sum(bl[c] for c in kids[v] if c in bl) /
                                              max(1, len([c for c in kids[v] if c in bl]))))
    pos = {}
    for dp, row in order.items():
        for i, v in enumerate(row): pos[v] = ((i + 0.5) / len(row), -dp)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    # edges: cross-edges amber dashed, primary light gray (draw() palette)
    for v in parents:
        for k, p in enumerate(parents[v]):
            if p not in pos: continue
            x1, y1 = pos[p]; x2, y2 = pos[v]
            if k > 0:
                ax.plot([x1, x2], [y1, y2], color="#EF9F27", lw=1.4,
                        ls=(0, (5, 4)), zorder=1)
            else:
                ax.plot([x1, x2], [y1, y2], color="#bdbcb5", lw=0.9,
                        alpha=0.9, zorder=1)

    # nodes: teal/red/gray, white ring (thicker if verified), id inside, conf above
    for v, (x, y) in pos.items():
        t = truth[v]
        col = "#888780" if t is None else ("#1D9E75" if t == 1 else "#E24B4A")
        lw = 2.5 if v in verify_set else 1.1
        ax.scatter([x], [y], s=460, color=col, edgecolors="white",
                   linewidths=lw, zorder=3)
        ax.text(x, y, str(v), ha="center", va="center",
                color="white", fontsize=7, zorder=4)
        try:
            cf = f"{float(conf[v]):.2f}"
        except (TypeError, ValueError):
            cf = ""
        if cf:
            ax.text(x, y + 0.28, cf, ha="center", va="bottom",
                    color="#444444", fontsize=6, zorder=4)

    ax.set_title(title + "   (teal=true, red=false, gray=uncheckable, "
                 "amber dashed=cross-edge)", color="#222222", fontsize=11)

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")

def plot_all_strategy_dags(dag: ClaimDAG, budget: int):
    """One subplot per strategy on the same DAG"""
    n_strats = len(STRATEGIES)
    ncols, nrows = 3, (n_strats + 2) // 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5))
    fig.patch.set_facecolor("#0f0f14")
    axes_flat = axes.flatten() if nrows > 1 else list(axes)

    pos = _dag_positions(dag)

    for ax, (name, fn) in zip(axes_flat, STRATEGIES.items()):
        verify_set = fn(dag, budget)
        m = evaluate(dag, verify_set)

        ax.set_facecolor("#16161e")
        ax.axis("off")
        _draw_dag(ax, dag, pos, verify_set)

        ax.set_title(
            f"{name}\nRecall={m['recall']:.2f}   "
            f"Reliability={m['reliability']:.3f}   "
            f"({len(verify_set)} checks)",
            color="#eeeeee", fontsize=9, pad=6,
        )
        ax.legend(handles=_dag_legend(), loc="upper right",
                  framealpha=0.2, labelcolor="#dddddd", fontsize=7)

    for ax in axes_flat[n_strats:]:
        ax.set_visible(False)

    fig.suptitle(
        f"All Strategies on Same DAG — budget={budget} checks",
        color="#eeeeee", fontsize=13, y=1.01, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "all_strategy_dags.png")
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
    grid (output of dag_model.run_warmup_checkpoint_experiment()).

    Pairs with sigma_high <= sigma_low are not simulated and are left blank.
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
        "Warm-up + Subtree-Checkpoint Policy (DAG) — (sigma_low, sigma_high) Sweep",
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
        "Policy Comparison (DAG): Offline vs. Adaptive vs. Warm-up + Checkpoint",
        color="#dddddd",
        fontsize=13,
        y=1.02,
    )

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")