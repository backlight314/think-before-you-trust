"""Regenerate the poster figures from the numbers reported in the paper.

    python poster/make_figures.py   ->  poster/figures/fig_*.pdf (+ .png)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# Poster palette (Columbia brand colours used in the .tex)
GRAY = "#6E7175"        # random baseline (reference, always dashed)
CONF = "#C75000"        # Accent1 — confidence / self-report
STRUCT = "#1D4F91"      # PrimaryBlue — structure
INK = "#222222"
MUTED = "#53565A"
GRID = "#D0D0CE"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig A — real LLMs, 10% budget (paper Table 5), ordered by error rate
# ---------------------------------------------------------------------------
models = ["Gemini", "Claude", "DeepSeek", "Qwen-80b", "Llama-70b",
          "GPT-120b", "GPT-20b", "Llama-8b"]
err = [7, 23, 44, 49, 54, 56, 74, 80]
rand = [0.944, 0.822, 0.682, 0.639, 0.705, 0.657, 0.576, 0.538]
unc = [0.960, 0.850, 0.660, 0.631, 0.661, 0.585, 0.445, 0.572]
dep = [0.947, 0.865, 0.758, 0.752, 0.777, 0.826, 0.752, 0.782]

fig, ax = plt.subplots(figsize=(11, 5.6))
x = range(len(models))
ax.plot(x, rand, color=GRAY, lw=2, ls="--", marker="o", ms=7, label="Random")
ax.plot(x, unc, color=CONF, lw=2.5, marker="s", ms=9,
        label="Confidence (uncertainty-first)")
ax.plot(x, dep, color=STRUCT, lw=2.5, marker="o", ms=9,
        label="Structure (dependency-aware)")
ax.set_xticks(list(x))
ax.set_xticklabels([f"{m}\n{e}% err" for m, e in zip(models, err)], fontsize=12)
ax.set_ylabel("Reliability at 10% budget")
ax.set_ylim(0.4, 1.0)
ax.annotate("confidence below random\non 5 of 8 models", xy=(6, 0.445),
            xytext=(3.2, 0.50), color=CONF, fontsize=13,
            arrowprops=dict(arrowstyle="->", color=CONF, lw=1.5))
ax.legend(loc="lower left", frameon=False, fontsize=13)
ax.set_title("Real LLM reasoning graphs (8 models × 45 graphs)",
             loc="left", fontsize=16, color=INK)
save(fig, "fig_real_llm")

# ---------------------------------------------------------------------------
# Fig B — simulation: cheap structure vs expensive optimizer
# (overconfident regime on synthetic DAGs; paper §4.1)
# ---------------------------------------------------------------------------
labels = ["Random", "Betweenness\n(cheap, structure only)",
          "Greedy-MC\n(Monte-Carlo optimizer)"]
vals = [0.880, 0.940, 0.937]
cols = [GRAY, STRUCT, MUTED]

fig, ax = plt.subplots(figsize=(11, 3.0))
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
y = [2, 1, 0]
ax.hlines(y, 0.80, vals, color=GRID, lw=2)
ax.scatter(vals, y, s=220, color=cols, zorder=3)
for v, yy in zip(vals, y):
    ax.text(v + 0.004, yy, f"{v:.3f}", va="center", fontsize=15,
            color=INK)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlim(0.80, 1.0)
ax.set_xlabel("Reliability (simulated DAGs, overconfident regime)")
ax.set_title("Simulation: tie with the optimizer (p = 0.50)", loc="left",
             fontsize=16, color=INK)
save(fig, "fig_simulation")

# ---------------------------------------------------------------------------
# Fig C — confidence-corruption stress test (paper Fig. 6)
# ---------------------------------------------------------------------------
regimes = ["As reported", "Overconfident", "Scrambled", "Confident on\nFALSE claims"]
conf_c = [0.94, 0.93, 0.85, 0.80]
betw_c = [0.90, 0.90, 0.90, 0.90]
rand_c = [0.85] * 4

fig, ax = plt.subplots(figsize=(11, 5))
xr = range(len(regimes))
ax.plot(xr, rand_c, color=GRAY, lw=2, ls="--", label="Random")
ax.plot(xr, conf_c, color=CONF, lw=2.5, marker="s", ms=9,
        label="Confidence-based")
ax.plot(xr, betw_c, color=STRUCT, lw=2.5, marker="o", ms=9,
        label="Structure (betweenness) — never reads confidence")
for xi, v in zip(xr, conf_c):
    ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", color=INK, fontsize=13)
ax.set_xticks(list(xr))
ax.set_xticklabels(regimes)
ax.set_ylim(0.75, 0.98)
ax.set_ylabel("Reliability at 20% budget")
ax.set_xlabel("Corruption applied to confidence only  →")
ax.legend(loc="lower left", frameon=False, fontsize=13)
ax.set_title("Stress test: corrupt only the confidence", loc="left",
             fontsize=16, color=INK)
save(fig, "fig_robustness")

print("wrote", sorted(os.listdir(OUT)))
