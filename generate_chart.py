"""Generate AUC-ROC progression chart for the README."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# All experiments in order
experiments = [
    ("Baseline\nLogReg 10feat", 0.647, "keep"),
    ("All 85\nfeatures", 0.742, "keep"),
    ("RandomForest\n300 trees", 0.753, "keep"),
    ("XGBoost", 0.696, "discard"),
    ("LightGBM", 0.691, "discard"),
    ("Feature sel.\ntop-25", 0.744, "discard"),
    ("Voting\nRF+GB+LR", 0.754, "keep"),
    ("Stacking\nRF+GB", 0.753, "discard"),
    ("GB sample\nweights", 0.733, "discard"),
    ("5-model\nensemble", 0.751, "discard"),
    ("SMOTE +\nensemble", 0.737, "discard"),
    ("Tuned\nRF1000+GB500", 0.767, "keep"),
    ("RF1500+GB800\nbalanced_sub", 0.774, "keep"),
    ("RF2000+GB1200\n(more compute)", 0.773, "discard"),
]

labels = [e[0] for e in experiments]
aucs = [e[1] for e in experiments]
statuses = [e[2] for e in experiments]

# Best AUC-ROC running line (only keeps)
best_line = []
current_best = 0
for auc, status in zip(aucs, statuses):
    if status == "keep" and auc > current_best:
        current_best = auc
    best_line.append(current_best)

# Colors
colors = ["#2ecc71" if s == "keep" else "#e74c3c" for s in statuses]
edge_colors = ["#1a9850" if s == "keep" else "#c0392b" for s in statuses]

fig, ax = plt.subplots(figsize=(16, 7))

# Background
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# Best line (stepped)
ax.step(range(len(best_line)), best_line, where="mid", color="#3498db",
        linewidth=2.5, alpha=0.6, linestyle="--", label="Best AUC-ROC", zorder=1)
ax.fill_between(range(len(best_line)), 0.6, best_line, step="mid",
                alpha=0.08, color="#3498db")

# Bars
bars = ax.bar(range(len(aucs)), aucs, color=colors, edgecolor=edge_colors,
              linewidth=1.5, width=0.7, zorder=2, alpha=0.85)

# Value labels on bars
for i, (bar, auc, status) in enumerate(zip(bars, aucs, statuses)):
    y_pos = bar.get_height() + 0.003
    weight = "bold" if status == "keep" else "normal"
    color = "#2ecc71" if status == "keep" else "#e74c3c"
    ax.text(i, y_pos, f"{auc:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight=weight, color=color)

# Baseline reference line
ax.axhline(y=0.647, color="#e67e22", linewidth=1, linestyle=":", alpha=0.5)
ax.text(len(aucs) - 0.5, 0.648, "baseline 0.647", ha="right", va="bottom",
        fontsize=8, color="#e67e22", alpha=0.7)

# Final best reference
ax.axhline(y=0.774, color="#2ecc71", linewidth=1, linestyle=":", alpha=0.5)
ax.text(len(aucs) - 0.5, 0.775, "best 0.774 (+19.6%)", ha="right", va="bottom",
        fontsize=8, color="#2ecc71", alpha=0.7, fontweight="bold")

# Axes
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="#c9d1d9")
ax.set_ylabel("AUC-ROC", fontsize=13, color="#c9d1d9", fontweight="bold")
ax.set_xlabel("Experiment", fontsize=13, color="#c9d1d9", fontweight="bold")
ax.set_ylim(0.6, 0.82)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.tick_params(axis="y", colors="#c9d1d9")
ax.tick_params(axis="x", colors="#c9d1d9")

# Grid
ax.grid(axis="y", color="#21262d", linewidth=0.5, alpha=0.8)
ax.set_axisbelow(True)

# Spines
for spine in ax.spines.values():
    spine.set_color("#30363d")

# Title
ax.set_title("Autoresearch: AUC-ROC across 14 experiments",
             fontsize=16, color="#f0f6fc", fontweight="bold", pad=15)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2ecc71", edgecolor="#1a9850", label="Keep (improved)"),
    Patch(facecolor="#e74c3c", edgecolor="#c0392b", label="Discard (no improvement)"),
    plt.Line2D([0], [0], color="#3498db", linewidth=2.5, linestyle="--",
               label="Best AUC-ROC", alpha=0.6),
]
legend = ax.legend(handles=legend_elements, loc="upper left", fontsize=10,
                   facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

plt.tight_layout()
plt.savefig("autoresearch_results.png", dpi=150, facecolor=fig.get_facecolor(),
            bbox_inches="tight", pad_inches=0.3)
print("Saved: autoresearch_results.png")
