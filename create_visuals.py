"""
create_visuals.py — Report Visualisations for the Two-Tier Cascade System

Reads results/eval_results.json and produces two publication-ready charts
saved to results/figures/. Run after main.py has completed.

Usage:
    python create_visuals.py

Outputs (PNG + PDF for LaTeX inclusion):
    tier1_recall_bar.png/pdf         — Per-class recall at Tier 1 (gate performance)
    tier2_metrics_grouped_bar.png/pdf — Per-class Precision / Recall / F1 at Tier 2 (LLM performance)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_JSON  = "results/eval_results.json"
OUT_DIR       = "results/figures"
ORDINAL_ORDER = ["minimal", "mild", "moderate", "severe"]

SEVERITY_COLORS = {
    "minimal":  "#4CAF50",
    "mild":     "#FFC107",
    "moderate": "#FF5722",
    "severe":   "#B71C1C",
}
TIER1_COLOR = "#1565C0"   # blue  — precision bars
TIER2_COLOR = "#00695C"   # teal  — recall bars
F1_COLOR    = "#6A1B9A"   # purple — F1 bars

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


def load_results(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found at '{path}'.\nRun main.py first."
        )
    with open(path) as f:
        return json.load(f)


def save(fig, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight")
    print(f"  Saved: {name}.png / .pdf")
    plt.close(fig)


def plot_tier1_recall(t1: dict) -> None:
    """Bar chart of per-class recall at Tier 1 with a 95% clinical target line."""
    recall   = t1["recall"]
    labels   = ORDINAL_ORDER
    values   = [recall[l] for l in labels]
    colors   = [SEVERITY_COLORS[l] for l in labels]
    x_labels = [l.capitalize() for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x_labels, values, color=colors, width=0.5, zorder=3)
    ax.axhline(95, color="black", linestyle="--", linewidth=1.2,
               label="95% clinical target", zorder=4)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Tier 1 — Per-Class Recall\n(Binary Sentinel Filter)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.legend(fontsize=10)
    fig.text(
        0.5, -0.02,
        f"Threshold p > {t1['threshold']:.2f}  |  "
        f"{t1['posts_in']} → {t1['posts_out']} posts  |  "
        f"{t1['filtered_out_pct']:.1f}% filtered out  |  "
        f"At-risk recall: {t1['recall']['at_risk']:.1f}%",
        ha="center", fontsize=9, color="grey",
    )
    save(fig, "tier1_recall_bar")


def plot_tier2_grouped_bar(t2: dict) -> None:
    """Grouped bar chart of Precision / Recall / F1 per severity class at Tier 2."""
    per_class = t2.get("per_class", {})
    if not per_class:
        print("  [Skip] No Tier 2 per-class data available yet.")
        return

    metrics      = ["precision", "recall", "f1"]
    metric_lbls  = ["Precision", "Recall", "F1"]
    bar_colors   = [TIER1_COLOR, TIER2_COLOR, F1_COLOR]
    x            = np.arange(len(ORDINAL_ORDER))
    width        = 0.22
    offsets      = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for metric, label, color, offset in zip(metrics, metric_lbls, bar_colors, offsets):
        vals = [per_class[l][metric] for l in ORDINAL_ORDER]
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=color, alpha=0.85, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.013,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in ORDINAL_ORDER], fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Tier 2 — Per-Class Precision / Recall / F1\n(Fine-Tuned Llama 3.1-8B + QLoRA)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.legend(fontsize=10)

    macro = t2["macro"]
    fig.text(
        0.5, -0.02,
        f"Macro — Precision: {macro['precision']:.4f}  |  "
        f"F1: {macro['f1_macro']:.4f}  |  "
        f"F1 (weighted): {macro['f1_weighted']:.4f}  |  "
        f"Ordinal MAE: {macro['ordinal_mae']}",
        ha="center", fontsize=9, color="grey",
    )
    save(fig, "tier2_metrics_grouped_bar")


def main():
    print("=" * 52 + "\n  Generating Report Visualisations\n" + "=" * 52)
    results = load_results(RESULTS_JSON)

    print("\nGenerating charts:")
    plot_tier1_recall(results["tier1"])
    plot_tier2_grouped_bar(results["tier2"])

    print(f"\nAll figures saved to '{OUT_DIR}/'")

if __name__ == "__main__":
    main()