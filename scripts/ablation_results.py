#!/usr/bin/env python3
"""
Aggregate ablation results into comparison tables and figures.

Reads results/ablation/metrics.json (produced by ablation_eval.py)
and generates:
  - results/ablation/comparison.md: Markdown comparison tables
  - results/ablation/figures/: Bar charts and radar plots

Usage:
  python scripts/ablation_results.py
  python scripts/ablation_results.py --metrics results/ablation/metrics.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "ablation"
DEFAULT_METRICS = RESULTS_DIR / "metrics.json"

EXPERIMENT_META = {
    "baseline_ref":                {"priority": "P0", "label": "baseline (ref)",    "group": "Core"},
    "ablation_dino":               {"priority": "P0", "label": "+DINO",             "group": "Core"},
    "ablation_synth":              {"priority": "P0", "label": "+Synth",            "group": "Core"},
    "ablation_train":              {"priority": "P0", "label": "+Train",            "group": "Core"},
    "ablation_full":               {"priority": "P0", "label": "Full",              "group": "Core"},
    "ablation_film_only":          {"priority": "P1", "label": "FiLM only",         "group": "Fusion"},
    "ablation_cross_only":         {"priority": "P1", "label": "Cross-Attn only",   "group": "Fusion"},
    "ablation_dino_none":          {"priority": "P1", "label": "No DINO fusion",    "group": "Fusion"},
    "ablation_both_strided":       {"priority": "P1", "label": "Both + strided",    "group": "Fusion"},
    "ablation_synth_legacy":       {"priority": "P2", "label": "Legacy synth",      "group": "Synthesis"},
    "ablation_synth_reflection2":  {"priority": "P2", "label": "Reflection2",       "group": "Synthesis"},
    "ablation_synth_advanced":     {"priority": "P2", "label": "Advanced",          "group": "Synthesis"},
    "ablation_synth_mixed5050":    {"priority": "P2", "label": "Mixed 50/50",       "group": "Synthesis"},
    "ablation_adam_manual":        {"priority": "P3", "label": "Adam + manual",     "group": "Training"},
    "ablation_adamw_cosine":       {"priority": "P3", "label": "AdamW + cosine",    "group": "Training"},
    "ablation_grad_clip":          {"priority": "P3", "label": "+ grad clip",       "group": "Training"},
    "ablation_ema":                {"priority": "P3", "label": "+ EMA",             "group": "Training"},
}

DATASET_ORDER = ["ceilnet_table2", "real20", "objects", "postcard", "wild", "sir2_withgt"]
METRIC_ORDER = ["PSNR", "SSIM", "NCC", "LMSE"]

# Map metrics: higher=better for PSNR/SSIM/NCC, lower=better for LMSE
METRIC_HIGHER_BETTER = {"PSNR": True, "SSIM": True, "NCC": True, "LMSE": False}


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def format_value(value: float, metric: str) -> str:
    if metric == "LMSE":
        return f"{value:.4f}"
    return f"{value:.4f}"


def delta_str(current: float, baseline: float, metric: str) -> str:
    diff = current - baseline
    if METRIC_HIGHER_BETTER.get(metric, True):
        if diff > 0.001:
            return f"+{diff:.4f}"
        elif diff < -0.001:
            return f"{diff:.4f}"
    else:
        if diff < -0.001:
            return f"{diff:.4f}"  # negative is good for LMSE
        elif diff > 0.001:
            return f"+{diff:.4f}"
    return " ~"


def generate_markdown(metrics: dict, output_path: Path):
    """Generate comparison tables in Markdown."""
    lines = []
    lines.append("# Ablation Experiment Results\n")

    # ---- Per-experiment per-dataset table ----
    lines.append("## Full Results (per dataset)\n")

    # Header
    header = "| Experiment | Priority |"
    for ds in DATASET_ORDER:
        for m in METRIC_ORDER:
            header += f" {ds} {m} |"
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(DATASET_ORDER) * len(METRIC_ORDER)) + "|")

    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    sorted_exps = sorted(
        [k for k in metrics if k in EXPERIMENT_META],
        key=lambda k: (priority_order.get(EXPERIMENT_META[k]["priority"], 9), k),
    )

    for exp_name in sorted_exps:
        meta = EXPERIMENT_META[exp_name]
        per_ds = metrics[exp_name]
        row = f"| **{meta['label']}** | {meta['priority']} |"
        for ds in DATASET_ORDER:
            if ds in per_ds:
                for m in METRIC_ORDER:
                    val = per_ds[ds].get(m, float("nan"))
                    row += f" {format_value(val, m)} |"
            else:
                row += " - |" * len(METRIC_ORDER)
        lines.append(row)

    lines.append("")

    # ---- P0: Core ablation with deltas ----
    baseline_ref = metrics.get("baseline_ref", {})
    if baseline_ref:
        lines.append("## P0: Core Improvement Ablation\n")
        lines.append("Each row shows absolute metrics and delta vs baseline (pre-trained checkpoint).\n")

        p0_exps = [k for k in sorted_exps if EXPERIMENT_META[k]["priority"] == "P0"]
        baseline = baseline_ref

        header = "| Experiment |"
        for ds in DATASET_ORDER:
            header += f" {ds} PSNR (Δ) |"
        lines.append(header)
        lines.append("|" + "---|" * (1 + len(DATASET_ORDER)) + "|")

        for exp_name in p0_exps:
            meta = EXPERIMENT_META[exp_name]
            per_ds = metrics[exp_name]
            row = f"| **{meta['label']}** |"
            for ds in DATASET_ORDER:
                if ds in per_ds and ds in baseline:
                    val = per_ds[ds].get("PSNR", float("nan"))
                    base_val = baseline[ds].get("PSNR", float("nan"))
                    d = delta_str(val, base_val, "PSNR")
                    row += f" {format_value(val, 'PSNR')} ({d}) |"
                else:
                    row += " - |"
            lines.append(row)
        lines.append("")

        # P0 SSIM row
        header = "| Experiment |"
        for ds in DATASET_ORDER:
            header += f" {ds} SSIM (Δ) |"
        lines.append(header)
        lines.append("|" + "---|" * (1 + len(DATASET_ORDER)) + "|")

        for exp_name in p0_exps:
            meta = EXPERIMENT_META[exp_name]
            per_ds = metrics[exp_name]
            row = f"| **{meta['label']}** |"
            for ds in DATASET_ORDER:
                if ds in per_ds and ds in baseline:
                    val = per_ds[ds].get("SSIM", float("nan"))
                    base_val = baseline[ds].get("SSIM", float("nan"))
                    d = delta_str(val, base_val, "SSIM")
                    row += f" {format_value(val, 'SSIM')} ({d}) |"
                else:
                    row += " - |"
            lines.append(row)
        lines.append("")

    # ---- Grouped summary (mean across datasets) ----
    lines.append("## Summary: Mean Across Datasets\n")
    header = "| Experiment | Priority | Group | PSNR | SSIM | NCC | LMSE |"
    lines.append(header)
    lines.append("|" + "---|" * 7 + "|")

    for exp_name in sorted_exps:
        meta = EXPERIMENT_META[exp_name]
        per_ds = metrics[exp_name]
        means = {}
        for m in METRIC_ORDER:
            vals = [per_ds[ds][m] for ds in DATASET_ORDER if ds in per_ds and m in per_ds[ds]]
            means[m] = sum(vals) / len(vals) if vals else float("nan")

        row = f"| **{meta['label']}** | {meta['priority']} | {meta['group']} |"
        for m in METRIC_ORDER:
            row += f" {format_value(means[m], m)} |"
        lines.append(row)
    lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"Markdown tables written to {output_path}")


def generate_figures(metrics: dict, output_dir: Path):
    """Generate bar charts using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available, skipping figures")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- P0 bar chart: per-dataset PSNR ----
    p0_exps = [k for k in metrics if EXPERIMENT_META.get(k, {}).get("priority") == "P0"]
    if len(p0_exps) < 2:
        print("[WARN] not enough P0 experiments for figures")
        return

    # baseline_ref first, then the rest
    p0_exps = sorted(
        p0_exps,
        key=lambda k: (0 if k == "baseline_ref" else 1, EXPERIMENT_META[k].get("label", k)),
    )
    labels = [EXPERIMENT_META[e]["label"] for e in p0_exps]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5"]

    for idx, ds in enumerate(DATASET_ORDER):
        ax = axes[idx]
        metric = "PSNR"
        values = []
        for exp_name in p0_exps:
            per_ds = metrics[exp_name]
            values.append(per_ds.get(ds, {}).get(metric, 0))

        bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor="white", linewidth=0.5)
        ax.set_title(f"{ds} - {metric}", fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30, labelsize=8)

        # Value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    # Hide extra subplot if any
    if len(DATASET_ORDER) < len(axes):
        for ax in axes[len(DATASET_ORDER):]:
            ax.set_visible(False)

    fig.suptitle("P0 Ablation: Per-Dataset PSNR", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "p0_psnr_by_dataset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_dir / 'p0_psnr_by_dataset.png'}")

    # ---- P0 Delta vs baseline ----
    if baseline_ref:
        baseline = baseline_ref
        other_exps = [e for e in p0_exps if e != "baseline_ref"]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(DATASET_ORDER))
        width = 0.2

        other_colors = ["#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5"]
        for i, exp_name in enumerate(other_exps):
            deltas = []
            for ds in DATASET_ORDER:
                base = baseline.get(ds, {}).get("PSNR", 0)
                cur = metrics[exp_name].get(ds, {}).get("PSNR", 0)
                deltas.append(cur - base)
            label = EXPERIMENT_META[exp_name]["label"]
            ax.bar(x + i * width, deltas, width, label=label, color=other_colors[i % len(other_colors)],
                   edgecolor="white", linewidth=0.5)

        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xticks(x + width * (len(other_exps) - 1) / 2)
        ax.set_xticklabels(DATASET_ORDER, fontsize=9)
        ax.set_ylabel("Δ PSNR (dB)")
        ax.set_title("P0 Ablation: PSNR Delta vs Baseline", fontweight="bold")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "p0_psnr_delta.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved: {output_dir / 'p0_psnr_delta.png'}")

    # ---- Per-group summary bar chart ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    group_order = ["Core", "Fusion", "Synthesis", "Training"]

    for gidx, group in enumerate(group_order):
        ax = axes[gidx]
        group_exps = [k for k in metrics
                      if EXPERIMENT_META.get(k, {}).get("group") == group]
        group_exps = sorted(group_exps, key=lambda k: EXPERIMENT_META[k].get("label", k))

        group_labels = [EXPERIMENT_META[e]["label"] for e in group_exps]
        group_values = []
        for exp_name in group_exps:
            per_ds = metrics[exp_name]
            vals = [per_ds[ds]["PSNR"] for ds in DATASET_ORDER
                    if ds in per_ds and "PSNR" in per_ds[ds]]
            group_values.append(sum(vals) / len(vals) if vals else 0)

        group_colors = plt.cm.Set2(np.linspace(0, 1, max(len(group_exps), 1)))
        bars = ax.barh(group_labels, group_values, color=group_colors, edgecolor="white")
        ax.set_title(f"{group} (mean PSNR)", fontweight="bold")
        ax.set_xlabel("PSNR (dB)")

        for bar, val in zip(bars, group_values):
            if val > 0:
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}", va="center", fontsize=8)

    fig.suptitle("Ablation Summary by Group", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "summary_by_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_dir / 'summary_by_group.png'}")

    # ---- Synthetic vs Real generalization scatter ----
    # Compare CEILNet (synthetic) vs mean of real datasets
    real_datasets = ["real20", "objects", "postcard", "wild"]
    fig, ax = plt.subplots(figsize=(10, 8))

    for exp_name, meta in EXPERIMENT_META.items():
        if exp_name not in metrics:
            continue
        per_ds = metrics[exp_name]
        ceilnet_psnr = per_ds.get("ceilnet_table2", {}).get("PSNR")
        real_psnrs = [per_ds[ds]["PSNR"] for ds in real_datasets
                      if ds in per_ds and "PSNR" in per_ds[ds]]
        if ceilnet_psnr is None or not real_psnrs:
            continue
        real_mean = sum(real_psnrs) / len(real_psnrs)

        color_map = {"P0": "#4472C4", "P1": "#ED7D31", "P2": "#70AD47", "P3": "#FFC000"}
        ax.scatter(ceilnet_psnr, real_mean,
                   c=color_map.get(meta["priority"], "gray"),
                   s=80, alpha=0.8, edgecolors="black", linewidth=0.5)
        ax.annotate(meta["label"], (ceilnet_psnr, real_mean),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("CEILNet Table2 PSNR (synthetic)")
    ax.set_ylabel("Mean PSNR on Real Datasets")
    ax.set_title("Synthetic vs Real Generalization", fontweight="bold")

    # Legend for priority colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4472C4", label="P0: Core"),
        Patch(facecolor="#ED7D31", label="P1: Fusion"),
        Patch(facecolor="#70AD47", label="P2: Synthesis"),
        Patch(facecolor="#FFC000", label="P3: Training"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "synth_vs_real.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_dir / 'synth_vs_real.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate ablation results tables and figures")
    parser.add_argument("--metrics", type=str, default=str(DEFAULT_METRICS),
                        help="Path to metrics.json")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR),
                        help="Output directory")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Error: metrics file not found: {metrics_path}")
        print("Run ablation_eval.py first to generate metrics.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metrics_path)
    print(f"Loaded metrics for {len(metrics)} experiments")

    # Generate Markdown tables
    generate_markdown(metrics, output_dir / "comparison.md")

    # Generate figures
    if not args.no_figures:
        figures_dir = output_dir / "figures"
        generate_figures(metrics, figures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
