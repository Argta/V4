"""Aggregate sweep results and generate comparison visualizations."""

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_sweep_summary(sweep_results, sweep_variable, save_dir):
    """Save a summary CSV from sweep results.

    Args:
        sweep_results: list of {value, metrics, variable} dicts
        sweep_variable: name of swept variable
        save_dir: directory to save summary.csv
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "summary.csv"

    rows = []
    for r in sweep_results:
        m = r.get("metrics")
        if m is None:
            continue
        rows.append({
            sweep_variable: r["value"],
            "rmse_deg": round(m.get("rmse", 0), 2),
            "mae_deg": round(m.get("mae", 0), 2),
            "accuracy_10deg": round(m.get("accuracy_10deg", 0) * 100, 1),
            "front_back_conf_pct": round(m.get("front_back_confusion", 0) * 100, 1),
            "n_frames": m.get("n_frames", 0),
        })

    if not rows:
        print("  No valid sweep results to summarize")
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Summary CSV saved: {csv_path}")

    # Print table
    print(f"\n  Sweep Results: {sweep_variable}")
    header = f"  {'Value':>10} | {'RMSE':>8} | {'MAE':>8} | {'Acc10':>7} | {'F/B%':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        v = row[sweep_variable]
        print(f"  {v:>10} | {row['rmse_deg']:>6.2f} deg | {row['mae_deg']:>6.2f} deg | "
              f"{row['accuracy_10deg']:>6.1f}% | {row['front_back_conf_pct']:>6.1f}%")


def plot_sweep_comparison(sweep_results, sweep_variable, save_path=None):
    """Generate comparison plot for parameter sweep.

    Shows RMSE, accuracy, and front-back confusion vs swept variable.

    Args:
        sweep_results: list of {value, metrics} dicts
        sweep_variable: name of swept variable
        save_path: optional save path
    """
    valid = [(r["value"], r["metrics"]) for r in sweep_results if r.get("metrics")]

    if len(valid) < 2:
        print("  Not enough valid results for comparison plot")
        return

    values = [v for v, _ in valid]
    rmses = [m["rmse"] for _, m in valid]
    accs = [m["accuracy_10deg"] * 100 for _, m in valid]
    fbs = [m["front_back_confusion"] * 100 for _, m in valid]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(values, rmses, "o-", color="steelblue", linewidth=2, markersize=8)
    axes[0].set_xlabel(sweep_variable)
    axes[0].set_ylabel("RMSE (deg)")
    axes[0].set_title("RMSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(values, accs, "s-", color="forestgreen", linewidth=2, markersize=8)
    axes[1].set_xlabel(sweep_variable)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy (within 10 deg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 105)

    axes[2].plot(values, fbs, "^-", color="darkorange", linewidth=2, markersize=8)
    axes[2].set_xlabel(sweep_variable)
    axes[2].set_ylabel("Confusion Rate (%)")
    axes[2].set_title("Front-Back Confusion")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 105)

    plt.suptitle(f"Parameter Sweep: {sweep_variable}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Sweep plot saved: {save_path}")
    plt.close(fig)


def plot_algorithm_comparison(algo_results, scene_name="", save_path=None):
    """Bar chart comparing multiple localization algorithms.

    Args:
        algo_results: list of (method_name, metrics) tuples
        scene_name: scene identifier
        save_path: optional save path
    """
    methods = [m for m, r in algo_results if "error" not in (r or {})]
    rmses = [r["rmse"] for _, r in algo_results if "rmse" in (r or {})]
    accs = [r["accuracy_10deg"] * 100 for _, r in algo_results if "accuracy_10deg" in (r or {})]

    if len(methods) < 1:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["steelblue", "forestgreen", "darkorange"]

    axes[0].bar(methods, rmses, color=colors[:len(methods)], edgecolor="white")
    axes[0].set_ylabel("RMSE (deg)")
    axes[0].set_title("RMSE by Algorithm")
    for i, v in enumerate(rmses):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    axes[1].bar(methods, accs, color=colors[:len(methods)], edgecolor="white")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy (10 deg) by Algorithm")
    axes[1].set_ylim(0, 105)
    for i, v in enumerate(accs):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

    plt.suptitle(f"Algorithm Comparison: {scene_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Comparison plot saved: {save_path}")
    plt.close(fig)
