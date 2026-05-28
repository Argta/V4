"""Visualization functions for localization evaluation."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_doa_trajectory(timestamps, doa_estimated, doa_truth,
                        scene_name="", save_path=None):
    """Plot estimated vs true DOA over time.

    Args:
        timestamps: (M,) time in seconds
        doa_estimated: (M,) estimated azimuth in degrees
        doa_truth: (M,) true azimuth in degrees
        scene_name: scene identifier for title
        save_path: optional save path
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(timestamps, doa_truth, "k-", linewidth=1.5, alpha=0.8, label="True DOA")
    ax.plot(timestamps, doa_estimated, "r--", linewidth=1.0, alpha=0.8, label="Estimated DOA")
    ax.fill_between(timestamps, doa_truth, doa_estimated, alpha=0.15, color="red")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Azimuth (deg)")
    ax.set_title(f"DOA Estimation: {scene_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-180, 180)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_histogram(errors_deg, scene_name="", save_path=None):
    """Plot histogram of angular errors.

    Args:
        errors_deg: array of angular errors in degrees
        scene_name: scene identifier
        save_path: optional save path
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(errors_deg, bins=40, range=(-180, 180), color="steelblue",
            edgecolor="white", alpha=0.85)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.axvline(x=np.mean(errors_deg), color="red", linestyle="--",
               linewidth=1, label=f"Mean = {np.mean(errors_deg):.1f}")

    ax.set_xlabel("Angular Error (deg)")
    ax.set_ylabel("Frame Count")
    ax.set_title(f"Error Distribution: {scene_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, scene_name="", save_path=None):
    """Plot 4-quadrant confusion matrix as heatmap.

    Args:
        cm: 4x4 confusion matrix
        scene_name: scene identifier
        save_path: optional save path
    """
    labels = ["Front-R", "Front-L", "Back-R", "Back-L"]
    fig, ax = plt.subplots(figsize=(5, 4.5))

    im = ax.imshow(cm, cmap="YlOrRd", origin="upper")

    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cm[i][j] > cm.max()/2 else "black")

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Estimated")
    ax.set_ylabel("Truth")
    ax.set_title(f"DOA Confusion Matrix: {scene_name}")
    plt.colorbar(im, ax=ax, shrink=0.8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
