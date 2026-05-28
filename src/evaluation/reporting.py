"""Text and CSV reporting for localization evaluation."""

import json
from pathlib import Path


def generate_report(metrics, scene_name="", save_path=None):
    """Generate a text report from evaluation metrics.

    Args:
        metrics: dict from compute_all_metrics()
        scene_name: scene identifier
        save_path: optional path to save .txt report

    Returns:
        report string
    """
    lines = [
        "=" * 48,
        f"  Localization Evaluation Report",
        f"  Scene: {scene_name}",
        "=" * 48,
        "",
        f"  Frames evaluated:     {metrics.get('n_frames', 0)}",
        f"  RMSE:                 {metrics.get('rmse', 0):.2f} deg",
        f"  MAE:                  {metrics.get('mae', 0):.2f} deg",
        f"  Accuracy (5 deg):     {metrics.get('accuracy_5deg', 0)*100:.1f}%",
        f"  Accuracy (15 deg):    {metrics.get('accuracy_15deg', 0)*100:.1f}%",
        f"  Accuracy (30 deg):    {metrics.get('accuracy_30deg', 0)*100:.1f}%",
        f"  Front-Back Confusion: {metrics.get('front_back_confusion', 0)*100:.1f}%",
        "",
        "  Confusion Matrix (rows=Truth, cols=Estimated):",
        "          Front-R  Front-L  Back-R  Back-L",
    ]

    cm = metrics.get("confusion_matrix", [])
    if cm:
        labels = ["Front-R", "Front-L", "Back-R", "Back-L"]
        for i, row in enumerate(cm):
            line = f"  {labels[i]:>7}  " + "".join(f"{v:>7}  " for v in row)
            lines.append(line)

    report = "\n".join(lines)
    print(report)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Report saved: {save_path}")

    return report


def save_csv(timestamps, doa_estimated, doa_truth, errors, save_path):
    """Save per-frame localization results as CSV.

    Args:
        timestamps: (M,) time in seconds
        doa_estimated: (M,) estimated azimuth in degrees
        doa_truth: (M,) true azimuth in degrees
        errors: (M,) angular error in degrees
        save_path: output CSV path
    """
    import csv

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "doa_truth_deg", "doa_estimated_deg", "error_deg"])
        for t, truth, est, err in zip(timestamps, doa_truth, doa_estimated, errors):
            writer.writerow([f"{t:.4f}", f"{truth:.2f}", f"{est:.2f}", f"{err:.2f}"])

    print(f"  CSV saved: {save_path}")


def save_evaluation_json(metrics, save_path):
    """Save evaluation metrics as JSON.

    Args:
        metrics: dict from compute_all_metrics()
        save_path: output JSON path
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  JSON saved: {save_path}")
