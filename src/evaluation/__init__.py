"""Error evaluation and reporting for localization results."""

from .metrics import (
    angular_error, rmse, mae, accuracy_within,
    front_back_confusion_rate, confusion_matrix_quadrants,
    compute_all_metrics,
)


def evaluate(estimated_doa, truth_doa, timestamps=None, scene_name=""):
    """Run full evaluation on localization results.

    Args:
        estimated_doa: (N,) estimated azimuth in degrees
        truth_doa: (N,) true azimuth in degrees
        timestamps: (N,) optional timestamps
        scene_name: optional scene identifier

    Returns:
        dict: metrics dict from compute_all_metrics
    """
    return compute_all_metrics(estimated_doa, truth_doa)
