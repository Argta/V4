"""Parameter sweep engine for binaural localization experiments."""

import copy
import time
from pathlib import Path
import numpy as np


def run_sweep(scene_config, sweep_variable, sweep_values, results_dir=None):
    """Run a parameter sweep across multiple values.

    Args:
        scene_config: base SceneConfig
        sweep_variable: name of variable to sweep
            Supported: "snr", "absorption", "distance", "max_order",
                       "noise_background_snr", "noise_sensor_snr"
        sweep_values: list of values
        results_dir: output directory for sweep results

    Returns:
        list of dicts: [{value, metrics, timestamp}, ...]
    """
    from src.pipeline.simulator import BinauralSimulator
    from src.output.saver import save_results
    from src.config.schema import SceneConfig

    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "sweeps"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Parameter Sweep: {sweep_variable}")
    print(f"  Values: {sweep_values}")
    print(f"  Base scene: {scene_config.name}")
    print(f"{'='*60}")

    all_results = []

    for i, val in enumerate(sweep_values):
        print(f"\n--- Sweep {i+1}/{len(sweep_values)}: {sweep_variable} = {val} ---")

        cfg = _apply_sweep_value(copy.deepcopy(scene_config), sweep_variable, val)
        cfg.evaluation.enabled = True
        cfg.output.visualize = False  # Skip per-run viz for speed

        sim = BinauralSimulator.__new__(BinauralSimulator)
        sim.cfg = cfg
        sim.fs = cfg.output.sample_rate
        sim.project_root = Path(__file__).resolve().parent.parent.parent

        try:
            stereo, trajectory, loc_result, eval_metrics = sim.run()

            if eval_metrics is not None:
                all_results.append({
                    "value": val,
                    "metrics": eval_metrics,
                    "variable": sweep_variable,
                })

            save_results(
                stereo, trajectory, f"{cfg.name}_sweep_{sweep_variable}_{val}",
                sim.fs,
                scene_config=cfg,
                results_dir=results_dir,
                loc_result=loc_result,
                eval_metrics=eval_metrics,
            )

        except Exception as e:
            print(f"  ERROR at {sweep_variable}={val}: {e}")
            all_results.append({
                "value": val,
                "metrics": None,
                "variable": sweep_variable,
                "error": str(e),
            })

    return all_results


def _apply_sweep_value(cfg, variable, value):
    """Apply a sweep value to the scene config."""
    if variable == "snr" or variable == "noise_background_snr":
        cfg.noise.enabled = True
        cfg.noise.background_snr_db = float(value)
    elif variable == "noise_sensor_snr":
        cfg.noise.enabled = True
        cfg.noise.sensor_snr_db = float(value)
    elif variable == "absorption":
        cfg.room.absorption = float(value)
    elif variable == "max_order":
        cfg.room.max_order = int(value)
    elif variable == "distance":
        # Adjust source position radius
        import math
        center = np.array(cfg.motion.params.get("center", cfg.microphone.head_center))
        if cfg.motion.type in ("circle", "semicircle"):
            cfg.motion.params["radius"] = float(value)
        elif cfg.motion.type == "static":
            # Move source position along x-axis
            cfg.motion.params["position"] = [
                center[0] + float(value), center[1], center[2]
            ]
    elif variable == "hrtf_mode":
        cfg.microphone.hrtf_mode = str(value)
    elif variable == "localization_method":
        cfg.localization.method = str(value)
    else:
        raise ValueError(f"Unknown sweep variable: {variable}")

    return cfg


def compare_algorithms(scene_config, methods=None, results_dir=None):
    """Compare multiple localization algorithms on the same scene.

    Args:
        scene_config: base SceneConfig
        methods: list of method names (default: all three)
        results_dir: output directory

    Returns:
        list of (method_name, metrics) tuples
    """
    if methods is None:
        methods = ["xcorr_itd", "gcc_phat", "srp_phat"]

    from src.pipeline.simulator import BinauralSimulator

    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "compare"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Algorithm Comparison")
    print(f"  Methods: {methods}")
    print(f"  Scene: {scene_config.name}")
    print(f"{'='*60}")

    results = []
    for method in methods:
        print(f"\n--- Method: {method} ---")
        cfg = copy.deepcopy(scene_config)
        cfg.localization.method = method
        cfg.evaluation.enabled = True
        cfg.output.visualize = False

        sim = BinauralSimulator.__new__(BinauralSimulator)
        sim.cfg = cfg
        sim.fs = cfg.output.sample_rate
        sim.project_root = Path(__file__).resolve().parent.parent.parent

        try:
            stereo, trajectory, loc_result, eval_metrics = sim.run()
            if eval_metrics:
                results.append((method, eval_metrics))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((method, {"error": str(e)}))

    return results
