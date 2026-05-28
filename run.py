#!/usr/bin/env python
"""Binaural acoustic simulation — unified entry point.

Usage:
    python run.py                                    # Interactive mode
    python run.py scenes/static_bowl.yaml             # Run a scene
    python run.py scenes/semicircle_voice.yaml --no-viz
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.simulator import BinauralSimulator
from src.output.saver import save_results
from src.output.visualizer import visualize, visualize_evaluation


def list_scenes():
    """Return list of (name, path) for all scene files."""
    scenes = []
    for pattern in ("*.yaml", "*.json"):
        for f in sorted((PROJECT_ROOT / "scenes").glob(pattern)):
            scenes.append((f.stem, f))
    return scenes


def interactive_menu():
    """Show interactive scene selector and run chosen scene."""
    scenes = list_scenes()
    if not scenes:
        print("No scene files found in scenes/")
        return

    print("\n" + "=" * 55)
    print("  Binaural Simulation — Scene Selector")
    print("=" * 55)
    for i, (name, _) in enumerate(scenes, 1):
        print(f"  [{i}] {name}")
    print(f"  [0] Exit")
    print("-" * 55)

    while True:
        try:
            choice = input("Select scene (0-{}): ".format(len(scenes))).strip()
            if not choice:
                continue
            idx = int(choice)
            if idx == 0:
                print("Exiting.")
                return
            if 1 <= idx <= len(scenes):
                break
        except (ValueError, EOFError, KeyboardInterrupt):
            print("Exiting.")
            return
        print(f"  Please enter 1-{len(scenes)} or 0 to exit")

    _, scene_path = scenes[idx - 1]
    run_scene(scene_path)


def run_scene(scene_path, hrtf_mode=None, source_gen=None):
    """Run a single scene simulation with optional evaluation."""
    no_viz = "--no-viz" in sys.argv

    sim = BinauralSimulator(scene_path)
    if hrtf_mode:
        sim.cfg.microphone.hrtf_mode = hrtf_mode
    if source_gen:
        sim.cfg.source.generator = source_gen
    stereo, trajectory, loc_result, eval_metrics, stereo_raw = sim.run()

    scene_name = sim.cfg.name
    wav_path, npz_path, mat_path = save_results(
        stereo, trajectory, scene_name, sim.fs,
        scene_config=sim.cfg,
        loc_result=loc_result,
        eval_metrics=eval_metrics,
        stereo_raw=stereo_raw,
    )

    if sim.cfg.output.visualize and not no_viz:
        # Standard visualization
        fig_path = str(Path(npz_path).with_suffix(".png"))
        visualize(sim.cfg, stereo, trajectory, sim.fs, save_path=fig_path)

        # Evaluation visualization (v3.0)
        if loc_result is not None and eval_metrics is not None:
            from src.localization import compute_truth_doa
            from src.evaluation.metrics import angular_error
            import numpy as np

            head_center = np.array(sim.cfg.microphone.head_center)
            truth_doa = compute_truth_doa(trajectory, head_center)
            loc_times = loc_result.timestamps
            truth_indices = np.clip(
                (loc_times * sim.fs).astype(int), 0, len(truth_doa) - 1
            )
            truth_per_frame = truth_doa[truth_indices]
            errors = angular_error(loc_result.doa_estimated, truth_per_frame)
            cm = np.array(eval_metrics.get("confusion_matrix", [[0]*4]*4))

            eval_fig_path = str(Path(npz_path).with_suffix("")) + "_eval.png"
            visualize_evaluation(
                scene_name, loc_times,
                loc_result.doa_estimated, truth_per_frame,
                errors, cm, eval_metrics,
                save_path=eval_fig_path,
            )

    print(f"\nDone! Output files:")
    print(f"  {wav_path}")
    print(f"  {npz_path}")
    if mat_path:
        print(f"  {mat_path}")
    # List evaluation outputs
    if loc_result is not None:
        import glob
        base = str(Path(npz_path).with_suffix(""))
        for ext in ["_loc.csv", "_metrics.json", "_eval.png"]:
            p = Path(base + ext)
            if p.exists():
                print(f"  {p}")


def run_sweep_mode(scene_path, sweep_var, sweep_values):
    """Run parameter sweep experiment."""
    from src.config.loader import load_scene
    from src.experiments import run_sweep, save_sweep_summary, plot_sweep_comparison

    cfg = load_scene(scene_path)
    values = [float(v) for v in sweep_values.split(",")]
    sweep_dir = PROJECT_ROOT / "results" / f"sweep_{sweep_var}_{cfg.name}"

    results = run_sweep(cfg, sweep_var, values, results_dir=sweep_dir)
    save_sweep_summary(results, sweep_var, sweep_dir)
    plot_sweep_comparison(results, sweep_var,
                          save_path=str(sweep_dir / "comparison.png"))

    print(f"\nSweep complete. Results: {sweep_dir}")


def run_compare_mode(scene_path, methods=None):
    """Run algorithm comparison on a scene."""
    from src.config.loader import load_scene
    from src.experiments import compare_algorithms, plot_algorithm_comparison

    cfg = load_scene(scene_path)
    method_list = methods.split(",") if methods else None
    compare_dir = PROJECT_ROOT / "results" / f"compare_{cfg.name}"

    results = compare_algorithms(cfg, methods=method_list, results_dir=compare_dir)
    plot_algorithm_comparison(results, cfg.name,
                              save_path=str(compare_dir / "comparison.png"))

    print(f"\nComparison complete. Results: {compare_dir}")


def main():
    if "--sweep" in sys.argv:
        idx = sys.argv.index("--sweep")
        scene_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else None
        sweep_var = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "snr"
        sweep_vals = sys.argv[idx + 2] if idx + 2 < len(sys.argv) else "-5,0,5,10,20"
        if scene_path is None or not scene_path.exists():
            print("Usage: python run.py <scene.yaml> --sweep <variable> <val1,val2,...>")
            sys.exit(1)
        run_sweep_mode(scene_path, sweep_var, sweep_vals)

    elif "--compare" in sys.argv:
        idx = sys.argv.index("--compare")
        scene_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else None
        methods = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if scene_path is None or not scene_path.exists():
            print("Usage: python run.py <scene.yaml> --compare [method1,method2,...]")
            sys.exit(1)
        run_compare_mode(scene_path, methods)

    elif len(sys.argv) >= 2:
        # Direct scene file provided
        scene_path = Path(sys.argv[1])
        if not scene_path.exists():
            print(f"Error: scene file not found: {scene_path}")
            sys.exit(1)
        # Optional --hrtf override
        hrtf_mode = None
        if "--hrtf" in sys.argv:
            idx = sys.argv.index("--hrtf")
            if idx + 1 < len(sys.argv):
                hrtf_mode = sys.argv[idx + 1]
        source_gen = None
        if "--source" in sys.argv:
            idx = sys.argv.index("--source")
            if idx + 1 < len(sys.argv):
                source_gen = sys.argv[idx + 1]
        run_scene(scene_path, hrtf_mode=hrtf_mode, source_gen=source_gen)
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()
