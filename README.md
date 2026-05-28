# Binaural Sound Source Localization Simulation System v3.0

A Python-based binaural room acoustics simulation system for studying sound source localization using interaural time differences (ITD) and interaural level differences (ILD).

## Features

- **True binaural simulation** — Independent RIR computation per ear with pyroomacoustics image-source method
- **HRTF spatialization** — Spherical head model (Woodworth ITD + Rayleigh ILD) for physically correct binaural cues
- **Doppler effect** — Time-varying resampling for moving sound sources
- **Multiple source types** — Bowl impact, chair sliding, human voice (formant synthesis), plus built-in sine/noise/chirp
- **Flexible trajectories** — Static, linear, semicircle, and circular motion paths
- **Self-contained scenes** — One YAML file = one complete simulation configuration
- **Automatic visualization** — Matplotlib 6-panel output: 3D scene, waveforms, spectrograms, ITD/ILD analysis
- **Portable** — All paths relative to project root, no hardcoded directories

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a demo scene (command line)
python run.py scenes/static_bowl.yaml

# Launch interactive MATLAB GUI
run_gui.bat
# Or in MATLAB: cd D:\shengxuedingwei2\matlab; binaural_gui
```

## MATLAB GUI

The interactive GUI (`matlab/binaural_gui.m`) provides:

- **3D Room Scene** — Interactive rotation/zoom, room wireframe, head+ears, trajectory, animated source
- **Playback Animation** — Source moves along trajectory, synchronized cursor across all views
- **Waveform Display** — Left (blue) / Right (red) channel overlay
- **ITD/ILD Analysis** — Time-varying interaural time and level differences
- **Run New Simulation** — Select a scene YAML, click to run Python simulation, auto-loads result
- **Speed Control** — 1x-20x animation speed slider
- **Result Browser** — Load any .mat result file from the results/ directory

## Project Structure

```
├── run.py                  # Entry point
├── scenes/                 # Scene definition files
│   ├── static_bowl.yaml    # Baseline binaural test
│   ├── moving_chair.yaml   # Linear motion + Doppler
│   ├── semicircle_voice.yaml  # Best binaural demo
│   ├── doppler_demo.yaml   # Pure Doppler pitch shift
│   └── binaural_circle.yaml   # Full 360-degree sweep
├── src/
│   ├── config/             # Configuration system (schema + YAML loader)
│   ├── signals/            # Signal generators (bowl, chair, voice, builtins)
│   ├── room/               # pyroomacoustics wrapper
│   ├── motion/             # Trajectory generation + Doppler effect
│   ├── spatial/            # HRTF + binaural processor
│   ├── pipeline/           # Main simulation orchestrator
│   └── output/             # File saver + visualizer
├── tests/                  # Unit tests
└── results/                # Simulation output (auto-created)
```

## Scene Configuration

Each scene YAML file defines:

| Section | Description |
|---------|-------------|
| `room` | Room dimensions, wall absorption, reflection order |
| `microphone` | Head center, head radius, ear positions (auto-computed) |
| `source` | Signal generator type, duration, generator-specific params |
| `motion` | Trajectory type (static/linear/semicircle/circle) and params |
| `output` | Sample rate, visualization toggle |

## Simulation Pipeline

```
Scene YAML → Source signal → Trajectory → Overlap-add segmentation
  → Per-segment binaural RIR (left + right ears)
  → HRTF (ITD fractional delay + ILD head shadow)
  → Doppler resampling → Stereo WAV + NPZ metadata + Visualization
```

## Architecture vs Old System (v1.0)

| Aspect | v1.0 | v3.0 |
|--------|------|------|
| Config | 12+ JSON files in 4 dirs | 1 YAML file per scene |
| Paths | Hardcoded D:\... | Relative from project root |
| Channels | Mono only | True binaural stereo |
| HRTF | None | Woodworth + Rayleigh model |
| Doppler | None | Continuous resampling |
| Visualization | MATLAB (broken) | Matplotlib |
| Segments | 20 fixed | 4-200 adaptive |
| Output | output_N.wav | {scene}_{timestamp}.wav |
