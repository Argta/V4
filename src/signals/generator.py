"""Signal generation with dynamic module loading.

Each generator module in this package exposes:
    generate(t, fs, cfg) -> np.ndarray

Where `t` is the time vector, `fs` is the sample rate,
and `cfg` is the source config`s params dict.
"""

from pathlib import Path
import importlib.util
import numpy as np


_GENERATORS_DIR = Path(__file__).resolve().parent


def _load_generator_module(name: str):
    """Dynamically load a generator module by name."""
    module_path = _GENERATORS_DIR / f"{name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Generator module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_source(source_config) -> np.ndarray:
    """Generate a source signal from a SourceConfig.

    Returns 1-D numpy array of float64 samples.
    """
    fs = source_config.sample_rate
    duration = source_config.duration
    generator_name = source_config.generator
    t = np.arange(0, duration, 1.0 / fs)

    # Try loading a custom generator module
    try:
        module = _load_generator_module(generator_name)
        signal = module.generate(t, fs, source_config.params)
        return signal.astype(np.float64)
    except FileNotFoundError:
        pass

    # Built-in fallback generators
    params = source_config.params

    if generator_name == "sine":
        freq = params.get("frequency", 440)
        amp = params.get("amplitude", 0.3)
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)

    elif generator_name == "chirp":
        f0 = params.get("f0", 300)
        f1 = params.get("f1", 3000)
        amp = params.get("amplitude", 0.5)
        instant_freq = f0 + (f1 - f0) * t / (2 * duration)
        return (amp * np.sin(2 * np.pi * instant_freq * t)).astype(np.float64)

    else:
        raise ValueError(
            f"Unknown generator: '{generator_name}'. "
            f"Available: bowl_impact, chair_sliding, human_voice, "
            f"sine, chirp"
        )