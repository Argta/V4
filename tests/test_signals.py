"""Tests for signal generation modules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config.schema import SourceConfig
from src.signals.generator import generate_source


def test_bowl_impact():
    cfg = SourceConfig(
        generator="bowl_impact",
        duration=0.3,
        sample_rate=16000,
        params={"impact_duration": 0.005},
    )
    sig = generate_source(cfg)
    assert len(sig) == int(0.3 * 16000)
    assert sig.dtype == np.float64
    assert np.max(np.abs(sig)) > 0


def test_chair_sliding():
    cfg = SourceConfig(
        generator="chair_sliding",
        duration=0.5,
        sample_rate=16000,
    )
    sig = generate_source(cfg)
    assert len(sig) == 8000
    assert np.max(np.abs(sig)) > 0


def test_human_voice():
    cfg = SourceConfig(
        generator="human_voice",
        duration=0.5,
        sample_rate=16000,
    )
    sig = generate_source(cfg)
    assert len(sig) == 8000
    assert np.max(np.abs(sig)) > 0


def test_builtin_sine():
    cfg = SourceConfig(
        generator="sine",
        duration=0.5,
        sample_rate=16000,
        params={"frequency": 440, "amplitude": 0.5},
    )
    sig = generate_source(cfg)
    assert len(sig) == 8000
    assert 0.4 < np.max(np.abs(sig)) <= 0.5


def test_builtin_chirp():
    cfg = SourceConfig(
        generator="chirp",
        duration=0.5,
        sample_rate=16000,
        params={"frequency": 440, "amplitude": 0.5},
    )
    sig = generate_source(cfg)
    assert len(sig) == 8000


if __name__ == "__main__":
    test_bowl_impact()
    test_chair_sliding()
    test_human_voice()
    test_builtin_sine()
    test_builtin_chirp()
    print("All signal tests passed!")
