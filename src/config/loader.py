"""Load and validate scene configuration from file (JSON or YAML)."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Union

from .schema import (
    SceneConfig, RoomConfig, MicrophoneConfig,
    SourceConfig, MotionConfig, OutputConfig,
    NoiseConfig, LocalizationConfig, EvaluationConfig,
    ExperimentConfig,
)


class ConfigError(Exception):
    """Configuration validation error."""


def _dict_to_room(d: dict) -> RoomConfig:
    return RoomConfig(
        dimensions=d["dimensions"],
        absorption=d.get("absorption", 0.3),
        max_order=d.get("max_order", 8),
        direct_only=d.get("direct_only", False),
    )


def _dict_to_mic(d: dict) -> MicrophoneConfig:
    return MicrophoneConfig(
        head_center=d["head_center"],
        head_radius=d.get("head_radius", 0.09),
        left_ear=d.get("left_ear"),
        right_ear=d.get("right_ear"),
        hrtf_mode=d.get("hrtf_mode", "analytical"),
        hrtf_dataset=d.get("hrtf_dataset", ""),
        hrtf_subject=d.get("hrtf_subject", 0),
        head_yaw_deg=d.get("head_yaw_deg", 0.0),
    )


def _dict_to_source(d: dict) -> SourceConfig:
    return SourceConfig(
        generator=d["generator"],
        duration=d["duration"],
        sample_rate=d.get("sample_rate", 44100),
        params={k: v for k, v in d.items()
                if k not in ("generator", "duration", "sample_rate")},
    )


def _dict_to_motion(d: dict) -> MotionConfig:
    return MotionConfig(
        enabled=d.get("enabled", False),
        type=d.get("type", "static"),
        params={k: v for k, v in d.items()
                if k not in ("enabled", "type", "head_yaw_speed")},
        head_yaw_speed=d.get("head_yaw_speed", 0.0),
    )


def _dict_to_output(d: dict) -> OutputConfig:
    return OutputConfig(
        sample_rate=d.get("sample_rate", 44100),
        visualize=d.get("visualize", True),
    )


def _dict_to_noise(d: dict) -> NoiseConfig:
    return NoiseConfig(
        enabled=d.get("enabled", False),
        background_snr_db=d.get("background_snr_db", 30.0),
        sensor_snr_db=d.get("sensor_snr_db", 40.0),
        noise_type=d.get("noise_type", "white"),
    )


def _dict_to_localization(d: dict) -> LocalizationConfig:
    freq_range = d.get("freq_range", [300, 3000])
    return LocalizationConfig(
        method=d.get("method", "gcc_phat"),
        frame_duration_ms=d.get("frame_duration_ms", 50.0),
        frame_hop_ms=d.get("frame_hop_ms", 25.0),
        freq_range=(freq_range[0], freq_range[1]) if freq_range else (300, 3000),
        head_yaw_speed=d.get("head_yaw_speed", 0.0),
        active_head=d.get("active_head", False),
    )


def _dict_to_evaluation(d: dict) -> EvaluationConfig:
    return EvaluationConfig(
        enabled=d.get("enabled", True),
        save_report=d.get("save_report", True),
        save_plots=d.get("save_plots", True),
    )


def _dict_to_experiment(d: dict) -> ExperimentConfig:
    return ExperimentConfig(
        enabled=d.get("enabled", False),
        type=d.get("type", "single"),
        sweep_variable=d.get("sweep_variable", ""),
        sweep_values=d.get("sweep_values", []),
    )


def load_scene(source: Union[str, Path, dict]) -> SceneConfig:
    """Load and validate a scene configuration.

    Accepts a file path (.json/.yaml/.yml) or a dict directly.
    """
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        if not path.exists():
            raise ConfigError(f"Scene file not found: {path}")

        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ConfigError(
                    "PyYAML is required for YAML scene files. "
                    "Install with: pip install pyyaml"
                )
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ConfigError(f"Unsupported file format: {suffix}. Use .json, .yaml, or .yml")

        if data is None:
            raise ConfigError("Empty scene file")

    required = ["name", "room", "microphone", "source"]
    for key in required:
        if key not in data:
            raise ConfigError(f"Missing required field: '{key}'")

    cfg = SceneConfig(
        name=data["name"],
        description=data.get("description", ""),
        room=_dict_to_room(data["room"]),
        microphone=_dict_to_mic(data["microphone"]),
        source=_dict_to_source(data["source"]),
        motion=_dict_to_motion(data.get("motion", {})),
        output=_dict_to_output(data.get("output", {})),
        noise=_dict_to_noise(data.get("noise", {})),
        localization=_dict_to_localization(data.get("localization", {})),
        evaluation=_dict_to_evaluation(data.get("evaluation", {})),
        experiment=_dict_to_experiment(data.get("experiment", {})),
    )

    return cfg
