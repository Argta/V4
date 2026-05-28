"""Scene configuration dataclasses."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class RoomConfig:
    """Room geometry and acoustic properties."""
    dimensions: List[float]      # [x, y, z] meters
    absorption: float = 0.3      # wall absorption coefficient 0-1
    max_order: int = 8           # reflection order (0 = direct only)
    direct_only: bool = False    # shortcut for absorption=1, max_order=0

    def __post_init__(self):
        if self.direct_only:
            self.absorption = 1.0
            self.max_order = 0
        if len(self.dimensions) != 3:
            raise ValueError(f"dimensions must be [x, y, z], got {self.dimensions}")


@dataclass
class MicrophoneConfig:
    """Binaural microphone array on a spherical head model."""
    head_center: List[float]        # [x, y, z] head center position
    head_radius: float = 0.09       # head radius in meters (~adult human)
    left_ear: Optional[List[float]] = None   # auto-computed if None
    right_ear: Optional[List[float]] = None  # auto-computed if None
    hrtf_mode: str = "analytical"   # analytical | parametric | measured
    hrtf_dataset: str = ""          # SOFA file path (measured mode)
    hrtf_subject: int = 0           # subject index (measured mode)
    head_yaw_deg: float = 0.0       # static head yaw (degrees, +=right)

    def __post_init__(self):
        if len(self.head_center) != 3:
            raise ValueError(f"head_center must be [x, y, z]")
        ear_spacing = self.head_radius * 0.83  # ears ~83% of radius from center
        if self.left_ear is None:
            self.left_ear = [
                self.head_center[0] - ear_spacing,
                self.head_center[1],
                self.head_center[2]
            ]
        if self.right_ear is None:
            self.right_ear = [
                self.head_center[0] + ear_spacing,
                self.head_center[1],
                self.head_center[2]
            ]


@dataclass
class SourceConfig:
    """Sound source signal configuration."""
    generator: str                 # module name in src/signals/
    duration: float                # signal duration in seconds
    sample_rate: int = 44100       # source sample rate
    params: Dict[str, Any] = field(default_factory=dict)  # generator-specific


@dataclass
class MotionConfig:
    """Source motion trajectory."""
    enabled: bool = False
    type: str = "static"           # static | linear | semicircle | circle
    params: Dict[str, Any] = field(default_factory=dict)
    head_yaw_speed: float = 0.0    # deg/s head rotation speed

    def __post_init__(self):
        valid_types = ("static", "linear", "semicircle", "circle", "spiral")
        if self.type not in valid_types:
            raise ValueError(f"motion type must be one of {valid_types}, got '{self.type}'")


@dataclass
class NoiseConfig:
    """Noise injection configuration."""
    enabled: bool = False
    background_snr_db: float = 30.0
    sensor_snr_db: float = 40.0
    noise_type: str = "white"


@dataclass
class LocalizationConfig:
    """Localization algorithm configuration."""
    method: str = "gcc_phat"          # xcorr_itd | gcc_phat | srp_phat
    frame_duration_ms: float = 50.0
    frame_hop_ms: float = 25.0
    freq_range: tuple = (300, 3000)
    head_yaw_speed: float = 0.0      # deg/s, head rotation speed
    active_head: bool = False        # enable active head-rotation FB detection


@dataclass
class EvaluationConfig:
    """Evaluation settings."""
    enabled: bool = True
    save_report: bool = True
    save_plots: bool = True


@dataclass
class ExperimentConfig:
    """Parameter sweep experiment settings."""
    enabled: bool = False
    type: str = "single"
    sweep_variable: str = ""
    sweep_values: list = field(default_factory=list)


@dataclass
class OutputConfig:
    """Output settings."""
    sample_rate: int = 44100
    visualize: bool = True


@dataclass
class SceneConfig:
    """Complete scene definition for a single simulation run."""
    name: str
    description: str = ""
    room: RoomConfig = field(default_factory=lambda: RoomConfig([5, 4, 3]))
    microphone: MicrophoneConfig = field(
        default_factory=lambda: MicrophoneConfig([2.5, 2.0, 1.5])
    )
    source: SourceConfig = field(
        default_factory=lambda: SourceConfig(generator="sine", duration=1.0)
    )
    motion: MotionConfig = field(default_factory=MotionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
