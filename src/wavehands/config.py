from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 360


@dataclass(frozen=True)
class HandTrackerConfig:
    max_num_hands: int = 2
    model_complexity: int = 0
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    inference_width: int = 256
    draw_landmarks: bool = False
    pinch_threshold_ratio: float = 0.38


@dataclass(frozen=True)
class SelectionConfig:
    hover_seconds: float = 0.08
    cooldown_seconds: float = 0.03
    stable_frames: int = 1


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 44100
    block_size: int = 64
    master_gain: float = 0.3
    attack_seconds: float = 0.01
    release_seconds: float = 0.035


@dataclass(frozen=True)
class MetricsConfig:
    enabled: bool = True
    log_interval_seconds: float = 2.0


@dataclass(frozen=True)
class UIConfig:
    panel_width: int = 320


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = CameraConfig()
    tracker: HandTrackerConfig = HandTrackerConfig()
    selection: SelectionConfig = SelectionConfig()
    audio: AudioConfig = AudioConfig()
    ui: UIConfig = UIConfig()
    metrics: MetricsConfig = MetricsConfig()


def default_config() -> AppConfig:
    return AppConfig()
