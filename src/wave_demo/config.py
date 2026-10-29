from dataclasses import dataclass, field, fields
from pathlib import Path
import os
import math
import tomllib


@dataclass(frozen=True)
class WaveConfig:
    fps: float = 20.0
    open_palm_threshold: float = 0.65
    use_landmark_openness: bool = True
    min_extended_fingers: int = 3
    arm_seconds: float = 0.15
    min_palm_pixels: float = 35.0
    window_seconds: float = 2.2
    segment_palm_widths: float = 0.4
    min_observations: int = 8
    min_duration_seconds: float = 0.4
    max_gap_seconds: float = 0.2
    smoothing_seconds: float = 0.08
    jump_palm_widths: float = 1.5


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    min_speech_seconds: float = 0.25
    pre_roll_seconds: float = 0.2
    end_silence_seconds: float = 0.7
    start_timeout_seconds: float = 10.0
    max_utterance_seconds: float = 15.0


@dataclass(frozen=True)
class SpeechConfig:
    whisper_repo: str = "Systran/faster-whisper-base.en"
    llm_repo: str = "mlx-community/Qwen3-1.7B-4bit"
    voice: str = "Samantha"
    words_per_minute: int = 180
    max_tokens: int = 96
    cpu_threads: int = 4
    processing_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class RuntimeConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    cooldown_seconds: float = 2.0
    release_seconds: float = 0.5
    observation_max_age_seconds: float = 0.25
    mirror_preview: bool = True


@dataclass(frozen=True)
class AppConfig:
    wave: WaveConfig = field(default_factory=WaveConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def project_root() -> Path:
    return Path(os.environ.get("WAVE_DEMO_HOME", Path(__file__).resolve().parents[2])).expanduser().resolve()


def load_config(path: Path | None = None) -> AppConfig:
    selected = path or project_root() / "config.toml"
    if path is not None and not selected.is_file():
        raise FileNotFoundError(f"Configuration not found: {selected}")
    data = tomllib.loads(selected.read_text()) if selected.exists() else {}
    types = {"wave": WaveConfig, "audio": AudioConfig, "speech": SpeechConfig, "runtime": RuntimeConfig}
    if unknown := data.keys() - types.keys():
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    sections = {}
    for name, cls in types.items():
        values = data.get(name, {})
        allowed = {f.name for f in fields(cls)}
        if unknown := values.keys() - allowed:
            raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
        section = cls(**values)
        for f in fields(section):
            value = getattr(section, f.name)
            expected = type(getattr(cls(), f.name))
            if expected is float:
                valid = type(value) in (float, int)
            else:
                valid = type(value) is expected
            if not valid:
                raise ValueError(f"{name}.{f.name} has invalid type")
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                if not math.isfinite(value):
                    raise ValueError(f"{name}.{f.name} must be finite")
                if value < 0 or (value == 0 and f.name != "camera_index"):
                    raise ValueError(f"{name}.{f.name} must be positive")
        sections[name] = section
    config = AppConfig(**sections)
    if config.audio.sample_rate != 16000:
        raise ValueError("The VAD/ASR pipeline requires sample_rate = 16000")
    if config.wave.open_palm_threshold > 1 or config.audio.vad_threshold > 1:
        raise ValueError("Confidence thresholds must be between 0 and 1")
    if not 1 <= config.wave.min_extended_fingers <= 4:
        raise ValueError("min_extended_fingers must be between 1 and 4")
    return config
