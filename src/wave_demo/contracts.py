"""Transport-neutral messages. Image coordinates are unmirrored and normalized.

Timestamps are monotonic seconds in an explicitly identified local clock domain.
Hand tracks are temporary camera tracks, never person identities.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class State(StrEnum):
    STARTING = "STARTING"
    READY = "READY"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    COOLDOWN = "COOLDOWN"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class WaveObservation:
    event_id: str
    source_id: str
    captured_at: float
    clock_domain: str
    hand_track_id: str | None
    region: tuple[float, float, float, float] | None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InteractionRequest:
    session_id: str
    observation: WaveObservation
    target_person_id: str | None = None


@dataclass(frozen=True)
class LifecycleEvent:
    kind: str
    timestamp: float
    session_id: str | None = None
    trigger_event_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpeechEvent:
    session_id: str
    kind: str
    text: str = ""
    error: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    # Routing result: validated call (or None), outcome and simulator state.
    command: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HandSample:
    captured_at: float
    landmarks: tuple[tuple[float, float, float], ...]
    open_palm_score: float
    handedness: str
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class Utterance:
    session_id: str
    samples: Any  # 1-D float32 numpy array, 16 kHz; core does not import numpy.
    ended_at: float  # Monotonic final voiced sample; no_speech uses timeout time.
    reason: str = "speech"
