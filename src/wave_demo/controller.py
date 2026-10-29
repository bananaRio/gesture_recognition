"""Single-owner, dependency-free interaction lifecycle and temporal guards."""

from collections.abc import Callable
from math import isfinite

from .config import AppConfig
from .contracts import InteractionRequest, LifecycleEvent, SpeechEvent, State, WaveObservation
from .policy import EngagementPolicy


_ACTIVE = {State.LISTENING, State.TRANSCRIBING, State.THINKING, State.SPEAKING}


class Controller:
    def __init__(
        self,
        config: AppConfig,
        clock_domain: str,
        event_sink: Callable[[LifecycleEvent], None],
    ) -> None:
        self.config = config
        self.policy = EngagementPolicy(clock_domain, config.runtime.observation_max_age_seconds)
        self.event_sink = event_sink
        self.state = State.STARTING
        self.session: InteractionRequest | None = None
        self.transcript = ""
        self.response = ""
        self.error = ""
        self.command: dict | None = None
        self.robot_state: dict | None = None
        self._trigger_event_id: str | None = None
        self._processing_started_at: float | None = None
        self._cooldown_started_at: float | None = None
        self._release_started_at: float | None = None
        self._last_hand_at: float | None = None
        self._last_hand_qualifying: bool | None = None

    @property
    def rejection_reason(self) -> str | None:
        return self.policy.rejection_reason

    def ready(self, now: float) -> None:
        """Finish startup once all devices and models are ready; never clear errors."""
        if self.state == State.STARTING:
            self._transition(State.READY, now, "startup_complete")

    def observe_hand(self, qualifying_palm: bool, now: float) -> None:
        """Observe an actual camera callback, including a fresh no-hand result.

        Calling tick while the camera is stalled cannot establish release.
        A gap breaks the run of observed absence even if both endpoints are empty.
        """
        if not isfinite(now):
            return
        if self._last_hand_at is not None and now <= self._last_hand_at:
            return
        gap_limit = min(
            self.config.wave.max_gap_seconds,
            self.config.runtime.observation_max_age_seconds,
        )
        if self._last_hand_at is None or now - self._last_hand_at > gap_limit:
            self._release_started_at = None
        self._last_hand_at = now
        self._last_hand_qualifying = qualifying_palm
        if qualifying_palm:
            self._release_started_at = None
        elif self._release_started_at is None:
            self._release_started_at = now

    def consider_wave(self, observation: WaveObservation, now: float) -> InteractionRequest | None:
        request = self.policy.consider(observation, self.state, now)
        if request is None:
            self._emit(
                "wave_rejected", now,
                {"event_id": observation.event_id, "reason": self.policy.rejection_reason},
            )
            return None
        self.session = request
        self._trigger_event_id = observation.event_id
        self.transcript = ""
        self.response = ""
        self.error = ""
        self.command = None
        self._processing_started_at = None
        self._cooldown_started_at = None
        self._release_started_at = None
        self._emit(
            "interaction_requested", now,
            {
                "source_id": observation.source_id,
                "hand_track_id": observation.hand_track_id,
                "target_person_id": request.target_person_id,
                "captured_at": observation.captured_at,
                "clock_domain": observation.clock_domain,
            },
        )
        self._transition(State.LISTENING, now, "wave_accepted")
        return request

    def on_worker_event(self, event: SpeechEvent, now: float) -> None:
        if (
            self.session is None
            or event.session_id != self.session.session_id
            or self.state not in _ACTIVE
        ):
            self._emit("worker_event_ignored", now, {
                "worker_session_id": event.session_id, "worker_kind": event.kind,
            })
            return

        kind = event.kind
        if kind == "timeout":
            self.fail(event.error or "Speech processing timed out; restart the application", now)
            return
        if kind == "transcribing" and self.state == State.LISTENING:
            self._processing_started_at = now
            self._transition(State.TRANSCRIBING, now, kind)
        elif kind == "thinking" and self.state in {State.TRANSCRIBING, State.THINKING}:
            if event.text:
                self.transcript = event.text
            self._transition(State.THINKING, now, kind)
        elif kind == "speaking" and self.state in {State.TRANSCRIBING, State.THINKING, State.SPEAKING}:
            self.response = event.text
            if event.command:
                self.command = event.command
                self.robot_state = event.command.get("state")
            self._transition(State.SPEAKING, now, kind)
        elif kind == "transcript":
            self.transcript = event.text
        elif kind == "response":
            self.response = event.text
        elif kind in {"completed", "cancelled", "failed"}:
            if kind == "failed":
                self.error = event.error or event.text or "Speech interaction failed"
            self._emit(kind, now, {"error": self.error, "timings": dict(event.timings)})
            self._enter_cooldown(now, kind)
            return
        else:
            self._emit("worker_event_ignored", now, {"worker_kind": kind, "reason": "unexpected_phase"})
            return
        self._emit("speech_progress", now, {
            "worker_kind": kind, "text": event.text, "timings": dict(event.timings),
            "command": {k: v for k, v in event.command.items() if k != "state"},
        })

    def tick(self, now: float) -> None:
        if not isfinite(now):
            return
        if (
            self.state in _ACTIVE
            and self._processing_started_at is not None
            and now - self._processing_started_at >= self.config.speech.processing_timeout_seconds
        ):
            # A native inference call may still be running. A terminal state
            # prevents overlapping sessions until the process is restarted.
            self.fail("Speech processing timed out; restart the application", now)
            return
        if self.state != State.COOLDOWN or self._cooldown_started_at is None:
            return
        released = (
            self._last_hand_qualifying is False
            and self._last_hand_at is not None
            and 0 <= now - self._last_hand_at <= self.config.runtime.observation_max_age_seconds
            and self._release_started_at is not None
            and self._last_hand_at - self._release_started_at >= self.config.runtime.release_seconds
        )
        if released and now - self._cooldown_started_at >= self.config.runtime.cooldown_seconds:
            self._transition(State.READY, now, "cooldown_and_release_complete")

    def fail(self, message: str, now: float) -> None:
        if self.state in {State.STOPPED, State.ERROR}:
            return
        self.error = message
        self._processing_started_at = None
        self._emit("fatal_error", now, {"error": message})
        self._transition(State.ERROR, now, "fatal_error")

    def cancel(self, now: float) -> None:
        if self.state in _ACTIVE:
            self._emit("cancelled", now)
            self._enter_cooldown(now, "cancelled")

    def close(self, now: float) -> None:
        if self.state != State.STOPPED:
            self.cancel(now)
            self._processing_started_at = None
            self._transition(State.STOPPED, now, "closed")

    def _enter_cooldown(self, now: float, reason: str) -> None:
        self._processing_started_at = None
        self._cooldown_started_at = now
        self._transition(State.COOLDOWN, now, reason)

    def _transition(self, state: State, now: float, reason: str) -> None:
        if self.state == state:
            return
        previous = self.state
        self.state = state
        self._emit("state_changed", now, {"previous": previous.value, "state": state.value, "reason": reason})

    def _emit(self, kind: str, now: float, details: dict | None = None) -> None:
        self.event_sink(LifecycleEvent(
            kind=kind,
            timestamp=now,
            session_id=self.session.session_id if self.session else None,
            trigger_event_id=self._trigger_event_id,
            details={} if details is None else details,
        ))
