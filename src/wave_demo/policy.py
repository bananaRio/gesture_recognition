"""Pure engagement gating; a camera hand track is never a person identity."""

from collections import OrderedDict
from math import isfinite
from uuid import uuid4

from .contracts import InteractionRequest, State, WaveObservation


class EngagementPolicy:
    def __init__(
        self,
        clock_domain: str,
        max_age_seconds: float = 0.25,
        enabled: bool = True,
        dedup_capacity: int = 1024,
    ) -> None:
        if not clock_domain or not isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError("A clock domain and positive freshness limit are required")
        if dedup_capacity < 1:
            raise ValueError("dedup_capacity must be positive")
        self.clock_domain = clock_domain
        self.max_age_seconds = max_age_seconds
        self.enabled = enabled
        self.dedup_capacity = dedup_capacity
        self.rejection_reason: str | None = None
        self._seen: OrderedDict[str, None] = OrderedDict()

    def consider(
        self, observation: WaveObservation, state: State, now: float
    ) -> InteractionRequest | None:
        self.rejection_reason = None
        if (not observation.event_id or not observation.source_id
                or observation.hand_track_id == ""):
            return self._reject("invalid_observation")
        if observation.event_id in self._seen:
            return self._reject("duplicate_event")

        # Consume an event even while busy or disabled: replaying it must not
        # create a later interaction when the application becomes ready.
        self._seen[observation.event_id] = None
        if len(self._seen) > self.dedup_capacity:
            self._seen.popitem(last=False)

        if not self.enabled:
            return self._reject("disabled")
        if observation.clock_domain != self.clock_domain:
            return self._reject("clock_domain_mismatch")
        if not isfinite(now) or not isfinite(observation.captured_at):
            return self._reject("invalid_timestamp")
        age = now - observation.captured_at
        if age < 0:
            return self._reject("future_observation")
        if age > self.max_age_seconds:
            return self._reject("stale_observation")
        if state != State.READY:
            return self._reject("not_ready")
        # This component has no body/person association evidence. In particular,
        # hand_track_id must never be copied into target_person_id.
        return InteractionRequest(session_id=str(uuid4()), observation=observation)

    def _reject(self, reason: str) -> None:
        self.rejection_reason = reason
        return None
