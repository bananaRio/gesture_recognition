"""Timestamped wave evidence, independent of cameras and dialogue policy.

MediaPipe supplies hand landmarks and an Open_Palm score. A hand qualifies as
open from that score or from extended fingers in the landmarks. While it stays
open, the detector tracks two signals over a short window: palm-centre
translation and fingertip swing relative to the wrist. Either signal must show
repeated left-right reversals (three alternating excursions, in palm widths)
to emit one observation, so a stationary open hand or a single swipe cannot.

The two motion signals are deliberately evaluated separately. A temporary hand
track describes continuity in one camera, not a person or an engagement intent.
"""

from collections import deque
from math import cos, dist, exp, hypot, isfinite, radians
from statistics import median
from uuid import uuid4

from .config import WaveConfig
from .contracts import HandSample, WaveObservation


def _extended_fingers(points: list[tuple[float, float, float]]) -> int:
    """Count extended non-thumb fingers in consistently scaled 3D coordinates.

    Joint straightness alone also describes a finger folded toward the wrist,
    so an extended fingertip must additionally reach farther than its PIP joint.
    This is geometric evidence, not a replacement classifier confidence score.
    """
    def straight(a, joint, b, minimum_degrees):
        first = tuple(x - y for x, y in zip(a, joint))
        second = tuple(x - y for x, y in zip(b, joint))
        denominator = hypot(*first) * hypot(*second)
        if denominator <= 1e-6:
            return False
        cosine = sum(x * y for x, y in zip(first, second)) / denominator
        return cosine <= cos(radians(minimum_degrees))

    wrist = points[0]
    return sum(
        straight(points[mcp], points[pip], points[dip], 150)
        and straight(points[pip], points[dip], points[tip], 140)
        and dist(points[tip], wrist) > 1.1 * dist(points[pip], wrist)
        for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12),
                                  (13, 14, 15, 16), (17, 18, 19, 20))
    )


class WaveDetector:
    def __init__(self, config: WaveConfig, source_id: str, clock_domain: str):
        self.config = config
        self.source_id = source_id
        self.clock_domain = clock_domain
        self.reset()

    def reset(self) -> None:
        """Discard tracking, accumulated evidence and the one-shot latch."""
        self._diagnostics = {
            "reason": "no_hand", "hand_visible": False, "open_palm_score": 0.0,
            "palm_width_px": 0.0, "samples": 0, "duration": 0.0,
            "translation_segments": 0, "wrist_segments": 0, "armed": False,
            "extended_fingers": 0, "open_hand_source": "none", "open_hand": False,
        }
        self.qualifying_palm = False
        self._last_timestamp: float | None = None
        self._lost_since: float | None = None
        self._track_id: str | None = None
        self._previous: tuple[float, float, float, int, int] | None = None
        self._handedness = ""
        self._opposite_count = 0
        self._fired = False
        self._last_qualified_at: float | None = None
        self._clear_motion()

    @property
    def diagnostics(self) -> dict:
        """Latest accepted-frame evidence, not an engagement probability."""
        return dict(self._diagnostics)

    def _clear_motion(self) -> None:
        self._armed_since: float | None = None
        self._widths: list[float] = []
        self._scale: float | None = None
        self._history: deque[tuple[float, float, float]] = deque()
        self._diagnostics.update(samples=0, duration=0.0, translation_segments=0,
                                 wrist_segments=0, armed=False)

    def _new_track(self) -> None:
        self._track_id = uuid4().hex
        self._previous = None
        self._handedness = ""
        self._opposite_count = 0
        self._last_qualified_at = None
        self._clear_motion()

    def _lose(self, timestamp: float, reason: str = "no_hand") -> None:
        self.qualifying_palm = False
        self._diagnostics["reason"] = reason
        # Missing/invalid positions still break all motion evidence. A brief
        # openness dip with continuous valid positions is handled before
        # reaching here and does not fabricate motion across missing frames.
        self._clear_motion()
        self._last_qualified_at = None
        if self._lost_since is None:
            self._lost_since = timestamp
        if timestamp - self._lost_since >= self.config.max_gap_seconds:
            self._track_id = None
            self._previous = None
            self._handedness = ""
            self._opposite_count = 0
            self._fired = False

    @staticmethod
    def _segments(values: list[float], threshold: float) -> int:
        """Count alternating excursions, ignoring sub-threshold jitter."""
        if not values:
            return 0
        anchor = extreme = values[0]
        direction = 0
        count = 0
        for value in values[1:]:
            if direction == 0:
                if abs(value - anchor) >= threshold:
                    direction = 1 if value > anchor else -1
                    extreme = value
                    count = 1
            elif (value - extreme) * direction >= 0:
                extreme = value
            elif (extreme - value) * direction >= threshold:
                direction *= -1
                extreme = value
                count += 1
        return count

    def update(self, sample: HandSample | None, now: float) -> WaveObservation | None:
        """Consume one result (or explicit loss); timestamps are monotonic seconds.

        ``now`` timestamps an explicit missing result. A present sample uses its
        capture time, rather than the possibly delayed time of this call. The
        caller applies the application's observation-age policy.
        """
        timestamp = sample.captured_at if sample is not None else now
        if not isfinite(timestamp):
            self.qualifying_palm = False
            self._diagnostics.update(reason="no_hand", hand_visible=False,
                                     open_palm_score=0.0, palm_width_px=0.0,
                                     extended_fingers=0, open_hand_source="none", open_hand=False)
            self._clear_motion()
            return None
        if self._last_timestamp is not None:
            if timestamp <= self._last_timestamp:
                return None  # Late/duplicate callbacks cannot alter current state.
            if timestamp - self._last_timestamp > self.config.max_gap_seconds:
                self._new_track()
                self._fired = False
                self._lost_since = None
        self._last_timestamp = timestamp
        if sample is None:
            self._diagnostics.update(hand_visible=False, open_palm_score=0.0, palm_width_px=0.0,
                                     extended_fingers=0, open_hand_source="none", open_hand=False)
            self._lose(timestamp)
            return None

        points = sample.landmarks
        if (len(points) != 21 or sample.width <= 0 or sample.height <= 0
                or not isfinite(sample.open_palm_score)
                or any(len(p) != 3 or not all(isfinite(v) for v in p) for p in points)):
            self._diagnostics.update(hand_visible=False, open_palm_score=0.0, palm_width_px=0.0,
                                     extended_fingers=0, open_hand_source="none", open_hand=False)
            self._lose(timestamp)
            return None
        x = [p[0] * sample.width for p in points]
        y = [p[1] * sample.height for p in points]
        # MediaPipe's normalized depth uses the same scale as image x.
        pixels = [(px, py, p[2] * sample.width) for px, py, p in zip(x, y, points)]
        extended_fingers = _extended_fingers(pixels)
        classifier_open = sample.open_palm_score >= self.config.open_palm_threshold
        landmark_open = (self.config.use_landmark_openness
                         and extended_fingers >= self.config.min_extended_fingers)
        open_hand = classifier_open or landmark_open
        open_hand_source = "classifier" if classifier_open else "landmarks" if landmark_open else "none"
        width = hypot(x[5] - x[17], y[5] - y[17])
        self._diagnostics.update(hand_visible=True, open_palm_score=sample.open_palm_score,
                                 palm_width_px=width, extended_fingers=extended_fingers,
                                 open_hand_source=open_hand_source, open_hand=open_hand)
        if width < self.config.min_palm_pixels:
            self._lose(timestamp, "too_small")
            return None
        grace = (self._scale is not None and self._last_qualified_at is not None
                 and timestamp - self._last_qualified_at <= self.config.max_gap_seconds + 1e-9)
        if not open_hand and not grace:
            self._lose(timestamp, "show_open_palm")
            return None

        center_x = sum(x[i] for i in (0, 5, 9, 13, 17)) / 5
        center_y = sum(y[i] for i in (0, 5, 9, 13, 17)) / 5
        if self._lost_since is not None:
            if timestamp - self._lost_since >= self.config.max_gap_seconds:
                self._track_id = None
                self._fired = False
            self._lost_since = None
        if self._track_id is None:
            self._new_track()
        if self._previous is not None:
            px, py, previous_width, frame_width, frame_height = self._previous
            discontinuity = (
                (sample.width, sample.height) != (frame_width, frame_height)
                or hypot(center_x - px, center_y - py)
                > max(self.config.jump_palm_widths * previous_width, 0.15 * sample.width)
                or abs(width / previous_width - 1.0) > 0.4
            )
            if discontinuity:
                self._new_track()
        if (self._handedness and sample.handedness
                and self._handedness != sample.handedness):
            self._opposite_count += 1
            # Handedness is secondary to spatial continuity. Single-frame label
            # flips must not erase an otherwise continuously observed wave.
            if self._opposite_count >= 3:
                self._new_track()
        else:
            self._opposite_count = 0
        if sample.handedness and (not self._handedness or self._opposite_count == 0):
            self._handedness = sample.handedness
        self._previous = center_x, center_y, width, sample.width, sample.height
        self.qualifying_palm = open_hand
        if open_hand:
            self._last_qualified_at = timestamp
        elif self._scale is None:
            # A geometric/handedness discontinuity must not inherit the old
            # track's openness grace and arm an unqualified new track.
            self._lose(timestamp, "show_open_palm")
            return None
        if self._fired:
            self._diagnostics["reason"] = "wave_detected"
            return None

        if self._armed_since is None:
            self._armed_since = timestamp
        self._widths.append(width)
        if timestamp - self._armed_since + 1e-9 < self.config.arm_seconds:
            self._diagnostics["reason"] = "arming"
            return None
        if self._scale is None:
            self._scale = median(self._widths)
        self._widths.clear()
        translation = center_x / self._scale
        rotation = (sum(x[i] for i in (8, 12, 16, 20)) / 4 - x[0]) / self._scale
        if self._history:
            previous_time, previous_translation, previous_rotation = self._history[-1]
            alpha = 1.0 - exp(-(timestamp - previous_time) / self.config.smoothing_seconds)
            translation = previous_translation + alpha * (translation - previous_translation)
            rotation = previous_rotation + alpha * (rotation - previous_rotation)
        self._history.append((timestamp, translation, rotation))
        cutoff = timestamp - self.config.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        segment_counts = {
            "translation_segments": self._segments([row[1] for row in self._history],
                                                    self.config.segment_palm_widths),
            "wrist_segments": self._segments([row[2] for row in self._history],
                                              self.config.segment_palm_widths),
        }
        self._diagnostics.update(
            reason="wave_more" if open_hand else "show_open_palm",
            samples=len(self._history), duration=timestamp - self._history[0][0],
            armed=True, **segment_counts,
        )
        if (len(self._history) < self.config.min_observations
                or timestamp - self._history[0][0] + 1e-9 < self.config.min_duration_seconds):
            return None

        # Openness uncertainty can retain measured positions, but cannot
        # itself authorize a trigger or a hand identity change.
        if not open_hand or self._opposite_count:
            return None

        for index, signal in ((1, "translation"), (2, "wrist_relative")):
            values = [row[index] for row in self._history]
            segments = self._segments(values, self.config.segment_palm_widths)
            if segments < 3:
                continue
            self._fired = True
            self._diagnostics["reason"] = "wave_detected"
            region = (
                max(0.0, min(1.0, min(p[0] for p in points))),
                max(0.0, min(1.0, min(p[1] for p in points))),
                max(0.0, min(1.0, max(p[0] for p in points))),
                max(0.0, min(1.0, max(p[1] for p in points))),
            )
            return WaveObservation(
                event_id=uuid4().hex,
                source_id=self.source_id,
                captured_at=timestamp,
                clock_domain=self.clock_domain,
                hand_track_id=self._track_id,
                region=region,
                evidence={
                    "signal": signal,
                    "segments": segments,
                    "reversals": segments - 1,
                    "amplitude_palm_widths": max(values) - min(values),
                    "segment_threshold_palm_widths": self.config.segment_palm_widths,
                    "duration_seconds": timestamp - self._history[0][0],
                    "observations": len(self._history),
                    "open_palm_score": sample.open_palm_score,
                    "extended_fingers": extended_fingers,
                    "open_hand_source": open_hand_source,
                    "handedness": sample.handedness,
                },
            )
        return None
