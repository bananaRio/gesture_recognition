"""Small CPU MediaPipe adapter; heavy dependencies load only on construction."""

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Callable

from .contracts import HandSample


def hand_sample(result, captured_at: float, width: int, height: int) -> HandSample | None:
    """Convert the first recognized hand; landmark coordinates stay unmirrored."""
    if not result.hand_landmarks:
        return None
    gestures = result.gestures[0] if result.gestures else []
    return HandSample(
        captured_at=captured_at,
        landmarks=tuple((float(p.x), float(p.y), float(p.z)) for p in result.hand_landmarks[0]),
        open_palm_score=max((float(g.score) for g in gestures if g.category_name == "Open_Palm"),
                            default=0.0),
        handedness=(result.handedness[0][0].category_name
                    if result.handedness and result.handedness[0] else ""),
        width=width,
        height=height,
    )


class HandRecognizer:
    def __init__(self, model_path: str | Path,
                 callback: Callable[[HandSample | None, float], None]):
        import cv2
        import mediapipe as mp

        self._cv2 = cv2
        self._mp = mp
        self._callback = callback
        self._lock = Lock()
        self._submit_lock = Lock()
        self._closed = False
        self._last_submitted_ms = -1
        self._last_completed_ms = -1
        self._capture_times: OrderedDict[int, float] = OrderedDict()
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            result_callback=self._receive,
        )
        self._recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

    def submit(self, frame, captured_at: float) -> bool:
        """Submit an unmirrored OpenCV BGR frame without waiting for inference.

        Returns false when closed or the timestamp is not newer. MediaPipe may
        itself drop a submitted frame while busy; this is normal backpressure.
        """
        timestamp_ms = int(captured_at * 1000)
        with self._submit_lock:
            with self._lock:
                if self._closed or timestamp_ms <= self._last_submitted_ms:
                    return False
                self._last_submitted_ms = timestamp_ms
                self._capture_times[timestamp_ms] = captured_at
                # Busy-frame drops do not invoke callbacks. Bound their metadata.
                while len(self._capture_times) > 128:
                    self._capture_times.popitem(last=False)
            rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            try:
                self._recognizer.recognize_async(image, timestamp_ms)
            except Exception:
                with self._lock:
                    self._capture_times.pop(timestamp_ms, None)
                raise
        return True

    def _receive(self, result, output_image, timestamp_ms: int) -> None:
        with self._lock:
            if self._closed or timestamp_ms <= self._last_completed_ms:
                return
            self._last_completed_ms = timestamp_ms
            captured_at = self._capture_times.pop(timestamp_ms, timestamp_ms / 1000)
            while self._capture_times and next(iter(self._capture_times)) < timestamp_ms:
                self._capture_times.popitem(last=False)
        sample = hand_sample(result, captured_at, output_image.width, output_image.height)
        self._callback(sample, captured_at)

    def close(self) -> None:
        """Stop native inference; safe to call more than once."""
        with self._submit_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._capture_times.clear()
            # Never hold the callback lock while waiting for native shutdown.
            self._recognizer.close()
