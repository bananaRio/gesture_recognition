"""Local microphone capture and bounded, session-scoped utterance detection.

PortAudio only copies samples into a bounded queue. Resampling and VAD run on
one background thread; audio outside an explicit listening session is dropped.
"""

from collections import deque
import math
import queue
import threading
import time

from .config import AudioConfig
from .contracts import Utterance


class UtteranceSegmenter:
    """Pure segmentation: caller supplies 512-sample frames and VAD scores.

    ``captured_at`` is the monotonic capture time of the frame's first sample.
    A candidate onset keeps its own frames so minimum-speech qualification does
    not remove the first phonemes when it exceeds the pre-roll duration.
    ``Utterance.ended_at`` identifies the last VAD-positive sample, before the
    trailing-silence endpointing delay; no-speech results use timeout time.
    """

    def __init__(self, config: AudioConfig):
        if config.sample_rate != 16000:
            raise ValueError("Silero VAD requires 16 kHz input")
        self.config = config
        self.cancel()

    def begin(self, session_id: str, started_at: float) -> None:
        self.cancel()
        self._session_id = session_id
        self._started_at = started_at

    def cancel(self) -> None:
        self._session_id = None
        self._started_at = 0.0
        self._pre_roll = deque(maxlen=math.ceil(self.config.pre_roll_seconds * 16000 / 512))
        self._candidate = []
        self._candidate_voiced = 0
        self._frames = []
        self._sample_count = 0
        self._silence_count = 0
        self._last_frame_end = None
        self._last_voiced_end = None
        self._audio_started_at = None
        self._speaking = False

    def tick(self, now: float) -> Utterance | None:
        if (self._session_id is not None and not self._speaking
                and now - self._started_at >= self.config.start_timeout_seconds):
            return self._finish(now, "no_speech")
        return None

    def feed(self, samples, vad_score: float, captured_at: float) -> Utterance | None:
        import numpy as np

        if self._session_id is None:
            return None
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (512,):
            raise ValueError("VAD frames must contain exactly 512 mono samples")
        if not math.isfinite(vad_score) or not 0.0 <= vad_score <= 1.0:
            raise ValueError("VAD score must be a finite probability")
        if not math.isfinite(captured_at):
            raise ValueError("Capture timestamp must be finite")
        if self._last_frame_end is not None and abs(captured_at - self._last_frame_end) > 0.1:
            self.cancel()
            raise RuntimeError("Audio capture discontinuity; discarded incomplete utterance")
        self._last_frame_end = captured_at + 512 / self.config.sample_rate
        frame = samples.copy()
        voiced = vad_score >= self.config.vad_threshold
        if voiced:
            self._last_voiced_end = self._last_frame_end
        if self._speaking:
            self._frames.append(frame)
            self._sample_count += len(frame)
            self._silence_count = 0 if voiced else self._silence_count + len(frame)
        elif voiced:
            if not self._candidate:
                self._candidate = list(self._pre_roll)
                self._audio_started_at = captured_at - len(self._pre_roll) * 512 / self.config.sample_rate
            self._candidate.append(frame)
            self._candidate_voiced += len(frame)
            if self._candidate_voiced >= self.config.min_speech_seconds * self.config.sample_rate:
                self._speaking = True
                self._frames = self._candidate
                self._sample_count = sum(len(part) for part in self._frames)
                self._candidate = []
        else:
            self._candidate = []
            self._candidate_voiced = 0
        self._pre_roll.append(frame)
        if self._speaking:
            if self._sample_count >= self.config.max_utterance_seconds * self.config.sample_rate:
                return self._finish(self._last_frame_end, "max_duration")
            if self._silence_count >= self.config.end_silence_seconds * self.config.sample_rate:
                return self._finish(self._last_frame_end, "speech")
        return self.tick(self._last_frame_end)

    def _finish(self, ended_at: float, reason: str) -> Utterance:
        import numpy as np

        samples = (np.concatenate(self._frames) if self._frames
                   else np.empty(0, dtype=np.float32))
        samples = samples[:int(self.config.max_utterance_seconds * self.config.sample_rate)]
        if len(samples) and self._last_voiced_end is not None:
            ended_at = min(
                self._last_voiced_end,
                self._audio_started_at + len(samples) / self.config.sample_rate,
            )
        result = Utterance(self._session_id, samples, ended_at, reason)
        self.cancel()
        return result


class AudioCapture:
    def __init__(self, config: AudioConfig, microphone=None):
        self.config = config
        self.microphone = microphone
        self._chunks = queue.Queue(maxsize=32)
        self._results = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._request = (0, None, 0.0)
        self._finished_epoch = -1
        self._fault = None
        self._stop = threading.Event()
        self._stream = None
        self._thread = None
        self._last_callback = 0.0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Audio capture is already started")
        import numpy as np
        import onnxruntime
        import sounddevice as sd
        import soxr
        from silero_vad import load_silero_vad

        self._np = np
        onnxruntime.disable_telemetry_events()
        self._vad = load_silero_vad(onnx=True)
        device = sd.query_devices(self.microphone, "input")
        self.device_name = device["name"]
        self._native_rate = int(device["default_samplerate"])
        self._resampler = soxr.ResampleStream(
            self._native_rate, self.config.sample_rate, 1, dtype="float32", quality="HQ"
        )
        self._stream = sd.InputStream(
            device=self.microphone, samplerate=self._native_rate,
            channels=1, dtype="float32", blocksize=0, callback=self._callback,
        )
        self._last_callback = time.monotonic()
        try:
            self._stream.start()
            self._thread = threading.Thread(target=self._run, name="wave-audio", daemon=True)
            self._thread.start()
        except Exception:
            self._stream.close()
            self._stream = None
            raise

    def _callback(self, indata, frames, time_info, status) -> None:
        now = time.monotonic()
        self._last_callback = now
        epoch, session_id, _ = self._request
        if session_id is None or epoch == self._finished_epoch:
            return
        if status:
            self._fault = f"Microphone capture error: {status}"
            return
        adc = float(time_info.inputBufferAdcTime)
        current = float(time_info.currentTime)
        captured_at = now + adc - current if adc and current else now - frames / self._native_rate
        try:
            self._chunks.put_nowait((epoch, captured_at, indata[:, 0].copy()))
        except queue.Full:
            self._fault = "Microphone input queue overflow; incomplete audio was discarded"

    def begin_session(self, session_id: str) -> None:
        if self._stream is None or self._stop.is_set():
            raise RuntimeError("Audio capture is not running")
        if self._fault:
            raise RuntimeError(self._fault)
        with self._lock:
            epoch, _, _ = self._request
            self._request = (epoch + 1, session_id, time.monotonic())

    def cancel_session(self) -> None:
        with self._lock:
            epoch, _, _ = self._request
            self._request = (epoch + 1, None, time.monotonic())

    def poll(self) -> Utterance | None:
        if self._fault:
            raise RuntimeError(self._fault)
        try:
            epoch, result = self._results.get_nowait()
        except queue.Empty:
            return None
        return result if epoch == self._request[0] else None

    def _deliver(self, epoch: int, result: Utterance) -> None:
        self._finished_epoch = epoch
        try:
            self._results.get_nowait()
        except queue.Empty:
            pass
        self._results.put_nowait((epoch, result))

    def _run(self) -> None:
        import torch

        np = self._np
        segmenter = UtteranceSegmenter(self.config)
        current_epoch = -1
        pending = np.empty(0, dtype=np.float32)
        first_sample_at = None
        output_samples = 0
        last_input_end = None
        try:
            while not self._stop.is_set():
                if self._fault:
                    return
                now = time.monotonic()
                if now - self._last_callback > 3.0:
                    raise RuntimeError("Microphone input stalled for more than three seconds")
                epoch, session_id, started_at = self._request
                if epoch != current_epoch:
                    current_epoch = epoch
                    segmenter.cancel()
                    self._vad.reset_states()
                    self._resampler.clear()
                    pending = np.empty(0, dtype=np.float32)
                    first_sample_at = last_input_end = None
                    output_samples = 0
                    if session_id is not None:
                        segmenter.begin(session_id, started_at)
                if result := segmenter.tick(now):
                    self._deliver(epoch, result)
                try:
                    chunk_epoch, captured_at, samples = self._chunks.get(timeout=0.05)
                except queue.Empty:
                    continue
                if (chunk_epoch != epoch or session_id is None
                        or epoch == self._finished_epoch):
                    continue
                if captured_at < started_at:
                    trim = min(len(samples), math.ceil((started_at - captured_at) * self._native_rate))
                    samples = samples[trim:]
                    captured_at += trim / self._native_rate
                if not len(samples):
                    continue
                if last_input_end is not None and abs(captured_at - last_input_end) > 0.1:
                    raise RuntimeError("Microphone capture gap; incomplete utterance was discarded")
                last_input_end = captured_at + len(samples) / self._native_rate
                if first_sample_at is None:
                    first_sample_at = captured_at
                resampled = self._resampler.resample_chunk(samples)
                pending = np.concatenate((pending, resampled))
                while len(pending) >= 512:
                    frame, pending = pending[:512], pending[512:]
                    score = float(self._vad(torch.from_numpy(frame.copy()), 16000).item())
                    frame_at = first_sample_at + output_samples / 16000
                    output_samples += 512
                    result = segmenter.feed(frame, score, frame_at)
                    if result is not None:
                        self._deliver(epoch, result)
                        break
        except Exception as exc:
            self._fault = f"Audio processing failed: {exc}"

    def close(self) -> None:
        self.cancel_session()
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
