from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wave_demo.audio import AudioCapture, UtteranceSegmenter
from wave_demo.config import AudioConfig


def feed(segmenter, count, score, start=0.0, value=1.0):
    result = None
    for i in range(count):
        current = segmenter.feed(np.full(512, value, np.float32), score, start + i * 0.032)
        if current is not None:
            result = current
    return result


def test_minimum_speech_does_not_clip_onset_and_emits_once():
    segmenter = UtteranceSegmenter(AudioConfig())
    segmenter.begin("session", 0.0)
    feed(segmenter, 10, 0.0, value=0.0)
    assert feed(segmenter, 8, 0.9, start=0.32, value=0.75) is None
    result = feed(segmenter, 22, 0.0, start=0.576, value=0.0)
    assert result.session_id == "session"
    assert result.reason == "speech"
    assert result.ended_at == pytest.approx(0.576)
    assert np.count_nonzero(result.samples == 0.75) == 8 * 512
    assert result.samples[:7 * 512].sum() == 0
    assert feed(segmenter, 30, 0.9, start=1.28) is None


def test_click_and_silence_do_not_create_an_utterance():
    segmenter = UtteranceSegmenter(AudioConfig())
    segmenter.begin("quiet", 0.0)
    feed(segmenter, 3, 0.9)
    feed(segmenter, 20, 0.0, start=0.096)
    result = segmenter.tick(10.0)
    assert result.reason == "no_speech"
    assert result.samples.size == 0
    assert segmenter.tick(11) is None


def test_new_session_cannot_receive_previous_audio():
    segmenter = UtteranceSegmenter(AudioConfig())
    segmenter.begin("old", 0.0)
    feed(segmenter, 10, 0.9, value=0.5)
    segmenter.begin("new", 1.0)
    feed(segmenter, 8, 0.9, start=1.0, value=0.25)
    result = feed(segmenter, 22, 0.0, start=1.256, value=0.0)
    assert result.session_id == "new"
    assert not np.any(result.samples == 0.5)


def test_duration_cap_bounds_recording():
    config = replace(AudioConfig(), max_utterance_seconds=0.3)
    segmenter = UtteranceSegmenter(config)
    segmenter.begin("long", 0.0)
    result = feed(segmenter, 12, 0.9)
    assert result.reason == "max_duration"
    assert result.samples.size == 4800
    assert result.ended_at == pytest.approx(0.3)


def test_capture_discontinuity_invalidates_partial_speech():
    segmenter = UtteranceSegmenter(AudioConfig())
    segmenter.begin("gap", 0.0)
    feed(segmenter, 8, 0.9)
    with pytest.raises(RuntimeError, match="discontinuity"):
        feed(segmenter, 1, 0.9, start=1.0)
    assert segmenter.tick(20.0) is None


def test_vad_input_validation():
    segmenter = UtteranceSegmenter(AudioConfig())
    segmenter.begin("invalid", 0.0)
    with pytest.raises(ValueError, match="512"):
        segmenter.feed(np.zeros(480), 0.9, 0.0)
    with pytest.raises(ValueError, match="probability"):
        segmenter.feed(np.zeros(512), float("nan"), 0.0)


def test_capture_callback_drops_samples_outside_session():
    capture = AudioCapture(AudioConfig())
    capture._native_rate = 48000
    audio = np.ones((480, 1), dtype=np.float32)
    timing = SimpleNamespace(inputBufferAdcTime=1.0, currentTime=1.01)
    capture._callback(audio, 480, timing, None)
    assert capture._chunks.empty()
    capture._request = (1, "active", 0.0)
    capture._callback(audio, 480, timing, None)
    epoch, _, copied = capture._chunks.get_nowait()
    audio[:] = 0
    assert epoch == 1
    assert copied.sum() == 480
    capture._finished_epoch = 1
    capture._callback(audio, 480, timing, None)
    assert capture._chunks.empty()


def test_capture_queue_overflow_is_visible_not_silently_corrupted():
    capture = AudioCapture(AudioConfig())
    capture._native_rate = 48000
    capture._request = (1, "active", 0.0)
    timing = SimpleNamespace(inputBufferAdcTime=1.0, currentTime=1.01)
    for _ in range(33):
        capture._callback(np.zeros((480, 1), np.float32), 480, timing, None)
    with pytest.raises(RuntimeError, match="overflow"):
        capture.poll()
