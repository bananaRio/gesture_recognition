"""Spawn ownership, cancellation, and preview coordinates without devices."""

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from types import SimpleNamespace

import numpy as np
import pytest

from wave_demo.app import describe_command, display_point, render_preview
from wave_demo.config import AppConfig
from wave_demo.contracts import HandSample, SpeechEvent, State, Utterance
from wave_demo.vision import hand_sample
from wave_demo.worker import SpeechWorker


class FakeBackend:
    def __init__(self, config, assets_dir):
        self.root = Path(assets_dir)
        self.owner = None
        self.cancelled = threading.Event()

    def load(self):
        self.owner = (os.getpid(), threading.get_ident())

    def warmup(self):
        assert self.owner == (os.getpid(), threading.get_ident())

    def process(self, utterance, emit):
        assert self.owner == (os.getpid(), threading.get_ident())
        emit(SpeechEvent(utterance.session_id, "transcribing"))
        if utterance.reason == "hang":
            (self.root / "started").write_text("started")
            time.sleep(30)
        if utterance.reason == "cooperative":
            (self.root / "started").write_text("started")
            self.cancelled.wait(timeout=30)
            return
        if utterance.reason in {"child", "orphan"}:
            child = subprocess.Popen([
                sys.executable, "-c",
                "import time,pathlib; time.sleep(1); pathlib.Path('late-playback').write_text('played')",
            ], cwd=self.root)
            (self.root / "started").write_text(str(child.pid))
            if utterance.reason == "orphan":
                os._exit(0)
            child.wait(timeout=10)
        (self.root / "processed").write_text(utterance.session_id)
        emit(SpeechEvent(utterance.session_id, "completed", text=json.dumps({
            "pid": os.getpid(), "thread": threading.get_ident(),
            "group": os.getpgrp(), "speech_imported": "wave_demo.speech" in sys.modules,
        })))

    def close(self):
        (self.root / "closed").write_text("closed")

    def cancel(self):
        self.cancelled.set()


class FailingBackend(FakeBackend):
    def load(self):
        raise RuntimeError("fixture model missing")


@pytest.fixture
def worker(tmp_path):
    instance = SpeechWorker(None, tmp_path, backend_factory=FakeBackend)
    yield instance
    instance.close()


def wait_for(path):
    deadline = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < deadline, f"Timed out waiting for {path}"
        time.sleep(0.01)


def test_inference_runs_in_its_own_process_and_group(worker, tmp_path):
    worker.start()
    assert worker.events.get(timeout=5) == ("ready", None)
    worker.submit(Utterance("one", [], 1.0))
    assert worker.events.get(timeout=5).kind == "transcribing"
    event = worker.events.get(timeout=5)
    assert (event.session_id, event.kind) == ("one", "completed")
    detail = json.loads(event.text)
    assert detail["pid"] != os.getpid()
    assert detail["pid"] == detail["group"]
    assert detail["speech_imported"] is False
    worker.close()
    assert not worker.is_alive
    assert (tmp_path / "closed").exists()


def test_cancel_kills_hung_job_and_discards_queued_job(worker, tmp_path):
    worker.start()
    assert worker.events.get(timeout=5) == ("ready", None)
    worker.submit(Utterance("hung", [], 1.0, "hang"))
    assert worker.events.get(timeout=5).kind == "transcribing"
    wait_for(tmp_path / "started")
    worker.submit(Utterance("queued", [], 2.0))
    started = time.monotonic()
    worker.cancel()
    assert time.monotonic() - started < 3
    assert not worker.is_alive
    assert not (tmp_path / "processed").exists()
    with pytest.raises(RuntimeError, match="cancelled"):
        worker.submit(Utterance("new", [], 3.0))
    with pytest.raises(queue.Empty):
        worker.events.get(timeout=0.1)


def test_close_stops_playback_descendant(worker, tmp_path):
    worker.start()
    assert worker.events.get(timeout=5) == ("ready", None)
    worker.submit(Utterance("speaking", [], 1.0, "child"))
    assert worker.events.get(timeout=5).kind == "transcribing"
    wait_for(tmp_path / "started")
    worker.close()
    worker.close()
    time.sleep(1.1)
    assert not (tmp_path / "late-playback").exists()


def test_cooperative_cancellation_releases_backend_without_termination(worker, tmp_path):
    worker.start()
    assert worker.events.get(timeout=5) == ("ready", None)
    worker.submit(Utterance("cooperative", [], 1.0, "cooperative"))
    assert worker.events.get(timeout=5).kind == "transcribing"
    wait_for(tmp_path / "started")
    worker.close()
    assert (tmp_path / "closed").exists()
    assert not (tmp_path / "processed").exists()


def test_cancel_before_start_is_terminal(worker):
    worker.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        worker.start()
    worker.close()


def test_cancel_stops_playback_even_after_worker_leader_exits(worker, tmp_path):
    worker.start()
    assert worker.events.get(timeout=5) == ("ready", None)
    worker.submit(Utterance("orphaned", [], 1.0, "orphan"))
    wait_for(tmp_path / "started")
    worker._process.join(timeout=5)
    assert not worker.is_alive
    worker.cancel()
    time.sleep(1.1)
    assert not (tmp_path / "late-playback").exists()


def test_startup_error_is_reported_without_hardware(tmp_path):
    worker = SpeechWorker(None, tmp_path, backend_factory=FailingBackend)
    try:
        worker.start()
        kind, detail = worker.events.get(timeout=5)
        assert kind == "fatal"
        assert "fixture model missing" in detail
    finally:
        worker.close()


@pytest.mark.parametrize("width,height", [(640, 480), (1280, 720)])
def test_preview_mirroring_preserves_coordinates_in_camera_domain(width, height):
    assert display_point(0.0, 0.0, width, height, False) == (0, 0)
    assert display_point(1.0, 1.0, width, height, False) == (width - 1, height - 1)
    assert display_point(0.0, 0.0, width, height, True) == (width - 1, 0)
    assert display_point(1.0, 1.0, width, height, True) == (0, height - 1)
    x, y = 0.25, 0.4
    unmirrored = display_point(x, y, width, height, False)
    mirrored = display_point(x, y, width, height, True)
    assert unmirrored[0] + mirrored[0] == width - 1
    assert unmirrored[1] == mirrored[1]
    assert (x, y) == (0.25, 0.4)


def test_command_description_names_the_call_or_why_nothing_ran():
    assert describe_command(None) == ""
    assert describe_command({"call": {"name": "look", "arguments": {"direction": "left"}},
                             "outcome": "executed"}) == 'look(direction="left")'
    assert describe_command({"call": {"name": "stop", "arguments": {}}, "outcome": "executed"}) == "stop()"
    assert describe_command({"call": None, "outcome": "vetoed_compound"}) == "No action (more than one request)"
    assert describe_command({"call": None, "outcome": "conversation"}) == "No action (conversation)"


@pytest.mark.parametrize("state", list(State))
def test_preview_renders_every_state_with_overlong_text(state):
    controller = SimpleNamespace(
        state=state, transcript="please look left " * 20, response="Okay. " * 200,
        error="Model failed at /a/very/long/path/without/spaces" * 5,
        command={"call": {"name": "look", "arguments": {"direction": "left"}}, "outcome": "executed"},
        robot_state={"direction": "left", "stopped": True, "greetings": 1, "last_action": "stop"},
    )
    sample = HandSample(1.0, ((0.5, 0.5, 0.0),) * 21, 0.9, "Right")
    diagnostics = {"reason": "wave_more", "hand_visible": True, "open_hand": True,
                   "open_hand_source": "landmarks", "palm_width_px": 80.0, "translation_segments": 2}
    canvas = render_preview(np.zeros((480, 640, 3), np.uint8), sample, controller, 30.0, AppConfig(), diagnostics)
    assert canvas.shape == (520, 1080, 3)


def test_hand_sample_keeps_first_hand_unmirrored_with_open_palm_score():
    point = SimpleNamespace(x=0.25, y=0.5, z=-0.1)
    result = SimpleNamespace(
        hand_landmarks=[[point] * 21],
        gestures=[[SimpleNamespace(category_name="None", score=0.9),
                   SimpleNamespace(category_name="Open_Palm", score=0.6)]],
        handedness=[[SimpleNamespace(category_name="Right")]],
    )
    sample = hand_sample(result, 12.5, 640, 480)
    assert sample.captured_at == 12.5
    assert sample.landmarks[0] == (0.25, 0.5, -0.1) and len(sample.landmarks) == 21
    assert (sample.open_palm_score, sample.handedness, sample.width, sample.height) == (0.6, "Right", 640, 480)
    assert hand_sample(SimpleNamespace(hand_landmarks=[]), 1.0, 640, 480) is None
