from math import pi, sin
import queue

from wave_demo.app import publish_latest, reset_detector_for_transition
from wave_demo.config import AppConfig
from wave_demo.controller import Controller
from wave_demo.contracts import State
from wave_demo.wave import WaveDetector
from test_wave import hand


def test_startup_wave_cannot_latch_out_ready_session():
    config = AppConfig()
    controller = Controller(config, "test", lambda event: None)
    detector = WaveDetector(config.wave, "camera:0", "test")
    ignored = []
    for i in range(40):
        t = i / 20
        observation = detector.update(hand(t, dx=80 * sin(4 * pi * t)), t)
        if observation:
            ignored.append(observation)
            assert controller.consider_wave(observation, t) is None
    assert len(ignored) == 1
    previous = controller.state
    controller.ready(2.0)
    reset_detector_for_transition(detector, previous, controller.state)
    # The user continues waving without lowering their hand between phases.
    for i in range(40, 80):
        t = i / 20
        observation = detector.update(hand(t, dx=80 * sin(4 * pi * t)), t)
        if observation:
            assert controller.consider_wave(observation, t) is not None
            break
    assert controller.state == State.LISTENING


def test_callback_queue_only_replaces_data_with_latest_result():
    pending = queue.Queue(maxsize=1)
    first, second = hand(1), hand(2)
    publish_latest(pending, first, 1)
    publish_latest(pending, second, 2)
    assert pending.get_nowait() == (second, 2)
    assert pending.empty()


def test_repeated_ready_frames_do_not_clear_motion():
    class Detector:
        resets = 0
        def reset(self):
            self.resets += 1
    detector = Detector()
    reset_detector_for_transition(detector, State.READY, State.READY)
    assert detector.resets == 0
    reset_detector_for_transition(detector, State.COOLDOWN, State.READY)
    assert detector.resets == 1
