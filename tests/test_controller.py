"""Fake-clock tests of event safety and interaction lifecycle, without devices."""

from dataclasses import replace

from wave_demo.config import AppConfig
from wave_demo.contracts import SpeechEvent, State, WaveObservation
from wave_demo.controller import Controller
from wave_demo.policy import EngagementPolicy


def observation(event_id="wave-1", at=1.0, domain="test-clock"):
    return WaveObservation(event_id, "webcam-0", at, domain, "hand-7", (0.1, 0.2, 0.4, 0.7))


def started():
    events = []
    controller = Controller(AppConfig(), "test-clock", events.append)
    controller.ready(0.0)
    request = controller.consider_wave(observation(), 1.0)
    assert request is not None
    return controller, request, events


def absent_until(controller, begin, end, step=0.05):
    count = round((end - begin) / step)
    for i in range(count + 1):
        now = begin + i * step
        controller.observe_hand(False, now)
        controller.tick(now)


def test_policy_keeps_person_unresolved_and_preserves_trigger():
    policy = EngagementPolicy("test-clock")
    obs = observation()
    request = policy.consider(obs, State.READY, 1.1)
    assert request is not None
    assert request.observation is obs
    assert request.target_person_id is None


def test_policy_rejects_stale_future_cross_clock_and_invalid_timestamps():
    examples = [
        (observation("stale", 1.0), 1.251, "stale_observation"),
        (observation("future", 2.0), 1.0, "future_observation"),
        (observation("domain", 1.0, "another-clock"), 1.0, "clock_domain_mismatch"),
        (observation("nan", float("nan")), 1.0, "invalid_timestamp"),
    ]
    policy = EngagementPolicy("test-clock")
    for obs, now, reason in examples:
        assert policy.consider(obs, State.READY, now) is None
        assert policy.rejection_reason == reason


def test_policy_disabled_and_busy_events_are_consumed_and_cache_is_bounded():
    policy = EngagementPolicy("test-clock", dedup_capacity=2)
    assert policy.consider(observation("busy"), State.LISTENING, 1.0) is None
    assert policy.consider(observation("busy"), State.READY, 1.0) is None
    assert policy.rejection_reason == "duplicate_event"
    policy.enabled = False
    assert policy.consider(observation("disabled"), State.READY, 1.0) is None
    assert policy.rejection_reason == "disabled"
    policy.enabled = True
    assert policy.consider(observation("third"), State.READY, 1.0) is not None
    assert len(policy._seen) == 2


def test_active_session_rejects_another_wave_and_has_immutable_correlation():
    controller, request, events = started()
    assert controller.state == State.LISTENING
    assert controller.consider_wave(observation("wave-2", 1.1), 1.1) is None
    controller.on_worker_event(SpeechEvent(request.session_id, "transcribing"), 1.3)
    controller.on_worker_event(SpeechEvent(request.session_id, "thinking", text="Hello"), 1.4)
    controller.on_worker_event(SpeechEvent(request.session_id, "speaking", text="Hi there"), 1.5)
    controller.on_worker_event(SpeechEvent(request.session_id, "completed"), 1.6)
    assert controller.state == State.COOLDOWN
    assert (controller.transcript, controller.response) == ("Hello", "Hi there")
    correlated = [event for event in events if event.session_id]
    assert all(event.session_id == request.session_id for event in correlated)
    assert all(event.trigger_event_id == "wave-1" for event in correlated)


def test_no_watchdog_during_listening_and_stage_events_do_not_reset_deadline():
    controller, request, _ = started()
    controller.tick(100.0)
    assert controller.state == State.LISTENING
    controller.on_worker_event(SpeechEvent(request.session_id, "transcribing"), 101.0)
    controller.on_worker_event(SpeechEvent(request.session_id, "transcribing"), 120.0)
    controller.on_worker_event(SpeechEvent(request.session_id, "thinking", text="Hi"), 125.0)
    controller.tick(130.99)
    assert controller.state == State.THINKING
    controller.tick(131.0)
    assert controller.state == State.ERROR
    controller.ready(132.0)
    controller.on_worker_event(SpeechEvent(request.session_id, "completed"), 133.0)
    assert controller.state == State.ERROR
    assert controller.consider_wave(observation("later", 134.0), 134.0) is None


def test_cooldown_requires_release_and_fresh_camera_observations():
    controller, request, _ = started()
    controller.on_worker_event(SpeechEvent(request.session_id, "completed"), 2.0)
    controller.observe_hand(True, 3.9)
    controller.tick(4.0)
    assert controller.state == State.COOLDOWN
    controller.observe_hand(False, 4.05)
    controller.tick(8.0)  # No callback cannot establish release.
    assert controller.state == State.COOLDOWN
    controller.observe_hand(False, 8.0)  # A camera gap restarts release.
    controller.tick(8.0)
    assert controller.state == State.COOLDOWN
    absent_until(controller, 8.05, 8.6)
    assert controller.state == State.READY


def test_release_completed_before_cooldown_must_still_be_fresh():
    controller, request, _ = started()
    controller.on_worker_event(SpeechEvent(request.session_id, "completed"), 2.0)
    absent_until(controller, 2.0, 2.6)
    controller.tick(4.0)
    assert controller.state == State.COOLDOWN
    absent_until(controller, 4.0, 4.6)
    assert controller.state == State.READY


def test_old_worker_results_cannot_mutate_a_new_session():
    controller, old_request, _ = started()
    controller.cancel(2.0)
    absent_until(controller, 2.0, 4.1)
    request = controller.consider_wave(observation("new", 4.2), 4.2)
    assert request is not None
    controller.on_worker_event(SpeechEvent(old_request.session_id, "failed", error="old error"), 4.3)
    controller.on_worker_event(SpeechEvent(old_request.session_id, "speaking", text="old reply"), 4.4)
    assert controller.state == State.LISTENING
    assert controller.error == controller.response == ""
    assert controller.session == request


def test_worker_failure_recovers_but_fatal_failure_and_close_do_not():
    controller, request, _ = started()
    controller.on_worker_event(SpeechEvent(request.session_id, "failed", error="ASR failed"), 2.0)
    assert controller.error == "ASR failed"
    assert controller.state == State.COOLDOWN
    absent_until(controller, 2.0, 4.1)
    assert controller.state == State.READY
    controller.fail("Camera disconnected", 4.2)
    controller.ready(4.3)
    controller.cancel(4.4)
    assert controller.state == State.ERROR
    controller.close(4.5)
    controller.ready(4.6)
    assert controller.state == State.STOPPED


def test_preview_mirroring_does_not_change_engagement_policy():
    config = AppConfig()
    for mirror in (True, False):
        adjusted = replace(config, runtime=replace(config.runtime, mirror_preview=mirror))
        controller = Controller(adjusted, "test-clock", lambda _: None)
        controller.ready(0.0)
        request = controller.consider_wave(observation(), 1.0)
        assert request is not None
        assert request.observation.region == (0.1, 0.2, 0.4, 0.7)


def test_repeated_or_out_of_order_absence_does_not_advance_release():
    controller, request, _ = started()
    controller.on_worker_event(SpeechEvent(request.session_id, "completed"), 2.0)
    controller.observe_hand(False, 4.0)
    controller.observe_hand(False, 4.0)
    controller.observe_hand(False, 3.0)
    controller.tick(4.6)
    assert controller.state == State.COOLDOWN


def test_spoken_command_is_kept_for_display_and_logged_without_state():
    controller, request, events = started()
    command = {"call": {"name": "look", "arguments": {"direction": "left"}}, "outcome": "executed",
               "state": {"direction": "left", "stopped": False, "greetings": 0, "last_action": "look"}}
    controller.on_worker_event(SpeechEvent(request.session_id, "transcribing"), 1.1)
    controller.on_worker_event(SpeechEvent(request.session_id, "thinking", text="Look left."), 1.2)
    controller.on_worker_event(SpeechEvent(request.session_id, "speaking", text="Looking left.",
                                           command=command), 1.3)
    assert controller.state == State.SPEAKING
    assert controller.command == command
    assert controller.robot_state["direction"] == "left"
    logged = [e.details["command"] for e in events if e.kind == "speech_progress" and e.details["command"]]
    assert logged == [{"call": command["call"], "outcome": "executed"}]
    controller.on_worker_event(SpeechEvent(request.session_id, "completed"), 1.4)
    absent_until(controller, 1.4, 3.5)
    assert controller.consider_wave(observation("next", 3.6), 3.6) is not None
    # A new session clears the last command but keeps the robot's state.
    assert controller.command is None
    assert controller.robot_state["direction"] == "left"
