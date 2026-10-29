"""Desktop composition root. Domain components have no GUI or robot dependency."""

from dataclasses import asdict
from collections import Counter
from datetime import datetime
from functools import cache
from pathlib import Path
import queue
import textwrap
import time
import uuid

from .assets import offline_environment, verify_assets
from .commands import Simulator
from .config import AppConfig
from .contracts import LifecycleEvent, SpeechEvent, State
from .event_log import EventLog
from .worker import SpeechWorker

WINDOW = "Wave to Command"
NO_ACTION = {
    "conversation": "No action (conversation)",
    "invalid_output": "No action (invalid model output)",
    "vetoed_discussion": "No action (sounded like a discussion)",
    "vetoed_compound": "No action (more than one request)",
}

# Preview colours are BGR. Accents carry meaning: green = microphone live,
# amber = processing, red = error, blue = the call that ran.
BACKGROUND, SURFACE, RULE = (23, 22, 20), (46, 44, 42), (52, 50, 47)
TEXT, SECONDARY, FAINT = (236, 238, 235), (186, 190, 186), (112, 116, 112)
GREEN, AMBER, RED, CALL = (126, 225, 130), (80, 190, 240), (110, 140, 255), (230, 200, 120)
PANEL, BAR, PAD, GUTTER = 440, 40, 24, 52
STATE_COLORS = {
    State.READY: TEXT, State.LISTENING: GREEN, State.TRANSCRIBING: AMBER,
    State.THINKING: AMBER, State.SPEAKING: AMBER, State.ERROR: RED,
}
STATE_HINTS = {
    State.STARTING: "Loading local models",
    State.READY: "Wave an open hand from side to side",
    State.LISTENING: "Say one command",
    State.TRANSCRIBING: "Turning speech into text",
    State.THINKING: "Choosing an action",
    State.SPEAKING: "Ava is replying",
    State.COOLDOWN: "Lower your hand to rearm",
}
READY_HINTS = {
    "no_hand": "Raise an open hand and wave it side to side",
    "too_small": "Move your hand a little closer",
    "show_open_palm": "Open your hand toward the camera",
    "arming": "Keep your palm open for a moment",
    "wave_more": "Wave left and right a few times",
    "wave_detected": "Wave detected",
    "stale_result": "Camera processing is delayed",
}
EXAMPLES = ("Look left", "Look right", "Face forward", "Say hello", "Stop", "What’s your status?")
GAZE = {"left": "left", "center": "ahead", "right": "right"}


def display_point(x: float, y: float, width: int, height: int, mirror: bool) -> tuple[int, int]:
    """Transform coordinates only for preview, never for observation messages."""
    return (round((1 - x if mirror else x) * (width - 1)), round(y * (height - 1)))


def describe_command(command: dict | None) -> str:
    if not command:
        return ""
    call = command.get("call")
    if call is None:
        return NO_ACTION.get(command.get("outcome"), "No action")
    arguments = ", ".join(f'{name}="{value}"' for name, value in call["arguments"].items())
    return f"{call['name']}({arguments})"


def publish_latest(results, sample, captured_at):
    """Native callbacks only publish data; the main thread owns all state."""
    try:
        results.put_nowait((sample, captured_at))
    except queue.Full:
        try:
            results.get_nowait()
        except queue.Empty:
            pass
        try:
            results.put_nowait((sample, captured_at))
        except queue.Full:
            pass


def reset_detector_for_transition(detector, previous: State, current: State):
    # In particular, an ignored startup/busy wave cannot latch the next READY.
    if previous != current and current in (State.READY, State.COOLDOWN, State.ERROR):
        detector.reset()


@cache
def _font():
    import cv2
    return cv2.FontFace("sans")


@cache
def _wrap_width(size: int, pixels: int) -> int:
    """Convert a pixel width to a textwrap budget using an average glyph width."""
    import cv2
    sample = "Could you turn your head to the right, please?"
    measured = cv2.getTextSize((0, 0), sample, (0, 0), _font(), size)[2]
    return max(8, pixels * len(sample) // measured)


def render_preview(frame, sample, controller, fps: float, config: AppConfig, diagnostics=None):
    import cv2
    import numpy as np

    diagnostics = diagnostics or {}
    state = controller.state
    width, height = config.runtime.width, config.runtime.height
    body = max(height, 480)
    canvas = np.full((body + BAR, width + PANEL, 3), BACKGROUND, dtype=np.uint8)
    camera = cv2.resize(frame, (width, height))
    if config.runtime.mirror_preview:
        camera = cv2.flip(camera, 1)
    if sample:
        points = [display_point(p[0], p[1], width, height, config.runtime.mirror_preview)
                  for p in sample.landmarks]
        # The skeleton turns green only when the detector counts the hand as open.
        bone, joint = ((140, 224, 115), (230, 250, 224)) if diagnostics.get("open_hand") else (SECONDARY, TEXT)
        for chain in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (5, 9, 10, 11, 12),
                      (9, 13, 14, 15, 16), (13, 17, 18, 19, 20), (0, 17)):
            for a, b in zip(chain, chain[1:]):
                cv2.line(camera, points[a], points[b], bone, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(camera, point, 3, joint, -1, cv2.LINE_AA)
    if state == State.LISTENING:
        cv2.rectangle(camera, (2, 2), (width - 3, height - 3), GREEN, 4)
    canvas[:height, :width] = camera

    def text(content, x, y, size=16, color=TEXT, weight=400, flags=cv2.PUT_TEXT_ALIGN_LEFT):
        """Draw at a baseline and return the x where following text continues."""
        return cv2.putText(canvas, content, (x, y), color, _font(), size, weight, flags)[0][0]

    def paragraph(content, x, y, color, max_lines, pixels):
        for line in textwrap.wrap(content, _wrap_width(16, pixels), max_lines=max_lines, placeholder=" …"):
            text(line, x, y, 16, color)
            y += 23
        return y

    # State, then one hint that says what to do next. A fatal error replaces the hint.
    x, column = width + PAD, PANEL - 2 * PAD
    cv2.circle(canvas, (x + 6, 39), 6, STATE_COLORS.get(state, FAINT), -1, cv2.LINE_AA)
    text(str(state), x + 22, 48, 26, TEXT, 700)
    if state == State.ERROR:
        y = paragraph(controller.error, x, 80, RED, 4, column)
    else:
        hint = STATE_HINTS.get(state, "")
        if state == State.READY:
            hint = READY_HINTS.get(diagnostics.get("reason"), hint)
        y = paragraph(hint, x, 80, SECONDARY, 2, column)

    # The latest exchange. Before the first one, show what can be said instead.
    call = (controller.command or {}).get("call")
    rows = [row for row in (
        ("You", controller.transcript, TEXT, 2),
        ("→", describe_command(controller.command), CALL if call else FAINT, 1),
        ("Ava", controller.response, TEXT, 3),
        ("Error", "" if state == State.ERROR else controller.error, RED, 2),
    ) if row[1]]
    y += 26
    for name, content, color, max_lines in rows:
        text(name, x, y, 14, FAINT, 600)
        y = paragraph(content, x + GUTTER, y, color, max_lines, column - GUTTER) + 10
    if not rows and state != State.ERROR:
        text("Try saying", x, y, 14, FAINT, 600)
        for index, example in enumerate(EXAMPLES):
            text(f"“{example}”", x + index // 3 * 172, y + 30 + index % 3 * 26, 16, SECONDARY)

    # Simulated Ava: a face plate whose eyes follow the gaze and dim while stopped.
    robot = controller.robot_state or Simulator().snapshot()
    bottom = body - PAD
    text("Simulated Ava", x, bottom - 64, 14, FAINT, 600)
    cv2.rectangle(canvas, (x + 12, bottom - 44), (x + 44, bottom), SURFACE, -1)
    cv2.rectangle(canvas, (x, bottom - 32), (x + 56, bottom - 12), SURFACE, -1)
    for corner in ((x + 12, bottom - 32), (x + 44, bottom - 32), (x + 12, bottom - 12), (x + 44, bottom - 12)):
        cv2.circle(canvas, corner, 12, SURFACE, -1, cv2.LINE_AA)
    shift = {"left": -7, "right": 7}.get(robot["direction"], 0)
    for eye in (-10, 10):
        cv2.ellipse(canvas, (x + 28 + eye + shift, bottom - 22), (4, 7), 0, 0, 360,
                    FAINT if robot["stopped"] else TEXT, -1, cv2.LINE_AA)
    gaze = GAZE.get(robot["direction"], robot["direction"])
    text(f"{'Stopped, looking' if robot['stopped'] else 'Looking'} {gaze}", x + 72, bottom - 25, 16, TEXT, 600)
    greetings = robot["greetings"]
    text(f"{greetings} greeting{'' if greetings == 1 else 's'}" if greetings else "No greetings yet",
         x + 72, bottom - 4, 14, FAINT)

    # Status bar: hand measurements on the left, loop rate and exit key on the right.
    y = body + 25
    cv2.line(canvas, (0, body), (width + PANEL, body), RULE, 1)
    if diagnostics.get("hand_visible"):
        hand = f"open ({diagnostics.get('open_hand_source')})" if diagnostics.get("open_hand") else "not open"
        palm = f"{float(diagnostics.get('palm_width_px', 0.0)):.0f} px"
    else:
        hand, palm = "not found", "–"
    cursor = PAD
    for name, value in (("Hand", hand), ("Palm", palm)):
        cursor = text(value, text(name, cursor, y, 13, FAINT) + 8, y, 13, SECONDARY) + 28
    cursor = text("Wave", cursor, y, 13, FAINT) + 14
    segments = min(3, max(diagnostics.get("translation_segments", 0), diagnostics.get("wrist_segments", 0)))
    for index in range(3):
        cv2.circle(canvas, (cursor + index * 14, y - 5), 4, GREEN if segments == 3 else SECONDARY,
                   -1 if index < segments else 1, cv2.LINE_AA)
    right = width + PANEL - PAD
    text("Esc to quit", right, y, 13, FAINT, flags=cv2.PUT_TEXT_ALIGN_RIGHT)
    right -= cv2.getTextSize((0, 0), "Esc to quit", (0, 0), _font(), 13)[2] + 24
    text(f"{fps:.0f} fps", right, y, 13, FAINT, flags=cv2.PUT_TEXT_ALIGN_RIGHT)
    return canvas


def run(config: AppConfig, root: Path, microphone=None, duration: float | None = None) -> int:
    offline_environment()
    if errors := verify_assets(root / "assets", config):
        raise RuntimeError("; ".join(errors))
    import cv2
    from .audio import AudioCapture
    from .controller import Controller
    from .vision import HandRecognizer
    from .wave import WaveDetector

    run_id = str(uuid.uuid4())
    clock_domain = f"{run_id}:monotonic"
    path = root / "logs" / f"{datetime.now():%Y%m%d-%H%M%S}-{run_id[:8]}.jsonl"
    log = EventLog(path)
    log(LifecycleEvent("run_started", time.monotonic(), details={
        "run_id": run_id, "clock_domain": clock_domain, "config": asdict(config),
    }))
    controller = Controller(config, clock_domain, log)
    detector = WaveDetector(config.wave, f"camera:{config.runtime.camera_index}", clock_domain)
    results = queue.Queue(maxsize=1)

    def callback(sample, captured_at):
        publish_latest(results, sample, captured_at)

    camera = recognizer = audio = worker = None
    exit_code = 0
    try:
        camera = cv2.VideoCapture(config.runtime.camera_index, cv2.CAP_AVFOUNDATION)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.runtime.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.runtime.height)
        camera.set(cv2.CAP_PROP_FPS, 30)
        if not camera.isOpened():
            raise RuntimeError("Camera unavailable. Grant Camera access to the launching app in macOS Settings.")
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Camera opened but returned no frame")
        audio = AudioCapture(config.audio, microphone)
        audio.start()
        recognizer = HandRecognizer(root / "assets" / "gesture_recognizer.task", callback)
        worker = SpeechWorker(config.speech, root / "assets")
        worker.start()
        print(f"Local event log: {path}", flush=True)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        start = previous = time.monotonic()
        last_submission = 0.0
        last_result = start
        latest_sample = None
        fps = 0.0
        last_state = controller.state
        diagnostics = {"reason": "no_hand", "hand_visible": False}
        last_diagnostic_log = 0.0
        diagnostic_reasons = Counter()
        peak_segments = 0

        def apply_state_change():
            nonlocal last_state, exit_code
            if controller.state == last_state:
                return
            reset_detector_for_transition(detector, last_state, controller.state)
            if controller.state in (State.COOLDOWN, State.ERROR):
                audio.cancel_session()
            if controller.state == State.ERROR:
                worker.cancel()
                exit_code = 1
            last_state = controller.state

        while True:
            ok, frame = camera.read()
            now = time.monotonic()
            if not ok:
                controller.fail("Camera stopped returning frames", now)
                exit_code = 1
                break
            fps = 0.9 * fps + 0.1 / max(now - previous, 1e-6)
            previous = now
            if duration is not None and now - start >= duration:
                break
            # Some camera drivers ignore requested dimensions. Fix the inference
            # resolution so pixel thresholds have the documented meaning.
            frame = cv2.resize(frame, (config.runtime.width, config.runtime.height))
            if now - last_submission >= 1 / config.wave.fps:
                recognizer.submit(frame, now)
                last_submission = now
            try:
                while True:
                    message = worker.events.get_nowait()
                    if isinstance(message, tuple):
                        kind, detail = message
                        if kind == "ready":
                            controller.ready(now)
                        elif kind == "fatal":
                            controller.fail(detail, now)
                    else:
                        controller.on_worker_event(message, now)
            except queue.Empty:
                pass
            if controller.state not in (State.ERROR, State.STOPPED) and not worker.is_alive:
                controller.fail("Speech worker exited unexpectedly; restart the application", now)
            if controller.state == State.STARTING and now - start > 120:
                controller.fail("Local models did not finish starting within 120 seconds", now)
            # Reset before consuming the first READY sample, not after it.
            apply_state_change()
            try:
                sample, captured_at = results.get_nowait()
                if 0 <= now - captured_at <= config.runtime.observation_max_age_seconds:
                    last_result = now
                    latest_sample = sample
                    observation = detector.update(sample, captured_at)
                    diagnostics = dict(detector.diagnostics)
                    diagnostics["callback_age_ms"] = (now - captured_at) * 1000
                    diagnostic_reasons[diagnostics["reason"]] += 1
                    peak_segments = max(peak_segments, diagnostics["translation_segments"],
                                        diagnostics["wrist_segments"])
                    controller.observe_hand(detector.qualifying_palm, captured_at)
                    if observation:
                        log(LifecycleEvent("wave_observed", now, trigger_event_id=observation.event_id,
                                           details=asdict(observation)))
                        decision_at = time.monotonic()
                        request = controller.consider_wave(observation, decision_at)
                        if request:
                            audio.begin_session(request.session_id)
                            armed_at = time.monotonic()
                            log(LifecycleEvent("microphone_armed", armed_at, request.session_id,
                                               observation.event_id,
                                               {"decision_to_listening_ms": (armed_at - decision_at) * 1000}))
                else:
                    detector.reset()
                    latest_sample = None
                    diagnostics = {"reason": "stale_result", "hand_visible": False,
                                   "callback_age_ms": (now - captured_at) * 1000}
                    diagnostic_reasons["stale_result"] += 1
            except queue.Empty:
                pass
            if now - last_result > config.wave.max_gap_seconds:
                detector.reset()
                latest_sample = None
                diagnostics = {"reason": "stale_result", "hand_visible": False,
                               "callback_age_ms": (now - last_result) * 1000}
                # Missing inference is not evidence that the user lowered their hand.
            if now - last_result > 5.0:
                controller.fail("Gesture recognizer has stopped producing current results", now)
            utterance = audio.poll()
            if utterance is not None and controller.state == State.LISTENING:
                if controller.session and utterance.session_id == controller.session.session_id:
                    controller.on_worker_event(SpeechEvent(utterance.session_id, "transcribing"), now)
                    worker.submit(utterance)
            controller.tick(now)
            apply_state_change()
            if now - last_diagnostic_log >= 1.0:
                log(LifecycleEvent("vision_status", now, details={
                    **diagnostics, "state": controller.state.value,
                    "reason_counts": dict(diagnostic_reasons), "peak_motion_segments": peak_segments,
                }))
                last_diagnostic_log = now
                diagnostic_reasons.clear()
                peak_segments = 0
            preview = render_preview(frame, latest_sample, controller, fps, config, diagnostics)
            cv2.imshow(WINDOW, preview)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        controller.fail(str(exc), time.monotonic())
        print(f"Error: {exc}", flush=True)
        exit_code = 1
    finally:
        controller.close(time.monotonic())
        closers = [resource.close for resource in (audio, worker, recognizer) if resource is not None]
        if camera is not None:
            closers.append(camera.release)
        closers.extend((cv2.destroyAllWindows, log.close))
        for closer in closers:
            try:
                closer()
            except Exception as exc:
                print(f"Cleanup error: {exc}", flush=True)
                exit_code = 1
    return exit_code
