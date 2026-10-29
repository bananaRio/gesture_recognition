"""Real MediaPipe model plus wave rules on synthetic motion of a public hand image.

Run: uv run python scripts/smoke_wave_model.py [--download]
No camera is opened. Translating one still image is synthetic motion: this checks
the model-to-detector wiring and that an open hand alone or a single swipe does
not trigger. It is not a live-camera or real-wave validation.
"""

import argparse
import hashlib
import json
import math
import urllib.request

from wave_demo.cli import prepare_environment
from wave_demo.config import load_config, project_root
from wave_demo.vision import hand_sample
from wave_demo.wave import WaveDetector

SOURCE = ("https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/"
          "right_hands.jpg?generation=1782185275057115")
EXPECTED_SHA256 = "4b5134daa4cb60465535239535f9f74c2842aba3aa5fd30bf04ef5678f93d87f"
FRAMES = 60
CASES = {
    # Horizontal offsets (pixels) of the hand, one per 20 FPS submission.
    "stationary_open_palm": ([140] * FRAMES, 0),
    "one_way_swipe": ([40 + round(i * 200 / (FRAMES - 1)) for i in range(FRAMES)], 0),
    "alternating_wave": ([140 + round(70 * math.sin(i * 0.35)) for i in range(FRAMES)], 1),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch the public fixture if absent")
    args = parser.parse_args()
    root = project_root()
    prepare_environment(root)
    config = load_config(root / "config.toml")
    fixture = root / ".cache" / "right_hands.jpg"
    if not fixture.exists():
        if not args.download:
            parser.error("Public test image is not cached; rerun with --download")
        with urllib.request.urlopen(SOURCE, timeout=30) as response:
            fixture.write_bytes(response.read(2_000_001))
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("Test image does not match MediaPipe's published hash")

    import cv2
    import mediapipe as mp
    import numpy as np

    image = cv2.imread(str(fixture))
    # The fixture shows two hands; keep the right half (one hand) on a white frame.
    crop = cv2.resize(image[:, image.shape[1] // 2:], None, fx=0.9, fy=0.9)
    height, width = crop.shape[:2]
    report = {"source": SOURCE, "camera_opened": False, "mediapipe": mp.__version__, "cases": {}}
    for name, (offsets, expected) in CASES.items():
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(root / "assets" / "gesture_recognizer.task"),
                                              delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6, min_tracking_confidence=0.6,
        )
        detector = WaveDetector(config.wave, "fixture", "fixture-clock")
        events, open_frames, scores = [], 0, []
        with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
            for index, offset in enumerate(offsets):
                frame = np.full((480, 640, 3), 255, dtype=np.uint8)
                frame[40:40 + height, offset:offset + width] = crop
                t = index / config.wave.fps
                result = recognizer.recognize_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    round(t * 1000))
                sample = hand_sample(result, t, 640, 480)
                if event := detector.update(sample, t):
                    events.append(event.evidence)
                open_frames += detector.diagnostics["open_hand"]
                scores.append(sample.open_palm_score if sample else 0.0)
        report["cases"][name] = {
            "frames": len(offsets), "open_hand_frames": open_frames,
            "open_palm_score_range": [round(min(scores), 3), round(max(scores), 3)],
            "wave_events": len(events), "expected_events": expected, "evidence": events,
            "passed": len(events) == expected and open_frames > 0,
        }
        print(name, json.dumps(report["cases"][name]), flush=True)
    report["passed"] = all(case["passed"] for case in report["cases"].values())
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "wave-model-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
