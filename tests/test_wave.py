"""Synthetic evidence tests: no camera/model dependencies required."""

from dataclasses import replace
from math import pi, sin
import unittest

from wave_demo.config import WaveConfig
from wave_demo.contracts import HandSample
from wave_demo.wave import WaveDetector


def hand(t, dx=0.0, tip_dx=0.0, dy=0.0, score=0.95, handedness="Right", scale=1.0):
    # Pixel geometry: a 70 px palm, independently movable fingertips.
    points = [(320.0, 270.0, 0.0) for _ in range(21)]
    for i, x in ((5, 285), (9, 308), (13, 332), (17, 355)):
        points[i] = (x, 230, 0)
    for i, x in ((8, 285), (12, 308), (16, 332), (20, 355)):
        points[i] = (x + tip_dx, 155, 0)
    points = tuple(((320 + (x - 320) * scale + dx) / 640,
                    (240 + (y - 240) * scale + dy) / 480, z)
                   for x, y, z in points)
    return HandSample(t, points, score, handedness)


def anatomical_hand(t, dx=0.0, extended=(0, 1, 2, 3), score=0.0):
    """A 72 px palm with plausible straight or folded 3D finger joints."""
    points = [(320.0, 300.0, 0.0) for _ in range(21)]
    points[1:5] = [(292, 283, 0), (276, 270, 0), (265, 253, 0), (255, 240, 0)]
    for finger, (mcp, x, lengths) in enumerate((
        (5, 282, (28, 22, 17)), (9, 307, (33, 24, 19)),
        (13, 332, (30, 22, 17)), (17, 354, (22, 18, 15)),
    )):
        points[mcp] = (x, 250, 0)
        if finger in extended:
            first, second, third = lengths
            points[mcp + 1:mcp + 4] = [
                (x, 250 - first, 0), (x, 250 - first - second, 0),
                (x, 250 - first - second - third, 0),
            ]
        else:
            points[mcp + 1:mcp + 4] = [(x, 235, -18), (x, 252, -25), (x, 265, -9)]
    normalized = tuple(((x + dx) / 640, y / 480, z / 640) for x, y, z in points)
    return HandSample(t, normalized, score, "Right")


def run(detector, generator, duration=2.0, fps=20, start=0.0):
    events = []
    for i in range(int(duration * fps) + 1):
        t = start + i / fps
        event = detector.update(generator(t), t)
        if event:
            events.append(event)
    return events


class WaveTests(unittest.TestCase):
    def setUp(self):
        self.detector = WaveDetector(WaveConfig(), "webcam:0", "test-monotonic")

    def test_translation_emits_one_observation_and_preserves_contract(self):
        events = run(self.detector, lambda t: hand(t, dx=80 * sin(2 * pi * 2 * t)), 3)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.source_id, "webcam:0")
        self.assertEqual(event.clock_domain, "test-monotonic")
        self.assertEqual(event.evidence["signal"], "translation")
        self.assertGreaterEqual(event.evidence["reversals"], 2)
        self.assertGreaterEqual(event.evidence["observations"], 8)
        self.assertGreaterEqual(event.evidence["duration_seconds"] + 1e-9, 0.4)
        self.assertTrue(self.detector.qualifying_palm)

    def test_stationary_wrist_wave_uses_separate_signal(self):
        events = run(self.detector, lambda t: hand(t, tip_dx=80 * sin(2 * pi * 2 * t)))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].evidence["signal"], "wrist_relative")

    def test_static_jitter_sweep_vertical_and_closed_hand_do_not_trigger(self):
        cases = {
            "still": lambda t: hand(t),
            "jitter": lambda t: hand(t, dx=8 * sin(t * 30), tip_dx=5 * sin(t * 21)),
            "sweep": lambda t: hand(t, dx=50 * t),
            "vertical": lambda t: hand(t, dy=70 * sin(2 * pi * 2 * t)),
            "closed": lambda t: hand(t, dx=80 * sin(2 * pi * 2 * t), score=0.1),
        }
        for name, generator in cases.items():
            with self.subTest(name=name):
                self.detector.reset()
                self.assertEqual(run(self.detector, generator), [])

    def test_frame_rate_variation_and_mirroring(self):
        for fps in (10, 15, 30):
            with self.subTest(fps=fps):
                self.detector.reset()
                events = run(self.detector, lambda t: hand(t, dx=-80 * sin(2 * pi * 1.5 * t)), fps=fps)
                self.assertEqual(len(events), 1)

    def test_missing_observations_cannot_join_three_separate_sweeps(self):
        events = []
        for i in range(45):
            t = i * 0.05
            phase = i % 15
            if phase == 14:
                sample = None
            else:
                direction = 1 if (i // 15) % 2 == 0 else -1
                sample = hand(t, dx=direction * phase * 9)
            event = self.detector.update(sample, t)
            if event:
                events.append(event)
        self.assertEqual(events, [])

    def test_uncertain_palm_breaks_motion_and_is_not_qualifying(self):
        run(self.detector, lambda t: hand(t, dx=70 * t), duration=0.5)
        self.detector.update(hand(0.55, score=0.3), 0.55)
        self.assertFalse(self.detector.qualifying_palm)
        events = run(self.detector, lambda t: hand(t, dx=-70 * (t - 0.6)), start=0.6, duration=0.6)
        self.assertEqual(events, [])

    def test_release_rearms_and_allocates_new_temporary_track(self):
        first = run(self.detector, lambda t: hand(t, dx=80 * sin(2 * pi * 2 * t)))
        self.detector.update(None, 2.05)
        self.detector.update(None, 2.3)
        second = run(self.detector, lambda t: hand(t, dx=80 * sin(2 * pi * 2 * t)), start=2.4)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first[0].event_id, second[0].event_id)
        self.assertNotEqual(first[0].hand_track_id, second[0].hand_track_id)

    def test_out_of_order_results_do_not_change_qualification(self):
        self.detector.update(hand(1.0), 1.0)
        self.assertTrue(self.detector.qualifying_palm)
        self.detector.update(None, 0.9)
        self.assertTrue(self.detector.qualifying_palm)
        self.detector.update(hand(1.0, score=0), 1.0)
        self.assertTrue(self.detector.qualifying_palm)

    def test_jump_and_handedness_switch_do_not_create_wave(self):
        for i in range(40):
            t = i * 0.05
            side = "Right" if i % 12 < 6 else "Left"
            dx = 140 if i % 12 < 6 else -140
            self.assertIsNone(self.detector.update(hand(t, dx=dx, handedness=side), t))

    def test_invalid_and_too_small_geometry_fail_closed(self):
        samples = [
            hand(0, scale=0.3),
            replace(hand(1), landmarks=((float("nan"), 0.0, 0.0),) * 21),
            replace(hand(2), landmarks=()),
        ]
        for sample in samples:
            self.assertIsNone(self.detector.update(sample, sample.captured_at))
            self.assertFalse(self.detector.qualifying_palm)

    def test_long_gap_clears_old_evidence(self):
        run(self.detector, lambda t: hand(t, dx=100 * t), duration=0.6)
        events = run(self.detector, lambda t: hand(t, dx=60 - 100 * (t - 1.1)), start=1.1, duration=0.6)
        self.assertEqual(events, [])

    def test_brief_pose_score_flicker_preserves_continuous_motion(self):
        events = run(self.detector, lambda t: hand(
            t, dx=80 * sin(2 * pi * 2 * t),
            score=0.60 if round(t * 20) % 15 == 7 else 0.95,
        ), duration=3)
        self.assertEqual(len(events), 1)

    def test_brief_handedness_flicker_does_not_destroy_spatial_track(self):
        events = run(self.detector, lambda t: hand(
            t, dx=80 * sin(2 * pi * 2 * t),
            handedness="Left" if round(t * 20) % 15 == 7 else "Right",
        ), duration=3)
        self.assertEqual(len(events), 1)

    def test_low_confidence_can_retain_motion_but_cannot_trigger(self):
        baseline = WaveDetector(self.detector.config, "camera", "test")
        trigger_time = run(baseline, lambda t: hand(t, dx=80 * sin(2 * pi * 2 * t)))[0].captured_at
        events = []
        for frame in range(round(trigger_time * 20) + 1):
            t = frame / 20
            event = self.detector.update(hand(
                t, dx=80 * sin(2 * pi * 2 * t), score=.6 if t == trigger_time else .95,
            ), t)
            if event:
                events.append(event)
        self.assertEqual(events, [])
        self.assertFalse(self.detector.qualifying_palm)
        self.assertTrue(self.detector.diagnostics["armed"])
        self.assertEqual(self.detector.diagnostics["reason"], "show_open_palm")
        next_time = trigger_time + .05
        self.assertIsNotNone(self.detector.update(
            hand(next_time, dx=80 * sin(2 * pi * 2 * next_time)), next_time,
        ))

    def test_sustained_closed_hand_clears_previously_armed_evidence(self):
        events = run(self.detector, lambda t: hand(
            t, dx=80 * sin(2 * pi * 2 * t), score=0.95 if t < .5 else .1,
        ), duration=2)
        self.assertEqual(events, [])
        self.assertFalse(self.detector.diagnostics["armed"])
        self.assertEqual(self.detector.diagnostics["samples"], 0)
        self.assertEqual(self.detector.diagnostics["reason"], "show_open_palm")
        # Reopening a now-stationary hand must not replay the preceding motion.
        self.assertEqual(run(self.detector, lambda t: hand(t), start=2.1, duration=1), [])

    def test_pose_grace_requires_initial_confident_arming(self):
        events = run(self.detector, lambda t: hand(t, dx=80 * sin(2 * pi * 2 * t), score=.6))
        self.assertEqual(events, [])
        self.assertFalse(self.detector.diagnostics["armed"])

    def test_diagnostics_explain_size_score_progress_and_loss(self):
        self.detector.update(hand(0, scale=.3), 0)
        self.assertEqual(self.detector.diagnostics["reason"], "too_small")
        self.assertAlmostEqual(self.detector.diagnostics["palm_width_px"], 21)
        self.assertTrue(self.detector.diagnostics["hand_visible"])
        self.detector.update(hand(.05, score=.3), .05)
        self.assertEqual(self.detector.diagnostics["reason"], "show_open_palm")
        self.detector.update(hand(.1), .1)
        self.assertEqual(self.detector.diagnostics["reason"], "arming")
        self.detector.update(hand(.3), .3)
        self.assertEqual(self.detector.diagnostics["reason"], "wave_more")
        self.assertTrue(self.detector.diagnostics["armed"])
        snapshot = self.detector.diagnostics
        snapshot["reason"] = "changed externally"
        self.assertEqual(self.detector.diagnostics["reason"], "wave_more")
        self.detector.update(None, .35)
        self.assertEqual(self.detector.diagnostics["reason"], "no_hand")
        self.assertFalse(self.detector.diagnostics["hand_visible"])
        self.assertFalse(self.detector.diagnostics["armed"])
        self.assertEqual(self.detector.diagnostics["samples"], 0)

    def test_open_landmarks_allow_slow_wave_without_classifier_score(self):
        events = run(self.detector, lambda t: anatomical_hand(t, dx=50 * sin(2 * pi * t)), 3)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].evidence["open_palm_score"], 0)
        self.assertEqual(events[0].evidence["extended_fingers"], 4)
        self.assertEqual(events[0].evidence["open_hand_source"], "landmarks")
        self.assertTrue(self.detector.diagnostics["open_hand"])
        self.assertEqual(self.detector.diagnostics["open_palm_score"], 0)

    def test_three_extended_fingers_qualify(self):
        events = run(self.detector, lambda t: anatomical_hand(
            t, dx=50 * sin(2 * pi * t), extended=(0, 1, 2),
        ), 3)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].evidence["extended_fingers"], 3)

    def test_static_open_fist_and_peace_sign_do_not_trigger(self):
        cases = [("static", (0, 1, 2, 3), 0), ("fist", (), 50), ("peace", (0, 1), 50)]
        for name, fingers, amplitude in cases:
            with self.subTest(name=name):
                self.detector.reset()
                events = run(self.detector, lambda t: anatomical_hand(
                    t, dx=amplitude * sin(2 * pi * t), extended=fingers,
                ), 3)
                self.assertEqual(events, [])
                self.assertEqual(self.detector.diagnostics["extended_fingers"], len(fingers))
                self.assertEqual(self.detector.qualifying_palm, len(fingers) >= 3)

    def test_landmark_gate_can_be_disabled_without_changing_raw_score(self):
        detector = WaveDetector(replace(WaveConfig(), use_landmark_openness=False), "camera", "test")
        self.assertEqual(run(detector, lambda t: anatomical_hand(t, dx=50 * sin(2 * pi * t))), [])
        self.assertEqual(detector.diagnostics["extended_fingers"], 4)
        self.assertEqual(detector.diagnostics["open_hand_source"], "none")
        self.assertFalse(detector.diagnostics["open_hand"])
        detector.update(anatomical_hand(2.05, score=.9), 2.05)
        self.assertEqual(detector.diagnostics["open_hand_source"], "classifier")
        self.assertTrue(detector.qualifying_palm)

    def test_missing_landmarks_clear_geometric_openness_and_motion(self):
        missing_samples = [None, replace(anatomical_hand(.4), landmarks=()),
                           replace(anatomical_hand(.4), landmarks=((0.1, 0.2),) * 21)]
        for missing in missing_samples:
            with self.subTest(missing=missing):
                self.detector.reset()
                run(self.detector, anatomical_hand, duration=.3)
                self.assertTrue(self.detector.diagnostics["armed"])
                self.assertIsNone(self.detector.update(missing, .4))
                diagnostics = self.detector.diagnostics
                self.assertEqual(diagnostics["extended_fingers"], 0)
                self.assertEqual(diagnostics["open_hand_source"], "none")
                self.assertFalse(diagnostics["open_hand"])
                self.assertFalse(diagnostics["armed"])
                self.assertEqual(diagnostics["samples"], 0)

    def test_closed_fingers_cannot_trigger_during_geometric_grace(self):
        baseline = WaveDetector(self.detector.config, "camera", "test")
        trigger_time = run(baseline, lambda t: anatomical_hand(t, dx=50 * sin(2 * pi * t)))[0].captured_at
        for frame in range(round(trigger_time * 20) + 1):
            t = frame / 20
            fingers = () if t == trigger_time else (0, 1, 2, 3)
            self.assertIsNone(self.detector.update(anatomical_hand(
                t, dx=50 * sin(2 * pi * t), extended=fingers,
            ), t))
        self.assertTrue(self.detector.diagnostics["armed"])
        self.assertFalse(self.detector.diagnostics["open_hand"])
        next_time = trigger_time + .05
        self.assertIsNotNone(self.detector.update(
            anatomical_hand(next_time, dx=50 * sin(2 * pi * next_time)), next_time,
        ))

    def test_sustained_folded_fingers_clear_geometric_motion(self):
        events = run(self.detector, lambda t: anatomical_hand(
            t, dx=50 * sin(2 * pi * t), extended=(0, 1, 2, 3) if t < .5 else (),
        ))
        self.assertEqual(events, [])
        self.assertFalse(self.detector.diagnostics["armed"])
        self.assertEqual(self.detector.diagnostics["samples"], 0)
        self.assertEqual(run(self.detector, anatomical_hand, start=2.1), [])

    def test_straight_joints_pointing_toward_wrist_are_not_open(self):
        sample = anatomical_hand(0)
        points = list(sample.landmarks)
        for mcp in (5, 9, 13, 17):
            x, y, z = points[mcp]
            for offset, dy in ((1, 8), (2, 16), (3, 24)):
                points[mcp + offset] = (x, y + dy / sample.height, z)
        self.detector.update(replace(sample, landmarks=tuple(points)), 0)
        self.assertEqual(self.detector.diagnostics["extended_fingers"], 0)
        self.assertFalse(self.detector.qualifying_palm)


if __name__ == "__main__":
    unittest.main()
