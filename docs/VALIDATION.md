# Validation record

Measured on October 28, 2026, on the M4 Pro MacBook Pro (24 GB, macOS 26.6.2,
native ARM64 Python 3.13.3). This record distinguishes automated/model checks
from the human acceptance protocol in [DEMO.md](DEMO.md). It covers the
wave-to-command app: MediaPipe and motion rules for activation, faster-whisper
`base.en` for transcription and Qwen3-1.7B (4-bit, MLX) for command routing.

## Completed checks

- **116 automated tests and 14 subtests pass**, covering the wave detector,
  command schema and model-output validation, vetoes, the simulator, the speech
  pipeline with fake models, controller/policy, audio segmentation,
  configuration, asset integrity, logging, actual spawned-worker cancellation
  and orphaned playback cleanup. Ruff passes.
- `gesture-demo check --no-hardware --models --hashes` passes: all 14 asset
  files match their SHA256 manifest; MediaPipe returns a callback; Silero scores
  a silent frame at 0.0017 speech probability; Whisper and Qwen load and warm in
  the spawned speech worker.
- **Real MediaPipe model, synthetic motion** (`scripts/smoke_wave_model.py`):
  the official MediaPipe hand image, translated across a white 640×480 frame at
  20 FPS, produced 0 waves held still, 0 waves for a one-way swipe and 1 wave for
  side-to-side motion (two reversals, 1.2 palm widths). The `Open_Palm` score
  stayed at 0.60–0.62, below the 0.65 threshold, so openness came from the
  landmark finger test. Translating a still image is not a real wave.
- **Real command routing** (`scripts/smoke_inference.py`, Python network access
  blocked): 19 of 19 cases correct. These are 12 demonstration texts from the
  prompt and 7 held-out paraphrases, including negated, hypothetical, compound
  and unsupported requests. Routing took 0.79–1.23 s per command after a
  7.5-second load and warmup. A synthesized spoken command, “Ava, please look to
  your right.”, was transcribed exactly, routed to `look(direction="right")`
  and confirmed aloud with “Looking right.”

## Choosing MediaPipe and motion rules

A paired practice comparison on the webcam (21 labeled trials, the same 16
frames per trial for both detectors) gave:

| Detector | Waves detected | False triggers | Median compute per clip |
| --- | --- | --- | --- |
| MediaPipe and motion rules | 2 of 2 | 0 of 19 | 79 ms |
| EmbeddingGemma 2 zero-shot (CPU) | 0 of 2 | 0 of 19 | 2,860 ms |

The negatives were a held open palm, a moving fist, a moving peace sign,
ordinary activity and an empty view. These are practice trials with operator
labels, not an acceptance batch. The EmbeddingGemma experiment was removed from
the project afterwards.

In an earlier 21-minute live session of the same MediaPipe detector
(`logs/20261028-144812-cbfe2f0b.jsonl`), 11 waves were observed and 9 started a
session. The other 2 arrived while a session or cooldown was in progress and
were correctly rejected. Every accepted wave qualified as open through the
landmark finger test, with classifier scores of only 0.53–0.64. Six used the
fingertip-swing signal and three palm translation.

## Integration fixes that still apply

- Use one OpenCV distribution (`opencv-contrib-python`) to avoid overlapping
  `cv2` installations.
- Constrain PyAV to versions 14–16 because faster-whisper 1.2's file decoder
  uses an API removed in a newer PyAV release.
- Isolate speech inference in a spawned process so OpenCV and PyAV do not load
  conflicting native AVFoundation classes into the same process. Terminating
  that worker also terminates its speech-output process group.
- Track release using captured observation times rather than GUI-delivery times.
- Reject cancelled jobs even if cancellation races with dispatch; native compute
  cannot resurrect a cancelled interaction.
- Static `Open_Palm` confidence is often near zero or just below threshold while
  a hand moves, so an alternative open-hand gate uses at least three extended
  fingers from the landmarks. Motion uses a 2.2-second window and three
  alternating excursions of at least 0.4 palm widths.
- Preserve motion through brief classifier uncertainty and one-frame handedness
  flips when landmark tracking remains continuous. Reset the detector on entry
  to `READY`, so an ignored startup wave cannot latch out subsequent gestures.

## Human acceptance: not yet measured

| Criterion | Status |
| --- | --- |
| ≥18/20 deliberate waves, two hands/two lighting conditions | Not run |
| Zero false activations during five eligible minutes | Not run |
| ≥11/12 correct live spoken commands, no wrong action | Not run |
| Live p95 decision-to-listening ≤300 ms | Not measured |
| Live p95 speech-end-to-confirmation ≤3 seconds | Not measured |
| Full hardware workflow after OS-level network disconnection | Not run |

These checks must be scored with a person at the webcam. Synthetic landmark,
image and speech tests are not substitutes for labeled trials. The project makes
no claim about multi-person engagement, robot motion, speaker identity, or
deployment on the robot.
