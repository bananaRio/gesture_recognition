# Wave to Command

A standalone laptop demonstration of **wave an open hand → speak one English voice command → a simulated Ava robot executes it and confirms aloud**. The camera supplies the activation event; no activation button is required. Each accepted wave starts one command session.

The target machine is an Apple Silicon Mac with native ARM64 Python 3.13; development targets the M4 Pro Mac with 24 GB RAM. This project does not require robot middleware, and no physical robot is commanded. Human webcam, microphone and end-to-end acceptance must be measured separately from automated checks; see [the demo protocol](docs/DEMO.md) and [validation evidence](docs/VALIDATION.md).

## Start

Run from the project directory in a native ARM64 terminal with `uv` available:

```bash
uv sync --locked
uv run gesture-demo setup
uv run gesture-demo check --models --hashes --speak
uv run gesture-demo run
```

Dependency installation and initial model setup require internet. Model assets occupy approximately 1.2 GB; allow at least **5 GB free** for setup, plus space for the Python environment and caches. `setup` downloads the models and records exact revisions, file sizes and SHA256 hashes in `assets/manifest.json`. Repeating setup verifies existing assets; `setup --refresh` explicitly resolves and downloads current revisions.

`check` opens the camera and microphone briefly, checks installed assets and voice availability, and writes `logs/latest-check.json`. The options above additionally load/warm the models, verify hashes and speak a test sentence. To inspect dependencies and models without opening camera or microphone:

```bash
uv run gesture-demo check --no-hardware --models --hashes
```

After installation and setup, recognition and speech inference use local models. There is no cloud model/API fallback or API key requirement. The gesture model and Whisper run on CPU; the command model uses **MLX on Apple Silicon, including its GPU runtime**.

Wait for **READY**, then wave an open palm from side to side. When **LISTENING** appears, lower your hand and say one command. The app passes through **TRANSCRIBING → THINKING → SPEAKING → COOLDOWN** and returns to READY after the cooldown and observed hand release. Further waves during a session do not start another one. Press **Escape** or close the preview to exit.

## Wave detection: MediaPipe and motion rules

MediaPipe's Gesture Recognizer finds the hand's 21 landmarks and scores the static `Open_Palm` pose. Our code decides whether the hand is waving:

1. **Open hand.** The hand qualifies when the `Open_Palm` score reaches 0.65 or when the landmarks show at least three extended fingers. In live trials the landmark test is what usually qualifies a waving hand, because motion blur keeps the classifier score near 0.6.
2. **Motion over time.** While the hand stays open, the detector tracks two signals in palm widths: palm-centre translation and fingertip swing relative to the wrist. It counts alternating left–right excursions of at least 0.4 palm widths within a 2.2-second window.
3. **Wave.** Three alternating excursions (two reversals) on either signal emit one wave observation.

An open hand held still, a single swipe, vertical movement, a closed fist and a two-finger pose do not trigger. A continuing wave emits once; the detector rearms only after the hand closes, is lowered or leaves the view. The preview shows whether a hand is found, whether it is open and why, its size, and wave progress out of three motion segments.

## Voice commands

After a wave, Silero VAD finds the end of one utterance and local Whisper (`base.en`) transcribes it. Qwen3-1.7B on MLX then chooses **one action name** from a fixed list and a short reply, as one JSON object. Code validates that object and executes the action against an in-memory Ava simulator. The spoken confirmation describes what actually executed.

| Say, for example | Action | Ava replies |
| --- | --- | --- |
| "Look left." / "Could you turn your head to the right?" | `look(direction="left" / "right")` | "Looking left." |
| "Face forward." / "Back to the middle, please." | `look(direction="center")` | "Looking straight ahead." |
| "Say hello." / "Greet our visitor." | `greet()` | "Hello! Nice to meet you." |
| "Stop." / "Stop moving, please." | `stop()` | "Stopping." |
| "What's your status?" / "Where are you looking?" | `get_status()` | "I am ready, looking left." |
| "How are you?" / "Go to the kitchen." | none | A short conversational reply. |

Malformed or out-of-schema model output never runs; it is not repaired or salvaged. Negated, quoted, hypothetical or explanatory mentions of a command ("don't look left", "what happens if I say stop?") and multi-action requests ("say hello then look right") are vetoed even if the model picks an action; Ava then explains or asks which action to do first. These vetoes are conservative guards, not a general semantic guarantee. There is no robot transport, shell execution or navigation function. The simulator's state is shown in the preview and passed to the model only as context for status questions.

## Devices and permissions

Grant Camera and Microphone access to the application that launches the process when macOS prompts. If access is denied, enable the relevant application under **System Settings → Privacy & Security → Camera / Microphone**, then quit and relaunch that application. Launching from Terminal rather than an IDE or agent app may have a different permission entry; if the app reports *Camera unavailable* from one launcher, run it from Terminal.

```bash
uv run gesture-demo devices
uv run gesture-demo check --camera 0 --microphone "REPLACE-WITH-INPUT-DEVICE"
uv run gesture-demo run --camera 0 --microphone "REPLACE-WITH-INPUT-DEVICE"
```

Use the exact input device name or numeric index printed by `devices`. Camera indices are selected separately. If the camera is unavailable, close other camera applications and retry `check` with the intended index. If the voice check fails, install the configured macOS voice or choose an installed voice in `config.toml`; `/usr/bin/say -v '?'` lists available voices. An ERROR state requires correcting the reported problem and restarting the app.

The microphone device stream remains **open while the app runs**, so the macOS microphone indicator can remain visible in READY. The application discards microphone samples outside its explicit listening session and does not use them for background speech recognition. Camera frames and audio samples are not saved. Transcripts and replies appear in the preview; event logs replace their text with character counts but record the chosen command.

## Configuration and logs

`config.toml` contains the starting thresholds: a 2.2-second wave window, three alternating movement segments of at least 0.4 palm widths, a 0.65 classifier threshold (or three extended fingers), 20 inference submissions per second, a 10-second speech-start timeout, 0.7-second end silence, a 15-second utterance limit, a 2-second cooldown and 0.5-second observed hand release. Brief classifier uncertainty preserves continuously observed positions for up to 200 ms; missing landmarks still reset motion evidence. A temporary camera hand track is not a person identity.

Tune on practice trials, then **freeze the configuration for an acceptance batch**. Each run records its effective configuration in the first JSONL event, including command-line overrides. Preserve that log together with `uv.lock` and `assets/manifest.json` when comparing results. An alternate configuration uses a global CLI option before the subcommand:

```bash
uv run gesture-demo --config /absolute/path/to/trial-config.toml run
```

The app prints its timestamped `logs/*.jsonl` path at startup. `wave_observed` records the wave evidence, `speech_progress` records each stage and the routed command, and once-per-second `vision_status` events record numeric hand diagnostics without images. Summarize one run with:

```bash
uv run gesture-demo report logs/REPLACE-WITH-RUN-FILENAME.jsonl
```

The report counts recorded events and gives p50/p95 values for available timing fields. `decision_to_listening_ms` measures policy consideration to microphone-session arming. `end_of_speech_to_playback_ms` measures the final voiced-sample time to launching the local `say` subprocess, including endpointing silence. **Subprocess launch is not measured acoustic output onset.** Wave recall, false activations and command accuracy require the manually labeled trials described in [DEMO.md](docs/DEMO.md).

## Components and future robot integration

| Boundary | Current responsibility |
| --- | --- |
| `HandRecognizer` / `WaveDetector` | Convert unmirrored MediaPipe samples into `WaveObservation` evidence with the motion rules above. |
| `EngagementPolicy` | Reject stale, duplicated, foreign-clock or busy-state observations; create an `InteractionRequest` when READY. |
| `Controller` | Own session state, session/trigger correlation, observed release, cooldown and lifecycle events. |
| `AudioCapture` | Segment session audio with Silero VAD. |
| `LocalSpeechBackend` / `commands.py` | Transcribe with Whisper, route to one validated command with Qwen/MLX, execute it on the simulator and speak through macOS `say`. |
| `app.py` | Connect devices, queues, preview and local event logging. |

Speech inference runs in a spawned process, keeping its native decoder libraries separate from OpenCV, and that process owns the simulator. Shutdown cancels queued/in-flight work and stops the worker's speech-output process group. A robot integration would replace `Simulator` with an adapter that sends the validated `ToolCall` to the robot host, which must own movement and navigation decisions.

`WaveObservation` carries an event ID, camera source, original capture time, explicit local monotonic clock domain, temporary hand track ID, normalized bounding region and measured evidence. Coordinates remain in the **original unmirrored camera frame**, even when the preview is mirrored. Scores and motion evidence do not establish engagement intent, identity or world position. `InteractionRequest.target_person_id` is `None` because this prototype has no person association. There is no ROS node, HTTP service, robot transport or multiperson inference in this project.

Not included: an EmbeddingGemma 2 zero-shot video classifier was tried and removed. It missed both labeled practice waves (MediaPipe caught both, with 0 of 19 false triggers) and took about 2.9 seconds per clip.

## Developer checks

```bash
uv run pytest
uv run ruff check src tests scripts
uv run python scripts/smoke_wave_model.py --download   # real MediaPipe model, synthetic motion; no camera
uv run python scripts/smoke_inference.py               # real Whisper + Qwen routing, synthetic speech; speaks once
```

These checks exercise temporal detection, command validation, audio segmentation, policy and lifecycle behavior with controlled inputs, plus the real models on synthetic hand motion and synthesized speech. Passing them does not establish webcam performance, human speech accuracy, audible playback or robot integration.
