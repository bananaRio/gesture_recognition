# Live demonstration and acceptance record

The demonstration claim is narrow: **on this laptop, one deliberate open-hand wave can start one English voice command, which a simulated Ava executes and confirms aloud, without an activation button**. Record actual successes and failures before describing reliability. Current automated evidence belongs in [VALIDATION.md](VALIDATION.md); this document is a procedure, not a completed test report.

## Prepare the run

Run from the project directory:

```bash
uv sync --locked
uv run gesture-demo setup
uv run gesture-demo check --models --hashes --speak
uv run gesture-demo run
```

Initial installation/setup needs internet. Subsequent inference uses the downloaded models locally. Close competing camera apps, select the intended microphone and grant macOS Camera/Microphone permissions to the launching application (Terminal works). Confirm that the check sentence is actually audible; a successful subprocess exit alone does not establish that the selected speaker volume/output is correct.

Use a seated single demonstrator at a comfortable webcam distance, with the palm fully visible. Start in even lighting. Spend a few practice attempts on hand size, movement amplitude and speech volume. If tuning is needed, change `config.toml`, restart and repeat practice. Freeze settings before scored trials; retain the run log, its `run_started` configuration, `uv.lock` and `assets/manifest.json`.

## Short presentation script

1. Wait until the preview says **READY**. Explain: “The laptop is waiting for a wave.” The microphone device is already open, but samples outside the listening session are discarded.
2. Hold an open hand toward the camera without moving it. Nothing should happen: an open hand alone is not a wave.
3. Wave it side to side through several clear reversals. On **LISTENING**, lower the hand and say: “Look to your left.” The preview shows the transcript, `look(direction="left")` and the simulated Ava looking left, and the Mac says “Looking left.”
4. During SPEAKING, wave once more to demonstrate that an active session is not restarted.
5. Lower the hand, wait through COOLDOWN until READY and repeat with “What's your status?”, “Say hello”, “Stop” and “Face forward”.
6. Show a vetoed request: wave, then say “Don't look right.” Ava should leave the simulator unchanged.
7. Exit with Escape. Keep the local log for review.

There is one command per activation. The command model cannot see the camera, identify a person, control a robot or perform navigation. A future robot should treat a detected wave as evidence for its engagement policy, not as a movement command.

## Scored trials

Use separate practice and scored attempts. Wait for READY before each deliberate wave; waves intentionally made while busy belong to the duplicate-trigger check, not the missed-wave denominator. Define success before starting: one deliberate wave produces exactly one LISTENING transition without a button.

Run **20 deliberate-wave trials**: five left-hand and five right-hand trials in each of two lighting conditions. Record the actual lighting and camera distance rather than calling them universally representative. A useful demo acceptance target is **at least 18 successes out of 20**, with no double activation. This target is a prospective criterion, not a measured result.

| Condition | Left-hand successes / 5 | Right-hand successes / 5 | Misses or double activations |
| --- | --- | --- | --- |
| Lighting A: describe | Not run | Not run | Not run |
| Lighting B: describe | Not run | Not run | Not run |

Assess **12 spoken command trials**, which may be twelve of the wave trials above: two each of look left, look right, face forward, say hello, stop and status, using your own wording. For each, label separately: activation succeeded; transcript preserved the intended command; the logged command matched it; the confirmation was audible; session ended and rearmed. A useful end-to-end target is **at least 11 of 12 correct commands**, with **no wrong action executed**. Retain failures in the denominator, including no-speech timeouts, incorrect transcriptions, wrong or missing actions and device errors.

| Trial | Spoken command / expected action | Wave activated once | Transcript usable | Logged command correct | Confirmation audible | Returned to READY | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1–12: fill one row per trial | Not run | Not run | Not run | Not run | Not run | Not run | Not run |

Then observe **five full minutes in READY with no intended activation**. Include typing, face touching, ordinary conversational hand movements, a stationary raised open hand and an empty camera view. Record the time actually spent eligible for activation. If a false activation occurs, record it, allow the session to finish, and resume the READY-time counter. The demo target is **zero observed false activations during five eligible minutes**. Zero in this short trial does not establish a zero long-term false-activation rate.

Check these lifecycle cases as well:

- Continue waving during LISTENING, THINKING and SPEAKING: no additional session should start.
- Keep the qualifying palm up after a response: rearming should wait for observed release and cooldown.
- Wave, then remain silent: after the configured speech-start timeout, the app should issue its local retry response and recover through cooldown.
- Say a negated, hypothetical or two-action request: no action should execute.
- Exit while a command is being processed or spoken: the app should close and stop playback. Restart before further scoring.

## Read the log without inventing labels

The application prints a path such as `logs/<timestamp>-<run-id>.jsonl`. Use the actual filename:

```bash
uv run gesture-demo report logs/REPLACE-WITH-RUN-FILENAME.jsonl
```

The report provides event counts and p50/p95 values for recorded software timings. Each `speech_progress` event at the speaking stage records the routed command (`call` and `outcome`), so commands can be matched to trials without storing what was said. The report cannot know how many unrecognized waves the person attempted or what they intended to say. Match manual trial IDs and times to the logged trigger/session IDs when reviewing outcomes. Calculate recall from labeled deliberate attempts; calculate false activations from labeled unintended starts divided by eligible observation time. Never treat `wave_observed` count as the total number of attempted waves.

`decision_to_listening_ms` measures policy consideration to microphone-session arming. `end_of_speech_to_playback_ms` measures the final voiced-sample time to `say` subprocess launch, including the silence used to detect the utterance end. It does **not** measure sound leaving the speaker. For a claim about perceived response latency, manually annotate an external recording that captures both the user's speech and the actual speaker output; recording is not built into this app.

The preview FPS is the display loop rate, not an inference latency benchmark. Logs omit transcript/reply contents and do not contain saved webcam frames or audio. Manual labels are therefore necessary to assess recognition quality.

## Scope when reporting to the PI

Report the machine, model revisions, frozen configuration, trial counts, observed failures and the boundaries used for timing. Describe unrun hardware or human trials as **not tested**, even if unit tests and model warmup pass. A successful laptop demonstration supports proceeding to a robot integration pilot; it does not establish person-aware engagement or the reliability needed to remove every fallback input.

Future integration should retain original capture timestamps and clock domains, unmirrored image coordinates and trigger-to-session correlation. A temporary hand track must remain distinct from a person ID; `target_person_id` stays `None` until an actual person-association component supplies evidence. The robot host owns motion; an adapter in place of the simulator would receive only validated tool calls. This prototype implements no middleware transport or robot behavior.
