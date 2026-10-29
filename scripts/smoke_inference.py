"""Real local ASR -> command routing -> TTS check using synthesized, non-user speech.

Routes fixed command texts through the local model, then runs one synthesized
spoken command end to end and speaks its confirmation. Does not open the webcam
or microphone. Outbound Python socket connections are rejected throughout model
load and inference.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import socket
import subprocess
import time

from wave_demo.assets import offline_environment, verify_assets
from wave_demo.cli import prepare_environment
from wave_demo.config import load_config, project_root
from wave_demo.contracts import Utterance
from wave_demo.speech import LocalSpeechBackend

# (text, expected function or None, expected look direction). The last seven are
# held-out paraphrases: they are not the prompt's demonstration texts.
CASES = [
    ("Please look left.", "look", "left"),
    ("Could you turn your head to the right?", "look", "right"),
    ("Please face forward.", "look", "center"),
    ("Please say hello to me.", "greet", None),
    ("Stop.", "stop", None),
    ("What is your current status?", "get_status", None),
    ("How are you?", None, None),
    ("Do not look left.", None, None),
    ("What happens if I say stop?", None, None),
    ("Explain the command look left.", None, None),
    ("Go to the kitchen.", None, None),
    ("Say hello then look right.", None, None),
    ("Ava, would you mind glancing to the left?", "look", "left"),
    ("Move your gaze back to the middle, please.", "look", "center"),
    ("Could you greet our visitor?", "greet", None),
    ("Stop moving, please.", "stop", None),
    ("Tell me where you are looking.", "get_status", None),
    ("It is important that you don't look to the right.", None, None),
    ("Please dance.", None, None),
]
SPOKEN = "Ava, please look to your right."


def main():
    root = project_root()
    config = load_config(root / "config.toml")
    prepare_environment(root)
    offline_environment()
    if errors := verify_assets(root / "assets", config):
        raise RuntimeError("; ".join(errors))
    fixture = root / ".cache" / "synthetic-command.aiff"
    subprocess.run(["/usr/bin/say", "-v", config.speech.voice, "-o", str(fixture)],
                   input=SPOKEN, text=True, check=True, timeout=15)
    blocked = []
    original_socket = socket.socket

    class OfflineSocket(original_socket):
        def connect(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                blocked.append(str(address))
                raise RuntimeError("Network access forbidden during offline smoke check")
            return super().connect(address)

        def connect_ex(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                blocked.append(str(address))
                raise RuntimeError("Network access forbidden during offline smoke check")
            return super().connect_ex(address)

    socket.socket = OfflineSocket
    backend = LocalSpeechBackend(config.speech, root / "assets")
    started = time.monotonic()
    events = []
    try:
        backend.load()
        backend.warmup()
        loaded = time.monotonic()
        cases = []
        for text, name, direction in CASES:
            began = time.monotonic()
            reply, command = backend._route(text, backend.cancellation_token, began + 30, {})
            expected = None if name is None else {
                "name": name, "arguments": {"direction": direction} if direction else {}}
            cases.append({"text": text, "expected": expected, "call": command["call"],
                          "outcome": command["outcome"], "reply": reply,
                          "routing_ms": round((time.monotonic() - began) * 1000),
                          "passed": command["call"] == expected and command["outcome"] != "invalid_output"})
            print(json.dumps(cases[-1]), flush=True)
        from faster_whisper.audio import decode_audio
        samples = decode_audio(str(fixture), sampling_rate=16000)
        backend.process(Utterance("synthetic-offline-smoke", samples, time.monotonic()), events.append)
        transcript = next((event.text for event in events if event.kind == "thinking"), "")
        speaking = next((event for event in events if event.kind == "speaking"), None)
        spoken_ok = bool(events and events[-1].kind == "completed" and speaking and speaking.command.get("call")
                         == {"name": "look", "arguments": {"direction": "right"}})
        success = spoken_ok and not blocked and all(case["passed"] for case in cases)
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": config.speech.llm_repo,
            "microphone_or_camera_used": False,
            "python_network_connections_blocked": True,
            "attempted_connections": blocked,
            "load_and_warmup_seconds": loaded - started,
            "routing_cases_passed": f"{sum(case['passed'] for case in cases)}/{len(cases)}",
            "cases": cases,
            "spoken_input": f"Synthesized {config.speech.voice} voice: {SPOKEN}",
            "transcript": transcript, "response": speaking.text if speaking else "",
            "events": [asdict(event) for event in events],
            "ok": success,
            "limitations": "Synthetic input; excludes gesture, microphone, human trials, and acoustic latency.",
        }
        (root / "logs").mkdir(exist_ok=True)
        (root / "logs" / "inference-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in ("routing_cases_passed", "transcript", "response",
                                                       "load_and_warmup_seconds", "ok")}, indent=2), flush=True)
        return 0 if success else 1
    finally:
        backend.close()
        socket.socket = original_socket
        fixture.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
