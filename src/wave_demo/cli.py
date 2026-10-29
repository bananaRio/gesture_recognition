"""Command-line entry points. Setup is the only command that accesses the network."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .assets import offline_environment, setup_assets, verify_assets
from .config import load_config, project_root


def prepare_environment(root: Path) -> None:
    for variable, subdir in (("MPLCONFIGDIR", "matplotlib"), ("HF_HOME", "huggingface"), ("TORCH_HOME", "torch")):
        path = root / ".cache" / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def check(config, root, microphone=None, hardware=True, models=False, speak=False, hashes=False):
    offline_environment()
    checks = {}

    def perform(name, function):
        started = time.monotonic()
        try:
            detail = function()
            checks[name] = {"ok": True, "detail": detail, "seconds": round(time.monotonic() - started, 3)}
        except Exception as exc:
            checks[name] = {"ok": False, "error": str(exc), "seconds": round(time.monotonic() - started, 3)}

    def dependencies():
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("The selected speech adapters require Apple Silicon macOS")
        return {name: importlib.metadata.version(name) for name in (
            "mediapipe", "opencv-contrib-python", "sounddevice", "soxr", "silero-vad",
            "onnxruntime", "faster-whisper", "mlx-lm",
        )}

    def assets():
        if errors := verify_assets(root / "assets", config, hashes):
            raise RuntimeError("; ".join(errors))
        return "Manifest and all asset sizes verified" + ("; SHA256 verified" if hashes else "")

    def voice():
        result = subprocess.run(["/usr/bin/say", "-v", "?"], capture_output=True, text=True, check=True)
        if not any(line.startswith(config.speech.voice + " ") for line in result.stdout.splitlines()):
            raise RuntimeError(f"Install the {config.speech.voice} voice in macOS Spoken Content settings")
        if speak:
            subprocess.run(["/usr/bin/say", "-v", config.speech.voice, "-r", str(config.speech.words_per_minute)],
                           input="The local speech prototype is ready for testing.", text=True, check=True, timeout=15)
        return {"voice": config.speech.voice, "playback_requested": speak}

    perform("dependencies", dependencies)
    perform("assets", assets)
    perform("voice", voice)
    if hardware:
        def camera():
            import cv2
            cap = cv2.VideoCapture(config.runtime.camera_index, cv2.CAP_AVFOUNDATION)
            try:
                if not cap.isOpened():
                    raise RuntimeError("Camera unavailable; grant Camera permission to the launching app")
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera returned no frame")
                return {"index": config.runtime.camera_index, "shape": list(frame.shape), "saved": False}
            finally:
                cap.release()

        def mic():
            import sounddevice as sd
            info = sd.query_devices(microphone, "input")
            blocks = []
            def callback(indata, frames, timing, status):
                blocks.append((frames, str(status)))  # Never retain samples.
            with sd.InputStream(device=microphone, channels=1, dtype="float32",
                                samplerate=info["default_samplerate"], callback=callback):
                time.sleep(0.5)
            if not blocks:
                raise RuntimeError("No microphone callbacks; check macOS Microphone permission")
            if any(status for _, status in blocks):
                raise RuntimeError(f"Microphone callback errors: {blocks}")
            return {"device": info["name"], "callbacks": len(blocks), "saved": False}

        perform("camera", camera)
        perform("microphone", mic)
    else:
        checks["hardware"] = {"status": "not_tested"}
    if models and checks["assets"]["ok"]:
        def model_load():
            from .worker import SpeechWorker
            from .vision import HandRecognizer
            from silero_vad import load_silero_vad
            import numpy as np
            import onnxruntime
            import queue
            import torch
            import threading
            onnxruntime.disable_telemetry_events()
            vad = load_silero_vad(onnx=True)
            probability = float(vad(torch.zeros(512), 16000).item())
            received = threading.Event()
            recognizer = HandRecognizer(root / "assets" / "gesture_recognizer.task", lambda *_: received.set())
            try:
                recognizer.submit(np.zeros((480, 640, 3), dtype=np.uint8), time.monotonic())
                if not received.wait(5):
                    raise RuntimeError("MediaPipe did not return a result")
            finally:
                recognizer.close()
            worker = SpeechWorker(config.speech, root / "assets")
            try:
                worker.start()
                deadline = time.monotonic() + 120
                while True:
                    try:
                        kind, detail = worker.events.get(timeout=0.5)
                    except queue.Empty:
                        if not worker.is_alive:
                            raise RuntimeError("Speech worker exited before model startup completed")
                        if time.monotonic() > deadline:
                            raise RuntimeError("Speech worker did not finish loading within 120 seconds")
                        continue
                    if kind == "ready":
                        break
                    raise RuntimeError(str(detail))
            finally:
                worker.close()
            return {"gesture_callback": True, "silero_silence_probability": probability,
                    "whisper_and_llm": "loaded and warmed locally"}
        perform("models", model_load)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "checks": checks, "ok": all(c.get("ok", True) for c in checks.values()),
        "human_acceptance_trials": "not_run",
    }
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    serialized = json.dumps(report, indent=2) + "\n"
    (logs / "latest-check.json").write_text(serialized)
    (logs / f"check-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json").write_text(serialized)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def summarize_log(path: Path) -> dict:
    import math
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    counts = {}
    values = {}
    for record in records:
        kind = record["kind"]
        counts[kind] = counts.get(kind, 0) + 1
        detail = record.get("details", {})
        metrics = detail.get("timings", {}) if kind == "completed" else {}
        if kind == "microphone_armed":
            metrics = {"decision_to_listening_ms": detail["decision_to_listening_ms"]}
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.setdefault(name, []).append(value)
    distributions = {}
    for name, samples in values.items():
        ordered = sorted(samples)
        distributions[name] = {"n": len(ordered), "p50": ordered[math.ceil(len(ordered) * .5) - 1],
                               "p95": ordered[math.ceil(len(ordered) * .95) - 1]}
    return {"events": counts, "metrics": distributions,
            "note": "Wave recall, false-activation rate, and answer quality require labeled human trials."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fully local wave detection and voice-command demo")
    parser.add_argument("--root", type=Path, default=project_root(), help="Project directory containing assets/config/logs")
    parser.add_argument("--config", type=Path, help="TOML configuration override")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="Download model assets; this command requires internet")
    setup.add_argument("--refresh", action="store_true", help="Resolve/download current model revisions")
    for name in ("check", "run"):
        command = sub.add_parser(name)
        command.add_argument("--camera", type=int, help="Camera index (default: 0)")
        command.add_argument("--microphone", help="Input device index or name; defaults to system input")
        if name == "check":
            command.add_argument("--no-hardware", action="store_true", help="Skip opening camera and microphone")
            command.add_argument("--models", action="store_true", help="Load and warm all models")
            command.add_argument("--speak", action="store_true", help="Play a short local TTS check")
            command.add_argument("--hashes", action="store_true", help="Verify all model file SHA256 hashes")
        else:
            command.add_argument("--duration", type=float, help="Exit after N seconds, for smoke testing")
    sub.add_parser("devices", help="List available microphone devices")
    report = sub.add_parser("report", help="Summarize a local event log")
    report.add_argument("log", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.root.expanduser().resolve()
        prepare_environment(root)
        config = load_config(args.config or root / "config.toml")
        if getattr(args, "camera", None) is not None:
            if args.camera < 0:
                raise ValueError("Camera index must be nonnegative")
            config = replace(config, runtime=replace(config.runtime, camera_index=args.camera))
        microphone = getattr(args, "microphone", None)
        if microphone is not None and microphone.isdecimal():
            microphone = int(microphone)
        if args.command == "setup":
            manifest = setup_assets(root / "assets", config, args.refresh)
            print(f"Assets ready: {len(manifest['files'])} verified files")
            return 0
        if args.command == "devices":
            import sounddevice as sd
            print(sd.query_devices())
            return 0
        if args.command == "check":
            return check(config, root, microphone, not args.no_hardware, args.models, args.speak, args.hashes)
        if args.command == "report":
            print(json.dumps(summarize_log(args.log), indent=2))
            return 0
        if args.duration is not None and args.duration <= 0:
            raise ValueError("Duration must be positive")
        from .app import run
        return run(config, root, microphone, args.duration)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
