"""Local-only ASR, MLX voice-command routing and native macOS speech output."""

import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from .commands import (ROUTING_EXAMPLES, SYSTEM_PROMPT, Simulator, is_command_discussion,
                       is_compound_request, parse_model_response)
from .config import SpeechConfig
from .contracts import SpeechEvent, Utterance


RETRY_PROMPT = "I didn't catch that. Please wave and try again."
INVALID_PROMPT = "Sorry, I couldn't turn that into a command. Please try again."
DISCUSSION_PROMPT = "I left everything unchanged because that sounded like a discussion or a negated command."
COMPOUND_PROMPT = "I can do one action at a time. Which one should I do first?"


class _Cancelled(Exception):
    pass


class LocalSpeechBackend:
    """Call load/warmup/process on one owning worker; cancel/close are thread-safe.

    Native ASR calls cannot be interrupted mid-kernel. Cancellation invalidates
    their results and token generation checks cancellation between tokens.
    The backend owns the simulated robot, so only executed commands change it.
    """

    def __init__(self, config: SpeechConfig, assets_dir: Path):
        self.config = config
        self.assets_dir = Path(assets_dir)
        self.simulator = Simulator()
        self._owner_thread = None
        self._loaded = False
        self._generation = 0
        self._closed = False
        self._lock = threading.Lock()
        self._say_process = None

    @property
    def cancellation_token(self) -> int:
        """Capture when dispatching a job, before handing it to the worker."""
        with self._lock:
            return self._generation

    def _assert_owner(self) -> None:
        if self._owner_thread != threading.get_ident():
            raise RuntimeError("Load, warmup and inference must use the same speech worker thread")

    def load(self) -> None:
        if self._closed:
            raise RuntimeError("Speech backend is closed")
        if self._loaded:
            self._assert_owner()
            return
        self._owner_thread = threading.get_ident()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        whisper_dir = self.assets_dir / "whisper"
        llm_dir = self.assets_dir / "llm"
        required = [whisper_dir / name for name in ("model.bin", "config.json", "tokenizer.json")]
        required += [llm_dir / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json")]
        missing = [str(path) for path in required if not path.is_file()]
        if not list(llm_dir.glob("*.safetensors")):
            missing.append(str(llm_dir / "*.safetensors"))
        if missing:
            raise FileNotFoundError("Missing local speech assets; run setup first: " + ", ".join(missing))
        voices = subprocess.run(
            ["/usr/bin/say", "-v", "?"], check=True, capture_output=True, text=True, timeout=5
        ).stdout
        if not any(line.startswith(self.config.voice + " ") for line in voices.splitlines()):
            raise RuntimeError(f"macOS voice {self.config.voice!r} is not installed")
        from faster_whisper import WhisperModel
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler

        self._whisper = WhisperModel(
            str(whisper_dir), device="cpu", compute_type="int8",
            cpu_threads=self.config.cpu_threads, num_workers=1, local_files_only=True,
        )
        self._llm, self._tokenizer = load(
            str(llm_dir), tokenizer_config={"local_files_only": True, "trust_remote_code": False}
        )
        self._stream_generate = stream_generate
        self._sampler = make_sampler(temp=0.0)
        self._loaded = True

    def warmup(self) -> None:
        self._assert_owner()
        if not self._loaded:
            raise RuntimeError("Load speech assets before warming up")
        import numpy as np

        segments, _ = self._whisper.transcribe(
            np.zeros(8000, dtype=np.float32), language="en", beam_size=1,
            temperature=0, vad_filter=False, condition_on_previous_text=False,
        )
        list(segments)
        for _ in self._stream_generate(
            self._llm, self._tokenizer, prompt=self._prompt("Hello."), max_tokens=2, sampler=self._sampler
        ):
            pass

    def _prompt(self, transcript: str) -> str:
        # The state is context for status questions, not an instruction.
        messages = [{"role": "system", "content": SYSTEM_PROMPT + "\nCurrent simulator state: "
                     + json.dumps(self.simulator.snapshot())}]
        for example, reply, action in ROUTING_EXAMPLES:
            messages.append({"role": "user", "content": example})
            messages.append({"role": "assistant", "content": json.dumps({"action": action, "reply": reply})})
        messages.append({"role": "user", "content": transcript})
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def _check(self, generation: int, deadline: float) -> None:
        if self._closed or generation != self._generation:
            raise _Cancelled()
        if time.monotonic() >= deadline:
            raise TimeoutError("Local speech processing timed out")

    def _route(self, transcript: str, generation: int, deadline: float, timings: dict) -> tuple[str, dict]:
        """Turn one transcript into at most one validated, executed command."""
        generation_started = time.monotonic()
        parts = []
        stream = self._stream_generate(
            self._llm, self._tokenizer, prompt=self._prompt(transcript),
            max_tokens=self.config.max_tokens, sampler=self._sampler,
        )
        try:
            for response in stream:
                self._check(generation, deadline)
                if not parts:
                    timings["first_token_ms"] = (time.monotonic() - generation_started) * 1000
                parts.append(response.text)
        finally:
            if hasattr(stream, "close"):
                stream.close()
        timings["generation_ms"] = (time.monotonic() - generation_started) * 1000
        try:
            call, reply = parse_model_response("".join(parts))
        except ValueError:
            # Never repair, salvage or execute malformed model output.
            return INVALID_PROMPT, {"call": None, "outcome": "invalid_output",
                                    "state": self.simulator.snapshot()}
        if call is not None and is_command_discussion(transcript):
            call, reply, outcome = None, DISCUSSION_PROMPT, "vetoed_discussion"
        elif call is not None and is_compound_request(transcript):
            call, reply, outcome = None, COMPOUND_PROMPT, "vetoed_compound"
        else:
            outcome = "executed" if call else "conversation"
        if call is not None:
            # Spoken confirmations describe what actually executed.
            reply = self.simulator.apply(call)
        return clean_response(reply), {"call": call.as_dict() if call else None, "outcome": outcome,
                                       "state": self.simulator.snapshot()}

    def process(self, utterance: Utterance, emit, *, cancellation_token: int | None = None) -> None:
        """Process one exchange using the token captured at worker dispatch.

        ``end_of_speech_to_playback_ms`` measures the last voiced capture sample
        to launching ``say``. It includes endpointing, queueing, ASR and command
        routing, but does not measure the speaker's acoustic onset delay.
        """
        self._assert_owner()
        if not self._loaded:
            raise RuntimeError("Speech backend is not loaded")
        generation = self.cancellation_token if cancellation_token is None else cancellation_token
        started = time.monotonic()
        deadline = started + self.config.processing_timeout_seconds
        timings = {}

        def send(kind, text="", error="", command=None):
            self._check(generation, deadline)
            emit(SpeechEvent(utterance.session_id, kind, text=text, error=error,
                             timings=dict(timings), command=command or {}))

        try:
            send("transcribing")
            transcript = ""
            if utterance.reason != "no_speech" and len(utterance.samples):
                segments, _ = self._whisper.transcribe(
                    utterance.samples, language="en", beam_size=1, temperature=0,
                    vad_filter=False, condition_on_previous_text=False,
                )
                parts = []
                for segment in segments:
                    self._check(generation, deadline)
                    parts.append(segment.text.strip())
                transcript = " ".join(parts).strip()
            timings["transcription_ms"] = (time.monotonic() - started) * 1000
            command = None
            if transcript:
                send("thinking", transcript)
                reply, command = self._route(transcript, generation, deadline, timings)
                if not reply:
                    raise RuntimeError("The local command model returned no spoken text")
            else:
                reply = RETRY_PROMPT
            send("speaking", reply, command=command)
            speech_started = time.monotonic()
            playback_started = self._speak(reply, generation, deadline)
            if utterance.reason != "no_speech" and len(utterance.samples):
                timings["end_of_speech_to_playback_ms"] = (playback_started - utterance.ended_at) * 1000
            timings["speech_ms"] = (time.monotonic() - speech_started) * 1000
            timings["total_ms"] = (time.monotonic() - started) * 1000
            send("completed", reply)
        except _Cancelled:
            return
        except TimeoutError as exc:
            if not self._closed and generation == self._generation:
                timings["total_ms"] = (time.monotonic() - started) * 1000
                emit(SpeechEvent(utterance.session_id, "timeout", error=str(exc), timings=timings))
        except Exception as exc:
            # No cloud fallback and no fabricated answer. The controller presents
            # this recoverable failure and invalidates the current interaction.
            if not self._closed and generation == self._generation:
                timings["total_ms"] = (time.monotonic() - started) * 1000
                emit(SpeechEvent(utterance.session_id, "failed", error=str(exc), timings=timings))

    def _speak(self, text: str, generation: int, deadline: float) -> float:
        """Speak safely and return monotonic subprocess launch time, not audio onset."""
        self._check(generation, deadline)
        with self._lock:
            self._check(generation, deadline)
            process = subprocess.Popen(
                ["/usr/bin/say", "-v", self.config.voice, "-r", str(self.config.words_per_minute)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            launched_at = time.monotonic()
            self._say_process = process
        try:
            _, stderr = process.communicate(text, timeout=max(0.1, deadline - time.monotonic()))
            self._check(generation, deadline)
            if process.returncode:
                raise RuntimeError("macOS speech output failed: " + (stderr or "unknown error").strip())
            return launched_at
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise TimeoutError("macOS speech output timed out") from None
        finally:
            with self._lock:
                if self._say_process is process:
                    self._say_process = None

    def cancel(self) -> None:
        with self._lock:
            self._generation += 1
            process = self._say_process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass

    def close(self) -> None:
        self._closed = True
        self.cancel()


def clean_response(text: str) -> str:
    """Remove accidental reasoning markup and control characters before TTS."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^>]*\|>", "", text)
    text = text.replace("`", "").replace("**", "")
    return " ".join(text.split()).strip()
