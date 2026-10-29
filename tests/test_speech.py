from types import SimpleNamespace
import subprocess
import threading
import time

import numpy as np
import pytest

from wave_demo.config import SpeechConfig
from wave_demo.contracts import Utterance
from wave_demo.speech import (COMPOUND_PROMPT, DISCUSSION_PROMPT, INVALID_PROMPT, LocalSpeechBackend,
                               RETRY_PROMPT, _Cancelled, clean_response)


def heard(backend, text):
    backend._whisper.transcribe = lambda *args, **kwargs: (iter([SimpleNamespace(text=f" {text} ")]), None)


def model_says(backend, output):
    # Stream the routing JSON in pieces, as MLX does.
    backend._stream_generate = lambda *args, **kwargs: iter(
        [SimpleNamespace(text=output[:10]), SimpleNamespace(text=output[10:])])


@pytest.fixture
def backend(tmp_path):
    instance = LocalSpeechBackend(SpeechConfig(), tmp_path)
    instance._owner_thread = threading.get_ident()
    instance._loaded = True
    instance._whisper = SimpleNamespace()
    heard(instance, "Could you look to the left?")
    instance.prompts = []
    instance._tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: (instance.prompts.append((messages, kwargs)), "prompt")[1])
    instance._llm = object()
    instance._sampler = object()
    model_says(instance, '{"action": "look_left", "reply": "Sure, looking to the left now!"}')
    return instance


def exchange(backend, monkeypatch, session="one"):
    spoken = []
    monkeypatch.setattr(backend, "_speak", lambda text, *args: (spoken.append(text), time.monotonic())[1])
    events = []
    backend.process(Utterance(session, np.ones(16000, np.float32), time.monotonic()), events.append)
    return events, spoken


def test_one_command_is_executed_and_confirmed_from_the_simulator(backend, monkeypatch):
    events, spoken = exchange(backend, monkeypatch)
    assert [event.kind for event in events] == ["transcribing", "thinking", "speaking", "completed"]
    assert events[1].text == "Could you look to the left?"
    # The confirmation describes what executed, not the model's free text.
    assert spoken == ["Looking left."]
    command = events[2].command
    assert command["call"] == {"name": "look", "arguments": {"direction": "left"}}
    assert command["outcome"] == "executed"
    assert command["state"]["direction"] == "left"
    assert backend.simulator.snapshot()["direction"] == "left"
    assert all(event.session_id == "one" for event in events)
    assert events[-1].timings["total_ms"] >= 0


def test_routing_prompt_is_non_thinking_and_includes_current_state(backend, monkeypatch):
    exchange(backend, monkeypatch)
    heard(backend, "Where are you looking?")
    model_says(backend, '{"action": "get_status", "reply": "Checking."}')
    _, spoken = exchange(backend, monkeypatch, "two")
    messages, options = backend.prompts[-1]
    assert options["enable_thinking"] is False and options["tokenize"] is False
    assert '"direction": "left"' in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Where are you looking?"}
    assert spoken == ["I am ready, looking left."]


def test_conversation_speaks_the_model_reply_without_an_action(backend, monkeypatch):
    heard(backend, "How are you?")
    model_says(backend, '{"action": "none", "reply": "I\'m ready to help."}')
    events, spoken = exchange(backend, monkeypatch)
    assert spoken == ["I'm ready to help."]
    assert events[2].command["call"] is None
    assert events[2].command["outcome"] == "conversation"
    assert backend.simulator.snapshot()["last_action"] is None


@pytest.mark.parametrize("transcript,output,reply,outcome", [
    ("Do not look left.", '{"action": "look_left", "reply": "Looking left."}', DISCUSSION_PROMPT,
     "vetoed_discussion"),
    ("Say hello then look right.", '{"action": "greet", "reply": "Hello!"}', COMPOUND_PROMPT, "vetoed_compound"),
    ("Look left.", 'Sure! {"action": "look_left", "reply": "OK"}', INVALID_PROMPT, "invalid_output"),
    ("Look left.", '{"action": "run_python", "reply": "OK"}', INVALID_PROMPT, "invalid_output"),
])
def test_vetoed_or_invalid_routing_never_executes(backend, monkeypatch, transcript, output, reply, outcome):
    heard(backend, transcript)
    model_says(backend, output)
    events, spoken = exchange(backend, monkeypatch)
    assert spoken == [reply]
    assert events[2].command["call"] is None
    assert events[2].command["outcome"] == outcome
    assert backend.simulator.snapshot()["last_action"] is None


def test_silence_does_not_call_asr_or_reply_model(backend, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No inference should run on a no-speech result")
    backend._whisper.transcribe = forbidden
    backend._stream_generate = forbidden
    spoken = []
    monkeypatch.setattr(backend, "_speak", lambda text, *args: (spoken.append(text), time.monotonic())[1])
    events = []
    backend.process(Utterance("quiet", np.empty(0), time.monotonic(), "no_speech"), events.append)
    assert spoken == [RETRY_PROMPT]
    assert [event.kind for event in events] == ["transcribing", "speaking", "completed"]


def test_cancel_during_generation_suppresses_playback_and_late_completion(backend, monkeypatch):
    def cancelled_stream(*args, **kwargs):
        yield SimpleNamespace(text='{"action": "stop", ')
        backend.cancel()
        yield SimpleNamespace(text='"reply": "Stopping."}')
    backend._stream_generate = cancelled_stream
    monkeypatch.setattr(backend, "_speak", lambda *args: pytest.fail("Cancelled text must not be spoken"))
    events = []
    backend.process(Utterance("cancelled", np.ones(512), time.monotonic()), events.append)
    assert [event.kind for event in events] == ["transcribing", "thinking"]
    assert backend.simulator.snapshot()["stopped"] is False


def test_native_inference_failure_is_reported(backend):
    def failing(*args, **kwargs):
        raise RuntimeError("local model error")
    backend._whisper.transcribe = failing
    events = []
    backend.process(Utterance("failed", np.ones(512), time.monotonic()), events.append)
    assert events[-1].kind == "failed"
    assert events[-1].error == "local model error"


def test_cancelled_dispatch_token_cannot_resurrect_job(backend, monkeypatch):
    token = backend.cancellation_token
    backend.cancel()
    assert backend.cancellation_token != token
    def forbidden(*args, **kwargs):
        pytest.fail("A cancelled dispatched job must not run inference or playback")
    backend._whisper.transcribe = forbidden
    backend._stream_generate = forbidden
    monkeypatch.setattr(backend, "_speak", forbidden)
    events = []
    backend.process(
        Utterance("stale", np.ones(512), time.monotonic()), events.append,
        cancellation_token=token,
    )
    assert events == []


def test_playback_latency_includes_endpointing_and_processing(backend, monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 4.0)
    monkeypatch.setattr(backend, "_speak", lambda *args: 4.0)
    events = []
    # Speech ended at 2.0; endpointing closed it later and the worker runs at 4.0.
    backend.process(Utterance("timing", np.ones(512), 2.0), events.append)
    assert events[-1].kind == "completed"
    assert events[-1].timings["end_of_speech_to_playback_ms"] == 2000.0


def test_inference_timeout_is_distinct_from_recoverable_failure(backend, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    def slow_transcribe(*args, **kwargs):
        clock[0] = 31.0
        return iter([SimpleNamespace(text="Late text")]), None
    backend._whisper.transcribe = slow_transcribe
    events = []
    backend.process(Utterance("timeout", np.ones(512), 0.0), events.append)
    assert [event.kind for event in events] == ["transcribing", "timeout"]


def test_speech_text_uses_stdin_not_command_arguments(backend, monkeypatch):
    calls = []
    class FakeProcess:
        returncode = 0
        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
        def communicate(self, text=None, timeout=None):
            calls.append(text)
            return None, ""
    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    text = "Literal `command` and $(command), with a quote: '."
    backend._speak(text, backend._generation, time.monotonic() + 5)
    assert calls[0][0] == ["/usr/bin/say", "-v", "Samantha", "-r", "180"]
    assert calls[0][1]["stdin"] == subprocess.PIPE
    assert calls[0][1].get("shell", False) is False
    assert calls[1] == text


def test_playback_timeout_kills_process(backend, monkeypatch):
    state = {"killed": False, "calls": 0}
    class FakeProcess:
        returncode = 0
        def __init__(self, *args, **kwargs):
            pass
        def communicate(self, *args, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                raise subprocess.TimeoutExpired("say", 1)
            return None, ""
        def kill(self):
            state["killed"] = True
    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    with pytest.raises(TimeoutError, match="speech output"):
        backend._speak("hello", backend._generation, time.monotonic() + 5)
    assert state["killed"]
    assert backend._say_process is None


def test_cancel_terminates_active_say(backend):
    terminated = []
    backend._say_process = SimpleNamespace(poll=lambda: None, terminate=lambda: terminated.append(True))
    token = backend._generation
    backend.cancel()
    assert terminated == [True]
    with pytest.raises(_Cancelled):
        backend._check(token, time.monotonic() + 10)


def test_strip_nonspoken_markup():
    assert clean_response("<think>hidden</think> **Hello**\nworld!<|end|>") == "Hello world!"
