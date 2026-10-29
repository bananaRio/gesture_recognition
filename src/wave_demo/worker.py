"""Spawn-isolated local speech inference.

OpenCV and PyAV bundle conflicting macOS FFmpeg classes. Speech dependencies
load only in the child. Cancellation is terminal and stops playback descendants.
"""

import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import threading
import time


class _SharedFlag:
    """One-way, lock-free flag that remains readable after an abrupt exit.

    multiprocessing.Event.wait uses a Condition whose dead waiter can hang
    Event.set after os._exit/SIGKILL. A shared byte avoids that shutdown hazard.
    """

    def __init__(self, context):
        self._value = context.RawValue("b", 0)

    def is_set(self):
        return bool(self._value.value)

    def set(self):
        self._value.value = 1


def _speech_process_main(config, assets_dir, jobs, events, cancelled, group_ready, backend_factory):
    backend = None
    try:
        os.setsid()
        group_ready.set()
        if cancelled.is_set():
            return
        if backend_factory is None:
            from .speech import LocalSpeechBackend
            backend_factory = LocalSpeechBackend
        backend = backend_factory(config, Path(assets_dir))

        def cancel_backend():
            while not cancelled.is_set():
                time.sleep(0.02)
            cancel = getattr(backend, "cancel", None)
            if cancel is not None:
                try:
                    cancel()
                except Exception:
                    # Parent termination remains the fallback for a backend
                    # that cannot interrupt its native operation cleanly.
                    pass

        threading.Thread(target=cancel_backend, name="speech-cancellation", daemon=True).start()
        backend.load()
        if cancelled.is_set():
            return
        backend.warmup()
        if cancelled.is_set():
            return
        events.put(("ready", None))
        while not cancelled.is_set():
            try:
                utterance = jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if utterance is None or cancelled.is_set():
                return
            backend.process(utterance, events.put)
    except BaseException as exc:
        if not cancelled.is_set():
            events.put(("fatal", f"Speech worker: {type(exc).__name__}: {exc}"))
    finally:
        if backend is not None:
            backend.close()


class SpeechWorker:
    """One serial worker with no speech-library imports in its parent.

    backend_factory is an optional importable callable for alternate backends
    and device-free spawned-process tests. LocalSpeechBackend is the default.
    """

    def __init__(self, config, assets_dir, *, backend_factory=None):
        context = mp.get_context("spawn")
        self.events = context.Queue()
        self._jobs = context.Queue(maxsize=1)
        self._cancelled = _SharedFlag(context)
        self._group_ready = _SharedFlag(context)
        self._process = context.Process(
            target=_speech_process_main,
            args=(config, str(assets_dir), self._jobs, self.events, self._cancelled,
                  self._group_ready, backend_factory),
            name="wave-speech", daemon=True,
        )
        self._started = False
        self._closed = False
        self._lock = threading.Lock()

    @property
    def pid(self):
        return self._process.pid if not self._closed else None

    @property
    def is_alive(self):
        return self._started and not self._closed and self._process.is_alive()

    def start(self):
        with self._lock:
            if self._closed or self._cancelled.is_set():
                raise RuntimeError("Speech worker is closed or cancelled")
            if self._started:
                raise RuntimeError("Speech worker is already started")
            self._process.start()
            self._started = True

    def submit(self, utterance):
        with self._lock:
            if self._closed or self._cancelled.is_set():
                raise RuntimeError("Speech worker is closed or cancelled")
            if not self._started or not self._process.is_alive():
                raise RuntimeError("Speech worker is not running")
            self._jobs.put_nowait(utterance)

    def _signal_worker(self, sig):
        pid = self._process.pid
        if pid is None:
            return
        try:
            # Never signal the parent's group: the group must be child-owned.
            # The child sets this flag only after setsid succeeds. The known
            # owned group may still contain `say` after its leader has crashed.
            if self._group_ready.is_set() or os.getpgid(pid) == pid:
                os.killpg(pid, sig)
            else:
                os.kill(pid, sig)
        except ProcessLookupError:
            pass

    def cancel(self):
        """Terminally stop inference and any active speech-playback child."""
        with self._lock:
            if self._closed:
                return
            self._cancelled.set()
            if not self._started:
                return
            # An idle worker exits its queue loop within 100 ms. A cooperative
            # backend also gets a chance to stop playback and release native
            # resources before process termination becomes necessary.
            self._process.join(timeout=0.35)
            # Also signal the known group after an abrupt leader exit, because
            # an orphaned playback subprocess may still be alive there.
            self._signal_worker(signal.SIGTERM)
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._signal_worker(signal.SIGKILL)
                self._process.join(timeout=2)

    def close(self):
        if self._closed:
            return
        self.cancel()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for channel in (self._jobs, self.events):
                # Do not wait for an audio feeder whose consumer was killed.
                channel.cancel_join_thread()
                channel.close()
            self._process.close()
