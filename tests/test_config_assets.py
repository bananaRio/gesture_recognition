import json
from pathlib import Path

import pytest

from wave_demo.assets import sha256, verify_assets
from wave_demo.config import AppConfig, load_config
from wave_demo.contracts import LifecycleEvent
from wave_demo.event_log import EventLog
from wave_demo.cli import summarize_log


def test_config_rejects_missing_explicit_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")


@pytest.mark.parametrize("content", [
    "[wave]\nfps = nan", "[wave]\nfps = 0", "[wave]\nunknown = 3",
    "[audio]\nsample_rate = 48000", "[runtime]\nmirror_preview = 1",
    "[wave]\nopen_palm_threshold = 1.5",
])
def test_invalid_configuration_does_not_silently_change_detection(tmp_path, content):
    path = tmp_path / "config.toml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_config(path)


def make_assets(root):
    config = AppConfig()
    files = {}
    for name in ("gesture_recognizer.task", "whisper/model.bin", "whisper/tokenizer.json",
                 "llm/config.json", "llm/tokenizer.json", "llm/model.safetensors"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {"models": {"whisper": {"repo": config.speech.whisper_repo},
                            "llm": {"repo": config.speech.llm_repo}}, "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_asset_integrity_detects_corruption_and_missing_tokenizer(tmp_path):
    make_assets(tmp_path)
    assert not verify_assets(tmp_path, AppConfig(), hashes=True)
    (tmp_path / "llm/model.safetensors").write_bytes(b"changed")
    assert any("checksum" in e for e in verify_assets(tmp_path, AppConfig(), hashes=True))
    (tmp_path / "whisper/tokenizer.json").unlink()
    assert any("tokenizer.json" in e for e in verify_assets(tmp_path, AppConfig()))


def test_asset_manifest_cannot_reference_external_file(tmp_path):
    manifest = make_assets(tmp_path)
    manifest["files"]["../outside"] = {"bytes": 7, "sha256": "wrong"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert any("Invalid asset path" in e for e in verify_assets(tmp_path, AppConfig()))


def test_log_omits_dialogue_and_report_uses_completed_timings_once(tmp_path):
    path: Path = tmp_path / "test.jsonl"
    log = EventLog(path)
    log(LifecycleEvent("speech_progress", 1, details={"text": "private utterance", "timings": {"x": 3}}))
    log(LifecycleEvent("completed", 2, details={"timings": {"x": 5}}))
    log.close()
    assert "private utterance" not in path.read_text()
    result = summarize_log(path)
    assert result["metrics"]["x"] == {"n": 1, "p50": 5, "p95": 5}
