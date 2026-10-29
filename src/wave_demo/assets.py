"""Explicit online setup; all runtime consumers use verified local paths."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
from datetime import datetime, timezone

from .config import AppConfig

GESTURE_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)


def offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(directory: Path, config: AppConfig, hashes: bool = False) -> list[str]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return ["Assets are not prepared. Run: uv run gesture-demo setup"]
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest["models"]["whisper"]["repo"] != config.speech.whisper_repo:
            return ["Whisper configuration differs from downloaded assets; rerun setup --refresh"]
        if manifest["models"]["llm"]["repo"] != config.speech.llm_repo:
            return ["Reply-model configuration differs from downloaded assets; rerun setup --refresh"]
        errors = []
        for relative, record in manifest["files"].items():
            path = directory / relative
            if not path.resolve().is_relative_to(directory.resolve()):
                errors.append(f"Invalid asset path: {relative}")
            elif not path.is_file() or path.stat().st_size != record["bytes"]:
                errors.append(f"Missing or incomplete asset: {relative}")
            elif hashes and sha256(path) != record["sha256"]:
                errors.append(f"Asset checksum mismatch: {relative}")
        for relative in ("gesture_recognizer.task", "whisper/model.bin", "whisper/tokenizer.json",
                         "llm/config.json", "llm/tokenizer.json"):
            if relative not in manifest["files"]:
                errors.append(f"Asset missing from manifest: {relative}")
        if not any(p.startswith("llm/") and p.endswith(".safetensors") for p in manifest["files"]):
            errors.append("No reply-model weights in manifest")
        return errors
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"Invalid asset manifest: {exc}"]


def setup_assets(directory: Path, config: AppConfig, refresh: bool = False) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    if not refresh and not verify_assets(directory, config, hashes=True):
        print("All model assets are already present and verified.", flush=True)
        return json.loads((directory / "manifest.json").read_text())
    if shutil.disk_usage(directory).free < 5 * 1024**3:
        raise RuntimeError("At least 5 GB of free disk space is required for model setup")
    from huggingface_hub import HfApi, snapshot_download

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    print("Downloading the official MediaPipe gesture model...", flush=True)
    target = directory / "gesture_recognizer.task"
    temporary = target.with_suffix(".download")
    try:
        with urllib.request.urlopen(GESTURE_URL, timeout=60) as source, temporary.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    models = {}
    api = HfApi(token=False)
    for name, repo in (("whisper", config.speech.whisper_repo), ("llm", config.speech.llm_repo)):
        revision = api.model_info(repo).sha
        print(f"Downloading {repo} at {revision}...", flush=True)
        snapshot_download(
            repo_id=repo, revision=revision, token=False, local_dir=directory / name,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.jinja", "*.tiktoken"],
        )
        models[name] = {"repo": repo, "revision": revision}
    records = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_file() and ".cache" not in relative.parts and path.name != "manifest.json":
            records[str(relative)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gesture_url": GESTURE_URL, "models": models, "files": records,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if errors := verify_assets(directory, config, hashes=True):
        raise RuntimeError("; ".join(errors))
    return manifest
