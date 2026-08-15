"""Cache locations and the handful of preferences worth remembering.

Deliberately tiny: the only thing we persist is what the user picked last time,
so the GUI can pre-select it. Nothing here is required for the tool to run — a
missing or corrupt settings file degrades to defaults rather than erroring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _xdg(env: str, fallback: Path) -> Path:
    raw = os.environ.get(env)
    return Path(raw).expanduser() if raw else fallback


HOME = Path.home()

# Follow the platform convention rather than scattering dotfiles in $HOME.
if os.name == "nt":
    _base_data = Path(os.environ.get("LOCALAPPDATA", HOME / "AppData" / "Local"))
    CONFIG_DIR = _base_data / "subtitle-generator"
    CACHE_DIR = CONFIG_DIR / "cache"
elif os.uname().sysname == "Darwin":  # type: ignore[attr-defined]
    CONFIG_DIR = HOME / "Library" / "Application Support" / "subtitle-generator"
    CACHE_DIR = HOME / "Library" / "Caches" / "subtitle-generator"
else:
    CONFIG_DIR = _xdg("XDG_CONFIG_HOME", HOME / ".config") / "subtitle-generator"
    CACHE_DIR = _xdg("XDG_CACHE_HOME", HOME / ".cache") / "subtitle-generator"

SETTINGS_PATH = CONFIG_DIR / "settings.json"

# Model weights live under the cache dir so a user can reclaim the space by
# deleting one directory, and so we never write into the package install.
WHISPER_CACHE = CACHE_DIR / "whisper"
DIARIZE_CACHE = CACHE_DIR / "diarization"

DEFAULTS: dict[str, Any] = {
    "ollama_model": None,  # last model the user chose; None = ask
    "whisper_model": "large-v3-turbo",  # overridable; any faster-whisper name works
    "target_language": "Chinese (Simplified)",
    "source_language": "Auto-detect",
    "identify_speakers": True,
    "speaker_sensitivity": "Balanced (recommended)",
    "keep_sidecar_files": True,
    "ollama_host": None,  # None = use OLLAMA_HOST env or the default port
}


def load() -> dict[str, Any]:
    """Read settings, falling back to defaults for anything missing or broken."""
    settings = dict(DEFAULTS)
    try:
        stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if isinstance(stored, dict):
        # Only accept keys we know about, so a stale file can't inject junk.
        settings.update({k: v for k, v in stored.items() if k in DEFAULTS})
    return settings


def save(settings: dict[str, Any]) -> None:
    """Persist settings, ignoring failures — this is a convenience, not state."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in settings.items() if k in DEFAULTS}
        SETTINGS_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


def ensure_cache_dirs() -> None:
    for path in (WHISPER_CACHE, DIARIZE_CACHE):
        path.mkdir(parents=True, exist_ok=True)
