"""Model asset management, built around a hard offline guarantee.

The tool is designed to run on a laptop with the wifi off. That means every
model must be fetched once, deliberately, by `subgen setup` — and that normal
runs must never touch the network, not even to check for updates.

Two mechanisms enforce that:

* `HF_HUB_OFFLINE` is set before any HuggingFace-backed library is imported, so
  faster-whisper resolves from the local cache instead of contacting the hub.
* `missing_assets()` is called before a job starts, so a missing model produces
  an immediate, actionable message rather than a request that hangs until it
  times out.

Downloads only ever happen inside `fetch_*`, which is only ever called from
setup. Nothing in the processing path can reach the network.
"""

from __future__ import annotations

import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import config

ProgressFn = Callable[[float, str], None]

# --- Diarization assets ------------------------------------------------------
# Pinned by URL rather than by a registry lookup, so resolving them needs no
# network round-trip beyond the download itself. If a URL ever rots, setup
# fails loudly with the URL in the message and these can be overridden by
# dropping files into the cache directory by hand.

SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SEGMENTATION_FILE = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"

# Trained jointly on Chinese and English, which matters here: a speaker
# embedding model trained on one language separates voices noticeably worse on
# the other, and those are the two languages we explicitly guarantee.
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)
EMBEDDING_FILE = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"


def enforce_offline() -> None:
    """Stop HuggingFace libraries from making network calls.

    Must run before faster-whisper is imported. Setting both variables covers
    old and new huggingface_hub releases, which renamed the flag partway
    through the 0.x series.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def allow_downloads() -> None:
    """Re-enable network access, used only by `subgen setup`."""
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(key, None)


@dataclass(frozen=True)
class Asset:
    label: str
    path: Path
    present: bool
    size_hint: str
    how_to_get: str


def segmentation_model() -> Path:
    return config.DIARIZE_CACHE / SEGMENTATION_FILE


def embedding_model() -> Path:
    return config.DIARIZE_CACHE / EMBEDDING_FILE


def whisper_cached(model: str) -> bool:
    """Whether a Whisper model is already in our local cache.

    faster-whisper stores HuggingFace snapshots as `models--<org>--<repo>`.
    We look for any directory mentioning the model name that actually contains
    weights, which stays correct as the naming convention evolves.
    """
    root = config.WHISPER_CACHE
    if not root.exists():
        return False

    # A local path or explicit repo the user supplied directly.
    candidate = Path(model).expanduser()
    if candidate.exists():
        return True

    needle = model.replace("/", "--").lower()
    for entry in root.rglob("*"):
        if not entry.is_dir():
            continue
        if needle not in entry.name.lower():
            continue
        if any(entry.rglob("*.bin")) or any(entry.rglob("*.safetensors")):
            return True
    return False


def diarization_ready() -> bool:
    return segmentation_model().exists() and embedding_model().exists()


def missing_assets(whisper_model: str, *, need_diarization: bool) -> list[Asset]:
    """Everything required for a run that is not on disk yet."""
    missing: list[Asset] = []

    if not whisper_cached(whisper_model):
        missing.append(
            Asset(
                label=f"Whisper model '{whisper_model}'",
                path=config.WHISPER_CACHE,
                present=False,
                size_hint=_whisper_size_hint(whisper_model),
                how_to_get=f"subgen setup --whisper-model {whisper_model}",
            )
        )

    if need_diarization and not diarization_ready():
        missing.append(
            Asset(
                label="Speaker diarization models",
                path=config.DIARIZE_CACHE,
                present=False,
                size_hint="~35 MB",
                how_to_get="subgen setup",
            )
        )

    return missing


def _whisper_size_hint(model: str) -> str:
    """Approximate download size, for the setup summary."""
    sizes = {
        "tiny": "~75 MB",
        "base": "~145 MB",
        "small": "~490 MB",
        "medium": "~1.5 GB",
        "large-v3": "~3.1 GB",
        "large-v3-turbo": "~1.6 GB",
    }
    return sizes.get(model, "size unknown")


# --- Downloading -------------------------------------------------------------


class DownloadError(RuntimeError):
    pass


def _download(url: str, destination: Path, *, progress: ProgressFn | None, label: str) -> Path:
    """Stream a file to disk, resolving redirects and verifying TLS.

    httpx rather than urllib deliberately: urllib validates certificates
    against OpenSSL's default store, which on macOS is frequently empty for
    python.org and pyenv builds. That produces a CERTIFICATE_VERIFY_FAILED on a
    perfectly good connection and is one of the most common setup failures for
    Python tools. httpx bundles certifi, so this works everywhere.
    """
    import httpx

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    try:
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,  # GitHub release assets redirect to a CDN
            timeout=httpx.Timeout(30.0, read=120.0),
            headers={"User-Agent": "subtitle-generator"},
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 16):
                    handle.write(chunk)
                    if progress and total:
                        progress(response.num_bytes_downloaded / total, label)
    except (httpx.HTTPError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(
            f"Could not download {label}.\n"
            f"  URL: {url}\n"
            f"  Reason: {exc}\n"
            "You need an internet connection for this one-time setup step."
        ) from exc

    partial.replace(destination)
    return destination


def fetch_diarization(*, progress: ProgressFn | None = None, force: bool = False) -> None:
    """Download the segmentation and embedding models into the cache."""
    config.ensure_cache_dirs()

    if not force and diarization_ready():
        if progress:
            progress(1.0, "Speaker models already present")
        return

    if force or not embedding_model().exists():
        _download(
            EMBEDDING_URL,
            embedding_model(),
            progress=progress,
            label="Speaker embedding model",
        )

    if force or not segmentation_model().exists():
        archive = config.DIARIZE_CACHE / "segmentation.tar.bz2"
        _download(
            SEGMENTATION_URL, archive,
            progress=progress, label="Speaker segmentation model",
        )
        try:
            with tarfile.open(archive, "r:bz2") as tar:
                _safe_extract(tar, config.DIARIZE_CACHE)
        except (tarfile.TarError, OSError) as exc:
            raise DownloadError(f"Could not unpack the segmentation model: {exc}") from exc
        finally:
            archive.unlink(missing_ok=True)

    if not diarization_ready():
        raise DownloadError(
            "Speaker models downloaded but the expected files are missing.\n"
            f"Looked for:\n  {segmentation_model()}\n  {embedding_model()}"
        )


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    """Extract without allowing paths to escape the destination directory."""
    root = destination.resolve()
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if not str(target).startswith(str(root)):
            raise DownloadError(f"Refusing to extract unsafe path: {member.name}")
    # `filter` is the modern, safe default; older Pythons ignore the kwarg.
    try:
        tar.extractall(destination, filter="data")
    except TypeError:  # pragma: no cover - Python < 3.12
        tar.extractall(destination)


def fetch_whisper(model: str, *, progress: ProgressFn | None = None) -> None:
    """Warm the Whisper cache by instantiating the model once.

    faster-whisper has no separate download API, so loading the model is how
    you fetch it. The instance is discarded immediately; the weights stay in
    the cache directory.
    """
    allow_downloads()
    config.ensure_cache_dirs()

    if progress:
        progress(0.0, f"Downloading Whisper '{model}' ({_whisper_size_hint(model)})")

    try:
        from faster_whisper import WhisperModel

        WhisperModel(
            model,
            device="cpu",
            compute_type="int8",
            download_root=str(config.WHISPER_CACHE),
        )
    except Exception as exc:
        raise DownloadError(
            f"Could not download Whisper model '{model}': {exc}"
        ) from exc
    finally:
        enforce_offline()

    if progress:
        progress(1.0, f"Whisper '{model}' ready")


def cache_summary() -> str:
    """Human-readable inventory, for `subgen setup --status`."""
    lines = [f"Cache directory: {config.CACHE_DIR}"]

    whisper_root = config.WHISPER_CACHE
    found = []
    if whisper_root.exists():
        for entry in sorted(whisper_root.iterdir()):
            if entry.is_dir() and (any(entry.rglob("*.bin")) or any(entry.rglob("*.safetensors"))):
                found.append(f"  - {entry.name} ({_dir_size(entry)})")
    lines.append("Whisper models:" if found else "Whisper models: none downloaded yet")
    lines.extend(found)

    ready = diarization_ready()
    lines.append(f"Speaker models: {'ready' if ready else 'not downloaded'}")
    return "\n".join(lines)


def _dir_size(path: Path) -> str:
    """Bytes on disk, counting each blob once.

    The HuggingFace cache stores weights in `blobs/` and symlinks them into
    `snapshots/`. Following those links counts every file twice and reports
    double the real size, which looks alarming for a multi-gigabyte model.
    """
    total = 0
    for entry in path.rglob("*"):
        if entry.is_symlink() or not entry.is_file():
            continue
        total += entry.stat().st_size

    gb = total / 1_000_000_000
    return f"{gb:.1f} GB" if gb >= 1 else f"{total / 1_000_000:.0f} MB"


def purge() -> None:
    """Delete every cached model. Exposed as `subgen setup --purge`."""
    shutil.rmtree(config.CACHE_DIR, ignore_errors=True)
