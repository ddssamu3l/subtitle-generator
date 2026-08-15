"""Speech recognition, behind a backend interface.

faster-whisper is the portable default and the only hard dependency: it runs
Whisper on CTranslate2, which means no PyTorch and a small install. On Apple
Silicon we transparently upgrade to mlx-whisper when it is present, because
CTranslate2 has no Metal backend and CPU inference on a long video is slow.

No model name is hardcoded as a requirement — whatever string the user supplies
is passed through to the backend, so models released after this was written
work without a code change.
"""

from __future__ import annotations

import sys
from typing import Callable

from . import config
from .transcript import Segment, Transcript, Word

ProgressFn = Callable[[float, str], None]


class TranscriptionError(RuntimeError):
    pass


# Offered in the GUI, roughly smallest/fastest to largest/most accurate. This
# is a convenience list, not a constraint: `--whisper-model` accepts any name
# faster-whisper recognises, including a local path or a HuggingFace repo id.
SUGGESTED_MODELS: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3-turbo",
    "large-v3",
)

# Turbo is large-v3 with a pruned decoder: four decoder layers instead of
# thirty-two. It transcribes at close to large-v3 accuracy for roughly a
# quarter of the compute and half the download, which matters here because
# CTranslate2 has no Metal backend and most users are transcribing on CPU.
# It is weaker at Whisper's own translate task, which is irrelevant to us —
# translation goes through a separate LLM.
DEFAULT_MODEL = "large-v3-turbo"


def _apple_silicon() -> bool:
    import platform

    return sys.platform == "darwin" and platform.machine() == "arm64"


def _mlx_available() -> bool:
    if not _apple_silicon():
        return False
    try:
        import mlx_whisper  # noqa: F401
    except Exception:
        return False
    return True


def active_backend() -> str:
    """Which engine will run, for display in the GUI."""
    return "mlx-whisper (GPU)" if _mlx_available() else "faster-whisper (CPU)"


def _mlx_repo(model: str) -> str:
    """Map a plain Whisper size onto an MLX community repo.

    Anything that already looks like a repo id or a filesystem path is passed
    through untouched, so a user can point at a model that does not exist yet.
    """
    if "/" in model:
        return model
    return f"mlx-community/whisper-{model}-mlx"


def transcribe(
    audio_path,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    progress: ProgressFn | None = None,
    duration: float = 0.0,
) -> Transcript:
    """Transcribe 16 kHz mono audio into a Transcript with word timings.

    Word-level timestamps are always requested: the speaker-alignment step
    needs them to split a segment at the point the speaker actually changes.
    """
    config.ensure_cache_dirs()
    if _mlx_available():
        return _transcribe_mlx(
            audio_path, model=model, language=language,
            progress=progress, duration=duration,
        )
    return _transcribe_faster_whisper(
        audio_path, model=model, language=language,
        progress=progress, duration=duration,
    )


def _transcribe_faster_whisper(
    audio_path, *, model: str, language: str | None,
    progress: ProgressFn | None, duration: float,
) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise TranscriptionError(
            "faster-whisper is not installed. Reinstall the tool with:\n"
            "  uv pip install -e ."
        ) from exc

    device, compute_type = _select_device()

    if progress:
        progress(0.0, f"Loading {model} model")

    try:
        engine = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            download_root=str(config.WHISPER_CACHE),
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Could not load Whisper model '{model}': {exc}\n"
            "If this is the first run, check your internet connection — the "
            "model downloads once and is cached afterwards."
        ) from exc

    try:
        segment_iter, info = engine.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            # Silence-trimming keeps Whisper from hallucinating text over long
            # pauses, which is the single most common source of junk subtitles.
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            beam_size=5,
            condition_on_previous_text=False,
        )
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    total = duration or getattr(info, "duration", 0.0) or 0.0
    segments: list[Segment] = []

    # faster-whisper returns a lazy generator; work happens as we consume it,
    # which is what makes incremental progress reporting possible at all.
    for raw in segment_iter:
        words = tuple(
            Word(
                start=float(w.start),
                end=float(w.end),
                text=w.word,
                probability=float(getattr(w, "probability", 1.0) or 1.0),
            )
            for w in (raw.words or [])
            if w.start is not None and w.end is not None
        )
        segments.append(
            Segment(
                start=float(raw.start),
                end=float(raw.end),
                text=(raw.text or "").strip(),
                words=words,
            )
        )
        if progress and total > 0:
            progress(min(float(raw.end) / total, 1.0), "Transcribing")

    if progress:
        progress(1.0, "Transcribing")

    return Transcript(
        segments=tuple(segments),
        language=getattr(info, "language", "") or "",
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=total,
    )


def _transcribe_mlx(
    audio_path, *, model: str, language: str | None,
    progress: ProgressFn | None, duration: float,
) -> Transcript:
    import mlx_whisper

    if progress:
        progress(0.0, f"Loading {model} model (GPU)")

    try:
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=_mlx_repo(model),
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"GPU transcription failed: {exc}\n"
            "Uninstall mlx-whisper to fall back to the CPU engine."
        ) from exc

    segments: list[Segment] = []
    for raw in result.get("segments", []):
        words = tuple(
            Word(
                start=float(w["start"]),
                end=float(w["end"]),
                text=w.get("word", ""),
                probability=float(w.get("probability", 1.0)),
            )
            for w in (raw.get("words") or [])
            if w.get("start") is not None and w.get("end") is not None
        )
        segments.append(
            Segment(
                start=float(raw.get("start", 0.0)),
                end=float(raw.get("end", 0.0)),
                text=(raw.get("text") or "").strip(),
                words=words,
            )
        )

    if progress:
        progress(1.0, "Transcribing")

    return Transcript(
        segments=tuple(segments),
        language=result.get("language", "") or "",
        duration=duration,
    )


def _select_device() -> tuple[str, str]:
    """Pick the best CTranslate2 device available.

    int8 on CPU is roughly 2x faster than float32 with no accuracy loss worth
    caring about for subtitles; float16 is the right choice on a real GPU.
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"
