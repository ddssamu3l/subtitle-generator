"""Speaker diarization: who spoke when.

We use sherpa-onnx rather than pyannote.audio deliberately. pyannote is a bit
more accurate, but it needs PyTorch (~2.5 GB) and a HuggingFace token with a
manually accepted licence — a wall that stops a good fraction of people from
ever getting the tool running. sherpa-onnx runs the same pyannote segmentation
network through onnxruntime with no token, no torch, and ~35 MB of weights.

Note that speaker identity is never shown to the viewer. The labels exist only
so the layout engine can tell "the same person continuing" from "somebody else
cutting in", which is what decides whether lines stack.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Callable

from . import models
from .transcript import SpeakerTurn

ProgressFn = Callable[[float, str], None]

TARGET_SAMPLE_RATE = 16000


class DiarizationError(RuntimeError):
    pass


def read_audio(path: Path):
    """Load 16 kHz mono PCM as normalised float32.

    sherpa-onnx does not expose a WAV reader in every build, and we already
    control the file's format because media.extract_audio produced it, so the
    stdlib reader is both sufficient and one less thing to depend on.
    """
    import numpy as np

    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, OSError) as exc:
        raise DiarizationError(f"Could not read extracted audio: {exc}") from exc

    if width != 2:
        raise DiarizationError(
            f"Expected 16-bit audio, got {width * 8}-bit. This is a bug in the "
            "audio extraction step."
        )

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if rate != TARGET_SAMPLE_RATE:
        raise DiarizationError(
            f"Expected {TARGET_SAMPLE_RATE} Hz audio, got {rate} Hz."
        )

    return samples


def is_available() -> bool:
    """Whether diarization can run right now, without downloading anything."""
    if not models.diarization_ready():
        return False
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return True


# Clustering cutoff used when the speaker count is unknown. Lower values split
# more eagerly. 0.5 is a reasonable default for real recordings, but it does
# merge genuinely similar voices — in testing, three different female speakers
# were clustered as one at 0.5 and only separated cleanly at 0.25. Since the
# right value depends entirely on the material, it is exposed to the user
# rather than buried here.
DEFAULT_THRESHOLD = 0.5

# Friendly presets for the GUI, mapping a description to a threshold.
SENSITIVITY_PRESETS: tuple[tuple[str, float], ...] = (
    ("Balanced (recommended)", 0.50),
    ("Sensitive — separates similar voices", 0.35),
    ("Most sensitive — many similar voices", 0.25),
    ("Conservative — fewer, clearer speakers", 0.70),
)


def diarize(
    audio_path: Path,
    *,
    num_speakers: int = -1,
    threshold: float = DEFAULT_THRESHOLD,
    progress: ProgressFn | None = None,
) -> list[SpeakerTurn]:
    """Return speaker turns sorted by start time.

    `num_speakers` of -1 means "work it out", which is the right default — the
    user picked a video, not a cast list. Any number of speakers is supported;
    nothing here assumes two.

    `threshold` is the clustering cutoff used when the count is unknown. Lower
    splits voices more eagerly. Note that forcing an exact `num_speakers` is
    less reliable than it sounds: the clusterer can produce a cluster that ends
    up with no surviving segments, yielding fewer labels than requested.
    """
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise DiarizationError(
            "sherpa-onnx is not installed. Reinstall the tool with:\n"
            "  uv pip install -e ."
        ) from exc

    if not models.diarization_ready():
        raise DiarizationError(
            "Speaker models are not downloaded yet. Run this once while online:\n"
            "  subgen setup"
        )

    clustering = (
        sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers)
        if num_speakers and num_speakers > 0
        else sherpa_onnx.FastClusteringConfig(threshold=threshold)
    )

    settings = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(models.segmentation_model())
            ),
            num_threads=2,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(models.embedding_model()),
            num_threads=2,
            provider="cpu",
        ),
        clustering=clustering,
        # Ignore blips shorter than this. Below ~0.2s the segmenter mostly
        # picks up breaths and mic noise, which would fragment real turns.
        min_duration_on=0.25,
        min_duration_off=0.4,
    )

    if not settings.validate():
        raise DiarizationError(
            "Speaker model configuration was rejected. The cached models may be "
            "corrupt — re-download them with:  subgen setup --force"
        )

    engine = sherpa_onnx.OfflineSpeakerDiarization(settings)

    samples = read_audio(audio_path)
    if engine.sample_rate != TARGET_SAMPLE_RATE:
        raise DiarizationError(
            f"Speaker model expects {engine.sample_rate} Hz but audio is "
            f"{TARGET_SAMPLE_RATE} Hz."
        )

    if progress:
        progress(0.0, "Identifying speakers")

    result = _process(engine, samples, progress)

    turns = [
        SpeakerTurn(start=float(s.start), end=float(s.end), speaker=int(s.speaker))
        for s in result
    ]
    turns.sort(key=lambda turn: (turn.start, turn.end))

    if progress:
        progress(1.0, "Identifying speakers")
    return turns


def _process(engine, samples, progress: ProgressFn | None):
    """Run diarization, using the progress callback when the build supports it."""

    def report(processed: int, total: int) -> int:
        if progress and total:
            progress(min(processed / total, 1.0), "Identifying speakers")
        return 0

    try:
        outcome = engine.process(samples, callback=report)
    except TypeError:
        # Older sherpa-onnx builds have no callback parameter.
        outcome = engine.process(samples)

    return outcome.sort_by_start_time()


def speaker_count(turns: list[SpeakerTurn]) -> int:
    return len({turn.speaker for turn in turns})
