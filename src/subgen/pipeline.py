"""Orchestration: one video in, one subtitled video out.

Both the CLI and the GUI drive this module, so all sequencing, progress
weighting and cancellation lives here rather than being duplicated in each
front end.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import align, diarize, media, models, subtitles, transcribe
from .languages import Language, whisper_code
from .translate import Translator
from .transcript import Cue

# Fractions of the total progress bar. Chosen to roughly match observed wall
# time so the bar advances at a believable rate rather than sitting at 90%.
PHASE_WEIGHTS = {
    "extract": 0.03,
    "transcribe": 0.42,
    "diarize": 0.12,
    "translate": 0.18,
    "render": 0.25,
}

StatusFn = Callable[[float, str], None]


class Cancelled(RuntimeError):
    """Raised when the user asks to stop. Not an error worth a traceback."""


@dataclass
class Options:
    target_language: Language | None = None
    source_language: Language | None = None
    whisper_model: str = transcribe.DEFAULT_MODEL
    ollama_model: str | None = None
    ollama_host: str | None = None
    identify_speakers: bool = True
    # False writes a subtitled copy and leaves the source alone (the default).
    # True burns into the original in place, which cannot be undone.
    replace_original: bool = False
    keep_sidecar_files: bool = True
    output_dir: Path | None = None
    subtitle_scale: float = 1.0
    # Clustering cutoff for speaker separation; lower splits more eagerly.
    speaker_sensitivity: float = diarize.DEFAULT_THRESHOLD
    # Exact speaker count when the user knows it; -1 means auto-detect.
    speaker_count: int = -1


@dataclass
class Result:
    source: Path
    output: Path | None = None
    subtitle_files: list[Path] = field(default_factory=list)
    cue_count: int = 0
    speaker_count: int = 0
    detected_language: str = ""
    translated: bool = False
    replaced_original: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.output is not None


class Progress:
    """Maps per-phase 0..1 callbacks onto a single overall fraction."""

    def __init__(self, report: StatusFn | None):
        self._report = report
        self._done = 0.0

    def phase(self, name: str) -> Callable[[float, str], None]:
        weight = PHASE_WEIGHTS[name]
        base = self._done

        def inner(fraction: float, label: str) -> None:
            if self._report:
                self._report(min(base + weight * max(0.0, min(fraction, 1.0)), 1.0), label)

        return inner

    def finish(self, name: str) -> None:
        self._done = min(self._done + PHASE_WEIGHTS[name], 1.0)

    def skip(self, name: str) -> None:
        """Redistribute an unused phase so the bar still reaches 100%."""
        self._done = min(self._done + PHASE_WEIGHTS[name], 1.0)


def output_path_for(source: Path, options: Options) -> Path:
    """Where the finished video goes. Never overwrites the source."""
    directory = options.output_dir or source.parent
    tag = options.target_language.code if options.target_language else "subtitled"
    suffix = source.suffix.lower()
    if suffix not in {".mp4", ".mkv", ".mov", ".webm", ".m4v"}:
        # Re-encoding into an exotic container invites muxer trouble; MP4 is
        # the safe destination for everything else.
        suffix = ".mp4"

    candidate = directory / f"{source.stem}.{tag}{suffix}"
    counter = 2
    while candidate.exists() and candidate.resolve() != source.resolve():
        candidate = directory / f"{source.stem}.{tag}.{counter}{suffix}"
        counter += 1
    return candidate


def preflight(options: Options) -> list[str]:
    """Problems that would stop a run, reported before any work begins."""
    problems: list[str] = []

    if not media.ffmpeg_path():
        problems.append("ffmpeg is not installed.\n" + media.install_hint())

    missing = models.missing_assets(
        options.whisper_model,
        need_diarization=options.identify_speakers,
    )
    for asset in missing:
        problems.append(
            f"{asset.label} is not downloaded ({asset.size_hint}).\n"
            f"Run this once while you have internet:  {asset.how_to_get}"
        )

    if options.target_language and not options.ollama_model:
        problems.append(
            "No translation model selected, so subtitles cannot be translated."
        )

    return problems


# Containers we are willing to re-encode into in place. Others (avi, wmv, flv)
# can technically hold H.264 but do so unreliably, and a broken result would
# have destroyed the original — so replacing is refused for them.
REPLACEABLE_CONTAINERS = frozenset({".mp4", ".mkv", ".mov", ".m4v", ".webm"})


def can_replace_in_place(source: Path) -> str | None:
    """None if replacing this file is safe, otherwise the reason it is not."""
    if source.suffix.lower() not in REPLACEABLE_CONTAINERS:
        return (
            f"{source.suffix or 'this format'} cannot be safely rewritten in "
            "place. A subtitled copy will be written instead."
        )
    return None


def process(
    source: Path,
    options: Options,
    *,
    report: StatusFn | None = None,
    cancel: threading.Event | None = None,
) -> Result:
    """Run the whole pipeline for one video."""
    result = Result(source=source)
    progress = Progress(report)

    def check_cancelled() -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    workspace = Path(tempfile.mkdtemp(prefix="subgen-"))

    try:
        check_cancelled()
        info = media.probe(source)
        result.detected_language = ""

        if not info.has_audio:
            result.error = f"{source.name} has no audio track to transcribe."
            return result

        # 1. Audio ------------------------------------------------------------
        audio = workspace / "audio.wav"
        media.extract_audio(source, audio, progress=progress.phase("extract"))
        progress.finish("extract")
        check_cancelled()

        # 2. Transcription ----------------------------------------------------
        transcript = transcribe.transcribe(
            audio,
            model=options.whisper_model,
            language=whisper_code(options.source_language),
            progress=progress.phase("transcribe"),
            duration=info.duration,
        )
        progress.finish("transcribe")
        result.detected_language = transcript.language
        check_cancelled()

        if not transcript.segments:
            result.error = f"No speech was found in {source.name}."
            return result

        # 3. Speakers ---------------------------------------------------------
        turns = []
        if options.identify_speakers and diarize.is_available():
            try:
                turns = diarize.diarize(
                    audio,
                    num_speakers=options.speaker_count,
                    threshold=options.speaker_sensitivity,
                    progress=progress.phase("diarize"),
                )
                result.speaker_count = diarize.speaker_count(turns)
            except diarize.DiarizationError as exc:
                # Losing speaker detection costs us stacked lines, not the run.
                if report:
                    report(0.0, f"Speaker detection unavailable: {exc}")
                turns = []
            progress.finish("diarize")
        else:
            progress.skip("diarize")
        check_cancelled()

        # 4. Cues -------------------------------------------------------------
        cues: list[Cue] = align.prepare(transcript, turns)
        if not cues:
            result.error = f"Nothing to subtitle in {source.name}."
            return result

        # 5. Translation ------------------------------------------------------
        target = options.target_language
        needs_translation = bool(
            target
            and options.ollama_model
            and not _already_in_target(transcript.language, target)
        )

        if needs_translation:
            translator = Translator(
                model=options.ollama_model or "",
                target=target,  # type: ignore[arg-type]
                host=options.ollama_host,
                source_hint=transcript.language or "",
            )
            cues = translator.translate_all(cues, progress=progress.phase("translate"))
            result.translated = True
            # Translated text has a different length, so reading time and
            # therefore the stacking overlaps must be recomputed.
            cues = align.apply_readability(cues)
            progress.finish("translate")
        else:
            progress.skip("translate")
        check_cancelled()

        result.cue_count = len(cues)

        # 6. Subtitles --------------------------------------------------------
        language_code = target.code if target else (transcript.language or "en")
        style = subtitles.Style.for_video(
            info.width, info.height,
            language_code=language_code,
            scale=options.subtitle_scale,
        )
        ass_file = workspace / "subtitles.ass"
        subtitles.write_ass(cues, style, ass_file)

        # 7. Burn in ----------------------------------------------------------
        # In replace mode we render beside the original under a temporary name
        # and only swap it in once the result has been verified. The original
        # is therefore never in a partially-written state, and a crash or a
        # cancellation mid-encode leaves it untouched.
        replacing = options.replace_original and can_replace_in_place(source) is None
        if replacing:
            destination = source.parent / f".subgen-render-{source.stem}{source.suffix}"
        else:
            destination = output_path_for(source, options)

        media.burn_in(
            source, ass_file, destination,
            duration=info.duration,
            source_bitrate=info.video_bitrate,
            width=info.width,
            height=info.height,
            progress=progress.phase("render"),
        )
        progress.finish("render")

        if replacing:
            problem = _verify_render(destination, info)
            if problem:
                destination.unlink(missing_ok=True)
                result.error = (
                    f"Refused to replace {source.name}: the new file failed its "
                    f"check ({problem}). Your original is untouched."
                )
                return result
            # os.replace is atomic within a filesystem, so the original is
            # either the old file or the new one, never a half-written mix.
            os.replace(destination, source)
            destination = source

        result.output = destination
        result.replaced_original = replacing

        # 8. Sidecars ---------------------------------------------------------
        if options.keep_sidecar_files:
            directory = destination.parent
            stem = destination.stem
            kept_ass = directory / f"{stem}.ass"
            shutil.copyfile(ass_file, kept_ass)
            kept_srt = subtitles.write_srt(cues, directory / f"{stem}.srt")
            result.subtitle_files = [kept_ass, kept_srt]

        if report:
            report(1.0, "Done")
        return result

    except Cancelled:
        result.error = "Cancelled."
        return result
    except (media.MediaError, transcribe.TranscriptionError) as exc:
        result.error = str(exc)
        return result
    except Exception as exc:  # noqa: BLE001 - surface anything to the UI
        result.error = f"Unexpected error processing {source.name}: {exc}"
        return result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _verify_render(rendered: Path, original: "media.VideoInfo") -> str | None:
    """Check a rendered file before it is allowed to overwrite the original.

    Returns None when the file looks sound, otherwise a short reason. This is
    the only thing standing between a truncated encode and the permanent loss
    of the user's video, so it errs towards refusing.
    """
    if not rendered.exists():
        return "the file was not written"

    if rendered.stat().st_size < 1024:
        return "the file is empty"

    try:
        info = media.probe(rendered)
    except media.MediaError as exc:
        return f"it is not readable: {exc}"

    if info.width != original.width or info.height != original.height:
        return (
            f"the picture size changed "
            f"({original.width}x{original.height} to {info.width}x{info.height})"
        )

    if original.has_audio and not info.has_audio:
        return "the audio track went missing"

    # A truncated encode is the realistic failure here, so compare running time
    # rather than trusting that ffmpeg exited cleanly.
    if original.duration > 0 and info.duration > 0:
        drift = abs(info.duration - original.duration)
        if drift > max(1.0, original.duration * 0.02):
            return (
                f"it is {drift:.1f}s shorter or longer than the original, "
                "which suggests the encode was cut short"
            )

    return None


def _already_in_target(detected: str, target: Language) -> bool:
    """Skip translation when Whisper produced the target language already.

    Chinese is the exception worth spelling out: Whisper reports plain "zh"
    whether the audio is Mandarin or Cantonese and regardless of which script
    it transcribed into, so we still run the translator to guarantee the
    Simplified/Traditional distinction the user asked for.
    """
    if not detected:
        return False
    if target.code.startswith("zh"):
        return False
    return detected.lower() == target.code.split("-")[0].lower()
