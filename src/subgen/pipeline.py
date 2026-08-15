"""Orchestration: one video in, one subtitled video out.

Both the CLI and the GUI drive this module, so all sequencing, progress
weighting and cancellation lives here rather than being duplicated in each
front end.
"""

from __future__ import annotations

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
    keep_sidecar_files: bool = True
    output_dir: Path | None = None
    subtitle_scale: float = 1.0


@dataclass
class Result:
    source: Path
    output: Path | None = None
    subtitle_files: list[Path] = field(default_factory=list)
    cue_count: int = 0
    speaker_count: int = 0
    detected_language: str = ""
    translated: bool = False
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
                turns = diarize.diarize(audio, progress=progress.phase("diarize"))
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
        destination = output_path_for(source, options)
        media.burn_in(
            source, ass_file, destination,
            duration=info.duration,
            progress=progress.phase("render"),
        )
        progress.finish("render")
        result.output = destination

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
