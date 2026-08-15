"""Command line entry point.

`subgen` with no arguments opens the GUI, which is the intended way to use the
tool. Everything else exists for setup, scripting and headless machines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import models

# Must happen before anything imports faster-whisper, so that model resolution
# never reaches for the network during a normal run.
models.enforce_offline()

from . import config, diarize, formats, languages, media, ollama, pipeline, transcribe  # noqa: E402


def _make_reporter():
    """A terminal progress bar that redraws only when it would actually change.

    Two reasons to throttle: redrawing on every chunk is wasted work, and when
    output is redirected to a file `\\r` is not honoured, so an unthrottled bar
    writes tens of thousands of lines into the log.
    """
    interactive = sys.stdout.isatty()
    state = {"percent": -1, "label": ""}

    def report(fraction: float, label: str) -> None:
        percent = int(max(0.0, min(fraction, 1.0)) * 100)
        if percent == state["percent"] and label == state["label"]:
            return
        state["percent"], state["label"] = percent, label

        if not interactive:
            # One line per phase change only, so logs stay readable.
            print(f"  {percent:3d}%  {label}", flush=True)
            return

        filled = percent * 30 // 100
        bar = "█" * filled + "░" * (30 - filled)
        print(f"\r  {bar} {percent:3d}%  {label[:38]:<38}", end="", flush=True)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subgen",
        description=(
            "Burn speaker-aware, meaning-first subtitles into videos, offline.\n"
            "Run with no arguments to open the graphical picker."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Supported formats: {formats.describe()}",
    )

    parser.add_argument("videos", nargs="*", type=Path, help="video files to process")
    parser.add_argument(
        "-l", "--lang", default=None,
        help="target subtitle language code (e.g. zh-Hans, en). Omit to keep the "
             "spoken language.",
    )
    parser.add_argument(
        "--spoken", default=None,
        help="override the spoken language instead of auto-detecting it",
    )
    parser.add_argument(
        "-m", "--model", default=None,
        help="Ollama model to translate with. Defaults to your remembered choice.",
    )
    parser.add_argument(
        "--whisper-model", default=None,
        help=f"transcription model (default: {transcribe.DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="where to write finished videos (default: next to the source)",
    )
    parser.add_argument(
        "--no-speakers", action="store_true",
        help="disable speaker detection, so lines never stack",
    )
    parser.add_argument(
        "--no-sidecars", action="store_true",
        help="do not write .ass/.srt files alongside the video",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="subtitle size multiplier (default: 1.0)",
    )

    parser.add_argument(
        "--setup", action="store_true",
        help="download the models needed to run offline, then exit",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="with --setup, re-download even if files already exist",
    )
    parser.add_argument(
        "--purge", action="store_true",
        help="delete every cached model and exit",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="show what is installed and what is missing, then exit",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="list the Ollama models available for translation, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    # `subgen setup` reads better than `subgen --setup`; accept both.
    if argv and argv[0] == "setup":
        argv = ["--setup"] + argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.purge:
        models.purge()
        print("Deleted all cached models.")
        return 0

    if args.status:
        return _status(args)

    if args.list_models:
        return _list_models()

    if args.setup:
        return _setup(args)

    if not args.videos:
        from . import gui

        return gui.run()

    return _headless(args)


# --- subcommands -------------------------------------------------------------


def _status(args: argparse.Namespace) -> int:
    settings = config.load()
    whisper_model = args.whisper_model or settings["whisper_model"]

    print(models.cache_summary())
    print()
    print(f"ffmpeg:          {media.ffmpeg_path() or 'NOT FOUND — ' + media.install_hint()}")
    print(f"Transcription:   {transcribe.active_backend()}")
    print(f"Speaker models:  {'ready' if diarize.is_available() else 'not downloaded'}")

    if not ollama.is_installed():
        print("Ollama:          not installed (translation unavailable)")
    elif not ollama.is_running():
        print("Ollama:          installed but not running")
    else:
        found = ollama.list_models(settings.get("ollama_host"))
        print(f"Ollama:          running, {len(found)} chat model(s) available")

    missing = models.missing_assets(whisper_model, need_diarization=True)
    if missing:
        print("\nMissing for offline use:")
        for asset in missing:
            print(f"  - {asset.label} ({asset.size_hint}) → {asset.how_to_get}")
    else:
        print("\nEverything needed is downloaded. This will run with no network.")
    return 0


def _list_models() -> int:
    if not ollama.is_installed():
        print("Ollama is not installed.\n")
        print(ollama.install_hint())
        return 1

    if not ollama.is_running() and not ollama.try_start_daemon():
        print("Ollama is installed but the daemon is not running. Start it with:")
        print("  ollama serve")
        return 1

    found = ollama.list_models()
    if not found:
        print("No chat models installed. Any of these would work:\n")
        print(ollama.pull_hint())
        return 1

    print("Models available for translation:\n")
    for model in found:
        print(f"  {model.describe()}")
    print("\nAny chat-capable model works — this list is whatever you have installed.")
    return 0


def _setup(args: argparse.Namespace) -> int:
    """One-time download so that later runs need no network at all."""
    settings = config.load()
    whisper_model = args.whisper_model or settings["whisper_model"]

    models.allow_downloads()
    print("Downloading models for offline use. This is only needed once.\n")

    show = _make_reporter()

    try:
        models.fetch_diarization(progress=show, force=args.force)
        print()
        models.fetch_whisper(whisper_model, progress=show)
        print()
    except models.DownloadError as exc:
        print(f"\n\n{exc}", file=sys.stderr)
        return 1
    finally:
        models.enforce_offline()

    print("\n" + models.cache_summary())

    if not ollama.is_installed():
        print(
            "\nOptional — translation needs Ollama, which we do not install for "
            "you because it registers a background service:\n"
        )
        print("  " + ollama.install_hint().replace("\n", "\n  "))
    elif not ollama.list_models():
        print("\nOptional — pull a chat model to enable translation:\n")
        print(ollama.pull_hint())

    print("\nSetup complete. You can now run offline.")
    return 0


def _headless(args: argparse.Namespace) -> int:
    settings = config.load()

    target = _resolve_language(args.lang, "target") if args.lang else None
    if args.lang and target is None:
        return 2

    spoken = _resolve_language(args.spoken, "spoken") if args.spoken else None
    if args.spoken and spoken is None:
        return 2

    videos: list[Path] = []
    for path in args.videos:
        if not path.exists():
            print(f"No such file: {path}", file=sys.stderr)
            return 2
        if not formats.is_supported(path):
            print(
                f"Unsupported format: {path.name}\nSupported: {formats.describe()}",
                file=sys.stderr,
            )
            return 2
        videos.append(path)

    options = pipeline.Options(
        target_language=target,
        source_language=spoken,
        whisper_model=args.whisper_model or settings["whisper_model"],
        ollama_model=args.model or settings.get("ollama_model"),
        ollama_host=settings.get("ollama_host"),
        identify_speakers=not args.no_speakers,
        keep_sidecar_files=not args.no_sidecars,
        output_dir=args.output_dir,
        subtitle_scale=args.scale,
    )

    problems = pipeline.preflight(options)
    blocking = [p for p in problems if "translation model" not in p]
    if blocking:
        for problem in blocking:
            print(f"\n{problem}", file=sys.stderr)
        return 1
    if problems and target:
        print(
            "Warning: no translation model available — subtitles will stay in "
            "the spoken language.\n",
            file=sys.stderr,
        )

    failures = 0
    for index, path in enumerate(videos, start=1):
        print(f"\n[{index}/{len(videos)}] {path.name}")

        result = pipeline.process(path, options, report=_make_reporter())
        print()

        if result.ok and result.output:
            detail = [f"{result.cue_count} lines"]
            if result.speaker_count:
                detail.append(f"{result.speaker_count} speakers")
            if result.translated:
                detail.append("translated")
            print(f"  → {result.output}  ({', '.join(detail)})")
            for extra in result.subtitle_files:
                print(f"  → {extra}")
        else:
            failures += 1
            print(f"  Failed: {result.error}", file=sys.stderr)

    return 1 if failures else 0


def _resolve_language(value: str, role: str) -> languages.Language | None:
    """Accept either a code (zh-Hans) or a label (Chinese (Simplified))."""
    found = languages.BY_CODE.get(value) or languages.BY_LABEL.get(value)
    if found:
        return found

    # Be forgiving about the common shorthands people actually type.
    aliases = {
        "zh": "zh-Hans", "chinese": "zh-Hans", "simplified": "zh-Hans",
        "zh-cn": "zh-Hans", "zh-hans": "zh-Hans",
        "zh-tw": "zh-Hant", "traditional": "zh-Hant", "zh-hant": "zh-Hant",
        "english": "en",
    }
    alias = aliases.get(value.strip().lower())
    if alias:
        return languages.BY_CODE[alias]

    print(f"Unknown {role} language: {value}", file=sys.stderr)
    print("\nAvailable:", file=sys.stderr)
    for language in languages.LANGUAGES:
        print(f"  {language.code:<8} {language.label}", file=sys.stderr)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
