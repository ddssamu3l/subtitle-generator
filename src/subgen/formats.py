"""Which video files we admit, and how we describe them to a file dialog.

This is the single source of truth for "can we process this?". The GUI picker
filters on it, and the CLI validates against it, so the two can never drift.
"""

from __future__ import annotations

# Container formats ffmpeg reads reliably and that carry a video stream we can
# burn subtitles into. Audio-only formats are deliberately excluded: the whole
# output of this tool is a video with subtitles rendered into the picture.
VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".ts",
    ".m2ts",
    ".mts",
    ".wmv",
    ".flv",
    ".ogv",
    ".3gp",
)


def is_supported(path) -> bool:
    """True if `path` has an extension we are willing to open."""
    from pathlib import Path

    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def tk_filetypes() -> list[tuple[str, str]]:
    """Filter spec for tkinter's file dialog.

    Tk wants space-separated globs in a single string. We put the combined
    "all supported" entry first so it is the default selection, then per-format
    entries for people who know what they are looking for.
    """
    combined = " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)
    types: list[tuple[str, str]] = [("Video files", combined)]

    # macOS's native dialog matches case-sensitively on some Tk builds, so also
    # offer uppercase variants rather than silently hiding .MP4 files.
    types.append(("Video files (uppercase)", combined.upper()))

    for ext in VIDEO_EXTENSIONS:
        types.append((f"{ext.lstrip('.').upper()} video", f"*{ext}"))

    types.append(("All files", "*"))
    return types


def describe() -> str:
    """Human-readable list, for error messages and --help text."""
    return ", ".join(ext.lstrip(".") for ext in VIDEO_EXTENSIONS)
