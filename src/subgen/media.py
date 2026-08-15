"""Everything that shells out to ffmpeg: probing, audio extraction, burn-in.

Two things in here are subtle and worth knowing about before editing:

1. ffmpeg filtergraph paths need brutal escaping (colons, backslashes and
   quotes all have meaning inside a filter argument, and Windows drive letters
   contain a colon). Rather than maintain that escaping, we copy the subtitle
   file into a scratch directory under a plain ASCII name and run ffmpeg with
   its working directory set there, so the filtergraph only ever sees
   `ass=subs.ass`. Input and output paths stay as ordinary argv entries, which
   need no escaping at all.

2. Hardware encoding is attempted and then verified, not assumed. VideoToolbox
   is present on every Mac but fails on some inputs, so a failed hardware run
   falls back to libx264 rather than surfacing an error.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ProgressFn = Callable[[float, str], None]


class MediaError(RuntimeError):
    """A problem with ffmpeg or with the file the user picked."""


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def require_ffmpeg() -> str:
    path = ffmpeg_path()
    if not path:
        raise MediaError(
            "ffmpeg was not found on your PATH.\n" + install_hint()
        )
    return path


def install_hint() -> str:
    """How to get ffmpeg on this platform."""
    if sys.platform == "darwin":
        if shutil.which("brew"):
            return "Install it with:  brew install ffmpeg"
        return (
            "Install Homebrew from https://brew.sh then run:  brew install ffmpeg\n"
            "Or download a build from https://ffmpeg.org/download.html"
        )
    if sys.platform.startswith("linux"):
        if shutil.which("apt"):
            return "Install it with:  sudo apt install ffmpeg"
        if shutil.which("dnf"):
            return "Install it with:  sudo dnf install ffmpeg"
        if shutil.which("pacman"):
            return "Install it with:  sudo pacman -S ffmpeg"
        return "Install ffmpeg with your distribution's package manager."
    if shutil.which("winget"):
        return "Install it with:  winget install Gyan.FFmpeg"
    return "Install ffmpeg from https://ffmpeg.org/download.html"


def install_command() -> list[str] | None:
    """The command we would run for a consented, assisted ffmpeg install.

    Returns None when we cannot install without sudo or a package manager,
    in which case the caller should show `install_hint()` instead. We never
    run anything requiring elevation on the user's behalf.
    """
    if sys.platform == "darwin" and shutil.which("brew"):
        return ["brew", "install", "ffmpeg"]
    if sys.platform.startswith("win") and shutil.which("winget"):
        return ["winget", "install", "--id", "Gyan.FFmpeg", "-e"]
    return None


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float  # seconds; 0.0 when ffprobe could not determine it
    width: int
    height: int
    has_audio: bool
    video_codec: str = ""
    audio_codec: str = ""
    video_bitrate: int = 0  # bits/sec; 0 when it could not be determined

    @property
    def duration_label(self) -> str:
        if self.duration <= 0:
            return "unknown length"
        minutes, seconds = divmod(int(self.duration), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


def probe(path: Path) -> VideoInfo:
    """Read stream metadata, and confirm the file is actually openable.

    This doubles as validation: a file with a video extension but corrupt
    contents fails here, before we spend minutes transcribing it.
    """
    probe_bin = ffprobe_path()
    if not probe_bin:
        raise MediaError("ffprobe was not found on your PATH.\n" + install_hint())

    try:
        result = subprocess.run(
            [
                probe_bin, "-v", "error",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"ffprobe timed out reading {path.name}.") from exc

    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else "unknown error"
        raise MediaError(f"Could not read {path.name}: {tail}")

    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        raise MediaError(f"ffprobe returned unreadable output for {path.name}.") from exc

    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video is None:
        raise MediaError(
            f"{path.name} has no video stream, so there is nothing to burn "
            "subtitles into."
        )

    duration = 0.0
    for candidate in (payload.get("format", {}).get("duration"), video.get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue

    return VideoInfo(
        path=path,
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        has_audio=audio is not None,
        video_codec=video.get("codec_name") or "",
        audio_codec=(audio or {}).get("codec_name") or "",
        video_bitrate=_video_bitrate(payload, video, audio, path, duration),
    )


def _video_bitrate(payload, video, audio, path: Path, duration: float) -> int:
    """Best available estimate of the source's video bitrate, in bits/sec.

    Needed so the re-encode can match the source instead of inventing a size.
    Containers vary in what they report, so this tries progressively weaker
    evidence rather than giving up at the first missing field.
    """

    def as_int(value) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    # 1. The video stream states it outright. MP4/MOV usually do.
    direct = as_int(video.get("bit_rate"))
    if direct > 0:
        return direct

    # 2. Some Matroska files put it in stream tags instead.
    for key in ("BPS", "BPS-eng"):
        tagged = as_int((video.get("tags") or {}).get(key))
        if tagged > 0:
            return tagged

    # 3. Container total, minus audio if we know it. Otherwise assume audio is
    #    a small slice, which is true for essentially all real video files.
    total = as_int(payload.get("format", {}).get("bit_rate"))
    if total > 0:
        audio_rate = as_int((audio or {}).get("bit_rate"))
        return max(total - audio_rate, int(total * 0.92)) if audio_rate else int(total * 0.92)

    # 4. Last resort: derive it from how big the file actually is.
    if duration > 0:
        try:
            return int(path.stat().st_size * 8 / duration * 0.92)
        except OSError:
            pass

    return 0


def extract_audio(source: Path, destination: Path, *, progress: ProgressFn | None = None) -> Path:
    """Decode to 16 kHz mono PCM — the format both Whisper and diarization want.

    Doing this once up front means the audio is decoded a single time instead
    of once per model, which matters for long files.
    """
    ffmpeg = require_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-y",
        "-i", str(source),
        "-vn",                  # drop video
        "-ac", "1",             # mono
        "-ar", "16000",         # 16 kHz
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(destination),
    ]
    _run(command, progress=progress, label="Extracting audio", total=0.0)

    if not destination.exists() or destination.stat().st_size == 0:
        raise MediaError(f"Failed to extract audio from {source.name}.")
    return destination


def _supports_encoder(name: str) -> bool:
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return name in result.stdout


SUBTITLE_BITRATE_HEADROOM = 1.10  # burned-in text is sharp and costs extra bits

# Used only when the source bitrate is unknown, keyed by pixel count.
_FALLBACK_BITRATES: tuple[tuple[int, int], ...] = (
    (3840 * 2160, 16_000_000),
    (2560 * 1440, 9_000_000),
    (1920 * 1080, 5_000_000),
    (1280 * 720, 2_500_000),
    (854 * 480, 1_200_000),
)

MIN_BITRATE = 400_000
MAX_BITRATE = 40_000_000


def target_bitrate(source_bitrate: int, width: int, height: int) -> int:
    """Choose an output bitrate that keeps the file close to the original size.

    Re-encoding at a fixed rate is what turns a modest 1.7GB source into a
    7.7GB output: the encoder is simply told to spend far more bits than the
    original ever used. Matching the source, with a little headroom for the
    hard edges of burned-in text, keeps the result about the same size and the
    same visual quality.
    """
    if source_bitrate > 0:
        chosen = int(source_bitrate * SUBTITLE_BITRATE_HEADROOM)
    else:
        pixels = max(width * height, 1)
        chosen = _FALLBACK_BITRATES[-1][1]
        for threshold, rate in _FALLBACK_BITRATES:
            if pixels >= threshold:
                chosen = rate
                break

    return max(MIN_BITRATE, min(chosen, MAX_BITRATE))


def burn_in(
    source: Path,
    subtitle_file: Path,
    destination: Path,
    *,
    duration: float = 0.0,
    source_bitrate: int = 0,
    width: int = 0,
    height: int = 0,
    progress: ProgressFn | None = None,
    prefer_hardware: bool = True,
) -> Path:
    """Render `subtitle_file` (ASS) permanently into the picture.

    The original file is never modified; we always write a new one. Audio is
    stream-copied, so only the video is re-encoded — and the video bitrate is
    derived from the source so the output does not balloon in size.
    """
    ffmpeg = require_ffmpeg()
    if not subtitle_file.exists():
        raise MediaError(f"Subtitle file {subtitle_file} is missing.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    # See the module docstring: copy the subtitles next to where ffmpeg will
    # run so the filtergraph needs no path escaping whatsoever.
    scratch = destination.parent / f".subgen-{destination.stem}"
    scratch.mkdir(parents=True, exist_ok=True)
    local_subs = scratch / "subs.ass"
    shutil.copyfile(subtitle_file, local_subs)

    target = target_bitrate(source_bitrate, width, height)

    attempts: list[list[str]] = []
    if prefer_hardware and sys.platform == "darwin" and _supports_encoder("h264_videotoolbox"):
        # VideoToolbox has no constant-quality mode worth relying on, so it is
        # driven by an explicit bitrate derived from the source.
        attempts.append([
            "-c:v", "h264_videotoolbox",
            "-b:v", str(target),
            "-maxrate", str(int(target * 1.5)),
            "-bufsize", str(int(target * 3)),
        ])
    # x264 is quality-driven, but CRF alone will happily spend far more bits
    # than the source used, so it is capped as well.
    attempts.append([
        "-c:v", "libx264", "-preset", "medium", "-crf", "21",
        "-maxrate", str(int(target * 1.4)),
        "-bufsize", str(int(target * 2.5)),
    ])

    # +faststart is an MP4-family muxer option; other containers reject it.
    container_args = (
        ["-movflags", "+faststart"]
        if destination.suffix.lower() in {".mp4", ".m4v", ".mov"}
        else []
    )

    last_error = ""
    try:
        for index, encoder_args in enumerate(attempts):
            command = [
                ffmpeg, "-hide_banner", "-nostdin", "-y",
                "-i", str(source.resolve()),
                "-vf", "ass=subs.ass",
                *encoder_args,
                "-c:a", "copy",
                *container_args,
                str(destination.resolve()),
            ]
            try:
                _run(
                    command,
                    progress=progress,
                    label="Rendering video",
                    total=duration,
                    cwd=scratch,
                )
                return destination
            except MediaError as exc:
                last_error = str(exc)
                # Fall through to the next encoder; the software path is last
                # and its failure is a real failure.
                if index == len(attempts) - 1:
                    raise MediaError(
                        f"Could not render subtitles into {source.name}.\n{last_error}"
                    ) from exc
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return destination


_TIME_RE = re.compile(r"out_time_ms=(\d+)")


def _run(
    command: list[str],
    *,
    progress: ProgressFn | None,
    label: str,
    total: float,
    cwd: Path | None = None,
) -> None:
    """Run ffmpeg, translating its progress output into fraction-complete calls.

    stderr goes to a temporary file rather than a pipe. With both streams piped,
    ffmpeg blocks once the 64KB stderr buffer fills while we are still blocked
    reading stdout, and neither process can proceed — a deadlock that libass
    reliably triggers, because it logs a font-matching line per subtitle event.
    Spooling stderr to a file gives it unlimited room and costs nothing.
    """
    full = command[:1] + ["-progress", "pipe:1", "-nostats"] + command[1:]

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errors:
        try:
            process = subprocess.Popen(
                full,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise MediaError(f"Could not start ffmpeg: {exc}") from exc

        assert process.stdout is not None
        try:
            for line in process.stdout:
                if progress is None or total <= 0:
                    continue
                match = _TIME_RE.search(line)
                if match:
                    seconds = int(match.group(1)) / 1_000_000
                    progress(min(seconds / total, 1.0), label)
        finally:
            process.stdout.close()
            process.wait()

        if process.returncode != 0:
            errors.seek(0)
            raise MediaError(_last_meaningful_line(errors.read().splitlines()))

    if progress and total > 0:
        progress(1.0, label)


def _last_meaningful_line(lines: Iterable[str]) -> str:
    """ffmpeg's real error is usually the last non-empty stderr line."""
    meaningful = [line.strip() for line in lines if line.strip()]
    return meaningful[-1] if meaningful else "ffmpeg failed with no error output."
