"""Render cues to ASS, with the stacked-dialogue layout.

Why ASS and not SRT: SRT has no way to say "put this line above that one". The
whole dialogue behaviour the tool exists to produce needs per-line positioning,
which only ASS gives us. An SRT is still written alongside as a convenience,
but it is a lossy export — it cannot represent stacking.

The layout algorithm
--------------------
Cues can overlap in time (see `align.apply_readability`). Naively emitting them
would draw two lines on top of each other, so instead:

1. Every cue start and end becomes a boundary on a shared timeline.
2. Between consecutive boundaries the set of visible cues is, by construction,
   constant. Each such interval is laid out independently.
3. Within an interval, visible cues are ordered by start time. The most recent
   one takes the bottom slot and older ones are pushed up above it, so a new
   speaker always appears where the eye already is.
4. Adjacent intervals in which a cue keeps the same slot are merged back
   together, so a line that never moves is emitted as one event rather than
   several identical ones.

Step 3 is what produces the requested behaviour: a line already on screen
rises to make room when somebody else cuts in, and drops away on its own.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .align import wrap_text
from .transcript import Cue

# More than this many lines at once stops being readable; the oldest is
# dropped from the stack rather than shrinking everything to fit.
MAX_STACK = 3


@dataclass(frozen=True)
class Style:
    """Visual parameters, all derived from the video's height so that a 4K and
    a 720p render look the same size relative to the picture."""

    width: int
    height: int
    font: str
    font_size: int
    margin_bottom: int
    line_height: int
    outline: float
    shadow: float

    @classmethod
    def for_video(
        cls, width: int, height: int, *, language_code: str = "en", scale: float = 1.0
    ) -> "Style":
        # Guard against a probe that returned nothing useful.
        width = width or 1920
        height = height or 1080

        font_size = max(16, int(round(height / 22.5 * scale)))
        return cls(
            width=width,
            height=height,
            font=default_font(language_code),
            font_size=font_size,
            margin_bottom=max(12, int(round(height * 0.045))),
            line_height=int(round(font_size * 1.25)),
            outline=max(1.0, round(font_size * 0.055, 1)),
            shadow=max(0.0, round(font_size * 0.02, 1)),
        )


# fontconfig language tags for the scripts that need a specific face.
_FONTCONFIG_LANG = {
    "zh-Hans": "zh-cn",
    "zh-Hant": "zh-tw",
    "zh": "zh-cn",
    "ja": "ja",
    "ko": "ko",
}

_font_cache: dict[str, str] = {}


def default_font(language_code: str) -> str:
    """Find a font that genuinely contains the glyphs we are about to draw.

    Hardcoding a face name does not work. libass resolves fonts through
    fontconfig, and fontconfig's view of the system is often not what you would
    expect — on macOS it frequently does not index PingFang SC at all, so
    naming it gets you a silent fallback to Verdana and a subtitle full of tofu
    boxes. The name being "correct" for the platform is irrelevant if the font
    stack cannot find it.

    So we ask fontconfig what it would actually pick for the language, which is
    the same question libass will ask later, and use its answer. That is
    self-correcting across platforms and across whatever fonts happen to be
    installed.
    """
    if language_code in _font_cache:
        return _font_cache[language_code]

    resolved = _query_fontconfig(language_code) or _platform_default(language_code)
    _font_cache[language_code] = resolved
    return resolved


def _query_fontconfig(language_code: str) -> str | None:
    """Ask fc-match which family covers this language. None if unavailable."""
    if not shutil.which("fc-match"):
        return None

    base = language_code.split("+")[0]
    lang_tag = _FONTCONFIG_LANG.get(base) or _FONTCONFIG_LANG.get(base.split("-")[0])

    # Latin script needs no coverage query — every system font has it, and
    # fontconfig's generic answer (often Verdana) is a worse subtitle face than
    # the platform's own UI font. macOS and Windows have a known-good choice, so
    # only fall through to fontconfig there when the script demands it.
    has_known_ui_font = sys.platform == "darwin" or sys.platform.startswith("win")
    if not lang_tag and has_known_ui_font:
        return None

    pattern = f":lang={lang_tag}" if lang_tag else "sans-serif"

    try:
        result = subprocess.run(
            ["fc-match", "--format=%{family[0]}", pattern],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    family = (result.stdout or "").strip()
    if result.returncode != 0 or not family:
        return None

    # fontconfig returns comma-separated aliases for some CJK faces; libass
    # wants a single family name.
    return family.split(",")[0].strip()


def _platform_default(language_code: str) -> str:
    """Last resort when fontconfig is unavailable, e.g. on Windows."""
    cjk = language_code.startswith(("zh", "ja", "ko"))

    if sys.platform == "darwin":
        if cjk:
            return "Hiragino Sans GB" if language_code.startswith("zh") else "Hiragino Sans"
        return "Helvetica Neue"
    if sys.platform.startswith("win"):
        if language_code.startswith("ja"):
            return "Yu Gothic"
        if language_code.startswith("ko"):
            return "Malgun Gothic"
        return "Microsoft YaHei" if cjk else "Segoe UI"
    if cjk:
        return "Noto Sans CJK SC"
    return "Noto Sans"


# --- Layout ------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """One cue, on screen, at one vertical slot, for one span of time."""

    cue: Cue
    start: float
    end: float
    slot: int
    lines: int


def layout(cues: list[Cue]) -> list[Placement]:
    """Assign every visible cue a vertical slot over time. See module docstring."""
    if not cues:
        return []

    ordered = sorted(cues, key=lambda cue: (cue.start, cue.end))
    line_counts = {id(cue): wrap_text(cue.display_text).count("\n") + 1 for cue in ordered}

    boundaries = sorted({round(t, 3) for cue in ordered for t in (cue.start, cue.end)})

    raw: list[Placement] = []
    for index in range(len(boundaries) - 1):
        span_start, span_end = boundaries[index], boundaries[index + 1]
        if span_end - span_start < 0.01:
            continue

        # Test the midpoint rather than the edges. Because boundaries are drawn
        # from the cue times themselves, a cue either covers a whole interval or
        # none of it, so this is exact — and unlike an edge comparison it cannot
        # be defeated by the boundary rounding above, which would otherwise make
        # a cue look absent during its own final interval and silently drop the
        # stacked line.
        midpoint = (span_start + span_end) / 2
        visible = [cue for cue in ordered if cue.start <= midpoint <= cue.end]
        if not visible:
            continue

        # Newest last, so that reversing puts the newest at slot 0 (bottom).
        visible.sort(key=lambda cue: (cue.start, cue.end))
        stack = visible[-MAX_STACK:]

        for slot, cue in enumerate(reversed(stack)):
            raw.append(
                Placement(
                    cue=cue,
                    start=span_start,
                    end=span_end,
                    slot=slot,
                    lines=line_counts[id(cue)],
                )
            )

    return _merge_placements(raw)


def _merge_placements(placements: list[Placement]) -> list[Placement]:
    """Join back-to-back intervals where a cue held the same slot."""
    if not placements:
        return []

    placements = sorted(placements, key=lambda p: (id(p.cue), p.start))
    merged: list[Placement] = []

    for placement in placements:
        if merged:
            previous = merged[-1]
            same_cue = previous.cue is placement.cue
            same_slot = previous.slot == placement.slot
            contiguous = abs(previous.end - placement.start) < 0.02
            if same_cue and same_slot and contiguous:
                merged[-1] = Placement(
                    cue=previous.cue,
                    start=previous.start,
                    end=placement.end,
                    slot=previous.slot,
                    lines=previous.lines,
                )
                continue
        merged.append(placement)

    # Safety net. `align` already refuses to create overlaps too short to read,
    # but a rounding edge could still leave a raised line on screen for a couple
    # of frames, which reads as a glitch. Dropping the raised placement leaves
    # the newer line in the bottom slot, i.e. a clean replacement.
    merged = [p for p in merged if p.slot == 0 or (p.end - p.start) >= 0.25]

    merged.sort(key=lambda p: (p.start, p.slot))
    return merged


def _baseline_for(placement: Placement, style: Style, stack_at: dict[int, int]) -> int:
    """Vertical pixel position of a slot, accounting for taller cues below it.

    A two-line cue in the bottom slot has to push the slot above it twice as
    far up as a one-line cue would, otherwise they collide.
    """
    offset = style.margin_bottom
    for lower_slot in range(placement.slot):
        offset += stack_at.get(lower_slot, 1) * style.line_height
    return style.height - offset


# --- ASS emission ------------------------------------------------------------


def _timestamp(seconds: float) -> str:
    """ASS wants H:MM:SS.cc with centisecond precision."""
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis == 100:  # rounding carried into the next second
        centis, secs = 0, secs + 1
        if secs == 60:
            secs, minutes = 0, minutes + 1
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape(text: str) -> str:
    """Make text safe for an ASS Dialogue line.

    Braces introduce override tags and would otherwise be swallowed or, worse,
    interpreted as formatting from the transcript.
    """
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\N")
    )


def build_ass(cues: list[Cue], style: Style) -> str:
    """Produce a complete ASS document for these cues."""
    placements = layout(cues)

    # How tall the cue in each slot is, per moment, so slots above it sit clear.
    header = f"""[Script Info]
; Generated by subtitle-generator
ScriptType: v4.00+
PlayResX: {style.width}
PlayResY: {style.height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font},{style.font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{style.outline},{style.shadow},2,20,20,{style.margin_bottom},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]
    centre_x = style.width // 2

    for placement in placements:
        # Heights of the cues sharing this instant, keyed by slot.
        concurrent = {
            other.slot: other.lines
            for other in placements
            if other.start < placement.end and other.end > placement.start
        }
        baseline = _baseline_for(placement, style, concurrent)

        text = _escape(wrap_text(placement.cue.display_text))
        # \pos disables libass collision handling, so our slots are final.
        # \an2 anchors the text block by its bottom centre.
        override = f"{{\\an2\\pos({centre_x},{baseline})}}"

        lines.append(
            "Dialogue: 0,"
            f"{_timestamp(placement.start)},{_timestamp(placement.end)},"
            f"Default,,0,0,0,,{override}{text}"
        )

    return "\n".join(lines) + "\n"


def write_ass(cues: list[Cue], style: Style, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_ass(cues, style), encoding="utf-8")
    return destination


# --- SRT (lossy convenience export) -----------------------------------------


def _srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        millis, secs = 0, secs + 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt(cues: list[Cue]) -> str:
    """Flatten cues into SRT.

    Overlaps cannot be expressed, so simultaneous cues are merged into one
    multi-line entry — the closest honest approximation the format allows.
    """
    ordered = sorted(cues, key=lambda cue: (cue.start, cue.end))
    blocks: list[str] = []
    index = 1

    consumed: set[int] = set()
    for position, cue in enumerate(ordered):
        if position in consumed:
            continue
        group = [cue]
        for other_position in range(position + 1, len(ordered)):
            other = ordered[other_position]
            if other.start < cue.end and other_position not in consumed:
                group.append(other)
                consumed.add(other_position)
            elif other.start >= cue.end:
                break

        start = min(item.start for item in group)
        end = max(item.end for item in group)
        text = "\n".join(wrap_text(item.display_text) for item in group)

        blocks.append(
            f"{index}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n{text}\n"
        )
        index += 1

    return "\n".join(blocks)


def write_srt(cues: list[Cue], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_srt(cues), encoding="utf-8")
    return destination
