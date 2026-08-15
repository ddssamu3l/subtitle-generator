"""The data that flows through the pipeline.

Transcription produces Words and Segments. Diarization produces SpeakerTurns.
Aligning the two produces Cues, which are what actually get rendered. Keeping
these as plain frozen dataclasses (rather than passing library-specific objects
around) is what lets the ASR and diarization backends be swapped freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0

    @property
    def midpoint(self) -> float:
        """Used for speaker assignment — more robust than either edge alone.

        Whisper's word boundaries drift by a few tens of milliseconds, so a
        word's start can land inside the previous speaker's turn even when the
        word plainly belongs to the next one. The midpoint is far less prone
        to that.
        """
        return (self.start + self.end) / 2


@dataclass(frozen=True)
class Segment:
    """One Whisper output segment, before any speaker information is applied."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class Transcript:
    segments: tuple[Segment, ...]
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0

    @property
    def words(self) -> tuple[Word, ...]:
        return tuple(word for segment in self.segments for word in segment.words)

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()


@dataclass(frozen=True)
class SpeakerTurn:
    """A contiguous stretch attributed to one speaker by diarization.

    Turns may overlap each other in time — that is exactly the case the
    stacked-subtitle layout exists to handle.
    """

    start: float
    end: float
    speaker: int

    def overlaps(self, start: float, end: float) -> float:
        """Seconds of overlap with the given interval; 0.0 when disjoint."""
        return max(0.0, min(self.end, end) - max(self.start, start))


@dataclass(frozen=True)
class Cue:
    """One subtitle line: what is said, when, and by whom.

    `speaker` is None when diarization was disabled or inconclusive, which the
    layout engine treats as "never stack this" — a single voice track.
    """

    start: float
    end: float
    text: str
    speaker: int | None = None
    words: tuple[Word, ...] = ()
    translated: str | None = None

    @property
    def display_text(self) -> str:
        """What actually gets rendered — the translation when we have one."""
        return self.translated if self.translated else self.text

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def with_translation(self, text: str) -> "Cue":
        return replace(self, translated=text)

    def with_timing(self, start: float, end: float) -> "Cue":
        return replace(self, start=start, end=end)


@dataclass
class Job:
    """One video the user asked us to process, plus the choices they made."""

    source: object  # pathlib.Path, kept loose to avoid a circular import
    target_language: object | None  # languages.Language, or None to keep original
    source_language: object | None = None  # None means auto-detect
    identify_speakers: bool = True
    cues: list[Cue] = field(default_factory=list)
