"""Turn a linear transcript plus speaker turns into readable, timed cues.

This module is where the dialogue behaviour is decided, so it is worth stating
the rule it implements:

  * A cue stays on screen long enough to actually be read, which is usually
    longer than the words took to say.
  * If the next thing said comes from the *same* speaker, the previous cue is
    cut short and replaced. One voice never stacks against itself.
  * If it comes from a *different* speaker, the previous cue is allowed to
    remain visible, and the two overlap. The layout engine renders that overlap
    as stacked lines, newest at the bottom.

So the overlaps that drive stacking are created here, by holding lines for
reading time — not by Whisper, which emits a single sequential stream and would
otherwise never produce two cues at once.
"""

from __future__ import annotations

from .transcript import Cue, Segment, SpeakerTurn, Transcript, Word

# Reading speed in characters per second. Latin script and CJK differ by a lot:
# a Chinese character carries far more meaning than a letter, so the same
# information takes fewer characters and needs proportionally more time each.
LATIN_CPS = 17.0
CJK_CPS = 7.0

MIN_DURATION = 1.0      # never flash a line faster than this
MAX_DURATION = 7.0      # never hold one longer than this
MAX_STACK_HOLD = 3.0    # cap on how long a line lingers into another speaker

# An overlap shorter than this is not perceived as two lines on screen — it is
# perceived as the subtitle glitching. Speech naturally overlaps by a few tens
# of milliseconds at almost every turn boundary, so without this floor nearly
# every speaker change would produce a two-frame double-line flicker. Below the
# threshold we cut cleanly to a replacement instead.
MIN_STACK_OVERLAP = 0.4

# Line-length budget before we wrap or split, again script-dependent.
LATIN_CHARS_PER_LINE = 42
CJK_CHARS_PER_LINE = 18
MAX_LINES = 2


def _is_cjk(text: str) -> bool:
    """Whether text is predominantly CJK, which changes every layout constant."""
    if not text:
        return False
    cjk = sum(
        1 for ch in text
        if 0x3040 <= ord(ch) <= 0x30FF      # kana
        or 0x3400 <= ord(ch) <= 0x4DBF      # CJK ext A
        or 0x4E00 <= ord(ch) <= 0x9FFF      # CJK unified
        or 0xAC00 <= ord(ch) <= 0xD7AF      # hangul
    )
    return cjk >= max(1, len(text.strip()) // 4)


def chars_per_line(text: str) -> int:
    return CJK_CHARS_PER_LINE if _is_cjk(text) else LATIN_CHARS_PER_LINE


def reading_time(text: str) -> float:
    """How long `text` needs to be on screen to be comfortably readable."""
    stripped = text.strip()
    if not stripped:
        return MIN_DURATION
    cps = CJK_CPS if _is_cjk(stripped) else LATIN_CPS
    return max(MIN_DURATION, min(len(stripped) / cps, MAX_DURATION))


# --- Speaker assignment ------------------------------------------------------


def assign_speaker(word: Word, turns: list[SpeakerTurn]) -> int | None:
    """Which speaker said this word.

    Preference order: the turn overlapping it most, then the nearest turn by
    midpoint. The fallback matters because diarization trims silence tightly
    and a word's first phoneme can land just outside its turn.
    """
    if not turns:
        return None

    best_turn: SpeakerTurn | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = turn.overlaps(word.start, word.end)
        if overlap > best_overlap:
            best_overlap, best_turn = overlap, turn

    if best_turn is not None:
        return best_turn.speaker

    midpoint = word.midpoint
    nearest = min(
        turns,
        key=lambda t: 0.0 if t.start <= midpoint <= t.end
        else min(abs(t.start - midpoint), abs(t.end - midpoint)),
    )
    return nearest.speaker


def _segment_speaker(segment: Segment, turns: list[SpeakerTurn]) -> int | None:
    """Speaker for a segment that arrived without word timings."""
    if not turns:
        return None
    best_turn, best_overlap = None, 0.0
    for turn in turns:
        overlap = turn.overlaps(segment.start, segment.end)
        if overlap > best_overlap:
            best_overlap, best_turn = overlap, turn
    return best_turn.speaker if best_turn else None


# --- Cue construction --------------------------------------------------------


def build_cues(transcript: Transcript, turns: list[SpeakerTurn]) -> list[Cue]:
    """Split segments wherever the speaker changes mid-sentence.

    Whisper happily runs two people's words into one segment when they trade
    lines quickly. Splitting on the word-level speaker boundary recovers the
    real dialogue structure.
    """
    cues: list[Cue] = []

    for segment in transcript.segments:
        if not segment.text.strip():
            continue

        if not segment.words:
            cues.append(
                Cue(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    speaker=_segment_speaker(segment, turns),
                )
            )
            continue

        run: list[Word] = []
        run_speaker: int | None = None

        for word in segment.words:
            speaker = assign_speaker(word, turns)
            if run and speaker != run_speaker:
                cues.append(_cue_from_words(run, run_speaker))
                run = []
            run_speaker = speaker
            run.append(word)

        if run:
            cues.append(_cue_from_words(run, run_speaker))

    merged = _merge_fragments(cues)
    return [cue for cue in merged if cue.text.strip()]


def _cue_from_words(words: list[Word], speaker: int | None) -> Cue:
    # Whisper emits a leading space on each Latin word and none for CJK, so
    # plain concatenation reproduces the correct spacing for both.
    text = "".join(word.text for word in words).strip()
    return Cue(
        start=words[0].start,
        end=words[-1].end,
        text=text,
        speaker=speaker,
        words=tuple(words),
    )


def _merge_fragments(cues: list[Cue], *, min_chars: int = 2) -> list[Cue]:
    """Reabsorb one- or two-character splinters into their neighbour.

    A single misassigned word at a speaker boundary would otherwise become its
    own cue, which looks like a glitch on screen.
    """
    if not cues:
        return []

    merged: list[Cue] = [cues[0]]
    for cue in cues[1:]:
        previous = merged[-1]
        tiny = len(cue.text.strip()) <= min_chars
        adjacent = cue.start - previous.end < 0.35
        if tiny and adjacent:
            merged[-1] = Cue(
                start=previous.start,
                end=max(previous.end, cue.end),
                text=f"{previous.text} {cue.text}".strip(),
                speaker=previous.speaker,
                words=previous.words + cue.words,
            )
            continue

        # Same speaker, effectively continuous, and short enough to combine.
        if (
            cue.speaker == previous.speaker
            and cue.start - previous.end < 0.25
            and len(previous.text) + len(cue.text) <= chars_per_line(cue.text) * MAX_LINES
        ):
            merged[-1] = Cue(
                start=previous.start,
                end=cue.end,
                text=f"{previous.text} {cue.text}".strip(),
                speaker=previous.speaker,
                words=previous.words + cue.words,
            )
            continue

        merged.append(cue)
    return merged


# --- Readability timing (this is what creates the stacking overlaps) ---------


def apply_readability(cues: list[Cue]) -> list[Cue]:
    """Hold each cue on screen long enough to read, within the dialogue rules.

    Extension is forward only — we never move a cue's start earlier, because
    that would put words on screen before they are spoken.
    """
    if not cues:
        return []

    ordered = sorted(cues, key=lambda cue: (cue.start, cue.end))
    adjusted: list[Cue] = []

    for index, cue in enumerate(ordered):
        wanted_end = cue.start + reading_time(cue.display_text)
        end = max(cue.end, wanted_end)

        next_same_speaker = _next_index(ordered, index, same_speaker_as=cue.speaker)
        next_any = ordered[index + 1] if index + 1 < len(ordered) else None

        if next_same_speaker is not None:
            # Replacement: this speaker is about to say something else, so the
            # current line must be gone by then.
            end = min(end, ordered[next_same_speaker].start - 0.04)

        if next_any is not None and next_any.speaker != cue.speaker:
            # Stacking: another voice cuts in. We may linger, but only for a
            # while — three lines of history on screen is unreadable.
            end = min(end, next_any.start + MAX_STACK_HOLD)

            # Either overlap enough to be read as two lines, or not at all.
            # Snapping rather than always trimming matters: a line whose reading
            # time lands just under the threshold would otherwise be cut short
            # by the full overlap, losing a noticeable slice of its time on
            # screen. So a near-miss is rounded up to a real stack, and only a
            # genuinely incidental overlap is removed.
            overlap = end - next_any.start
            if 0 < overlap < MIN_STACK_OVERLAP:
                if overlap >= MIN_STACK_OVERLAP / 2:
                    end = next_any.start + MIN_STACK_OVERLAP
                else:
                    end = next_any.start - 0.04

        if cue.speaker is None and next_any is not None:
            # No diarization means no way to tell replace from stack, so fall
            # back to conventional single-line behaviour.
            end = min(end, next_any.start - 0.04)

        end = max(end, cue.start + 0.4)  # never produce a zero-length cue
        adjusted.append(cue.with_timing(cue.start, end))

    return adjusted


def _next_index(cues: list[Cue], index: int, *, same_speaker_as: int | None) -> int | None:
    if same_speaker_as is None:
        return None
    for offset in range(index + 1, len(cues)):
        if cues[offset].speaker == same_speaker_as:
            return offset
    return None


# --- Length control ----------------------------------------------------------


def split_long_cues(cues: list[Cue]) -> list[Cue]:
    """Break cues that would not fit in two lines into consecutive cues.

    Splitting uses word timings so each piece keeps honest timing rather than
    an interpolated guess.
    """
    result: list[Cue] = []

    for cue in cues:
        budget = chars_per_line(cue.text) * MAX_LINES
        if len(cue.text) <= budget or len(cue.words) < 2:
            result.append(cue)
            continue

        pieces = max(2, -(-len(cue.text) // budget))  # ceiling division
        per_piece = max(1, len(cue.words) // pieces)

        for start in range(0, len(cue.words), per_piece):
            chunk = list(cue.words[start:start + per_piece])
            if not chunk:
                continue
            # Avoid leaving a one-word orphan as the final cue.
            remaining = len(cue.words) - (start + per_piece)
            if 0 < remaining <= 2:
                chunk.extend(cue.words[start + per_piece:])
                result.append(_cue_from_words(chunk, cue.speaker))
                break
            result.append(_cue_from_words(chunk, cue.speaker))

    return result


def wrap_text(text: str) -> str:
    """Insert a line break so a cue renders as at most two balanced lines."""
    stripped = text.strip()
    limit = chars_per_line(stripped)
    if len(stripped) <= limit:
        return stripped

    if _is_cjk(stripped):
        # CJK has no spaces to break on. Breaking at the exact midpoint splits
        # words and separates a pronoun from its verb, so prefer the punctuation
        # mark nearest the middle and only fall back to a hard split.
        middle = len(stripped) // 2
        best, best_distance = None, len(stripped)
        for position, char in enumerate(stripped):
            if char in "，。！？；：、,.!?;:" and 0 < position < len(stripped) - 1:
                distance = abs(position - middle)
                if distance < best_distance:
                    best, best_distance = position + 1, distance

        # Only honour it if it does not leave a badly lopsided pair of lines.
        if best is not None and best_distance <= len(stripped) * 0.3:
            return f"{stripped[:best]}\n{stripped[best:]}"
        return f"{stripped[:middle]}\n{stripped[middle:]}"

    words = stripped.split()
    target = len(stripped) / 2
    line, length = [], 0
    for index, word in enumerate(words):
        line.append(word)
        length += len(word) + 1
        if length >= target and index < len(words) - 1:
            return " ".join(line) + "\n" + " ".join(words[index + 1:])
    return stripped


def prepare(transcript: Transcript, turns: list[SpeakerTurn]) -> list[Cue]:
    """Full pipeline from raw transcript to render-ready cues."""
    cues = build_cues(transcript, turns)
    cues = split_long_cues(cues)
    return apply_readability(cues)
