"""Tests for the dialogue layout rules.

These encode the behaviour the tool exists to produce, so they are worth
reading as a specification:

  - a second speaker interrupting stacks below, pushing the first line up
  - the same speaker continuing replaces, and never stacks
  - overlaps too brief to perceive become clean replacements, not flicker
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subgen import align, subtitles  # noqa: E402
from subgen.transcript import Cue, Segment, SpeakerTurn, Transcript, Word  # noqa: E402


def cue(start, end, text, speaker=None):
    return Cue(start=start, end=end, text=text, speaker=speaker)


# --- stacking ----------------------------------------------------------------


def test_interrupting_speaker_stacks_below_and_pushes_first_line_up():
    cues = [
        cue(0.0, 2.0, "Hello there, how are you?", speaker=0),
        cue(1.5, 3.0, "I am doing fine.", speaker=1),
    ]
    placements = subtitles.layout(cues)

    # Look at a single instant inside the overlap rather than filtering by the
    # placement's own span, which merging may have extended past it.
    overlapping = [p for p in placements if p.start <= 1.75 < p.end]
    assert len(overlapping) == 2, "both lines should be on screen during the overlap"

    by_speaker = {p.cue.speaker: p.slot for p in overlapping}
    assert by_speaker[1] == 0, "the interrupting speaker takes the bottom slot"
    assert by_speaker[0] == 1, "the earlier speaker is pushed up one slot"


def test_stacked_lines_get_distinct_vertical_positions():
    cues = [
        cue(0.0, 2.0, "First speaker line", speaker=0),
        cue(1.5, 3.0, "Second speaker line", speaker=1),
    ]
    style = subtitles.Style.for_video(1920, 1080)
    events = [
        line for line in subtitles.build_ass(cues, style).splitlines()
        if line.startswith("Dialogue:")
    ]
    positions = {line.split("\\pos(")[1].split(")")[0] for line in events}
    assert len(positions) == 2, f"expected two vertical positions, got {positions}"


def test_same_speaker_never_stacks():
    cues = [
        cue(0.0, 2.0, "First half of the thought", speaker=0),
        cue(2.1, 4.0, "and the second half", speaker=0),
    ]
    placements = subtitles.layout(align.apply_readability(cues))
    assert all(p.slot == 0 for p in placements), "one voice must never stack on itself"


def test_third_speaker_does_not_produce_a_fourth_line():
    cues = [
        cue(0.0, 5.0, "One", speaker=0),
        cue(1.0, 5.0, "Two", speaker=1),
        cue(2.0, 5.0, "Three", speaker=2),
        cue(3.0, 5.0, "Four", speaker=3),
    ]
    placements = subtitles.layout(cues)
    at_end = [p for p in placements if p.start <= 4.0 < p.end]
    assert len(at_end) <= subtitles.MAX_STACK


# --- flicker prevention ------------------------------------------------------


def test_tiny_natural_overlap_becomes_a_clean_replacement():
    # Turn boundaries routinely overlap by a few tens of milliseconds. That must
    # not draw two lines for two frames.
    cues = [
        cue(0.0, 3.59, "A line that ran slightly long", speaker=0),
        cue(3.52, 6.0, "The reply", speaker=1),
    ]
    adjusted = align.apply_readability(cues)
    first = adjusted[0]
    assert first.end <= 3.52, "a sub-perceptual overlap should be trimmed away"

    placements = subtitles.layout(adjusted)
    assert all(p.slot == 0 for p in placements), "no stacking for a trimmed overlap"


def test_genuine_overlap_survives():
    # Fast speech: the first line still needs reading time when the reply starts.
    cues = [
        cue(0.0, 0.9, "A properly long line that clearly needs several seconds to read",
            speaker=0),
        cue(1.1, 4.0, "The reply", speaker=1),
    ]
    adjusted = align.apply_readability(cues)
    assert adjusted[0].end > 1.1 + align.MIN_STACK_OVERLAP - 0.01, (
        "a line still needing reading time must persist into the next speaker"
    )
    assert any(p.slot == 1 for p in subtitles.layout(adjusted))


def test_near_threshold_overlap_snaps_up_instead_of_being_cut_short():
    # A line whose reading time overruns by just under the threshold should be
    # rounded up to a real stack, not robbed of that reading time entirely.
    cues = [
        cue(0.0, 0.9, "Short but dense line here", speaker=0),
        cue(1.1, 3.0, "The reply", speaker=1),
    ]
    adjusted = align.apply_readability(cues)
    assert adjusted[0].end == 1.1 + align.MIN_STACK_OVERLAP


def test_no_diarization_means_no_stacking():
    cues = [
        cue(0.0, 2.0, "Some narration", speaker=None),
        cue(1.5, 3.5, "More narration", speaker=None),
    ]
    placements = subtitles.layout(align.apply_readability(cues))
    assert all(p.slot == 0 for p in placements)


# --- text handling -----------------------------------------------------------

def test_cjk_wraps_at_punctuation_not_mid_clause():
    wrapped = align.wrap_text("嗨，你可算来了。我还以为你放鸽子了呢。")
    assert "\n" in wrapped
    before = wrapped.split("\n")[0]
    assert before.endswith(("。", "，")), f"broke mid-clause: {wrapped!r}"


def test_latin_wraps_on_word_boundaries():
    text = "This is a fairly long subtitle line that will need to be wrapped somewhere"
    wrapped = align.wrap_text(text)
    assert "\n" in wrapped
    assert wrapped.replace("\n", " ") == text, "wrapping must not alter the words"


def test_cjk_reading_time_is_slower_per_character():
    assert align.reading_time("这是一个测试句子在这里") > align.reading_time("abcdefghijk")


def test_ass_escapes_braces_so_they_are_not_read_as_tags():
    style = subtitles.Style.for_video(1280, 720)
    body = subtitles.build_ass([cue(0.0, 2.0, "Use {braces} here", speaker=0)], style)
    assert "\\{braces\\}" in body


def test_timestamp_rounding_does_not_produce_sixty_seconds():
    assert subtitles._timestamp(59.999) == "0:01:00.00"
    assert subtitles._timestamp(3661.5) == "1:01:01.50"


# --- speaker assignment ------------------------------------------------------


def test_segment_splits_where_the_speaker_changes():
    words = [
        Word(0.0, 0.5, " Hello"),
        Word(0.5, 1.0, " there"),
        Word(1.2, 1.6, " Hi"),
        Word(1.6, 2.0, " back"),
    ]
    transcript = Transcript(
        segments=(Segment(0.0, 2.0, "Hello there Hi back", tuple(words)),)
    )
    turns = [SpeakerTurn(0.0, 1.1, 0), SpeakerTurn(1.15, 2.0, 1)]

    cues = align.build_cues(transcript, turns)
    assert len(cues) == 2, f"expected a split at the speaker change, got {cues}"
    assert cues[0].speaker == 0 and cues[0].text == "Hello there"
    assert cues[1].speaker == 1 and cues[1].text == "Hi back"


def test_word_assignment_uses_midpoint_not_edges():
    # This word starts inside speaker 0's turn but is mostly inside speaker 1's.
    word = Word(0.95, 1.60, " borderline")
    turns = [SpeakerTurn(0.0, 1.0, 0), SpeakerTurn(1.0, 2.0, 1)]
    assert align.assign_speaker(word, turns) == 1


def test_empty_input_is_handled():
    assert subtitles.layout([]) == []
    assert align.apply_readability([]) == []
    assert align.build_cues(Transcript(segments=()), []) == []
