"""Meaning-first subtitle translation through a local LLM.

The point of doing this with an LLM rather than a translation model is that a
translation model sees one cue at a time. It cannot know that "他" three lines
ago is the same person, that a character has been speaking sarcastically for
the last minute, or that a line is a callback to a joke. It produces defensible
sentences that add up to something nobody would say.

So every request carries a window of surrounding dialogue and the speaker
structure, and the prompt asks for the meaning to survive, not the wording.

Everything here runs on the user's machine via Ollama. No text leaves it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from . import ollama
from .languages import Language
from .transcript import Cue

ProgressFn = Callable[[float, str], None]

# How many cues we translate per request, and how much surrounding dialogue
# rides along. Small batches keep the model honest about line counts; the
# context window is what buys continuity across batches.
BATCH_SIZE = 12
CONTEXT_BEFORE = 6
CONTEXT_AFTER = 4


class TranslationError(RuntimeError):
    pass


SYSTEM_PROMPT = """\
You are a professional subtitle translator working on {target_name} \
({target_endonym}) subtitles for a video.

Your job is to carry across what was meant, not what was said. A viewer \
watching with your subtitles should have the same experience as a native \
speaker watching without them.

The source text was produced by automatic speech recognition, so it is not a \
clean script. Expect misheard words that sound like the right ones, mangled \
names and technical terms, dropped or doubled words, missing punctuation, and \
sentences arbitrarily split across lines.

Principles, in order of priority:

0. REPAIR, THEN TRANSLATE. Before translating a line, work out what the speaker \
almost certainly said, using the surrounding dialogue as evidence. If a word is \
clearly a mishearing of something that fits the conversation, translate the \
intended word. If a line is a fragment of a sentence that continues on the next \
line, translate it so the fragments read naturally together. If punctuation is \
missing or wrong, infer the real sentence boundaries.
   Repair only what the context genuinely supports. Do NOT invent facts, names, \
numbers, places or events that are not there. When a line is garbled and the \
context does not tell you what was meant, translate it as directly as you can \
rather than inventing something plausible. A slightly awkward faithful line is \
much better than a fluent wrong one.
   Critically: NEVER move meaning between lines. Each subtitle appears while \
those exact words are being spoken, so a line must contain only what its own \
input line says. When a sentence is split across lines, translate each part \
where it sits — do not finish the sentence early, do not pull words back from a \
later line, and do not push words forward into one. Reading a word before it is \
spoken is worse than an awkward break.
1. MEANING OVER WORDS. Translate the intent, the tone and the subtext. If a \
literal rendering would be confusing, stiff, or would lose the point, discard \
it and write what the speaker actually meant.
2. CULTURAL ADAPTATION. Idioms, jokes, references and figures of speech should \
be replaced with whatever a {target_name} speaker would naturally say in that \
situation. Do not explain a joke; land it. Do not gloss an idiom; substitute \
one.
3. REGISTER AND RELATIONSHIP. Match how formal, rude, warm, hesitant or \
authoritative the speaker is being, using the politeness levels, pronouns and \
sentence-final particles that {target_name} uses to signal exactly that.
4. CONSISTENCY. Names, terminology, and how characters address one another must \
stay stable across the whole video. Use the surrounding dialogue to keep track.
5. SUBTITLE ECONOMY. A viewer has seconds to read while watching. Prefer short, \
natural, spoken-sounding lines. Trim filler that adds nothing.

Hard rules:
- Output EXACTLY one translated line per numbered input line. Never merge, \
split, reorder, drop, or add lines.
- Preserve the numbering format exactly: "<number>. <translation>".
- Output ONLY the numbered lines. No preamble, no commentary, no notes, no \
explanation, no romanisation, no original text.
- Never output the source language. Every line must be in {target_name}.
- If a line is only a sound, a name, or is untranslatable, render it naturally \
in {target_name} rather than leaving it as-is.
- Lines marked [context] are for your understanding only. Do NOT translate or \
output them.
"""


@dataclass
class Translator:
    model: str
    target: Language
    host: str | None = None
    source_hint: str = ""

    def translate_all(
        self, cues: list[Cue], *, progress: ProgressFn | None = None
    ) -> list[Cue]:
        """Translate every cue, returning new Cues carrying the translation."""
        if not cues:
            return []

        result = list(cues)
        batches = [
            list(range(start, min(start + BATCH_SIZE, len(cues))))
            for start in range(0, len(cues), BATCH_SIZE)
        ]

        for batch_number, indices in enumerate(batches):
            translations = self._translate_batch(result, indices)
            for index, text in zip(indices, translations):
                result[index] = result[index].with_translation(text)

            if progress:
                progress(
                    (batch_number + 1) / len(batches),
                    f"Translating to {self.target.label}",
                )

        return result

    # --- one batch ----------------------------------------------------------

    def _translate_batch(self, cues: list[Cue], indices: list[int]) -> list[str]:
        prompt = self._build_prompt(cues, indices)
        expected = len(indices)
        originals = [cues[index].text for index in indices]

        for attempt in range(2):
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        target_name=self.target.label,
                        target_endonym=self.target.endonym,
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            if attempt == 1:
                # The first try came back malformed. Restate the one constraint
                # that matters most, as a fresh turn rather than a re-prompt.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Output exactly {expected} lines, numbered 1 to "
                            f"{expected}, in {self.target.label} only, with no "
                            "other text whatsoever."
                        ),
                    }
                )

            try:
                raw = ollama.chat(
                    self.model,
                    messages,
                    host=self.host,
                    temperature=0.3 if attempt == 0 else 0.1,
                    num_ctx=8192,
                )
            except ollama.OllamaError as exc:
                if attempt == 1:
                    raise TranslationError(str(exc)) from exc
                continue

            parsed = _parse_numbered(raw, expected)
            if parsed and self._looks_translated(parsed, originals):
                return parsed

        # Both attempts failed the format or language check. Rather than lose
        # the batch, fall back to translating each line on its own — slower,
        # and without cross-line context, but it reliably returns something.
        return self._translate_individually(cues, indices)

    def _translate_individually(self, cues: list[Cue], indices: list[int]) -> list[str]:
        outputs: list[str] = []
        for index in indices:
            original = cues[index].text
            try:
                raw = ollama.chat(
                    self.model,
                    [
                        {
                            "role": "system",
                            "content": (
                                f"Translate the user's subtitle line into "
                                f"{self.target.label} ({self.target.endonym}). "
                                "Convey the intended meaning naturally rather "
                                "than word-for-word. Reply with the translation "
                                "and nothing else."
                            ),
                        },
                        {"role": "user", "content": original},
                    ],
                    host=self.host,
                    temperature=0.2,
                )
                cleaned = _strip_wrapper(raw)
                outputs.append(cleaned or original)
            except ollama.OllamaError:
                # Keep the original rather than dropping the line entirely; a
                # subtitle in the wrong language beats a missing subtitle.
                outputs.append(original)
        return outputs

    # --- prompt construction -------------------------------------------------

    def _build_prompt(self, cues: list[Cue], indices: list[int]) -> str:
        first, last = indices[0], indices[-1]
        before = cues[max(0, first - CONTEXT_BEFORE):first]
        after = cues[last + 1:last + 1 + CONTEXT_AFTER]

        parts: list[str] = []
        if self.source_hint:
            parts.append(f"The spoken language is {self.source_hint}.")

        speakers = {cue.speaker for cue in cues if cue.speaker is not None}
        if len(speakers) > 1:
            parts.append(
                f"This is a conversation between {len(speakers)} speakers. Each "
                "line is tagged with who is speaking; use it to keep pronouns, "
                "names and levels of formality consistent."
            )

        if before:
            parts.append(
                "Dialogue immediately before this section, for continuity "
                "(already translated where shown):"
            )
            parts.append(
                "\n".join(
                    f"[context] {_speaker_tag(cue)}{cue.text}"
                    + (f"  ->  {cue.translated}" if cue.translated else "")
                    for cue in before
                )
            )

        parts.append(
            f"Translate these {len(indices)} lines into {self.target.label}. "
            "Reply with exactly this many numbered lines and nothing else:"
        )
        parts.append(
            "\n".join(
                f"{position}. {_speaker_tag(cues[index])}{cues[index].text}"
                for position, index in enumerate(indices, start=1)
            )
        )

        if after:
            parts.append("Dialogue that follows, for context only — do not translate:")
            parts.append(
                "\n".join(f"[context] {_speaker_tag(cue)}{cue.text}" for cue in after)
            )

        return "\n\n".join(parts)

    def _looks_translated(self, produced: list[str], originals: list[str]) -> bool:
        """Reject output that is obviously the source echoed back.

        Only meaningful for languages with a distinct script — for English to
        Spanish there is nothing to check, so we accept and move on.
        """
        if not self.target.script_ranges:
            identical = sum(
                1 for new, old in zip(produced, originals)
                if new.strip() == old.strip()
            )
            return identical < max(1, len(produced) // 2)

        scripted = sum(1 for line in produced if self.target.uses_script(line))
        return scripted >= max(1, int(len(produced) * 0.6))


def _speaker_tag(cue: Cue) -> str:
    """Speaker prefix used only inside the prompt, never in the output."""
    return "" if cue.speaker is None else f"(speaker {cue.speaker + 1}) "


_NUMBERED = re.compile(r"^\s*(\d{1,3})\s*[.．、:：)\]]\s*(.+?)\s*$")


def _parse_numbered(raw: str, expected: int) -> list[str] | None:
    """Pull `n. text` lines out of a response, tolerating chatter around them.

    Returns None when the count does not match, which triggers a retry — a
    silently short batch would desynchronise every subsequent line.
    """
    found: dict[int, str] = {}

    for line in _strip_think(raw).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("[context]"):
            continue
        match = _NUMBERED.match(stripped)
        if not match:
            continue
        number = int(match.group(1))
        if 1 <= number <= expected and number not in found:
            found[number] = _clean_line(match.group(2))

    if len(found) != expected:
        return None
    return [found[n] for n in range(1, expected + 1)]


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove reasoning blocks that some models emit despite think=False."""
    cleaned = _THINK_BLOCK.sub("", text)
    # An unterminated opening tag means the whole rest is reasoning.
    if "<think>" in cleaned.lower():
        cleaned = cleaned[: cleaned.lower().index("<think>")]
    return cleaned


def _clean_line(text: str) -> str:
    """Strip decorations models like to add around a translated line."""
    cleaned = text.strip()
    cleaned = re.sub(r"^\(speaker\s*\d+\)\s*", "", cleaned, flags=re.IGNORECASE)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'“”「」":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _strip_wrapper(raw: str) -> str:
    """Reduce a single-line response to just the translation."""
    text = _strip_think(raw).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    match = _NUMBERED.match(lines[0])
    return _clean_line(match.group(2) if match else lines[0])


def preflight(model: str, target: Language, host: str | None = None) -> str | None:
    """Cheap check that this model can produce the target language at all.

    Returns None on success, or a human-readable reason it looks unsuitable.
    This is advisory — the user is free to proceed anyway, because the check
    can only ever be a sample of one.
    """
    try:
        raw = ollama.chat(
            model,
            [
                {
                    "role": "system",
                    "content": (
                        f"Translate into {target.label} ({target.endonym}). "
                        "Reply with only the translation."
                    ),
                },
                {"role": "user", "content": "Good morning. How did you sleep?"},
            ],
            host=host,
            temperature=0.0,
            timeout=180.0,
        )
    except ollama.OllamaError as exc:
        return str(exc)

    answer = _strip_wrapper(raw)
    if not answer:
        return f"'{model}' returned an empty translation."
    if not target.uses_script(answer):
        return (
            f"'{model}' did not reply in {target.label} "
            f"(it produced: {answer[:60]!r}). It may not support this language "
            "well."
        )
    return None
