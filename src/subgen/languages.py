"""Target languages offered in the GUI.

Translation is done by a general-purpose LLM rather than a fixed set of
trained language pairs, so this list is presentation, not capability — adding
an entry here is enough to support a new language. English and Simplified
Chinese lead the list because they are the two we explicitly guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str  # BCP-47-ish tag, also passed to Whisper when used as source
    label: str  # what the GUI shows
    endonym: str  # native name, included in the translation prompt
    script_ranges: tuple[tuple[int, int], ...] = ()  # for output sanity checks

    def uses_script(self, text: str) -> bool:
        """Whether `text` contains characters from this language's script.

        Used to catch a translator that echoed the source back untranslated.
        Languages written in Latin script declare no ranges and always pass,
        since "contains Latin characters" proves nothing.
        """
        if not self.script_ranges:
            return True
        return any(
            any(low <= ord(ch) <= high for low, high in self.script_ranges)
            for ch in text
        )


# CJK Unified Ideographs + Extension A + compatibility forms.
_HAN = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))
_KANA = ((0x3040, 0x309F), (0x30A0, 0x30FF))
_HANGUL = ((0xAC00, 0xD7AF), (0x1100, 0x11FF))
_CYRILLIC = ((0x0400, 0x04FF),)
_ARABIC = ((0x0600, 0x06FF),)
_DEVANAGARI = ((0x0900, 0x097F),)
_THAI = ((0x0E00, 0x0E7F),)
_HEBREW = ((0x0590, 0x05FF),)

LANGUAGES: tuple[Language, ...] = (
    Language("en", "English", "English"),
    Language("zh-Hans", "Chinese (Simplified)", "简体中文", _HAN),
    Language("zh-Hant", "Chinese (Traditional)", "繁體中文", _HAN),
    Language("ja", "Japanese", "日本語", _KANA + _HAN),
    Language("ko", "Korean", "한국어", _HANGUL),
    Language("es", "Spanish", "Español"),
    Language("fr", "French", "Français"),
    Language("de", "German", "Deutsch"),
    Language("pt", "Portuguese", "Português"),
    Language("it", "Italian", "Italiano"),
    Language("ru", "Russian", "Русский", _CYRILLIC),
    Language("ar", "Arabic", "العربية", _ARABIC),
    Language("hi", "Hindi", "हिन्दी", _DEVANAGARI),
    Language("vi", "Vietnamese", "Tiếng Việt"),
    Language("th", "Thai", "ไทย", _THAI),
    Language("id", "Indonesian", "Bahasa Indonesia"),
    Language("tr", "Turkish", "Türkçe"),
    Language("pl", "Polish", "Polski"),
    Language("nl", "Dutch", "Nederlands"),
    Language("he", "Hebrew", "עברית", _HEBREW),
)

BY_CODE = {lang.code: lang for lang in LANGUAGES}
BY_LABEL = {lang.label: lang for lang in LANGUAGES}

# Shown in the per-video dropdown above the real languages: transcribe only,
# leaving whatever was spoken as the subtitle text.
KEEP_ORIGINAL_LABEL = "Keep original (no translation)"
KEEP_ORIGINAL_CODE = "__original__"

# What Whisper should assume is being spoken. "auto" lets it detect, which is
# right almost always; the override exists for noisy or code-switched audio.
AUTODETECT_LABEL = "Auto-detect"
AUTODETECT_CODE = "auto"


def label_choices() -> list[str]:
    """Target-language dropdown values, with the passthrough option first."""
    return [KEEP_ORIGINAL_LABEL] + [lang.label for lang in LANGUAGES]


def source_choices() -> list[str]:
    """Spoken-language dropdown values, with auto-detect first."""
    return [AUTODETECT_LABEL] + [lang.label for lang in LANGUAGES]


def resolve(label: str) -> Language | None:
    """Map a dropdown label back to a Language, or None for the passthrough."""
    if label in (KEEP_ORIGINAL_LABEL, AUTODETECT_LABEL):
        return None
    return BY_LABEL.get(label)


def whisper_code(lang: Language | None) -> str | None:
    """Whisper wants a plain ISO-639-1 code and does not know about scripts.

    Simplified and Traditional Chinese are both just "zh" to the recognizer;
    the distinction only matters at translation time.
    """
    if lang is None:
        return None
    return lang.code.split("-")[0]
