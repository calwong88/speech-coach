"""Word type and CrisperWhisper token cleanup. No AI or GPU imports, so tests stay fast."""
import re
from dataclasses import dataclass

# Matches a filler tag like [UM] or [UH]. The outer ( ) makes re.split keep the tags.
_FILLER_TAG = re.compile(r"(\[(?:UM|UH)\])")

MAX_PATTERN_WORDS = 3  # longest repeating patter we check for
MAX_REPEATS = 4        # real speech doesn't repeat a short patter 5+ times in a row
MIN_WORD_SOUND_S = 0.1  # a real word needs at least this much sound after it starts


@dataclass(frozen=True)
class Word:
    text: str
    start: float # seconds from the start of the recording
    end: float


def split_token(raw:str) -> list[str]:
    """Turn one raw CrisperWhisper token into clean words, in spoken order.

    ",[UM]I"    -> ["um", "I"]
    ",[UH][UH]" -> ["uh", "uh"]
    ",me."      -> ["me"]
    """
    words: list[str] = []
    for piece in _FILLER_TAG.split(raw):
        if _FILLER_TAG.fullmatch(piece):
            words.append((piece.strip("[]")).lower())  # "[UM]" -> "um"
        else:
            cleaned = piece.strip(" ,.?!")
            if cleaned:
                words.append(cleaned)
    return words


def drop_words_after(words: list[Word], end_s:float) -> list[Word]:
    """Drop words that start after the audio has gone quiet: the model invested them."""
    return [w for w in words if w.start < end_s]


def drop_loops(words: list[Word]) -> tuple[list[Word], list[Word]]:
    """Split words into (kept, dropped) dropping repetition loops the model invented."""
    texts = [w.text.lower() for w in words]
    kept: list[Word] = []
    dropped: list[Word] = []
    i = 0
    while i < len(words):
        loop_len = _loop_length(texts, i)
        if loop_len:
            dropped.extend(words[i : i + loop_len])
            i += loop_len  # skip the whole loop
        else:
            kept.append(words[i])
            i += 1
    return kept, dropped


def _loop_length(texts: list[str], i: int) -> int:
    """How many words a loop starting at postition i covers, or 0 if there's no loop."""
    for size in range(1, MAX_PATTERN_WORDS + 1):
        pattern = texts[i : i + size]
        if len(pattern) < size:
            break  # not enough words left for a pattern this long
        repeats = 1
        while texts[i + repeats * size : i + (repeats + 1) * size] == pattern:
            repeats += 1
        if repeats > MAX_REPEATS:
            return repeats * size
    return 0


def drop_words_after(words: list[Word], end_s: float) -> list[Word]:
    """Drop words that start too late to be real: in the silence after speech ends,
    or in the last fraction of a second of fading sound."""
    return [w for w in words if w.start < end_s - MIN_WORD_SOUND_S]
