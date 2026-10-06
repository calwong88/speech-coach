"""Word type and CrisperWhisper token cleanup. No AI or GPU imports, so tests stay fast."""
import re
from dataclasses import dataclass

# Matches a filler tag like [UM] or [UH]. The outer ( ) makes re.split keep the tags.
_FILLER_TAG = re.compile(r"(\[(?:UM|UH)\])")


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


