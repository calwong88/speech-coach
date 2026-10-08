"""Metrics engine: deterministic measurements from a list of timed words.

No AI here. The same input always gives the same numbers, so results can be
tracked over time and tested exactly.
"""
from collections import Counter
from dataclasses import dataclass

from speech_coach.words import Word

VOCAL_FILLERS = {"um", "uh"}

# Only *possible* fillers: "like" in "I'd like to" is a real word, so these are
# reported separately and never counted in the filler score.
DISCOURSE_MARKERS: list[tuple[str, ...]] = [
    ("you", "know"),
    ("i", "mean"),
    ("like",),
    ("basically",),
    ("actually",),
    ("literally",),
]


@dataclass(frozen=True)
class SpeechMetrics:
    duration_s: float                 # first word start -> last word end
    word_count: int                   # spoken words, vocal fillers excluded
    words_per_minute: float
    filler_count: int
    fillers_per_minute: float
    filler_breakdown: dict[str, int]  # e.g. {"um": 3, "uh": 2}
    possible_fillers: dict[str, int]  # e.g. {"like": 1, "you know": 1
    repetitions: list[str]            # e.g. ["i'm"] for "I'm, I'm"


def analyze(words: list[Word]) -> SpeechMetrics:
    if not words:
        return SpeechMetrics(0.0, 0, 0.0, 0, 0.0, {}, {}, [])

    tokens = [w.text.lower() for w in words]
    spoken = [t for t in tokens if t not in VOCAL_FILLERS]
    fillers = Counter(t for t in tokens if t in VOCAL_FILLERS)
    filler_count = sum(fillers.values())

    duration_s = words[-1].end - words[0].start
    minutes = duration_s / 60

    return SpeechMetrics(
        duration_s=round(duration_s, 2),
        word_count=len(spoken),
        words_per_minute=_per_minute(len(spoken), minutes),
        filler_count=filler_count,
        fillers_per_minute=_per_minute(filler_count, minutes),
        filler_breakdown=dict(fillers),
        possible_fillers=dict(_count_phrases(spoken, DISCOURSE_MARKERS)),
        repetitions=_repetitions(spoken),
    )

def _per_minute(count: int, minutes: float) -> float:
    return round(count / minutes, 1) if minutes > 0 else 0.0


def _count_phrases(tokens: list[str], phrases: list[tuple[str, ... ]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for i in range(len(tokens)):
        for phrase in phrases:
            if tuple(tokens[i : i + len(phrase)]) == phrase:
                counts[" ".join(phrase)] += 1
    return counts


def _repetitions(spoken: list[str]) -> list[str]:
    # Pair each word with the next one: [a, b, c] -> (a,b), (b, c)
    return [a for a, b in zip(spoken, spoken[1:]) if a == b]


def speaking_rate(word_count: int, duration_s: float, pause_s: float) -> float:
    """Words per minute while actually talking, with pause time removed."""
    talking_s = duration_s - pause_s
    return _per_minute(word_count, talking_s / 60) if talking_s > 0 else 0.0


def _repetitions(spoken: list[str]) -> list[str]:
    # Pair each word with the next one: [a, b, c] -> (a, b), (b, c).
    # Repeated single letters are usually a spelled-out acronym ("A A A"), so they 
    # don't count, except "i": "I, I think" is a real stumble.
    return[
        a for a, b in zip(spoken, spoken[1:])
        if a == b and (len(a) > 1 or a == "i")
    ]