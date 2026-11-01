"""Mark fillers, repeated words and longer pauses inside the transcript. No AI."""
from speech_coach.metrics import VOCAL_FILLERS, is_repetition
from speech_coach.pauses import Pause
from speech_coach.words import Word

ANNOTATE_PAUSE_S = 1.0  # show pauses at least this long; 0.5 s is too cluttered to read
LEGEND = "Key: [uh] filler   {and} repeated word   (pause 2.4s) pause of 1 s or more"


def annotate(words: list[Word], pauses: list[Pause]) -> str:
    # Each event is (time, kind, position, text). Sorting puts everything in spoken order:
    # kind 0 (a word) comes before kind 1 (a pause) at the same time, and `position`
    # keeps words that share a timestamp, like "um" and "I", in their original order.
    events: list[tuple[float, int, int, str]] = []
    previous: str | None = None
    for position, word in enumerate(words):
        lower = word.text.lower()
        if lower in VOCAL_FILLERS:
            text = f"[{word.text}]"
        else:
            text = f"{{{word.text}}}" if previous and is_repetition(previous, lower) else word.text
            previous = lower
        events.append((word.start, 0, position, text))
    for pause in pauses:
        if pause.duration >= ANNOTATE_PAUSE_S:
            events.append((pause.start, 1, 0, f"(pause {pause.duration:.1f}s)"))
    return " ".join(text for *_, text in sorted(events))