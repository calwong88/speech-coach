"""Compare two attempts at the same question. Pure arithmetic, no AI."""
from dataclasses import dataclass

from speech_coach.metrics import SpeechMetrics, speaking_rate
from speech_coach.pauses import LONG_PAUSE_S, Pause


@dataclass(frozen=True)
class AttemptStats:
    duration_s:  float
    fillers_per_minute: float
    long_pauses: int
    repetitions: int
    speaking_rate: float


def attempt_stats(m: SpeechMetrics, pauses: list[Pause]) -> AttemptStats:
    pause_s = sum(p.duration for p in pauses)
    return AttemptStats(
        duration_s=m.duration_s,
        fillers_per_minute=m.fillers_per_minute,
        long_pauses=sum(p.duration >= LONG_PAUSE_S for p in pauses),
        repetitions=len(m.repetitions),
        speaking_rate=speaking_rate(m.word_count, m.duration_s, pause_s),
    )


# For these, lower is better. Duration and speasking rate have no "better" direction
# on their own: whether shorter or faster is good depends on the question.
LOWER_IS_BETTER = {"fillers_per_minute", "long_pauses", "repetitions"}

# (label, field, format) for each row of the table
ROWS = [
    ("Fillers/min", "fillers_per_minute", "{:.1f}"),
    ("Long pauses", "long_pauses", "{}"),
    ("Repetitions", "repetitions", "{}"),
    ("Duration", "duration_s", "{:.0f} s"),
    ("Speaking rate", "speaking_rate", "{:.0f} wpm"),
]

def verdict(field: str, before: float, after: float) -> str:
    if field not in LOWER_IS_BETTER:
        return ""
    if after < before:
        return "better"
    if after > before:
        return "worse"
    return "same"


def comparison_table(before: AttemptStats, after: AttemptStats) -> str:
    lines = [f"{'':16}{'Before':>10}{'Now':>10}"]
    for label, field, fmt in ROWS:
        b, a = getattr(before, field), getattr(after, field)
        lines.append(f"{label:16}{fmt.format(b):>10}{fmt.format(a):>10} {verdict(field, b, a)}")
    return "\n".join(lines)
