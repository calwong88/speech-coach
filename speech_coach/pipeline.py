"""Turn a recording into an analysis: the steps shared by the CLI, experiments and later the web app."""
from dataclasses import dataclass
from pathlib import Path

from speech_coach.annotate import annotate
from speech_coach.corrections import apply_corrections
from speech_coach.metrics import SpeechMetrics, analyze
from speech_coach.pauses import Pause, find_pauses, load_audio, speech_span
from speech_coach.transcriber import Transcriber
from speech_coach.words import Word, drop_loops, drop_words_after


@dataclass(frozen=True)
class Analysis:
    transcript: str             # corrected text, for display and the coach
    annotated: str              # corrected text with fillers, repeats and pauses marked
    metrics: SpeechMetrics      # computer from the original words
    pauses: list[Pause]
    dropped_after_speech: list[Word]
    dropped_loops: list[Word]


def analyze_recording(wav: Path, transcriber: Transcriber) -> Analysis:
    samples, rate = load_audio(wav)
    words = transcriber.transcribe(wav)

    span = speech_span(samples, rate)
    end_s = span[1] if span else None
    kept = drop_words_after(words, end_s) if end_s is not None else words
    late = words[len(kept):]
    kept, looped = drop_loops(kept)

    pauses = find_pauses(samples, rate)
    duration_s = span[1] - span[0] if span else None
    return Analysis(
        transcript=apply_corrections(" ".join(w.text for w in kept)),
        annotated=apply_corrections(annotate(kept, pauses)),
        metrics=analyze(kept, duration_s),
        pauses=pauses,
        dropped_after_speech=late,
        dropped_loops=looped,
    )
