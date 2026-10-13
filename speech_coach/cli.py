from datetime import datetime
from pathlib import Path

from speech_coach.metrics import SpeechMetrics, analyze, speaking_rate
from speech_coach.pauses import LONG_PAUSE_S, Pause, speech_end
from speech_coach.recorder import record_until_enter
from speech_coach.transcriber import Transcriber
from speech_coach.words import drop_words_after, drop_loops
from speech_coach.pipeline import analyze_recording

RECORDINGS = Path("data/recordings")


def print_scorecard(transcript: str, m: SpeechMetrics, pauses: list[Pause]) -> None:
    long_pauses = [p for p in pauses if p.duration >= LONG_PAUSE_S]
    print("\n=== Transcript ===")
    print(transcript)
    print("\n=== Scorecard ===")
    print(f"Duration:          {m.duration_s:.1f} s")
    print(f"Pace:              {m.words_per_minute:.0f} words/minute ({m.word_count} words)")
    pause_s = sum(p.duration for p in pauses)
    print(f"Time pausing:      {pause_s:.1f} s of {m.duration_s:.1f} s")
    print(f"Speaking rate:     {speaking_rate(m.word_count, m.duration_s, pause_s):.0f} words/minute (pauses removed)")
    print(f"Fillers:           {m.filler_count} ({m.fillers_per_minute:.1f}/minute) {m.filler_breakdown}")
    print(f"Possible fillers:  {m.possible_fillers or 'none'}")
    print(f"Repetitions:       {', '.join(m.repetitions) or 'none'}")
    print(f"Pauses >= 0.5 s:   {len(pauses)} (long >= {LONG_PAUSE_S:.0f} s: {len(long_pauses)})")


def main() -> None:
    print("Loading speech model...")
    transcriber = Transcriber()  # fail fast:: load the model before you speak

    wav = RECORDINGS / f"{datetime.now():%Y-%m-%d_%H%M%S}.wav"
    record_until_enter(wav)
    result = analyze_recording(wav, transcriber)

    if result.dropped_after_speech:
        late = " ".join(w.text for w in result.dropped_after_speech)
        print(f"\n(Dropped {len(result.dropped_after_speech)} word(s) heard after you stopped talking: {late})")
    if result.dropped_loops:
        loop = " ".join(w.text for w in result.dropped_loops[:6])
        print(f"(Removed a reptition loop of {len(result.dropped_loops)} words: {loop} ...)")
    
    print_scorecard(result.transcript, result.metrics, result.pauses)

if __name__ == "__main__":
    main()