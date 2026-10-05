from datetime import datetime
from pathlib import Path

from speech_coach.metrics import SpeechMetrics, analyze, speaking_rate
from speech_coach.pauses import LONG_PAUSE_S, Pause, find_pauses, load_audio, speech_end
from speech_coach.recorder import record_until_enter
from speech_coach.transcriber import Transcriber
from speech_coach.words import drop_words_after

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

    wav = RECORDINGS / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.wav"
    record_until_enter(wav)

    samples, rate = load_audio(wav)
    words = transcriber.transcribe(wav)

    end_s = speech_end(samples, rate)
    if end_s is not None:
        kept = drop_words_after(words, end_s)
        dropped = words[len(kept):]
        if dropped:
            print(f"\n(Dropped {len(dropped)} word(s) heard after you stopped talking: "
                  f"{' '.join(w.text for w in dropped)})")
        words = kept

    transcript = " ".join(w.text for w in words)
    print_scorecard(transcript, analyze(words), find_pauses(samples, rate))

if __name__ == "__main__":
    main()