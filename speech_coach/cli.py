import argparse
import threading
from datetime import datetime
from pathlib import Path

from speech_coach.coach import Coach, CoachFeedback, Problem
from speech_coach.metrics import SpeechMetrics, analyze, speaking_rate
from speech_coach.pauses import LONG_PAUSE_S, Pause, speech_end
from speech_coach.pipeline import analyze_recording
from speech_coach.recorder import record_until_enter
from speech_coach.transcriber import Transcriber
from speech_coach.words import drop_loops, drop_words_after
from speech_coach.history import HISTORY_FILE, append_session, session_record

RECORDINGS = Path("data/recordings")
DEFAULT_QUESTION = "Tell me about yourself."
SPEAKER_CONTEXT = (
    "Career changer from banking (accounts processing at BMO Private Wealth), "
    "targeting entry-level IAM and cybersecurity roles."
)
MIN_WORDS_FOR_COACHING = 15  #below this there's nothing real to coach

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


def print_feedback(feedback: CoachFeedback, problems: list[Problem]) -> None:
    """Show the coach's feedback, hiding any quote that failed verification."""
    failed = {problem.field for problem in problems}
    print("\n=== Coach ===")
    if "worked_quote" not in failed:
        print(f'What worked:   "{feedback.worked_quote}"')
        print(f"               {feedback.worked_why}")
    print(f"Fix next:     {feedback.fix_next}")
    if "original_sentence" not in failed:
        print(f'You said:      "{feedback.original_sentence}"')
        print(f'Try:           "{feedback.improved_sentence}"')
    print(f"Next attempt: {feedback.next_attempt}")
    if failed:
        print("\n(Some feedback was hidden because it couldn't be verified against your transcript.)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Practise a spoken answer and get feedback.")
    parser.add_argument(
        "--think", action="store_true", help="slower, more detailed coaching"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading speech model...")
    transcriber = Transcriber()  # fail fast:: load the model before you speak

    coach = Coach(think=args.think)
    # Load the coach in the background while you think and speak
    threading.Thread(target=coach.warm_up, daemon=True).start()

    question = input(f"\nQuestion (Enter for '{DEFAULT_QUESTION}'): ").strip() or DEFAULT_QUESTION
    print(f"\nQuestion: {question}")

    started = datetime.now()
    wav = RECORDINGS / f"{started:%Y-%m-%d_%H%M%S}.wav"
    record_until_enter(wav)
    result = analyze_recording(wav, transcriber)

    if result.dropped_after_speech:
        late = " ".join(w.text for w in result.dropped_after_speech)
        print(f"\n(Dropped {len(result.dropped_after_speech)} word(s) heard after you stopped talking: {late})")
    if result.dropped_loops:
        loop = " ".join(w.text for w in result.dropped_loops[:6])
        print(f"(Removed a reptition loop of {len(result.dropped_loops)} words: {loop} ...)")

    print_scorecard(result.transcript, result.metrics, result.pauses)

    feedback: CoachFeedback | None = None
    problems: list[Problem] = []
    if result.metrics.word_count < MIN_WORDS_FOR_COACHING:
        print(f"\n(Only {result.metrics.word_count} words heard: too short for coaching. "
              "Check your microphone, or give a longer answer.)")
    else:
        print("\nCoach is reviewing your answer...")
        feedback, problems = coach.review(
            question, result.transcript, result.metrics, result.pauses, SPEAKER_CONTEXT
        )
        print_feedback(feedback, problems)

    append_session(session_record(
        when=started,
        question=question,
        recording=wav,
        transcript=result.transcript,
        metrics=result.metrics,
        pauses=result.pauses,
        feedback=feedback,
        problems=problems,
        think=coach.think,
        model=coach.model,
    ))
    print(f"\n(Saved to {HISTORY_FILE})")


if __name__ == "__main__":
    main()
