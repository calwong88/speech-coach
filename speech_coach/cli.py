import argparse
import threading

from speech_coach.api_coach import ApiCoach
from speech_coach.coach import Coach
from speech_coach.compare import AttemptStats, comparison_table
from speech_coach.history import HISTORY_FILE
from speech_coach.practice import (
    analyze_attempt,
    coach_attempt,
    new_recording_path,
    save_attempt,
)
from speech_coach.questions import Question, load_questions, resolve
from speech_coach.recorder import record_until_enter
from speech_coach.report import format_feedback, format_scorecard
from speech_coach.transcriber import Transcriber


def ask_question(bank: list[Question]) -> Question:
    categories = ", ".join(sorted({q.category for q in bank}))
    print("\nEnter = 'Tell me about yourself.' | r = random | "
          f"r <category> ({categories}) | or type your own question.")
    question = resolve(input("> "), bank)
    print(f"\nQuestion ({question.category}): {question.text}")
    if question.target_s:
        low, high = question.target_s
        print(f"Target length: {low}-{high} s")
    return question


def practice_once(question: Question, transcriber: Transcriber, coach: Coach) -> AttemptStats | None:
    wav = new_recording_path()
    record_until_enter(wav)

    attempt = analyze_attempt(question, wav, transcriber)
    for note in attempt.notes:
        print(f"\n({note})")
    a = attempt.analysis
    print("\n" + format_scorecard(a.annotated, a.metrics, a.pauses, question.target_s))

    if attempt.coachable:
        print("\nCoach is reviewing your answer...")
        shown = len(attempt.notes)
        coach_attempt(attempt, coach)
        for note in attempt.notes[shown:]:  # only notes added by the coach step
            print(f"\n({note})")
        if attempt.feedback:
            print("\n" + format_feedback(attempt.feedback, attempt.problems))

    save_attempt(attempt, coach)
    print(f"\n(saved to {HISTORY_FILE})")
    return attempt.stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Practise a spoken answer and get feedback."
    )
    parser.add_argument(
        "--think", action="store_true", help="slower, more detailed coaching"
    )
    parser.add_argument(
        "--coach",
        choices=["local", "api"],
        default="local",
        help="local (default, private) or api (Claude API; name and employer redacted)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank = load_questions()

    print("Loading speech model...")
    transcriber = Transcriber()  # fail fast: load the model before you speak

    if args.coach == "api":
        coach = ApiCoach()
        print("Coach: Claude API (only redacted text leaves this machine)")
    else:
        coach = Coach(think=args.think)
    # Load the coach in the background while you think and speak
    threading.Thread(target=coach.warm_up, daemon=True).start()

    try:
        question = ask_question(bank)
        previous: AttemptStats | None = None
        attempt = 1
        while True:
            print(f"\n--- Attempt {attempt} ---")
            current = practice_once(question, transcriber, coach)
            if current and previous:
                print("\n=== Compared with your last attempt ===")
                print(comparison_table(previous, current))
            previous = current or previous

            choice = (
                input("\nEnter = same question again | n = new question | q = quit: ")
                .strip()
                .lower()
            )
            if choice == "q":
                break
            if choice == "n":
                question = ask_question(bank)
                previous, attempt = None, 1
            else:
                attempt += 1
    except KeyboardInterrupt:
        pass  # Ctrl + C quits cleanly too
    print("\nSession over. Good practice.")


if __name__ == "__main__":
    main()
