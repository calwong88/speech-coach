"""Experiment: compare coach models on the same saved sessions, rated blind.

Step 1: python -m experiments.coach_compare generate   (runs every coach, saves results)
Step 2: python -m experiments.coach_compare rate       (you rate blind, then models are revealed)
"""
import json
import random
import sys
import time
from pathlib import Path

from speech_coach.api_coach import ApiCoach
from speech_coach.cli import MIN_WORDS_FOR_COACHING, SPEAKER_CONTEXT, print_feedback
from speech_coach.coach import Coach, CoachFeedback, Problem
from speech_coach.history import load_history
from speech_coach.pipeline import analyze_recording
from speech_coach.redact import redact
from speech_coach.transcriber import Transcriber

RECORDINGS = Path("data/recordings")
RESULTS = Path("data/comparison/results.json")
RATED = Path("data/comparison/rated.json")
N_SESSIONS = 5


def make_coaches() -> dict[str, Coach]:
    return {
        "qwen:14b": Coach(model="qwen3:14b"),
        "qwen3.5:9b": Coach(model="qwen3.5:9b-q4_K_M"),
        "claude-sonnet": ApiCoach(),
    }


def pick_sessions() -> list[tuple[Path, str]]:
    """The newest saved sessions that are long enough to coach, one per recoridng."""
    seen: set[str] = set()
    picked = []
    for record in reversed(load_history()):
        wav = RECORDINGS / record["recording"]
        too_short = record["metrics"]["word_count"] < MIN_WORDS_FOR_COACHING
        if record["recording"] in seen or not wav.exists() or too_short:
            continue
        seen.add(record["recording"])
        picked.append((wav, record["question"]))
        if len(picked) == N_SESSIONS:
            break
    return picked


def generate() -> None:
    transcriber = Transcriber()
    sessions = [(wav, q, analyze_recording(wav, transcriber)) for wav, q in pick_sessions()]
    context = redact(SPEAKER_CONTEXT)

    results = []
    for name, coach in make_coaches().items():
        coach.warm_up() # load before timing, so the times measure reviews, not loading
        calls = 0
        original_ask = coach._ask

        def counting_ask(messages, ask=original_ask):
            nonlocal calls
            calls += 1
            return ask(messages)

        coach._ask = counting_ask # count attempts: 2 means the retry was needed

        for wav, question, analysis in sessions:
            calls = 0
            transcript = redact(analysis.transcript)
            start = time.perf_counter()
            try:
                feedback, problems = coach.review(
                    redact(question), transcript, analysis.metrics, analysis.pauses, context
                )
                error = None
            except Exception as exc:  # noqa: BLE001 (recording the failure and keep going)
                feedback, problems, error = None, [], str(exc)
            result = {
                "model": name,
                "recording": wav.name,
                "question": redact(question),
                "transcript": transcript,
                "seconds": round(time.perf_counter() - start, 1),
                "attempts": calls,
                "unverified": sorted({p.field for p in problems}),
                "error": error,
                "feedback": feedback.model_dump() if feedback else None,
            }
            results.append(result)
            print(f"{name:14} {wav.name}: {result['seconds']}s, attempts={calls}, "
                  f"unverified={result['unverified']}, error={error}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} reviews to {RESULTS}. Next: python -m experiments.coach_compare rate")


def ask_rating(label: str) -> int:
    while True:
        answer = input(f"\nRate feedback {label} for usefulness, 1 (useless) to 5 (very useful): ").strip()
        if answer in {"1", "2", "3", "4", "5"}:
            return int(answer)
        print("Please type a number form 1 to 5.")


def rate() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    by_recording: dict[str, list[dict]] = {}
    for r in results:
        if r["feedback"]:
            by_recording.setdefault(r["recording"], []).append(r)

    for n, group in enumerate(by_recording.values(), 1):
        random.shuffle(group)  # blind: the order says nothing about the model
        print(f"\n\n################ Answer {n} of {len(by_recording)}")
        print(f"Question: {group[0]['question']}")
        print(f"Transcript: {group[0]['transcript']}")
        for label, r in zip("ABC", group):
            print(f"\n=============== Feedback {label} ===============")
            problems = [Problem(field, "unverified") for field in r["unverified"]]
            print_feedback(CoachFeedback(**r["feedback"]), problems)
            r["rating"] = ask_rating(label)

    RATED.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summarize(results)


def summarize(results: list[dict]) -> None:
    print("\n\n######## Results: the models are revealed")
    print(f"{'Model':16}{'Rating':>8}{'Seconds':>9}{'Retries':>9}{'Unverified':>12}{'Errors':>8}")
    for model in dict.fromkeys(r["model"] for r in results):
        rows = [r for r in results if r["model"] == model]
        rated = [r["rating"] for r in rows if "rating" in r]
        rating = sum(rated) / len(rated) if rated else 0
        seconds = sum(r["seconds"] for r in rows) / len(rows)
        retries = sum(r["attempts"] > 1 for r in rows)
        unverified = sum(bool(r["unverified"]) for r in rows)
        errors = sum(r["error"] is not None for r in rows)
        print(f"{model:16}{rating:>8.1f}{seconds:>9.1f}{retries:>9}{unverified:>12}{errors:>8}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "generate":
        generate()
    elif command == "rate":
        rate()
    else:
        raise SystemExit("Usage: python -m experiments.coach_compare [generate|rate]")
