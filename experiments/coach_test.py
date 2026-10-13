"""Experiment: real coach feedback on saved recordings, thinking off vs on."""
import subprocess
import time
from pathlib import Path

from speech_coach.coach import Coach, check_quotes
from speech_coach.pipeline import analyze_recording
from speech_coach.transcriber import Transcriber

QUESTION = "Tell me about yourself."
RECORDINGS = sorted(Path("data/recordings").glob("*.wav"))[-3:]  # newest three


def gpu_memory_used() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nohead"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    transcriber = Transcriber()
    analyses = [analyze_recording(wav, transcriber) for wav in RECORDINGS]

    for think in (False, True):
        coach = Coach(think=think)
        coach.warm_up()
        print(f"\n######## Thinking {'ON' if think else 'OFF'} | GPU memory: {gpu_memory_used()}")

        for wav, analysis in zip(RECORDINGS, analyses):
            start = time.perf_counter()
            try:
                feedback = coach.review(QUESTION, analysis.transcript, analysis.metrics, analysis.pauses)
            except Exception as error:  # an experiment should report failures, not stop on them
                print(f"\n--- {wav.name}: FAILED: {error}")
                continue
            seconds = time.perf_counter() - start
            problems = check_quotes(feedback, analysis.transcript)
            verdict = "PASS" if not problems else f"FAIL {problems}"
            print(f"\n--- {wav.name}: {seconds:.1f}s | quote check: {verdict}")
            print(feedback.model_dump_json(indent=2))


if __name__ == "__main__":
    main()