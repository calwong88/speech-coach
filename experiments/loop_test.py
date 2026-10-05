"""Experiment: does turning off previous-text conditioning stop the 'A S A S' loop?"""
from pathlib import Path

from speech_coach.transcriber import Transcriber

WAV = Path("data/loop_case.wav")

CONFIGS = [
    ("Default", {}),
    ("No conditioning", {"condition_on_previous_text": False}),
]


def main() -> None:
    if not WAV.exists():
        raise SystemExit(f"{WAV} not found. Copy the loop recording there first.")

    transcriber = Transcriber()
    for label, overrides in CONFIGS:
        words = transcriber.transcribe(WAV, **overrides)
        fillers = sum(w.text.lower() in {"um", "uh"} for w in words)
        print(f"\n{label}: {len(words)} words, {fillers} fillers")
        print(" " + " ".join(w.text for w in words))

if __name__ == "__main__":
    main()