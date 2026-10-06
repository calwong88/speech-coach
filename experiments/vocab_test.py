"""Experiment: can a vocabulary hint fix my terms without losing fillers?"""
from pathlib import Path

from speech_coach.transcriber import Transcriber

WAV = Path("data/loop_case.wav")

VOCAB = (
    "Calvin Wong, BMO, Private Wealth, CompTIA, Security+, SecAI+, SC-300, "
    "cybersecurity, IAM, Okta, Entra ID, AML, KYC, FATCA"
)

CONFIGS = [
    ("Baseline", {}),
    ("Hotwords", {"hotwords": VOCAB}),
    ("Initial prompt", {"initial_prompt": f"Interview practice. Terms: {VOCAB}."}),
]

CHECK_TERMS = ["bmo", "sc-300", "comptia", "security+", "secai+"]


def main() -> None:
    if not WAV.exists():
        raise SystemExit(f"{WAV} not found.")

    transcriber = Transcriber()
    for label, overrides in CONFIGS:
        words = transcriber.transcribe(WAV, **overrides)
        text = " ".join(w.text for w in words)
        fillers = sum(w.text.lower() in {"um", "uh"} for w in words)
        found = [term for term in CHECK_TERMS if term in text.lower()]
        print(f"\n{label}: {len(words)} words, {fillers} fillers, terms found: {found}")
        print(" " + text)


if __name__ == "__main__":
    main()