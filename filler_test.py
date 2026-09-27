"""Experiment: does a filler-heavy prompt make Whisper keep 'um' and 'uh'?"""
from pathlib import Path

from smoke_test import WhisperModel

WAV = Path("data/smoke_test.wav") # reuse the last recording: same audio for every model
MODELS = ['large-v3-turbo', "nyrahealth/faster_CrisperWhisper"]
FILLER_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."


def transcribe_text(model, wav:Path) -> str:
    segments, _info = model.transcribe(str(wav), language="en")
    return " ".join(segment.text.strip() for segment in segments)


def main() -> None:
    if not WAV.exists():
        raise SystemExit(f"{WAV} not found. Record one first: python smoke_test.py")
    
    for name in MODELS:
        model = WhisperModel(name, device="cuda", compute_type="float16")
        print(f"\n{name}:\n {transcribe_text(model, WAV)}")
        del model # release the GPU memory before loading the next model


if __name__ == "__main__":
    main()