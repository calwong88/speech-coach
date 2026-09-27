"""Experiment: does a filler-heavy prompt make Whisper keep 'um' and 'uh'?"""
from smoke_test import WhisperModel, record   # reuse the DLL setup + recorder

FILLER_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."


def transcribe_text(model, wav, prompt=None) -> str:
    segments, _info = model.transcribe(str(wav), initial_prompt=prompt)
    return " ".join(segment.text.strip() for segment in segments)


def main() -> None:
    wav = record(15)
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    print("\nDefault:     ", transcribe_text(model, wav))
    print("\nWith prompt: ", transcribe_text(model, wav, FILLER_PROMPT))


if __name__ == "__main__":
    main()