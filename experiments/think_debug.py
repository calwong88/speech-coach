"""Why did --think return an empty response? Replays the newest recording."""

import time
from pathlib import Path

import ollama

from speech_coach.coach import (
    FEEDBACK_SCHEMA,
    NUM_CTX,
    NUM_PREDICT,
    build_messages,
)
from speech_coach.pipeline import analyze_recording
from speech_coach.transcriber import Transcriber

QUESTION = "What is an access review, and why do companies do them?"
MODELS = ["qwen3:14b", "qwen3.5:9b-q4_K_M"]


def main() -> None:
    wav = max(Path("data/recordings").glob("*.wav"))
    analysis = analyze_recording(wav, Transcriber())
    messages = build_messages(QUESTION, analysis.transcript, analysis.metrics, analysis.pauses)
    print("recording", wav.name)

    for model in MODELS:
        start = time.perf_counter()
        response = ollama.chat(
            model=model,
            messages=messages,
            format=FEEDBACK_SCHEMA,
            think=True,
            options={"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        )
        seconds = time.perf_counter() - start
        print(
            f"{model}: {seconds:.1f}s | {response.done_reason} | "
            f"output {response.eval_count} tokens | content {len(response.message.content or '')} chars"
        )


if __name__ == "__main__":
    main()
