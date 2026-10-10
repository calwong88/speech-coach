"""Smoke test: can Python talk to the local Ollama model?"""
import time

import ollama

MODEL = "qwen3:14b"
QUESTION = (
    "In one sentence, what makes a strong answer to "
    "'Tell me about yourself' in a job interview?"
)


def main() -> None:
    start = time.perf_counter()
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": QUESTION}],
        think=False,  # qwen3 "thinks" before answer by default; skip it for speed
    )
    print(response.message.content)
    print(f"\nAnswered in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()