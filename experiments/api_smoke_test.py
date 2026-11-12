"""Smoke test: can Python reach the Claude API with the key from .env?"""
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()  # copies ANTHROPIC_API_KEY from .env into the environment
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

start = time.perf_counter()
response = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=100,
    messages=[{"role": "user", "content": "In one sentence, what makes a strong answer to 'Tell me about yourself'?"}],
)
print(response.content[0].text)
print(
    f"\n{time.perf_counter() - start:.1f}s | stop_reason={response.stop_reason} | "
    f"tokens in/out: {response.usage.input_tokens}/{response.usage.output_tokens}"
)
