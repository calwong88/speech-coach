"""Optional coach using the Claude API. Only redacted text ever leaves this machine."""
import os

import anthropic
from dotenv import load_dotenv

from speech_coach.coach import Coach, CoachError, CoachFeedback, Problem
from speech_coach.metrics import SpeechMetrics
from speech_coach.pauses import Pause
from speech_coach.redact import redact

API_MODEL = "claude-sonnet-5-5"  # not a Covered Model (those require 30-day retention)
MAX_TOKENS = 2048


class ApiCoach(Coach):
    def __init__(self, model: str = API_MODEL) -> None:
        load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise CoachError("ANTHROPIC_API_KEY not found: add it to .env")
        super().__init__(model=model, think=False)
        self._client = anthropic.Anthropic()

    def warm_up(self) -> None:
        """Nothing to load: the model runs on Anthropic's servers."""


    def review(
        self,
        question: str,
        transcript: str,
        m: SpeechMetrics,
        pauses: list[Pause],
        context: str = "",
    ) -> tuple[CoachFeedback, list[Problem]]:
        # Redact everything that goes into the prompt; checks then run on the redacted text.
        return super().review(redact(question), redact(transcript), m, pauses, redact(context))

    def _ask(self, messages: list[dict[str, str]]) -> CoachFeedback:
        system, conversation = messages[0]["content"], messages[1:]
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=conversation,
            output_format=CoachFeedback,
        )
        if response.stop_reason != "end_turn" or response.parsed_output is None:
            raise CoachError(f"unusable API response (stop_reason={response.stop_reason})")
        return response.parsed_output
