import json

import pytest

from speech_coach.coach import CoachFeedback
from speech_coach.metrics import SpeechMetrics

pytest.importorskip("anthropic")  # skip this file where the API SDK can't load (e.g. app control)

from speech_coach.api_coach import ApiCoach  # must come after the skip check

METRICS = SpeechMetrics(30.0, 60, 120.0, 0, 0.0, {}, {}, [])
FEEDBACK = CoachFeedback(
    main_point="m", worked_quote="I work at [EMPLOYER]", worked_why="b", fix_next="c",
    original_sentence="Hi I'm [NAME]", improved_sentence="e",
    ended_clearly=True, ending_note="g", next_attempt="f",
)


class SpyCoach(ApiCoach):
    """Records what would be sent, instead of calling the API."""

    def __init__(self) -> None:  # deliberately skips ApiCoach.__init__: no key, no client
        self.sent: list = []

    def _ask(self, messages):
        self.sent.append(messages)
        return FEEDBACK


def test_name_and_employer_never_leave_the_machine() -> None:
    coach = SpyCoach()
    coach.review(
        "Tell me about yourself.",
        "Hi I'm Calvin Wong and I work at BMO",
        METRICS,
        [],
        context="Accounts processing at BMO Private Wealth",
    )
    everything_sent = json.dumps(coach.sent)
    assert "Calvin" not in everything_sent
    assert "BMO" not in everything_sent
