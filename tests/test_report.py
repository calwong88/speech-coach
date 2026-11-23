from speech_coach.coach import Problem
from speech_coach.report import format_feedback, format_scorecard
from tests.test_coach import METRICS, PAUSES, make_feedback


def test_unverified_quote_is_hidden_but_advice_stays() -> None:
    text = format_feedback(make_feedback(), [Problem("worked_quote", "not in the transcript")])
    assert "What worked" not in text
    assert "Main point:" in text and "Next attempt:" in text
    assert "couldn't be verified" in text


def test_verified_feedback_shows_every_section() -> None:
    text = format_feedback(make_feedback(), [])
    for label in ("Main point:", "What worked:", "Fix next:", "You said:", "Try:", "Ending:", "Next attempt:"):
        assert label in text


def test_scorecard_shows_the_length_verdict() -> None:
    assert "within the 45-60 s target" in format_scorecard("hello", METRICS, PAUSES, (45, 60))

