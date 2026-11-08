import random

import pytest

from speech_coach.questions import (
    DEFAULT_QUESTION,
    Question,
    find,
    length_verdict,
    load_questions,
    resolve,
)

BANK = load_questions()


def test_bank_file_is_valid() -> None:
    # Guards the TOML file itself: a typo there fails here, not mid-practice.
    assert len(BANK) >= 20
    for q in BANK:
        assert q.text.strip()
        if q.target_s is not None:
            low, high = q.target_s
            assert 0 < low < high


def test_enter_gives_the_default_question() -> None:
    q = resolve("", BANK)
    assert q.text == DEFAULT_QUESTION
    assert q.category == "opener"


def test_r_picks_from_the_bank() -> None:
    assert resolve("r", BANK, random.Random(0)) in BANK


def test_r_with_a_category_stays_in_it() -> None:
    assert resolve("r explain", BANK, random.Random(0)).category == "explain"


def test_unknown_category_falls_bank_to_any_question() -> None:
    assert resolve("r nonsense", BANK, random.Random(0)) in BANK


def test_typed_bank_question_keeps_its_target() -> None:
    q = resolve("what is mfa, and why is it important?", BANK)
    assert q.category == "explain"
    assert q.target_s == (45, 60)


def test_your_own_question_is_custom_with_no_target() -> None:
    assert resolve("What's your favorite colour?", BANK) == Question(
        "What's your favorite colour?", "custom", None
    )


def test_find_returns_none_when_missing() -> None:
    assert find(BANK, "not a real question") is None


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (50.0, "shorter than the 60-90 s target"),
        (75.0, "within the 60-90 s target"),
        (102.0, "longer than the 60-90 s target"),
    ],
)
def test_length_verdict(duration: float, expected: str) -> None:
    assert length_verdict(duration, (60, 90)) == expected


def test_no_target_means_no_verdict() -> None:
    assert length_verdict(102.0, None) == ""