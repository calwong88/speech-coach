import pytest

from speech_coach.words import Word, drop_words_after, split_token


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("So", ["So"]),
        (",think", ["think"]),
        (",[UM]I", ["um", "I"]),
        (",[UM]like", ["um", "like"]),
        (",[UM][UH]", ["um", "uh"]),
        (",me", ["me"]),
        (",I'm", ["I'm"]),
        ("[UM]", ["um"]),
    ],
)
def test_split_token(raw: str, expected: list[str]) -> None:
    assert split_token(raw) == expected


def test_drop_words_after_removes_invented_ending() -> None:
    # Real timestampes from your hallucination recording
    words = [Word("help", 13.01, 13.29), Word("me", 13.29, 13.51), Word("The", 13.51, 13.85)]
    assert [w.text for w in drop_words_after(words, 13.5)] == ["help", "me"]