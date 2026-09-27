import pytest

from speech_coach.words import split_token


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