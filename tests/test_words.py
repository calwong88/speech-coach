import pytest

from speech_coach.words import Word, drop_words_after, split_token, drop_loops


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


def words_from(*texts: str) -> list[Word]:
    return [Word(t, float(i), float(i + 1)) for i, t in enumerate(texts)]


def test_drop_loops_removes_your_real_loop() -> None:
    # The actual pattern from data/loop_case.wav
    words = words_from("Policy", "A", "D", *["A", "S"] * 30, "And", "Procedures")
    kept, dropped = drop_loops(words)
    assert [w.text for w in kept] == ["Policy", "A", "D", "And", "Procedures"]
    assert len(dropped) == 60


def test_drop_loops_keeps_real_repetitions() -> None:
    words = words_from("I'm", "I'm", "building", "the", "the", "the", "app")
    kept, dropped = drop_loops(words)
    assert kept == words
    assert dropped == []


def test_pattern_repeated_exactly_max_times_is_kept() -> None:
    words = words_from(*["you", "know"] * 4)
    assert drop_loops(words)[1] == []


def test_word_starting_in_fading_sound_is_dropped() -> None:
    # Real timestamps from 2026-10-05_222913.wav, where speech ends at 144.42
    words = [
        Word("management", 143.9, 144.38),
        Word("The", 144.38, 144.48),
        Word("The", 144.48, 144.84),
    ]
    assert [w.text for w in drop_words_after(words, 144.42)] == ["management"]


def test_short_real_last_word_is_kept() -> None:
    # "no" starts 0.15 s before the sound ends: enough to be real
    words = [Word("no", 10.0, 10.2)]
    assert drop_words_after(words, 10.15) == words

    