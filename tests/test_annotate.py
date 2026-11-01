from speech_coach.annotate import annotate
from speech_coach.pauses import Pause
from speech_coach.words import Word


def words_from(*texts: str) -> list[Word]:
    return [Word(t, float(i), float(i + 1)) for i, t in enumerate(texts)]


def test_fillers_are_bracketed() -> None:
    assert annotate(words_from("So", "um", "I"), []) == "So [um] I"


def test_repetition_across_a_filler_is_marked() -> None:
    words = words_from("I'm", "um", "I'm", "building")
    assert annotate(words, []) == "I'm [um] {I'm} building"


def test_spelled_out_letters_are_not_marked() -> None:
    assert annotate(words_from("A", "A", "A"), []) == "A A A"


def test_pause_goes_after_the_word_before_it() -> None:
    # words start at 0, 1, 2; the silence starts at 1.2 s, during "b"
    text = annotate(words_from("a", "b", "c"), [Pause(1.2, 2.6)])
    assert text == "a b (pause 1.4s) c"


def test_short_pauses_are_not_shown() -> None:
    assert annotate(words_from("a", "b"), [Pause(0.5, 1.1)]) == "a b"


def test_words_sharing_a_timestamp_keep_their_order() -> None:
    # A split token: "um" and "I" share one timestamp, as CrisperWhsiper produces
    words = [Word("um", 0.5, 1.5), Word("I", 0.5, 1.5)]
    assert annotate(words, []) == "[um] I"


