from speech_coach.metrics import analyze, speaking_rate
from speech_coach.words import Word


def make_words(*texts: str, seconds_each: float = 0.5) -> list[Word]:
    """Build a fake transcript: each word lasts 'seconds_each', back to back."""
    return [
        Word(text=t, start=i * seconds_each, end=(i + 1) * seconds_each)
        for i, t in enumerate(texts)
    ]


def test_empty_transcript_gives_zero() -> None:
    m = analyze([])
    assert m.word_count == 0
    assert m.words_per_minute == 0.0


def test_vocal_fillers_counted_and_exclued_from_word_count() -> None:
    m = analyze(make_words("So", "um", "I", "uh", "think", "um"))
    assert m.filler_breakdown == {"um": 2, "uh": 1}
    assert m.filler_count == 3
    assert m.word_count == 3

def test_words_per_minute() -> None:
    # 60 words x 0.5 s = 30 s = half a minute -> 120 wpm
    m = analyze(make_words(*[f"w{i}" for i in range(60)]))
    assert m.words_per_minute == 120.0


def test_fillers_per_minute() -> None:
    # 58 words + 2 fillers = 60 tokens x 0.5 s = 30 s -> 2 fillers / 0.5 min = 4.0
    texts = [f"w{i}" for i in range(58)] + ["um", "uh"]
    m = analyze(make_words(*texts))
    assert m.fillers_per_minute == 4.0


def test_discourse_markers_are_possible_fillers_not_scored() -> None:
    m = analyze(make_words("I", "like", "you", "know", "basically"))
    assert m.possible_fillers == {"like": 1, "you know": 1, "basically": 1}
    assert m.filler_count == 0


def test_repetition_detected_across_a_filler() -> None:
    m = analyze(make_words("I'm", "um", "I'm", "building", "this"))
    assert m.repetitions == ["i'm"]


def test_repetition_is_case_insensitive() -> None:
    m = analyze(make_words("The", "the", "plan"))
    assert m.repetitions == ["the"]


def test_speaking_rate_exluces_pause_time() -> None:
    # 60 words over 60 s, 30 s of it pausing -> 60 words in 30 s of talking = 120 wpm
    assert speaking_rate(60, 60.0, 30.0) == 120.0


def test_speaking_rate_when_all_pauses_is_zero() -> None:
    assert speaking_rate(10, 5.0, 5.0) == 0.0


def test_spelled_out_letters_are_not_repetitions() -> None:
    m = analyze(make_words("CompTIA", "A", "A", "A", "plus"))
    assert m.repetitions == []


def test_repeated_i_still_counts() -> None:
    m = analyze(make_words("I", "I", "think"))
    assert m.repetitions == ["i"]

