import numpy as np
import pytest

from speech_coach.pauses import find_pauses, speech_end, speech_span

RATE = 16_000

def tone(seconds: float) -> np.ndarray:
    """Fake 'speech': a steady 220 Hz tone."""
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def test_finds_a_pause_between_speech() -> None:
    audio = np.concatenate([tone(1.0), silence(1.0), tone(1.0)])
    pauses = find_pauses(audio, RATE)
    assert len(pauses) == 1
    assert pauses[0].start == pytest.approx(1.0, abs=0.05)
    assert pauses[0].duration == pytest.approx(1.0, abs=0.05)


def test_short_gap_is_not_a_pause() -> None:
    audio = np.concatenate([tone(1.0), silence(0.3), tone(1.0)])
    assert find_pauses(audio, RATE) == []


def test_leading_and_trailing_silence_ignored() -> None:
    audio = np.concatenate([silence(1.0), tone(1.0), silence(1.5)])
    assert find_pauses(audio, RATE) == []


def test_pure_silence_has_no_pauses() -> None:
    assert find_pauses(silence(2.0), RATE) == []


def test_speech_end_ignores_trailing_silence() -> None:
    audio = np.concatenate([tone(1.0), silence(1.0)])
    assert speech_end(audio, RATE) == pytest.approx(1.0, abs=0.05)


def test_speech_end_is_none_for_pure_silence() -> None:
    assert speech_end(silence(1.0), RATE) is None


def test_speech_span_skips_leading_and_trailing_silence() -> None:
    audio = np.concatenate([silence(0.5), tone(1.0), silence(0.5)])
    start, end = speech_span(audio, RATE)
    assert start == pytest.approx(0.5, abs=0.05)
    assert end == pytest.approx(1.5, abs=0.05)


def test_speech_span_is_none_for_pure_silence() -> None:
    assert speech_span(silence(1.0), RATE) is None


def test_pauses_always_fit_inside_the_speech_span() -> None:
    audio = np.concatenate([silence(1.0), tone(1.0), silence(1.5), tone(1.0), silence(1.0)])
    start, end = speech_span(audio, RATE)
    assert sum(p.duration for p in find_pauses(audio, RATE)) <= end - start