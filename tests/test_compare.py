from speech_coach.compare import AttemptStats, attempt_stats, comparison_table, verdict
from speech_coach.metrics import SpeechMetrics
from speech_coach.pauses import Pause


def test_fewer_fillers_is_better_more_is_worse() -> None:
    assert verdict("fillers_per_minute", 6.0, 3.1) == "better"
    assert verdict("fillers_per_minute", 3.1, 6.0) == "worse"
    assert verdict("long_pauses", 2, 2) == "same"


def test_duration_gets_no_verdict() -> None:
    assert verdict("duration_s", 144.0, 91.0) == ""


def test_attempt_stats_counts_long_pauses_and_repetitions() -> None:
    m = SpeechMetrics(60.0, 120, 120.0, 3, 3.0, {"um": 3}, {}, ["i", "the"])
    stats = attempt_stats(m, [Pause(1.0, 1.6), Pause(10.0, 12.5)])
    assert stats.long_pauses == 1
    assert stats.repetitions == 2


def test_table_shows_both_attempts_and_the_verdict() -> None:
    before = AttemptStats(144.0, 6.0, 4, 2, 150.0)
    after = AttemptStats(91.0, 3.1, 1, 2, 148.0)
    table = comparison_table(before, after)
    fillers_row = next(line for line in table.splitlines() if line.startswith("Fillers/min"))
    assert "6.0" in fillers_row and "3.1" in fillers_row and "better" in fillers_row

