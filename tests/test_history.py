from datetime import datetime
from pathlib import Path

from speech_coach.coach import CoachFeedback, Problem
from speech_coach.history import append_session, load_history, session_record
from speech_coach.metrics import SpeechMetrics
from speech_coach.pauses import Pause

METRICS = SpeechMetrics(30.0, 60, 120.0, 2, 4.0, {"um": 2}, {}, [])
FEEDBACK = CoachFeedback(
    worked_quote="a", worked_why="b", fix_next="c",
    original_sentence="d", improved_sentence="e", next_attempt="f",
)


def make_record(**changes) -> dict:
    fields = dict(
        when=datetime(2026, 10, 19, 21, 30),
        question="Why IAM?",
        recording=Path("data/recordings/x.wav"),
        transcript="hello",
        metrics=METRICS,
        pauses=[Pause(1.0, 3.5)],
        feedback=FEEDBACK,
        problems=[],
        think=False,
        model="qwen3:14b",
    )
    return session_record(**(fields | changes))


def test_record_summarises_pauses_and_keeps_metrics() -> None:
    record = make_record()
    assert record["pauses"] == {"count": 1, "long_count": 1, "total_s": 2.5}
    assert record["metrics"]["filler_count"] == 2
    assert record["recording"] == "x.wav"


def test_unverified_fields_are_recorded() -> None:
    record = make_record(problems=[Problem("worked_quote", "not in the transcript")])
    assert record["coach"]["unverified_fields"] == ["worked_quote"]


def test_session_without_coaching_has_no_coach_section() -> None:
    assert make_record(feedback=None)["coach"] is None


def test_sessions_round_trip_through_the_files(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    append_session(make_record(question="Q1"), path)
    append_session(make_record(question="Q2"), path)
    assert [r["question"] for r in load_history(path)] == ["Q1", "Q2"]


def test_missing_history_file_is_empty(tmp_path: Path) -> None:
    assert load_history(tmp_path / "missing.jsonl") == []