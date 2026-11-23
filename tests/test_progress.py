from speech_coach.progress import session_points, trend_summary


def record(when, fpm=4.0, words=60, category="opener", pause_s=15.0, long_count=1, ended=None, **old):
    r = {
        "timestamp": when,
        "category": category,
        "metrics": {"word_count": words, "fillers_per_minute": fpm, "duration_s": 60.0},
        "pauses": {"count": 5, "long_count": long_count, "total_s": pause_s},
        "coach": None if ended is None else {"feedback": {"ended_clearly": ended}},
    }
    if old.get("v1"):
        del r["category"]  # schema v1 had no category
    return r


def test_too_short_answers_are_left_out() -> None:
    assert session_points([record("2026-11-01T20:00:00", words=5)]) == []


def test_old_records_without_category_are_read() -> None:
    assert session_points([record("2026-10-01T20:00:00", v1=True)])[0].category == "unknown"


def test_category_filter() -> None:
    records = [record("2026-11-01T20:00:00", category="explain"), record("2026-11-02T20:00:00")]
    assert [p.category for p in session_points(records, "explain")] == ["explain"]


def test_pause_percentage() -> None:
    assert session_points([record("2026-11-01T20:00:00", pause_s=15.0)])[0].pause_pct == 25.0


def test_points_are_sorted_oldest_first() -> None:
    records = [record("2026-11-02T20:00:00"), record("2026-11-01T20:00:00")]
    assert [p.when.day for p in session_points(records)] == [1,2]


def test_trend_compares_first_and_latest_sessions() -> None:
    records = [record(f"2026-11-0{d}T20:00:00", fpm=fpm, ended=e)
                for d, fpm, e in [(1, 8.0, False), (2, 8.0, False), (3, 2.0, True), (4, 2.0, True)]]
    summary = trend_summary(session_points(records))
    assert "8.0 -> 2.0" in summary
    assert "2 of your latest 2" in summary


def test_trend_needs_at_least_two_sessions() -> None:
    assert "Not enough" in trend_summary(session_points([record("2026-11-01T20:00:00")]))


def test_records_breaking_the_pause_invariant_are_left_out() -> None:
    assert session_points([record("2026-11-01T00:10:00", pause_s=70.0)]) == []
