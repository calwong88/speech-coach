"""Progress over time, computed from the history file. Pure functions: the web page only draws."""
from dataclasses import dataclass
from datetime import datetime

from speech_coach.metrics import MIN_WORDS_FOR_COACHING


@dataclass(frozen=True)
class SessionPoint:
    when: datetime
    category: str
    fillers_per_minute: float
    pause_pct: float
    long_pauses:int
    duration_s: float
    ended_clearly: bool | None  # None: no coaching, or recorded before this field existed


def session_points(records: list[dict], category: str | None = None) -> list[SessionPoint]:
    """One point per real practice answer, oldest first. Reads every history schema version."""
    points = []
    for r in records:
        m = r["metrics"]
        if m["word_count"] < MIN_WORDS_FOR_COACHING:
            continue  # microphone tests and false starts aren't practice
        record_category = r.get("category", "unknown")  # schema v1 had no category
        if category and record_category != category:
            continue
        coach = r.get("coach") or {}
        duration = m["duration_s"]
        points.append(SessionPoint(
            when=datetime.fromisoformat(r["timestamp"]),
            category=record_category,
            fillers_per_minute=m["fillers_per_minute"],
            pause_pct=round(100 * r["pauses"]["total_s"] / duration, 1) if duration else 0.0,
            long_pauses=r["pauses"]["long_count"],
            duration_s=duration,
            ended_clearly=coach.get("feedback", {}).get("ended_clearly"),
        ))
    return sorted(points, key=lambda p: p.when)


def trend_summary(points: list[SessionPoint], n: int = 5) -> str:
    """Your first sessions compared with your latest, in words."""
    if len(points) < 2:
        return "Not enough practice sessions yet. Practise a few more times to see a trend."
    k = min(n, len(points) // 2)
    first, latest = points[:k], points[-k:]

    def avg(group: list[SessionPoint], attr: str) -> float:
        return sum(getattr(p, attr) for p in group) / len(group)

    lines = [
        f"Your first {k} sessions vs your latest {k} ({len(points)} sessions in total):",
        f"Fillers per minute:  {avg(first, 'fillers_per_minute'):.1f} -> {avg(latest, 'fillers_per_minute'):.1f}",
        f"Time pausing:        {avg(first, 'pause_pct'):.0f}% -> {avg(latest, 'pause_pct'):.0f}%",
        f"Long pauses:         {avg(first, 'long_pauses'):.1f} -> {avg(latest, 'long_pauses'):.1f}",
    ]
    endings = [p.ended_clearly for p in latest if p.ended_clearly is not None]
    if endings:
        lines.append(f"Ended clearly:       {sum(endings)} of your latest {len(endings)} coached answers")
    return "\n".join(lines)

