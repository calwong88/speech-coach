"""Practice history: one JSON line per session, appended to data/history.jsonl."""
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from speech_coach.coach import CoachFeedback, Problem
from speech_coach.metrics import SpeechMetrics
from speech_coach.pauses import LONG_PAUSE_S, Pause

HISTORY_FILE = Path("data/history.jsonl")
SCHEMA_VERSION = 1


def session_record(
        *,
        when: datetime,
        question: str,
        recording: Path,
        transcript: str,
        metrics: SpeechMetrics,
        pauses: list[Pause],
        feedback: CoachFeedback | None,
        problems: list[Problem],
        think: bool,
        model: str,
) -> dict:
    """Everything worth keeping about one practice session, as plain JSON-ready data."""
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": when.isoformat(timespec="seconds"),
        "question": question,
        "recording": recording.name,
        "transcript": transcript,
        "metrics": asdict(metrics),
        "pauses": {
            "count": len(pauses),
            "long_count": sum(p.duration >= LONG_PAUSE_S for p in pauses),
            "total_s": round(sum(p.duration for p in pauses), 2),
        },
        "coach": None if feedback is None else {
            "model": model,
            "think": think,
            "feedback": feedback.model_dump(),
            "unverified_fields": sorted({p.field for p in problems}),
        },
    }


def append_session(record: dict, path: Path = HISTORY_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_history(path: Path = HISTORY_FILE) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

