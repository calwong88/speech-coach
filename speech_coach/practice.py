"""One practice attempt, from recording to saved session. Shared by the CLI and the web app."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from speech_coach.coach import Coach, CoachFeedback, Problem
from speech_coach.compare import AttemptStats, attempt_stats
from speech_coach.history import append_session, session_record
from speech_coach.pipeline import Analysis, analyze_recording
from speech_coach.questions import Question, coaching_context
from speech_coach.transcriber import Transcriber

RECORDINGS = Path("data/recordings")
MIN_WORDS_FOR_COACHING = 15  # below this there's nothing real to coach
SPEAKER_CONTEXT = (
    "Career changer from banking (accounts processing at BMO Private Wealth), "
    "targeting entry-level IAM and cybersecurity roles."
)


@dataclass
class Attempt:
    question: Question
    wav: Path
    started: datetime
    analysis: Analysis
    feedback: CoachFeedback | None = None
    problems: list[Problem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # messages for the person, in order

    @property
    def coachable(self) -> bool:
        return self.analysis.metrics.word_count >= MIN_WORDS_FOR_COACHING

    @property
    def stats(self) -> AttemptStats | None:
        """For comparing attempts; none if the answer was too short to compare."""
        return attempt_stats(self.analysis.metrics, self.analysis.pauses) if self.coachable else None


def new_recording_path() -> Path:
    return RECORDINGS / f"{datetime.now():%Y-%m-%d_%H%M%S}.wav"


def analyze_attempt(question: Question, wav: Path, transcriber: Transcriber) -> Attempt:
    """Step 1: measure the answer. Deterministic, and no AI coach yet."""
    attempt = Attempt(question, wav, datetime.now(), analyze_recording(wav, transcriber))
    a = attempt.analysis
    if a.dropped_after_speech:
        late = " ".join(w.text for w in a.dropped_after_speech)
        attempt.notes.append(f"Dropped {len(a.dropped_after_speech)} word(s) heard after you stopped talking: {late}")
    if a.dropped_loops:
        loop = " ".join(w.text for w in a.dropped_loops[:6])
        attempt.notes.append(f"Removed a repetition loop of {len(a.dropped_loops)} words: {loop} ...")
    if not attempt.coachable:
        attempt.notes.append(
            f"Only {a.metrics.word_count} words heard: too short for coaching."
            " Check your microphone, or give a longer answer."
        )
    return attempt


def coach_attempt(attempt: Attempt, coach: Coach) -> None:
    """Step 2: ask the coach, if there's enough to coach. A failure becomes a note, never a crash."""
    if not attempt.coachable:
        return
    a = attempt.analysis
    try:
        attempt.feedback, attempt.problems = coach.review(
            attempt.question.text, a.transcript, a.metrics, a.pauses,
            coaching_context(attempt.question, SPEAKER_CONTEXT)
        )
    except Exception as error:  # noqa: BLE001 (the coach is optional: report it, keep the session)
        attempt.notes.append(f"The coach couldn't give feedback this time: {error}")


def save_attempt(attempt: Attempt, coach: Coach) -> None:
    """Step 3: append the session to the history file."""
    a = attempt.analysis
    append_session(session_record(
        when=attempt.started,
        question=attempt.question.text,
        category=attempt.question.category,
        recording=attempt.wav,
        transcript=a.transcript,
        metrics=a.metrics,
        pauses=a.pauses,
        feedback=attempt.feedback,
        problems=attempt.problems,
        think=coach.think,
        model=coach.model,
    ))
