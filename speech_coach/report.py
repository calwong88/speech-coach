"""Turn results into display text. Used by the CLI (printed) and the web app (shown in a page)."""
from speech_coach.annotate import LEGEND
from speech_coach.coach import CoachFeedback, Problem
from speech_coach.metrics import SpeechMetrics, speaking_rate
from speech_coach.pauses import LONG_PAUSE_S, MIN_PAUSE_S, Pause
from speech_coach.questions import length_verdict


def format_scorecard(
        transcript: str, m: SpeechMetrics, pauses: list[Pause], target_s: tuple[int, int] | None = None
) -> str:
    long_pauses = sum(p.duration >= LONG_PAUSE_S for p in pauses)
    pause_s = sum(p.duration for p in pauses)
    pause_pct = pause_s / m.duration_s if m.duration_s else 0
    verdict = length_verdict(m.duration_s, target_s)
    lines = [
        "=== Transcript ===",
        transcript,
        LEGEND,
        "",
        "=== Scorecard ===",
        f"Duration:         {m.duration_s:.1f} s" + (f" ({verdict}" if verdict else ""),
        f"Pace:             {m.words_per_minute:.0f} words/win ({m.word_count} words)",
        f"Time pausing:     {pause_s:.1f} s of {m.duration_s:.1f} s ({pause_pct:.0%})",
        f"Speaking rate:    {speaking_rate(m.word_count, m.duration_s, pause_s):.0f} words/min (pauses removed)",
        f"Fillers:          {m.filler_count} ({m.fillers_per_minute:.1f}/min) {m.filler_breakdown}",
        f"Possible fillers: {m.possible_fillers or 'none'}",
        f"Repetitions:      {', '.join(m.repetitions) or 'none'}",
        f"Pauses >= {MIN_PAUSE_S} s: {len(pauses)} (long >= {LONG_PAUSE_S:.0f} s: {long_pauses})",
    ]
    return "\n".join(lines)


def format_feedback(feedback: CoachFeedback, problems: list[Problem]) -> str:
    """The coach's feedback, hiding any quote that failed verification."""
    failed = {problem.field for problem in problems}
    lines = ["=== Coach ===", f"Main point:   {feedback.main_point}"]
    if "worked_quote" not in failed:
        lines += [f'What worked:  "{feedback.worked_quote}"', f"              {feedback.worked_why}"]
    lines.append(f"Fix next:     {feedback.fix_next}")
    if "original_sentence" not in failed:
        lines += [
            f'You said:     "{feedback.original_sentence}"',
            f'Try:          "{feedback.improved_sentence}"',
        ]
    ending = "finished clearly" if feedback.ended_clearly else "trailed off"
    lines += [f"Ending:       {ending}. {feedback.ending_note}", f"Next attempt: {feedback.next_attempt}"]
    if failed:
        lines += ["", "(Some feedback was hidden because it couldn't be verified against your transcript.)"]
    return "\n".join(lines)

