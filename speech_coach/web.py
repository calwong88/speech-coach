"""Practise in the browser. Run: python -m speech_coach.web, then use http://localhost:7860"""
import os
import shutil
import threading
from dataclasses import asdict, fields

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from speech_coach.auth import make_checker
from speech_coach.cli import parse_args
from speech_coach.coach import Coach, CoachError
from speech_coach.compare import comparison_table
from speech_coach.history import load_history
from speech_coach.practice import (
    RECORDINGS,
    analyze_attempt,
    coach_attempt,
    new_recording_path,
    save_attempt,
)
from speech_coach.progress import SessionPoint, session_points, trend_summary
from speech_coach.questions import DEFAULT_QUESTION, load_questions, resolve
from speech_coach.report import format_feedback, format_scorecard
from speech_coach.transcriber import Transcriber

BANK = load_questions()
CATEGORIES = ["any", *sorted({q.category for q in BANK})]
LOCAL = "Local (private)"
CLAUDE = "Claude API (name and employer redacted)"


def load_coaches(args) -> dict[str, Coach]:
    """The local coach always; Claude only if its key and SDK are available (fail closed)."""
    coaches = {LOCAL: Coach(think=args.think)}
    try:
        from speech_coach.api_coach import ApiCoach  # load the API SDK only if it can be used
        coaches[CLAUDE] = ApiCoach()
    except (ImportError, CoachError) as error:
        print(f"(Claude coach not available: {error})")
    return coaches


def build_app(transcriber: Transcriber, coaches: dict[str, Coach], default_coach: str) -> gr.Blocks:
    def random_question(category: str) -> str:
        return resolve("r" if category == "any" else f"r {category}", BANK).text

    def show_progress(category: str):
        points = session_points(load_history(), None if category == "any" else category)
        columns = [f.name for f in fields(SessionPoint)]
        table = pd.DataFrame([asdict(p) for p in points], columns=columns)
        return trend_summary(points), table, table, table

    def review(audio_path, question_text, coach_label, previous, previous_question):
        if not audio_path:
            yield "Record an answer first.", "", previous, previous_question
            return
        coach = coaches[coach_label]
        question = resolve(question_text, BANK)
        if question.text != previous_question:
            previous = None  # a new question: nothing to compare against

        RECORDINGS.mkdir(parents=True, exist_ok=True)
        wav = new_recording_path()
        shutil.copy(audio_path, wav)  # keep it with your other recordings

        # Step 1: the scorecard, shown straight away.
        attempt = analyze_attempt(question, wav, transcriber)
        a = attempt.analysis
        notes = "".join(f"({note})\n" for note in attempt.notes)
        scorecard = (
            f"Question ({question.category}): {question.text}\n\n{notes}"
            + format_scorecard(a.annotated, a.metrics, a.pauses, question.target_s)
        )
        waiting = f"{coach_label} is reviewing your answer..." if attempt.coachable else ""
        yield scorecard, waiting, previous, previous_question

        # Steps 2 and 3: coach, then save.
        shown = len(attempt.notes)
        coach_attempt(attempt, coach)
        save_attempt(attempt, coach)
        parts = [f"Coach: {coach_label}"] + [f"({note})" for note in attempt.notes[shown:]]
        if attempt.feedback:
            parts.append(format_feedback(attempt.feedback, attempt.problems))
        stats = attempt.stats
        if stats and previous:
            parts.append("=== Compared with your last attempt ===\n" + comparison_table(previous, stats))
        parts.append("(Saved to your history.)")
        yield scorecard, "\n\n".join(parts), stats or previous, question.text

    with gr.Blocks(title="Speech Coach", analytics_enabled=False) as app:
        gr.Markdown("# Speech Coach")

        with gr.Tab("Practice"):
            with gr.Row():
                category = gr.Dropdown(CATEGORIES, value="any", label="Category")
                random_button = gr.Button("Random question")
            question_box = gr.Textbox(DEFAULT_QUESTION, label="Question (pick one, or type your own)")
            coach_choice = gr.Radio(list(coaches), value=default_coach, label="Coach")
            audio = gr.Audio(sources=["microphone"], type="filepath", label="Your answer")
            submit = gr.Button("Get feedback", variant="primary")
            scorecard_box = gr.Textbox(label="Scorecard", lines=16)
            coach_box = gr.Textbox(label="Coach", lines=14)
            previous_stats = gr.State(None)
            previous_question = gr.State("")

            random_button.click(random_question, inputs=category, outputs=question_box)
            submit.click(
                review,
                inputs=[audio, question_box, coach_choice, previous_stats, previous_question],
                outputs=[scorecard_box, coach_box, previous_stats, previous_question],
            )

        with gr.Tab("Progress"):
            with gr.Row():
                progress_category = gr.Dropdown(CATEGORIES, value="any", label="Category")
                refresh_button = gr.Button("Refresh")
            summary_box = gr.Textbox(label="Your progress", lines=6)
            fillers_plot = gr.LinePlot(x="when", y="fillers_per_minute", title="Fillers per minute")
            pausing_plot = gr.LinePlot(x="when", y="pause_pct", title="Time pausing (%)")
            long_plot = gr.LinePlot(x="when", y="long_pauses", title="Long pauses (2 s or more)")

            progress_outputs = [summary_box, fillers_plot, pausing_plot, long_plot]
            refresh_button.click(show_progress, inputs=progress_category, outputs=progress_outputs)
            progress_category.change(show_progress, inputs=progress_category, outputs=progress_outputs)

        app.load(show_progress, inputs=progress_category, outputs=progress_outputs)
    return app


def main() -> None:
    args = parse_args()
    load_dotenv()
    username = os.environ.get("SPEECH_COACH_USER")
    password_hash = os.environ.get("SPEECH_COACH_PASSWORD_HASH")
    if not (username and password_hash):
        raise SystemExit("Login not configured: set SPEECH_COACH_USER and SPEECH_COACH_PASSWORD_HASH in .env")

    print("Loading speech model...")
    transcriber = Transcriber()
    coaches = load_coaches(args)
    default = CLAUDE if args.coach == "api" and CLAUDE in coaches else LOCAL
    threading.Thread(target=coaches[LOCAL].warm_up, daemon=True).start()
    build_app(transcriber, coaches, default).launch(
        server_name="127.0.0.1",  # only this computer; Tailscale forwards to it (Part B)
        auth=make_checker(username, password_hash),
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
