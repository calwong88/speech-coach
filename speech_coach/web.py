"""Practise in the browser. Run: python -m speech_coach.web, then use http://localhost:7860"""
import shutil
import threading

import gradio as gr

from speech_coach.cli import make_coach, parse_args
from speech_coach.coach import Coach
from speech_coach.compare import comparison_table
from speech_coach.practice import (
    RECORDINGS,
    analyze_attempt,
    coach_attempt,
    new_recording_path,
    save_attempt,
)
from speech_coach.questions import DEFAULT_QUESTION, load_questions, resolve
from speech_coach.report import format_feedback, format_scorecard
from speech_coach.transcriber import Transcriber

BANK = load_questions()
CATEGORIES = ["any", *sorted({q.category for q in BANK})]


def build_app(transcriber: Transcriber, coach: Coach) -> gr.Blocks:
    def random_question(category: str) -> str:
        return resolve("r" if category == "any" else f"r {category}", BANK).text

    def review(audio_path, question_text, previous, previous_question):
        if not audio_path:
            yield "Record an answer first.", "", previous, previous_question
            return
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
        waiting = "Coach is reviewing your answer..." if attempt.coachable else ""
        yield scorecard, waiting, previous, previous_question

        # Steps 2 and 3: coach, then save.
        shown = len(attempt.notes)
        coach_attempt(attempt, coach)
        save_attempt(attempt, coach)
        parts = [f"({note})" for note in attempt.notes[shown:]]
        if attempt.feedback:
            parts.append(format_feedback(attempt.feedback, attempt.problems))
        stats = attempt.stats
        if stats and previous:
            parts.append("=== Compared with your last attempt ===\n" + comparison_table(previous, stats))
        parts.append("(Saved to your history.)")
        yield scorecard, "\n\n".join(parts), stats or previous, question.text

    with gr.Blocks(title="Speech Coach", analytics_enabled=False) as app:
        gr.Markdown("# Speech Coach")
        with gr.Row():
            category = gr.Dropdown(CATEGORIES, value="any", label="Category")
            random_button = gr.Button("Random question")
        question_box = gr.Textbox(DEFAULT_QUESTION, label="Question (pick one, or type your own)")
        audio = gr.Audio(sources=["microphone"], type="filepath", label="Your answer")
        submit = gr.Button("Get feedback", variant="primary")
        scorecard_box = gr.Textbox(label="Scorecard", lines=16)
        coach_box = gr.Textbox(label="Coach", lines=14)
        previous_stats = gr.State(None)
        previous_question = gr.State("")

        random_button.click(random_question, inputs=category, outputs=question_box)
        submit.click(
            review,
            inputs=[audio, question_box, previous_stats, previous_question],
            outputs=[scorecard_box, coach_box, previous_stats, previous_question],
        )
    return app


def main() -> None:
    args = parse_args()
    print("Loading speech model...")
    transcriber = Transcriber()
    coach = make_coach(args)
    threading.Thread(target=coach.warm_up, daemon=True).start()
    build_app(transcriber, coach).launch(server_name="127.0.0.1", inbrowser=True)


if __name__ == "__main__":
    main()
