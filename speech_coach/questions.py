"""Question bank: practice questions by category, each with an optional target length."""
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path

QUESTIONS_FILE = Path(__file__).parent / "questions.toml"
DEFAULT_QUESTION = "Tell me about yourself."


@dataclass(frozen=True)
class Question:
    text: str
    category: str                    # e.g. "opener", or "custom" for a typed question
    target_s: tuple[int, int] | None # (min, max) seconds, or None for no target


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    with path.open("rb") as f:   # tomllib requires binary mode
        data = tomllib.load(f)
    bank = []
    for category, section in data["categories"].items():
        target = tuple(section["target_s"]) if "target_s" in section else None
        bank.extend(Question(text, category, target) for text in section["questions"])
    return bank


def find(bank: list[Question], text: str) -> Question | None:
    """The bank question matching this text (ignoring case), or None."""
    wanted = text.strip().lower()
    return next((q for q in bank if q.text.lower() == wanted), None)


def resolve(raw: str, bank: list[Question], rng: random.Random | None = None) -> Question:
    """Turn what you typed at the prompt into a Question.

    ""            -> the default question
    "r"           -> a random question from any category
    "r explain"   -> a random question from that category (any category if it doesn't exist)
    anything else -> that question; a bank question keeps its category and target
    """
    rng = rng or random.Random()
    text = raw.strip() or DEFAULT_QUESTION
    lowered = text.lower()
    if lowered == "r" or lowered.startswith("r "):
        category = lowered[2:].strip()
        pool = [q for q in bank if q.category == category] or bank
        return rng.choice(pool)
    return find(bank, text) or Question(text, "custom", None)


def length_verdict(duration_s: float, target_s: tuple[int, int] | None) -> str:
    if target_s is None:
        return ""
    low, high = target_s
    if duration_s < low:
        return f"shorter than the {low}-{high} s target"
    if duration_s > high:
        return f"longer than the {low}-{high} s target"
    return f"within the {low}-{high} s target"
