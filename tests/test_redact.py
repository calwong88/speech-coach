from speech_coach.redact import redact


def test_name_and_employer_are_replaced() -> None:
    text = "Hi my name is Calvin Wong and I work at Bank of Montreal (BMO)"
    assert redact(text) == "Hi my name is [NAME] and I work at [EMPLOYER] ([EMPLOYER])"


def test_possessive_is_still_caught() -> None:
    assert redact ("BMO's private wealth team") == "[EMPLOYER]'s private wealth team"


def test_whole_words_only() -> None:
    assert redact("BMOX is not a match") == "BMOX is not a match"


def test_text_without_personal_details_is_unchanged() -> None:
    assert redact("I verify identity documents") == "I verify identity documents"
