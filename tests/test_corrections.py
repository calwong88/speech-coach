from speech_coach.corrections import apply_corrections


def test_fixes_your_real_mishearings() -> None:
    text = "I work at Bemo and hold the Microsoft sc 300 and Security A I Plus"
    assert apply_corrections(text) == "I work at BMO and hold the Microsoft SC-300 and SecAI+"


def test_longest_phrase_wins() -> None:
    assert apply_corrections("security ai plus and security plus") == "SecAI+ and Security+"


def test_only_whole_words_are_replaced() -> None:
    assert apply_corrections("stop bemoaning it") == "stop bemoaning it"


def test_spelled_out_acronyms_are_joined() -> None:
    assert apply_corrections("the a m l and k y c checks") == "the AML and KYC checks"


def test_longer_correction_beats_acronym() -> None:
    assert apply_corrections("I hold security a i plus") == "I hold SecAI+"