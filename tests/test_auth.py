from speech_coach.auth import hash_password, make_checker, verify_password


def test_right_password_verifies_and_wrong_one_does_not() -> None:
    stored = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", stored)
    assert not verify_password("correct horse battery stapler", stored)


def test_same_password_gets_a_different_salt_each_time() -> None:
    first, second = hash_password("same"), hash_password("same")
    assert first != second
    assert verify_password("same", first) and verify_password("same", second)


def test_malformed_stored_hash_fails_closed() -> None:
    assert not verify_password("anything", "not-a-real-hash")


def test_login_needs_both_the_right_user_and_password() -> None:
    check = make_checker("calvin", hash_password("pw"))
    assert check("calvin", "pw")
    assert not check("someone", "pw")
    assert not check("calvin", "wrong")


