"""Login for the web app: one user; the password is stored only as a salted scrypt hash."""
import hashlib
import hmac
import secrets
from collections.abc import Callable

SCRYPT = {"n": 2**14, "r":8, "p": 1}  # cost settings: slow enough to resist guessing


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return 'scrypt:<salt>:<hash>', safe to store. The password itself is never stored."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **SCRYPT)
    return f"scrypt:{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split(":")
    except ValueError:
        return False  # malformed: fail closed
    if scheme != "scrypt":
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **SCRYPT)
    return hmac.compare_digest(digest.hex(), digest_hex)


def make_checker(username: str, stored_hash: str) -> Callable[[str, str], bool]:
    """The function Gradio calls with whatever someone types into the login form."""
    def check(user: str, password: str) -> bool:
        user_ok = hmac.compare_digest(user.encode(), username.encode())
        password_ok = verify_password(password, stored_hash)
        return user_ok and password_ok
    return check
