"""Password hashing and session tokens, using only Python's standard library.

Passwords are never stored. Only a salted PBKDF2-SHA256 hash is saved. Session tokens are random; only their
SHA-256 hash is saved, so a copy of the database cannot be used to log in.
"""
import hashlib
import hmac
import secrets

ITERATIONS = 600_000      # PBKDF2 rounds for NEW hashes (stored hashes remember their own number)
MIN_PASSWORD_LENGTH = 8


def hash_password(password: str, iterations: int = None) -> str:
    iterations = iterations or ITERATIONS
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored) -> bool:
    """True if the password matches. If `stored` is None (unknown user) it still does the same amount of work,
    so response time does not reveal which usernames exist."""
    if not stored:
        hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), b"0" * 16, ITERATIONS)
        return False
    try:
        scheme, iters, salt_hex, digest_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


def check_password_strength(password: str, username: str = "") -> None:
    """Raise ValueError with a plain-English reason if the password is too weak."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"the password must have at least {MIN_PASSWORD_LENGTH} characters")
    if not password.strip():
        raise ValueError("the password must not be only spaces")
    if username and password.lower() == username.lower():
        raise ValueError("the password must not be the same as the username")


def new_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
