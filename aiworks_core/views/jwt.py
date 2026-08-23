from datetime import datetime, timedelta, timezone

import jwt
from django.conf import settings


def create_token(
        user_id: int,
        token_type: str,
        *,
        expires_in: timedelta | None = None,
        no_expiry: bool = False,
        **extra_claims: str | int | bool,
) -> str:
    """Create a scoped JWT token.

    Args:
        user_id: The ID of the user the token is for.
        token_type: A label identifying the token purpose (e.g. "preview", "reset").
        expires_in: Optional expiry duration. Defaults to 1 hour if not provided.
        no_expiry: If True, omit the exp claim entirely so the token never expires.
        **extra_claims: Additional key-value pairs to embed in the token payload.

    Returns:
        A signed JWT string.
    """
    from django.conf import settings

    now = datetime.now(timezone.utc)
    payload = {
        "user_id": user_id,
        "type": token_type,
        "iat": now,
    }
    if not no_expiry:
        payload["exp"] = now + (expires_in if expires_in is not None else timedelta(hours=1))
    payload.update(extra_claims)
    return jwt.encode(payload, settings.JWT_SIGNING_KEY, algorithm="HS256")


def decrypt_and_validate_token(token: str, expected_type: str) -> dict | None:
    """Decrypt and validate a scoped JWT token.

    Verifies the token signature and that it has not expired. Validates that the
    token's type matches ``expected_type`` and that a ``user_id`` claim is present.
    Returns the full decoded payload so callers can perform additional validation
    (e.g. checking a session_id embedded in the payload).

    Args:
        token: The JWT string to validate.
        expected_type: The token type the caller expects (must match the "type" claim).

    Returns:
        The decoded payload dict if valid, or None if the token is invalid,
        expired, missing required claims, or the type does not match.
    """

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SIGNING_KEY,
            algorithms=["HS256"],
            options={"require": ["iat", "user_id", "type"]},
        )
    except jwt.PyJWTError:
        return None

    if payload.get("type") != expected_type:
        return None

    return payload
