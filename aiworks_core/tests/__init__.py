def _make_user(
        username="testuser", email="test@example.com", password="pass123", tier="pro"
):
    """Create a test user with verified email."""
    from ..models import User
    return User.objects.create_user(
        username=username,
        email=email,
        password=password,
        **({"tier": tier} if tier else {}),
        email_verified=True,
    )


def _make_session(user, session_id="sess1", title="Should we pivot?", **kwargs):
    """Create a test session."""
    from ..models import Session
    m2m_fields = {}
    for field in ("knowledge_bases", "mcp_servers"):
        if field in kwargs:
            m2m_fields[field] = kwargs.pop(field)
    if "session_type" not in kwargs:
        kwargs["session_type"] = "test_session"
    session = Session.objects.create(
        id=session_id,
        user=user,
        session_title=title,
        **kwargs,
    )
    for field, value in m2m_fields.items():
        getattr(session, field).set(value)
    return session


def _auth_header(user):
    """Generate an HTTP authorization header with JWT token for the given user."""
    from rest_framework_simplejwt.tokens import RefreshToken
    refresh = RefreshToken.for_user(user)
    return {"HTTP_AUTHORIZATION": f"Bearer {refresh.access_token}"}


# ---------------------------------------------------------------------------
# Mock patch helpers
# ---------------------------------------------------------------------------


# Export all test helper functions and commonly used imports
__all__ = [
    # Helper functions
    "_make_user",
    "_make_session",
    "_auth_header",
]
