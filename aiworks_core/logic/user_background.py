"""User background synthesis.

Generates and persists an LLM-synthesised background paragraph for a user,
combining their profile context and memory entries into a single coherent
paragraph.
"""

import logging

from ..utils import async_to_sync

from .llm import invoke_llm
from ..models import MemoryEntry, User

logger = logging.getLogger(__name__)


def generate_user_background(user: User) -> str:
    """Generate and persist an LLM-synthesised background paragraph for *user*.

    Combines the user's ``profile_context`` and all ``MemoryEntry`` rows into a
    single coherent paragraph.  The result is saved to
    ``user.llm_generated_background`` and also returned.

    Returns an empty string immediately (without calling the LLM) when the user
    has neither a profile_context nor any memory entries.

    Raises:
        Exception: Re-raises any exception thrown by the LLM invocation.
        ValueError: When the LLM returns an empty background paragraph.
    """
    profile_ctx = (user.profile_context or "").strip()
    memory_entries = list(
        MemoryEntry.objects.filter(user=user).values_list("content", flat=True)
    )

    if not profile_ctx and not memory_entries:
        return ""

    memories_text = (
        "\n".join(f"- {m}" for m in memory_entries) if memory_entries else "(none)"
    )
    profile_text = profile_ctx or "(none)"

    logger.info(
        "generate_user_background | user=%s | profile_len=%d | memories=%d",
        user.pk,
        len(profile_text),
        len(memory_entries),
    )

    try:
        background = async_to_sync(invoke_llm)(
            "user_background",
            system_message_template_name="user_background_synthesis.system_message",
            user_message_template_name="user_background_synthesis.prompt_template",
            template_params={
                "profile_context": profile_text,
                "memories": memories_text,
            },
            parse_json=False,
        )
        background = (background or "").strip()
    except Exception as exc:
        logger.exception(
            "generate_user_background | error for user=%s: %s", user.pk, exc
        )
        raise

    if not background:
        raise ValueError(
            f"generate_user_background | LLM returned empty background for user {user.pk}"
        )

    User.objects.filter(pk=user.pk).update(llm_generated_background=background)
    user.llm_generated_background = background
    logger.info(
        "generate_user_background | user=%s | saved %d chars",
        user.pk,
        len(background),
    )

    return background
