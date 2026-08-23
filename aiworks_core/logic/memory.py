import logging
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import transaction
from django.db.models import Q

from ..utils import async_to_sync
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential, before_sleep_log
from .llm import invoke_llm
from .prompt_compression import compress_prompt
from .user_background import generate_user_background
from ..models import (
    MemoryEntry,
    User,
    Session,
    SiteConfiguration,
)
from ..utils import chunked

logger = logging.getLogger(__name__)


def _get_lib_schema_path(filename: str) -> str:
    """Return an absolute path to a schema file bundled in aiworks_core/resources/schema/."""
    return str(Path(__file__).resolve().parent.parent / "resources" / "schema" / filename)


# ---------------------------------------------------------------------------
# Memory generation
# ---------------------------------------------------------------------------

_MEMORY_GENERATION_TARGET_RATIO = 0.6  # compress to 60% of max_entries


def _format_session_as_text_for_memory_extraction(source_session: Session) -> str:
    """Format a session as a text string for memory extraction.

    The host application may override this via the
    ``AIWORKS_CORE_FORMAT_SESSION_CALLBACK`` setting to extract additional
    domain-specific fields.
    """
    callback = getattr(settings, "AIWORKS_CORE_FORMAT_SESSION_CALLBACK", None)
    if callback:
        return callback(source_session)

    lines = [
        f"Session type: {source_session.session_type}",
        f"Title: {source_session.session_title}",
    ]
    if source_session.agent_result:
        lines.append(f"Result: {source_session.agent_result[:500]}")

    return "\n".join(lines)


def generate_memories_for_user(user, since_dt) -> int:
    """Extract new memories from sessions created since ``since_dt`` and store them.

    Returns the number of new memories stored.
    """
    cfg = SiteConfiguration.get_solo()

    sessions = list(
        Session.objects.filter(
            user=user,
            created_at__gte=since_dt,
        )
    )

    if not sessions:
        return 0

    new_count = 0
    for batch in chunked(sessions, 10):
        try:
            new_count += _extract_memories_from_batch(user, batch, cfg)
        except Exception as exp:
            logger.exception("memory_extraction | error processing batch for user=%s: %s", user.pk, exp)

    logger.info(
        "memory_extraction | user=%s | added %d new memories", user.pk, new_count
    )

    return new_count


# noinspection PyTypeChecker
@retry(
    retry=retry_if_not_exception_type(ValueError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.ERROR, exc_info=True),
)
def _extract_memories_from_batch(user, sessions: list[Any], cfg) -> int:
    compressed_sessions = [
        async_to_sync(compress_prompt)(_format_session_as_text_for_memory_extraction(s)) for s in sessions
    ]
    sessions_data = "\n\n".join(compressed_sessions)

    existing_memory_qs = list(
        MemoryEntry.objects.filter(user=user).values_list("content", flat=True)
    )
    existing_memories_text = (
        "\n".join(f"- {m}" for m in existing_memory_qs)
        if existing_memory_qs
        else "(none)"
    )
    profile_ctx = (user.profile_context or "").strip() or "(none)"

    new_memories = async_to_sync(invoke_llm)(
        "memory_generation",
        system_message_template_name="memory_extraction.system_message",
        user_message_template_name="memory_extraction.prompt_template",
        template_params={
            "profile_context": profile_ctx,
            "existing_memories": existing_memories_text,
            "sessions_data": sessions_data,
        },
        parse_json=True,
        schema_file_path=_get_lib_schema_path("memory_generation_schema.json"),
    )

    if not isinstance(new_memories, list):
        raise Exception(f"memory_extraction | unexpected LLM output type: {type(new_memories)}")

    new_memories = [str(m).strip() for m in new_memories if str(m).strip()]

    if new_memories:
        entries = [MemoryEntry(user=user, content=m) for m in new_memories]
        MemoryEntry.objects.bulk_create(entries)

        total_count = MemoryEntry.objects.filter(user=user).count()
        if total_count > cfg.memory_max_entries:
            _compress_memories_for_user(user, cfg)

    return len(new_memories)


def _compress_memories_for_user(user, cfg):
    """Compress a user's memories when they exceed the configured limit."""
    all_memories = list(
        MemoryEntry.objects.filter(user=user).values_list("content", flat=True)
    )
    existing_memories_text = "\n".join(f"- {m}" for m in all_memories)
    profile_ctx = (user.profile_context or "").strip() or "(none)"
    target_count = max(5, int(cfg.memory_max_entries * _MEMORY_GENERATION_TARGET_RATIO))

    compressed_memories = async_to_sync(invoke_llm)(
        "memory_compression",
        system_message_template_name="memory_compression.system_message",
        user_message_template_name="memory_compression.prompt_template",
        template_params={
            "profile_context": profile_ctx,
            "existing_memories": existing_memories_text,
            "target_count": target_count,
        },
        parse_json=True,
        schema_file_path=_get_lib_schema_path("memory_compression_schema.json"),
    )

    if not isinstance(compressed_memories, list):
        logger.warning(
            "memory_compression | unexpected LLM output type: %s",
            type(compressed_memories),
        )
        return
    compressed_memories = [
        str(m).strip() for m in compressed_memories if str(m).strip()
    ]
    if not compressed_memories:
        return

    with transaction.atomic():
        MemoryEntry.objects.filter(user=user).delete()
        entries = [MemoryEntry(user=user, content=m) for m in compressed_memories]
        MemoryEntry.objects.bulk_create(entries)
    logger.info(
        "memory_compression | user=%s | compressed %d -> %d memories",
        user.pk,
        len(all_memories),
        len(entries),
    )


def run_memory_generation_job(since_dt):
    """Synchronous wrapper — called by the watchdog.

    Iterates over users who have had sessions since ``since_dt`` and extracts
    new memories.  Updating ``SiteConfiguration.memory_last_run`` is the
    responsibility of the caller (``_handle_memory_generation`` in apps.py).
    """
    # Find users with at least one relevant session since the last run
    # Use set() for Python-level deduplication (SQLite doesn't support DISTINCT ON)
    user_ids = list(
        set(
            Session.objects.filter(created_at__gte=since_dt).values_list(
                "user_id", flat=True
            )
        )
    )

    if not user_ids:
        return

    # Memory generation is a Pro-only feature — skip free-tier users.
    users = list(User.objects.filter(pk__in=user_ids, tier=User.TIER_PRO))

    total_new = 0
    for user in users:
        try:
            count = generate_memories_for_user(user, since_dt)
            total_new += count
        except Exception as exc:
            logger.exception("memory_generation | error for user=%s: %s", user.pk, exc)

    logger.info(
        "memory_generation | processed %d users | %d new memories total",
        len(users),
        total_new,
    )


def run_user_background_calculation():
    """Synchronous wrapper — called by the watchdog.

    Iterates over users and recalculate the user background for them.
    """
    user_with_memories = list(
        set(
            MemoryEntry.objects.values_list(
                "user_id", flat=True
            )
        )
    )

    users = list(
        (
                User.objects.filter(
                    pk__in=user_with_memories
                ) |
                User.objects.filter(
                    Q(profile_context__isnull=False) & Q(profile_context__gt="")
                )
        ).filter(
            Q(llm_generated_background__isnull=True) | Q(llm_generated_background="")
        )
    )

    for user in users:
        try:
            generate_user_background(user)
        except Exception as exc:
            logger.exception("user_background_generation | error for user=%s: %s", user.pk, exc)

    logger.info(
        "user_background_generation | processed %d users",
        len(users)
    )


def import_memories(user, imported_text):
    existing_qs = list(
        MemoryEntry.objects.filter(user=user).values_list("content", flat=True)
    )
    existing_memories_text = (
        "\n".join(f"- {m}" for m in existing_qs)
        if existing_qs
        else "(none)"
    )

    new_memories = async_to_sync(invoke_llm)(
        "memory_import",
        system_message_template_name="memory_import.system_message",
        user_message_template_name="memory_import.prompt_template",
        template_params={
            "profile_context": user.profile_context,
            "existing_memories": existing_memories_text,
            "imported_text": imported_text,
        },
        parse_json=True,
        schema_file_path=_get_lib_schema_path("memory_import_schema.json"),
    )

    if not isinstance(new_memories, list):
        logger.warning(
            "memory_import | unexpected LLM output type: %s", type(new_memories)
        )
        raise ValueError("Incorrect return type, not a list")

    new_memories = [str(m).strip() for m in new_memories if str(m).strip()]

    with transaction.atomic():
        if new_memories:
            MemoryEntry.objects.filter(user=user).delete()
            entries = [MemoryEntry(user=user, content=m) for m in new_memories]
            MemoryEntry.objects.bulk_create(entries)

            logger.info(
                "memory_import | user=%s | replaced with %d memories",
                user.pk,
                len(new_memories),
            )
        else:
            logger.info(
                "memory_import | user=%s | Ignored memory-import. Empty list returned",
                user.pk,
            )

    final_memories_count = MemoryEntry.objects.filter(user=user).count()
    return final_memories_count
