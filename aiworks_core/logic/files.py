"""File state management for AttachedFile and TextAttachedFile.

Provides utilities for:
- Reading binary data (DB first, cold storage fallback)
- Lazy text extraction with TextAttachedFile caching
- Setting binary content with cascade delete of TextAttachedFile
- Daily offload to cold storage
"""

import base64
import logging

from ..utils import sync_to_async, async_to_sync
from django.db import transaction
from .cold_storage import (
    ColdStorageManager,
    get_attached_file_data,
    store_attached_file,
    clear_attached_file
)
from .document_parser import can_extract_text, extract_text_from_binary
from .llm import invoke_llm
from .logic_utils import is_text_file
from ..models import AttachedFile
from ..models import TextAttachedFile

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Binary data access
# ------------------------------------------------------------------


def get_binary_data(attached_file: AttachedFile) -> bytes | None:
    """Return raw binary data for an AttachedFile.

    Read path: DB binary_content field first, then cold storage fallback.
    Returns None if not found anywhere.
    """
    if attached_file.binary_content:
        return bytes(attached_file.binary_content)

    if not attached_file.pk:
        return None

    if not ColdStorageManager.is_enabled():
        return None

    return get_attached_file_data(attached_file)


# ------------------------------------------------------------------
# Text content (lazy extraction)
# ------------------------------------------------------------------


def get_text_content(attached_file: AttachedFile) -> str | None:
    """Return extracted text for an AttachedFile, creating TextAttachedFile lazily.

    - For extractable types: extracts text, creates TextAttachedFile, returns text.
      Subsequent calls return the cached TextAttachedFile.content.
    - For non-extractable types: returns base64-encoded string, no TextAttachedFile created.
    - Returns None if binary data is not available.
    """
    try:
        return attached_file.text_attached_file.content
    except TextAttachedFile.DoesNotExist:
        pass

    binary = get_binary_data(attached_file)
    if not binary:
        return None

    if not can_extract_text(attached_file.name) and not is_text_file(attached_file.name):
        return base64.b64encode(binary).decode("utf-8")

    text = extract_text_from_binary(binary, attached_file.name)

    if attached_file.pk:
        TextAttachedFile.objects.create(
            attached_file=attached_file,
            content=text,
            summary="",
        )

    return text


# ------------------------------------------------------------------
# Binary content mutation
# ------------------------------------------------------------------


def set_binary_content(attached_file: AttachedFile, data: bytes | None, offload_immediately=False) -> None:
    """Set binary_content on an AttachedFile, deleting any existing TextAttachedFile."""
    if not attached_file.pk:
        raise Exception("Attached file is not saved to DB yet. Cannot save content")

    try:
        attached_file.text_attached_file.delete()
    except TextAttachedFile.DoesNotExist:
        pass

    clear_attached_file(attached_file)

    attached_file.binary_content = data

    if offload_immediately and store_attached_file(attached_file):
        attached_file.binary_content = None

    attached_file.save(update_fields=["binary_content"])


# ------------------------------------------------------------------
# Offload to cold storage
# ------------------------------------------------------------------


def offload_to_cold_storage() -> dict:
    """Move all AttachedFile binary_content from DB to cold storage and clear DB.

    Returns dict with offloaded/failed/skipped counts.
    Skips entirely if cold storage is disabled (coldstorage_type == 'none').
    """
    result = {"offloaded": 0, "failed": 0, "skipped": 0}

    config = ColdStorageManager.is_enabled()
    if not config:
        logger.info("offload_to_cold_storage | cold storage disabled, skipping")
        return result

    queryset = AttachedFile.objects.filter(
        binary_content__isnull=False,
        binary_content__gt=b"",
    )

    for af in queryset.iterator():
        try:
            with transaction.atomic():
                af = AttachedFile.objects.select_for_update().get(pk=af.pk)
                if af.binary_content:
                    logger.info(
                        "offload_to_cold_storage | id=%s | file=%s | offloading to cold storage",
                        af.pk,
                        str(af)
                    )
                    if store_attached_file(af):
                        af.binary_content = None
                        af.save(update_fields=["binary_content"])
                        result["offloaded"] += 1
                    else:
                        result["failed"] += 1
                else:
                    result["skipped"] += 1
        except Exception:
            result["failed"] += 1
            logger.exception(
                "offload_to_cold_storage | id=%s | failed to offload",
                af.pk,
            )

    logger.info(
        "offload_to_cold_storage | offloaded=%d failed=%d skipped=%d",
        result["offloaded"],
        result["failed"],
        result["skipped"],
    )
    return result


async def summarize_file(file_id: int) -> str | None:
    """Generate and store an LLM summary for a TextAttachedFile.

    Skips if the summary is already populated. Uses the fast LLM type.
    Raises on any error — no fallback.
    """

    file_obj = await sync_to_async(AttachedFile.objects.get)(id=file_id)
    if not file_obj:
        raise Exception(f"File not found for ID={file_id}")

    # If this file is not text extractable we can't summarize it either
    if not can_extract_text(file_obj.name):
        return None

    text = await sync_to_async(get_text_content)(file_obj)
    if not text:
        return None

    taf, created = await sync_to_async(TextAttachedFile.objects.get_or_create)(
        attached_file=file_obj,
        defaults={"content": text, "summary": ""},
    )
    if not created and taf.summary:
        logger.debug(
            "summarize_file | file=%s | already summarised, skipping", file_id
        )
        return taf.summary

    logger.info(
        "summarize_file | file=%s name=%s content_len=%d",
        file_id,
        file_obj.name,
        len(text),
    )
    content_preview = text[:3000]
    summary = await invoke_llm(
        "summarize_document",
        system_message_template_name="summarize_document.system_message",
        user_message_template_name="summarize_document.prompt_template",
        template_params={"name": file_obj.name, "content": content_preview},
        parse_json=False,
    )
    taf.summary = summary.strip()
    await sync_to_async(taf.save)(update_fields=["summary"])
    logger.info(
        "summarize_file | file=%s | saved summary_len=%d",
        file_id,
        len(taf.summary),
    )
    return taf.summary


def get_file_summary(attached_file: AttachedFile) -> str | None:
    try:
        return async_to_sync(summarize_file)(attached_file.id)
    except Exception as exp:
        logger.exception(f"Was unable to calculate the file summary. Err={exp}")
        return None
