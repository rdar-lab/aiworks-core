"""Purge unused generated images from sessions.

After LLM generation finishes, some generated [UUID].png images may not be used
in the final output (e.g., the LLM generated alternatives or extra images).
This module detects and removes such orphaned images.

Generic approach:
1. Collect all UUID.png files from output files (candidates)
2. Scan all text-based output files for UUID.png references; remove referenced from candidates
3. Also scan session.agent_result
4. Delete remaining candidates
"""
import logging
import re

from .files import get_binary_data
from .logic_utils import is_text_file
from ..models import AttachedFile, Session

logger = logging.getLogger(__name__)

UUID_PNG_PATTERN = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\.png$", re.IGNORECASE)
UUID_PNG_IN_TEXT = re.compile(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\.png", re.IGNORECASE)


def _is_uuid_png_file(filename: str) -> bool:
    """Return True if filename matches [UUID].png pattern."""
    return UUID_PNG_PATTERN.match(filename) is not None


def _collect_uuid_png_candidates(session: Session) -> set[str]:
    """Collect all UUID.png filenames from output files (candidates for deletion).

    Filenames are lowercased for case-insensitive matching against references.
    """
    candidates = set()
    for f in session.attached_files.filter(file_type=AttachedFile.FILE_TYPE_OUTPUT):
        if _is_uuid_png_file(f.name):
            candidates.add(f.name.lower())
    return candidates


def _collect_text_output_files(session: Session) -> dict[str, str]:
    """Collect text content from text-based output AttachedFiles.

    Returns a dict mapping filename to text content.
    Uses get_binary_data + decode to avoid caching.
    """
    result = {}
    for f in session.attached_files.filter(file_type=AttachedFile.FILE_TYPE_OUTPUT):
        if not is_text_file(f.name):
            continue
        binary = get_binary_data(f)
        if binary:
            try:
                result[f.name] = binary.decode("utf-8")
            except UnicodeDecodeError:
                pass
    return result


def _extract_uuid_png_refs(text: str) -> set[str]:
    """Extract all UUID.png references from text content."""
    return {match.group(0).lower() for match in UUID_PNG_IN_TEXT.finditer(text)}


def purge_unused_images(session: Session) -> int:
    """Delete orphaned [UUID].png image files that are not referenced anywhere.

    Generic approach:
    1. Collect all UUID.png files from output files (candidates)
    2. Scan all text-based output files for UUID.png references; remove from candidates
    3. Also scan session.agent_result
    4. Delete remaining candidates

    Args:
        session: The session whose unused images should be purged.

    Returns:
        Number of orphaned images deleted.
    """
    if not session.pk:
        return 0

    session.refresh_from_db()

    candidates = _collect_uuid_png_candidates(session)
    if not candidates:
        return 0

    text_files = _collect_text_output_files(session)

    for filename, content in text_files.items():
        refs = _extract_uuid_png_refs(content)
        candidates -= refs

    if session.agent_result:
        refs = _extract_uuid_png_refs(session.agent_result)
        candidates -= refs

    if not candidates:
        logger.info("purge_unused_images | session=%s | no orphaned images found", session.id)
        return 0

    orphaned_pks = [
        f.pk
        for f in session.attached_files.filter(file_type=AttachedFile.FILE_TYPE_OUTPUT)
        if f.name.lower() in candidates
    ]

    count, _ = AttachedFile.objects.filter(pk__in=orphaned_pks).delete()
    logger.info(
        "purge_unused_images | session=%s | deleted %d orphaned images",
        session.id,
        count,
    )
    return count
