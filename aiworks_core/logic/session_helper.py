import io
import logging
import zipfile
from io import BytesIO
from typing import cast

from django.db import models as db_models

from .files import get_binary_data
from .logic_utils import make_safe_filename
from .user_background import generate_user_background
from ..models import (
    MemoryEntry,
    Session, AttachedFile,
    KnowledgeBase,
    MCPServer,
)

logger = logging.getLogger(__name__)

# Limit the data to include on get_formatted_files
SESSION_FILE_SIZE_INCLUDE_LIMIT_CHARS = 200_000


def get_effective_user_background(user) -> str:
    """Return the LLM-generated background for *user*, generating it lazily if needed.

    Checks the cached ``user.llm_generated_background`` first.  If empty but
    the user has a ``profile_context`` or memory entries, calls
    ``generate_user_background`` to synthesise and persist the background.

    Returns an empty string when the user has neither profile context nor
    memory entries (nothing to synthesise).

    Raises whatever exception ``generate_user_background`` raises on LLM
    failure or empty result.
    """
    has_profile = bool((user.profile_context or "").strip())
    has_memories = MemoryEntry.objects.filter(user=user).exists()
    if not has_profile and not has_memories:
        return ""
    user_bg = (user.llm_generated_background or "").strip()
    if not user_bg:
        user_bg = generate_user_background(user)
    return user_bg


def get_effective_context(session: Session) -> str:
    """Return the effective user context for *session*.

    When ``session.include_user_context`` is False, returns an empty string.
    Otherwise returns ``get_effective_user_background`` wrapped as
    ``"User Background\n{bg}"``, or empty string if there is nothing
    to synthesise.
    """
    if not session.include_user_context:
        return ""
    user_bg = get_effective_user_background(session.user)
    if not user_bg.strip():
        return ""
    return f"User Background\n{user_bg}"


def attach_to_session(session: Session, attachments_data: dict | None) -> bool:
    if attachments_data:
        files = attachments_data.get("files", [])
        if files:
            for af in files:
                af.session = session
                af.save()

        attached_session_ids = attachments_data.get("attached_session_ids", [])
        if attached_session_ids:
            for sid in attached_session_ids:
                try:
                    attached_session = Session.objects.filter(
                        db_models.Q(id=sid)
                        & (
                                db_models.Q(user=session.user)
                                | db_models.Q(is_public=True)
                        )
                    ).first()
                    session.attached_sessions.add(attached_session)
                except Session.DoesNotExist:
                    logger.warning(f"Attach to session failed, for session {sid} - session not found or not permitted")

        kb_ids = attachments_data.get("knowledge_base_ids", [])
        if kb_ids:
            kbs = KnowledgeBase.objects.filter(id__in=kb_ids, user=session.user)
            for kb in kbs:
                session.knowledge_bases.add(kb)

        mcp_ids = attachments_data.get("mcp_server_ids", [])
        if mcp_ids:
            mcps = MCPServer.objects.filter(id__in=mcp_ids, user=session.user)
            for mcp in mcps:
                session.mcp_servers.add(mcp)

        desktop_tunnel_servers = attachments_data.get("desktop_tunnel_servers", {})
        if desktop_tunnel_servers:
            existing_tunnel_servers = getattr(session, "desktop_tunnel_servers", {})
            for tunnel_id, tunnel_servers in desktop_tunnel_servers.items():
                if tunnel_id in existing_tunnel_servers:
                    existing_tunnel_servers[tunnel_id] = list(set(existing_tunnel_servers[tunnel_id] + tunnel_servers))
                else:
                    existing_tunnel_servers[tunnel_id] = tunnel_servers
            session.desktop_tunnel_servers = existing_tunnel_servers

        session.save(update_fields=["desktop_tunnel_servers", "updated_at"])
        return bool(kb_ids or mcp_ids or desktop_tunnel_servers)
    return False


def format_session_as_text(source_session: Session) -> str:
    """Format a session as a text string for attachment.

    The host application may override this via the
    ``AIWORKS_CORE_FORMAT_SESSION_AS_TEXT_CALLBACK`` setting.
    """
    import importlib
    from django.conf import settings

    callback = getattr(settings, "AIWORKS_CORE_FORMAT_SESSION_AS_TEXT_CALLBACK", None)
    if callback:
        module_path, func_name = callback.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name)(source_session)

    lines = []
    return "\n".join(lines)


def parse_session_to_file(source_session: Session) -> tuple[AttachedFile, list[AttachedFile]]:
    """Convert a source session to an AttachedFile and return its output files.

    Returns a tuple of (main_session_file, additional_output_files).
    The main session file contains the formatted text representation of the session.
    Additional output files are the non-hidden output files from the source session.
    """
    file_name = f"session_{source_session.id}.txt"
    content = format_session_as_text(source_session)

    af = AttachedFile(
        name=file_name,
        file_type=AttachedFile.FILE_TYPE_INPUT,
        binary_content=content.encode("utf-8"),
    )
    return af, _get_session_output_files(source_session)


def _get_session_output_files(source_session: Session) -> list[AttachedFile]:
    """Return non-hidden output files from a session.

    The host application may inject additional files via the
    ``AIWORKS_CORE_GET_SESSION_OUTPUT_FILES_CALLBACK`` setting.
    """
    import importlib
    from django.conf import settings

    result_files = list(
        source_session.attached_files.filter(
            file_type=AttachedFile.FILE_TYPE_OUTPUT,
            is_hidden=False,
        )
    )

    callback = getattr(settings, "AIWORKS_CORE_GET_SESSION_OUTPUT_FILES_CALLBACK", None)
    if callback:
        module_path, func_name = callback.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        extra_files = func(source_session)
        if extra_files:
            result_files.extend(extra_files)

    return result_files


def get_formatted_files(session: Session) -> str:
    """Format attached files and attached sessions for AI prompt.

    Returns a string containing:
    - Direct input attached files
    - Attached sessions (via M2M) converted to text with their output files
    """
    from .files import get_text_content

    parts = []

    input_files = session.attached_files.filter(file_type=AttachedFile.FILE_TYPE_INPUT)
    for f in input_files:
        text = get_text_content(f)
        if text:
            if len(text) > SESSION_FILE_SIZE_INCLUDE_LIMIT_CHARS:
                parts.append(f"FILE: {f.name}\nCONTENT: (Content too big to include here)")
            else:
                parts.append(f"FILE: {f.name}\nCONTENT: {text}")

    for attached_session in session.attached_sessions.all():
        attached_session = cast(Session, attached_session)
        if attached_session.user == session.user or attached_session.is_public:
            session_file, output_files = parse_session_to_file(attached_session)

            text = get_text_content(session_file)
            if text:
                if len(text) > SESSION_FILE_SIZE_INCLUDE_LIMIT_CHARS:
                    parts.append(f"FILE: {session_file.name}\nCONTENT: (Content too big to include here)")
                else:
                    parts.append(f"FILE: {session_file.name}\nCONTENT: {text}")

            for output_file in output_files:
                output_text = get_text_content(output_file)
                if output_text:
                    prefixed_name = f"/session_{attached_session.id}/{output_file.name.lstrip('/')}"
                    if len(output_text) > SESSION_FILE_SIZE_INCLUDE_LIMIT_CHARS:
                        parts.append(f"FILE: {prefixed_name}\nCONTENT: (Content too big to include here)")
                    else:
                        parts.append(f"FILE: {prefixed_name}\nCONTENT: {output_text}")

    return "\n\n".join(parts)


def create_download_zip(session) -> BytesIO:
    """Create a ZIP file of session output files.

    The host application may inject additional files via the
    ``AIWORKS_CORE_CREATE_DOWNLOAD_ZIP_CALLBACK`` setting.
    """
    import importlib
    from django.conf import settings

    output_files = session.attached_files.filter(
        file_type=AttachedFile.FILE_TYPE_OUTPUT,
        is_hidden=False,
    ).exclude(is_hidden=True)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        included_files = set()
        for attached_file in output_files:
            path_parts = attached_file.name.split("/")
            safe_parts = [
                make_safe_filename(part) for part in path_parts
            ]
            safe_name = "/".join(safe_parts).lstrip("/")

            data = get_binary_data(attached_file)
            if data:
                zip_file.writestr(safe_name, data)

            included_files.add(safe_name)

        callback = getattr(settings, "AIWORKS_CORE_CREATE_DOWNLOAD_ZIP_CALLBACK", None)
        if callback:
            module_path, func_name = callback.rsplit(":", 1)
            mod = importlib.import_module(module_path)
            func = getattr(mod, func_name)
            func(zip_file, session, included_files)

    zip_buffer.seek(0)
    return zip_buffer
