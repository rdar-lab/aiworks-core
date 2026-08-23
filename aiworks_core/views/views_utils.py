import json
import logging
import mimetypes
import os

from django.db import models as db_models
from django.http import HttpResponse
from rest_framework.exceptions import (
    PermissionDenied,
    ValidationError as DRFValidationError,
)

from aiworks_core.views.jwt import decrypt_and_validate_token
from ..models import KnowledgeBase, MCPServer, Tunnel
from ..models import (
    Session,
    AttachedFile,
    User
)

logger = logging.getLogger(__name__)

# MIME types that are served inline (all others are served as attachment downloads).
_ATTACHMENT_INLINE_TYPES = {
    "text/html",
    "text/css",
    "application/javascript",
    "text/javascript",
    "image/svg+xml",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/x-icon",
    "font/woff",
    "font/woff2",
    "application/font-woff",
    "application/font-woff2",
}


def require_pro(user, feature_name: str):
    """Raise PermissionDenied if the user is not on the Pro tier."""
    if not user.is_pro:
        logger.info(
            "pro_gate.denied | user=%s | feature=%s", user.username, feature_name
        )
        raise PermissionDenied(
            detail=f"'{feature_name}' requires a Pro account. Please upgrade."
        )


def parse_files_from_request(request, mandatory=True):
    from ..logic.document_parser import ALLOWED_UPLOAD_EXTENSIONS, validate_file_size

    uploaded = request.FILES.getlist("files")
    if not uploaded:
        if mandatory:
            raise DRFValidationError({"error": "No files provided"})
        else:
            return [], []

    for f in uploaded:
        ext = os.path.splitext(f.name.lower())[1]
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
            raise DRFValidationError(
                {"error": f"File type {ext!r} is not allowed. Allowed types: {allowed}"}
            )

    parsed = []
    errors = []

    for f in uploaded:
        try:
            raw_data = f.read()
            validate_file_size(raw_data, f.name)
            ext = os.path.splitext(f.name.lower())[1]
            if ext in ALLOWED_UPLOAD_EXTENSIONS:

                af = AttachedFile(
                    name=f.name,
                    file_type=AttachedFile.FILE_TYPE_INPUT,
                    binary_content=raw_data,
                )
                parsed.append(af)
                logger.debug(
                    "parse_files_from_request | file=%s | size=%d",
                    f.name,
                    len(raw_data),
                )
            else:
                logger.warning(
                    "parse_files_from_request | skipped disallowed file type: %s",
                    f.name,
                )
                errors.append({"name": f.name, "error": "File type not allowed"})
        except ValueError as exc:
            logger.warning(
                "parse_files_from_request | validation error for %s: %s",
                f.name,
                exc,
            )
            errors.append({"name": f.name, "error": str(exc)})
        except Exception as exc:
            logger.exception(
                "parse_files_from_request | error for %s: %s", f.name, exc
            )
            errors.append({"name": f.name, "error": "Failed to process file"})

    return parsed, errors


def upload_files(request, additional_file_fields: dict):
    parsed, errors = parse_files_from_request(request)

    created = []
    for attached_file in parsed:
        try:
            AttachedFile.objects.filter(
                **additional_file_fields, name=attached_file.name
            ).delete()
            for key, value in additional_file_fields.items():
                setattr(attached_file, key, value)
            attached_file.save()
            created.append({"name": attached_file.name})
        except Exception as exc:
            logger.exception(
                "upload_files | failed to save file %s: %s",
                attached_file.name,
                exc,
            )
            errors.append({"name": attached_file.name, "error": "Failed to save file"})

    return created, errors


def make_content_disposition(filename: str) -> str:
    """Return a Content-Disposition header value for *filename*.

    Uses RFC 5987 ``filename*=`` encoding for non-ASCII names so Hebrew and other
    Unicode titles produce valid download filenames on all browsers.
    """
    from ..logic.logic_utils import make_safe_filename
    safe_ascii = make_safe_filename(filename)
    try:
        safe_ascii.encode("ascii")
        return f'attachment; filename="{safe_ascii}"'
    except UnicodeEncodeError:
        pct = "".join(f"%{b:02X}" for b in filename.encode("utf-8"))
        return f'attachment; filename="{safe_ascii}"; filename*=UTF-8\'\'{pct}'


def prepare_attachments_data(request, eff_data):
    """Validate and build attachments data for the rerun action.

    Returns a dict suitable for passing to ``rerun_session(attachments_data=...)``.
    Raises ``DRFValidationError`` on validation failure.
    """

    raw_files = request.FILES.getlist("files") if hasattr(request, 'FILES') else []

    if not request.user.is_pro and len(raw_files) > 1:
        raise DRFValidationError(
            {"error": "Free plan allows 1 file per session. Please upload only 1 file."}
        )
    parsed_files = []
    if raw_files:
        try:
            parsed_files, file_errors = parse_files_from_request(
                request, mandatory=False
            )
        except DRFValidationError as exc:
            # parse_files_from_request raises DRFValidationError for disallowed
            # extensions. Normalize to the same {error, parse_errors} response
            # shape so callers always see a consistent structure.
            err_detail = exc.detail
            if isinstance(err_detail, dict):
                error_msg = err_detail.get("error", "File validation failed")
            elif isinstance(err_detail, list):
                error_msg = (
                    str(err_detail[0]) if err_detail else "File validation failed"
                )
            else:
                error_msg = str(err_detail)
            raise DRFValidationError(
                {
                    "error": "One or more files could not be uploaded.",
                    "parse_errors": [{"name": "", "error": error_msg}],
                }
            )
        if file_errors:
            # Return only safe, user-facing error descriptions — never raw
            # exception messages or stack traces.
            safe_errors = [
                {
                    "name": e.get("name", ""),
                    "error": e.get("error", "Upload failed"),
                }
                for e in file_errors
            ]
            raise DRFValidationError(
                {
                    "error": "One or more files could not be uploaded.",
                    "parse_errors": safe_errors,
                }
            )

    raw_attached_session_ids = eff_data.get("attachedSessionIds", [])
    if not isinstance(raw_attached_session_ids, list):
        raise DRFValidationError({"error": "attachedSessionIds must be an array of session IDs."})
    for sid in raw_attached_session_ids:
        if not isinstance(sid, str) or not sid:
            raise DRFValidationError({"error": "attachedSessionIds must be non-empty strings."})

    attached_session_ids = []
    for sid in raw_attached_session_ids:
        if not sid:
            continue
        if not Session.objects.filter(
                db_models.Q(id=sid, user=request.user)
                | db_models.Q(id=sid, is_public=True)
        ).exists():
            raise DRFValidationError({"attachedSessionIds": f"Session {sid} not found."})
        attached_session_ids.append(sid)

    kb_ids = eff_data.get("knowledgeBaseIds", [])
    if kb_ids:
        found_kb_ids = set(
            KnowledgeBase.objects.filter(id__in=kb_ids, user=request.user).values_list("id", flat=True)
        )
        unknown_kbs = [i for i in kb_ids if i not in found_kb_ids]
        if unknown_kbs:
            raise DRFValidationError(
                {"knowledgeBaseIds": f"Knowledge bases not found or not owned by you: {unknown_kbs}"}
            )

    mcp_ids = eff_data.get("mcpServerIds", [])
    if mcp_ids:
        found_mcp_ids = set(
            MCPServer.objects.filter(id__in=mcp_ids, user=request.user).values_list("id", flat=True)
        )
        unknown_mcps = [i for i in mcp_ids if i not in found_mcp_ids]
        if unknown_mcps:
            raise DRFValidationError(
                {"mcpServerIds": f"MCP servers not found or not owned by you: {unknown_mcps}"}
            )

    desktop_tunnel_servers = eff_data.get("desktopTunnelServers", {})
    desktop_tunnel_ids = list(desktop_tunnel_servers.keys())
    if desktop_tunnel_ids:
        found_tunnel_ids = set(
            Tunnel.objects.filter(tunnel_id__in=desktop_tunnel_ids, user=request.user).values_list("tunnel_id",
                                                                                                   flat=True)
        )
        unknown_tunnels = [i for i in desktop_tunnel_ids if i not in found_tunnel_ids]
        if unknown_tunnels:
            raise DRFValidationError(
                {"desktopTunnelServers": f"Tunnels not found or not owned by you: {unknown_tunnels}"}
            )

    attachments_data = {
        "files": parsed_files,
        "attached_session_ids": attached_session_ids,
        "knowledge_base_ids": kb_ids,
        "mcp_server_ids": mcp_ids,
        "desktop_tunnel_servers": desktop_tunnel_servers,
    }
    if not any([
        attachments_data.get("files"),
        attachments_data.get("attached_session_ids"),
        attachments_data.get("knowledge_base_ids"),
        attachments_data.get("mcp_server_ids"),
        attachments_data.get("desktop_tunnel_servers"),
    ]):
        attachments_data = None

    return attachments_data


def parse_effective_data(request) -> dict:
    """Return the effective session-creation payload as a plain dict.

    When the request is multipart/form-data, the client may encode session
    fields as a JSON blob in a ``data`` form field and add supported
    non-file fields (for example ``attachedSessionIds``) alongside it.
    For plain JSON requests, ``request.data`` is already the dict we want.
    """
    if "data" in request.data:
        try:
            payload = json.loads(request.data["data"])
        except (json.JSONDecodeError, ValueError):
            raise DRFValidationError(
                {"data": "Invalid JSON in session data field."}
            )

        if not isinstance(payload, dict):
            raise DRFValidationError(
                {"data": "AppSession data field must decode to a JSON object."}
            )

        # Merge attachedSessionIds from multipart form fields if not already
        # present in the JSON blob (either approach is supported).
        #
        # Three supported representations (in priority order):
        #   1. Inside the JSON blob: {"attachedSessionIds": ["id1", "id2"]}
        #   2. Repeated form fields: attachedSessionIds=id1&attachedSessionIds=id2
        #   3. Single field with JSON array string: attachedSessionIds=["id1","id2"]
        if (
                "attachedSessionIds" not in payload
                and "attachedSessionIds" in request.data
        ):
            if hasattr(request.data, "getlist"):
                attached_session_ids = request.data.getlist(
                    "attachedSessionIds"
                )
            else:
                attached_session_ids = [request.data.get("attachedSessionIds")]

            attached_session_ids = [
                value for value in attached_session_ids if value not in (None, "")
            ]

            # Handle a single value that is itself a JSON array string
            if len(attached_session_ids) == 1:
                raw_value = attached_session_ids[0]
                if isinstance(raw_value, str):
                    try:
                        parsed_value = json.loads(raw_value)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        parsed_value = raw_value
                    if isinstance(parsed_value, list):
                        attached_session_ids = parsed_value
                    else:
                        attached_session_ids = [raw_value]

            payload["attachedSessionIds"] = attached_session_ids

        return payload
    return request.data


def download_session_attachment(session: Session, attachment_path) -> HttpResponse:
    from ..logic.files import get_binary_data

    # --- Path validation ------------------------------------------------
    # Strip leading slashes; reject any path traversal attempts.
    clean_path = attachment_path.lstrip("/")
    if ".." in clean_path.split("/"):
        return HttpResponse(status=400)

    # --- File lookup ----------------------------------------------------
    attached_file = AttachedFile.objects.filter(
        session=session,
        name=clean_path,
        file_type=AttachedFile.FILE_TYPE_OUTPUT,
        is_hidden=False,
    ).first()

    if not attached_file:
        return HttpResponse(status=404)

    # --- File content ---------------------------------------------------
    raw_data = get_binary_data(attached_file)

    # --- MIME type ------------------------------------------------------
    mime_type, _ = mimetypes.guess_type(clean_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    response = HttpResponse(raw_data, content_type=mime_type)

    # Force a download for file types the browser cannot display inline.
    if mime_type not in _ATTACHMENT_INLINE_TYPES:
        filename = clean_path.rsplit("/", 1)[-1]
        response["Content-Disposition"] = make_content_disposition(filename)

    response["X-Frame-Options"] = "ALLOWALL"

    return response


def jwt_session_attachment(request, session_id, token, attachment_path):
    """Serve a specific output attachment file for a session.

    Authentication is via a scoped preview JWT token in the URL path. The token
    allows the browser to authenticate all sub-resources (CSS, JS, images) loaded
    by an iframe without any custom request headers.

    Only non-hidden, output-type ``AttachedFile`` records are served.
    """
    # --- Authentication -------------------------------------------------
    payload = decrypt_and_validate_token(token, "preview")
    if payload is None:
        return HttpResponse(status=401)
    user = User.objects.filter(pk=payload["user_id"]).first()
    if user is None:
        return HttpResponse(status=401)

    # Validate the session id is the same as in the token that was generated
    if str(payload.get("session_id")) != session_id:
        return HttpResponse(status=403)

    # --- Session lookup -------------------------------------------------
    try:
        session = Session.objects.get(id=session_id, is_deleted=False)
    except Session.DoesNotExist:
        return HttpResponse(status=404)

    # --- Public token must match public session -------------------------
    # If the token was generated for a public session, the session must still be public.
    # This invalidates tokens when a session is re-privated.
    if payload.get("is_public") and not session.is_public:
        return HttpResponse(status=403)

    # --- Access check ---------------------------------------------------
    # For public sessions, any authenticated user can access.
    # For private sessions, only the session owner can access.
    if not session.is_public and session.user_id != user.pk:
        return HttpResponse(status=403)

    return download_session_attachment(session, attachment_path)
