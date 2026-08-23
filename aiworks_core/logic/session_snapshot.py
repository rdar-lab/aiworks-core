"""Session snapshot logic.

Provides create_snapshot and restore_snapshot functions for capturing and
restoring point-in-time session state.

ZIP structure
-------------
Each snapshot is stored in cold storage as::

    session_snapshots/{session_id}/{snapshot_id}.zip

The ZIP contains::

    session.json   – JSON dict of session text/state fields
    files.json     – JSON array of file metadata records
    files/{idx}/   – one directory per attached file
        meta.json  – {name, file_type, is_hidden}
        data.bin   – raw binary file content

The host application configures which session fields to snapshot via the
``AIWORKS_CORE_SNAPSHOT_SESSION_FIELDS`` and
``AIWORKS_CORE_SNAPSHOT_FIELDS_TO_CLEAR`` Django settings.
"""

import io
import json
import logging
import zipfile
from typing import Optional

from django.conf import settings

from .cold_storage import ColdStorageManager
from .files import get_binary_data
from ..models import AttachedFile, Session, SessionSnapshot

logger = logging.getLogger(__name__)

SNAPSHOTS_FOLDER = "session_snapshots"

_SESSION_FIELDS = getattr(
    settings,
    "AIWORKS_CORE_SNAPSHOT_SESSION_FIELDS",
    ["agent_result", "session_title"],
)

_FIELDS_TO_CLEAR_ON_RESTORE = getattr(
    settings,
    "AIWORKS_CORE_SNAPSHOT_FIELDS_TO_CLEAR",
    [],
)


def _build_zip(session: Session) -> bytes:
    """Build the snapshot ZIP for *session* and return raw bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # --- session.json ---
        session_data = {field: getattr(session, field) for field in _SESSION_FIELDS}
        # Convert non-serialisable values
        for key, val in session_data.items():
            if hasattr(val, "isoformat"):
                session_data[key] = val.isoformat()
        zf.writestr("session.json", json.dumps(session_data))

        # --- files ---
        files = list(session.attached_files.all())
        files_meta = []
        for idx, af in enumerate(files):
            folder = f"files/{idx}"
            meta = {
                "name": af.name,
                "file_type": af.file_type,
                "is_hidden": af.is_hidden,
            }
            zf.writestr(f"{folder}/meta.json", json.dumps(meta))

            # Retrieve raw binary using the same approach as the download attachment view:
            # try cold storage first, then derive from the content field.
            raw_data = get_binary_data(af)
            if raw_data is None:
                raw_data = "".encode("utf-8")

            zf.writestr(f"{folder}/data.bin", raw_data)
            files_meta.append(meta)

        zf.writestr("files.json", json.dumps(files_meta))

    return buf.getvalue()


def create_snapshot(session: Session) -> Optional[SessionSnapshot]:
    """Create and persist a snapshot of *session*.

    This function is intended to be called immediately after a successful
    executor run.  It is a best-effort operation: failures are logged but
    never propagated to the caller so they never interrupt the executor
    lifecycle.

    Snapshots require cold storage to be enabled; if cold storage is
    disabled the call is silently skipped and ``None`` is returned.

    Returns the created ``SessionSnapshot`` instance, or ``None`` when the
    session type is not eligible for snapshots, cold storage is disabled,
    or an error occurs.
    """

    snapshot_supported_sessions = getattr(
        settings,
        "AIWORKS_CORE_SNAPSHOT_SUPPORTED_SESSION_TYPES",
        [],
    )
    if session.session_type not in snapshot_supported_sessions:
        return None

    if not ColdStorageManager.is_enabled():
        logger.debug(
            "session_snapshot.create_snapshot | session=%s | cold storage disabled, skipping",
            session.id,
        )
        return None

    try:
        snapshot = SessionSnapshot.objects.create(session=session)
        zip_data = _build_zip(session)
        manager = ColdStorageManager.get_instance()
        manager.store(
            f"{SNAPSHOTS_FOLDER}/{session.id}",
            f"{snapshot.id}.zip",
            zip_data,
        )
        logger.info(
            "session_snapshot.create_snapshot | session=%s | snapshot=%s | zip_bytes=%d",
            session.id,
            snapshot.id,
            len(zip_data),
        )
        return snapshot

    except Exception:
        logger.exception(
            "session_snapshot.create_snapshot | session=%s | failed to create snapshot",
            session.id,
        )
        return None


def restore_snapshot(session: Session, snapshot: SessionSnapshot) -> None:
    """Restore *session* to the state captured in *snapshot*.

    Steps:
    1. Verify cold storage is enabled.
    2. Load and validate the snapshot ZIP from cold storage.
    3. Parse ``session.json`` and restore session fields; clear cached fields.
    4. Delete all current ``AttachedFile`` records for the session.
    5. Re-create ``AttachedFile`` records from the ZIP and push binary
       data back to cold storage.

    Raises an exception on any failure so the caller can surface a
    meaningful error to the user.
    """
    if snapshot.session_id != session.id:
        raise ValueError(
            f"Snapshot {snapshot.id} does not belong to session {session.id}"
        )

    if not ColdStorageManager.is_enabled():
        raise Exception("Cold storage is not enabled; snapshot cannot be restored.")

    manager = ColdStorageManager.get_instance()
    folder = f"{SNAPSHOTS_FOLDER}/{session.id}"
    file_name = f"{snapshot.id}.zip"

    if not manager.exists(folder, file_name):
        raise Exception(
            f"Snapshot ZIP not found in cold storage for snapshot {snapshot.id}."
        )

    zip_data = manager.read(folder, file_name).read()
    _restore_from_zip(session, zip_data)


def _validate_zip_integrity(zf: zipfile.ZipFile, files_meta: list) -> None:
    """Validate that the ZIP contains all required entries.

    Raises ``ValueError`` if any required entry is missing or malformed.
    """
    names = set(zf.namelist())

    if "session.json" not in names:
        raise ValueError("Snapshot ZIP is missing 'session.json'")
    try:
        json.loads(zf.read("session.json"))
    except Exception as exc:
        raise ValueError(f"Snapshot ZIP has invalid 'session.json': {exc}") from exc

    if "files.json" not in names:
        raise ValueError("Snapshot ZIP is missing 'files.json'")

    for idx in range(len(files_meta)):
        data_bin_path = f"files/{idx}/data.bin"
        if data_bin_path not in names:
            raise ValueError(f"Snapshot ZIP is missing '{data_bin_path}'")


def _restore_from_zip(session: Session, zip_data: bytes) -> None:
    """Apply full state restoration from *zip_data*."""
    with zipfile.ZipFile(io.BytesIO(zip_data), mode="r") as zf:
        # Load metadata first for integrity validation — before any destructive ops
        session_data = json.loads(zf.read("session.json"))
        files_meta = json.loads(zf.read("files.json"))

        _validate_zip_integrity(zf, files_meta)

        # --- Restore session fields ---
        for field, value in session_data.items():
            if field in _SESSION_FIELDS and hasattr(session, field):
                setattr(session, field, value if value is not None else "")
        # Clear cached computed fields
        for field in _FIELDS_TO_CLEAR_ON_RESTORE:
            setattr(session, field, "")
        session.save(
            update_fields=_SESSION_FIELDS + _FIELDS_TO_CLEAR_ON_RESTORE + ["updated_at"]
        )

        # --- Wipe existing attached files ---
        session.attached_files.all().delete()

        # --- Restore files ---
        for idx, meta in enumerate(files_meta):
            folder = f"files/{idx}"
            raw_data = zf.read(f"{folder}/data.bin")
            name = meta.get("name", f"file_{idx}")
            is_hidden = meta.get("is_hidden", False)

            af = AttachedFile(
                session=session,
                name=name,
                binary_content=raw_data,
                file_type=meta.get("file_type", AttachedFile.FILE_TYPE_INPUT),
                is_hidden=meta.get("is_hidden", is_hidden),
            )
            af.save()
