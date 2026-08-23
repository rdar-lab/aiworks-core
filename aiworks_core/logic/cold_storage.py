"""Cold storage manager for raw file data.

Provides an abstraction layer for storing binary file data outside the
database.  Three backends are supported:

* ``none``  – data is discarded (no-op); useful when cold storage is not
  needed or not configured.
* ``local`` – data is written to a local directory on disk.
* ``s3``    – data is stored in an AWS S3 bucket.

The active backend is determined by the ``coldstorage_type`` field on the
``SiteConfiguration`` singleton.  Call :func:`ColdStorageManager.get_instance`
to obtain the current backend.  Call :func:`ColdStorageManager.reset` to
clear the cached instance (required after configuration changes and in tests).
"""

import io
import logging
import os
import shutil
from abc import ABC, abstractmethod
from typing import Optional, cast

import boto3
from botocore.exceptions import ClientError
from ..models import SiteConfiguration

logger = logging.getLogger(__name__)

# Folder name used for AttachedFile binary data in cold storage.
ATTACHED_FILES_FOLDER = "attached_files"


class ColdStorageManager(ABC):
    """Abstract base class for cold storage backends.

    Usage::

        manager = ColdStorageManager.get_instance()
        manager.store("my_folder", "file.bin", b"binary data")
        data = manager.read("my_folder", "file.bin").read()
        manager.remove("my_folder", "file.bin")
    """

    _instance: Optional["ColdStorageManager"] = None

    @staticmethod
    def reset() -> None:
        """Clear the cached singleton instance.

        Must be called after changing ``SiteConfiguration.coldstorage_*``
        fields, and in test ``tearDown`` methods.
        """
        ColdStorageManager._instance = None

    @staticmethod
    def is_offline() -> bool:
        """Return ``True`` when cold storage has been marked offline."""
        return SiteConfiguration.get_solo().coldstorage_offline

    @staticmethod
    def get_instance() -> "ColdStorageManager":
        """Return the active ``ColdStorageManager`` backend.

        Raises:
            Exception: If cold storage is marked offline.
            Exception: If ``coldstorage_type`` contains an unknown value.
        """
        if ColdStorageManager.is_offline():
            raise Exception("Cold storage is offline")

        if ColdStorageManager._instance is None:
            config = SiteConfiguration.get_solo()
            manager_type = config.coldstorage_type

            if manager_type == "none":
                ColdStorageManager._instance = NoneColdStorageManager()
            elif manager_type == "local":
                ColdStorageManager._instance = LocalColdStorageManager()
            elif manager_type == "s3":
                ColdStorageManager._instance = S3StorageManager()
            else:
                raise Exception(
                    f"Unknown cold storage manager type {manager_type!r}"
                )

        return cast("ColdStorageManager", ColdStorageManager._instance)

    @staticmethod
    def is_enabled() -> bool:
        """Return ``True`` when cold storage is active (not ``none`` and not offline)."""
        try:
            cfg = SiteConfiguration.get_solo()
            if cfg.coldstorage_offline:
                return False
            return cfg.coldstorage_type != "none"
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def store(self, folder: str, file_name: str, data: bytes) -> None:
        """Persist *data* under *folder*/*file_name*."""
        raise NotImplementedError()

    @abstractmethod
    def list(self, folder: str):
        """Yield the names of all files stored under *folder*."""
        raise NotImplementedError()

    @abstractmethod
    def read(self, folder: str, file_name: str) -> io.IOBase:
        """Return a file-like object whose ``.read()`` yields the raw bytes.

        Raises:
            Exception: If the file does not exist.
        """
        raise NotImplementedError()

    @abstractmethod
    def remove(self, folder: str, file_name: str) -> None:
        """Delete *folder*/*file_name* from storage."""
        raise NotImplementedError()

    @abstractmethod
    def exists(self, folder: str, file_name: str) -> bool:
        """Return ``True`` if *folder*/*file_name* exists in storage."""
        raise NotImplementedError()

    @abstractmethod
    def clear_all(self) -> None:
        """Remove **all** objects from this storage backend.

        Intended for use in tests and administrative tooling only.
        """
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# Concrete backends
# ---------------------------------------------------------------------------


class NoneColdStorageManager(ColdStorageManager):
    """No-op backend — all operations are silently discarded."""

    def store(self, folder: str, file_name: str, data: bytes) -> None:
        pass

    def list(self, folder: str):
        return []

    def read(self, folder: str, file_name: str) -> io.IOBase:
        raise Exception(f"File {file_name!r} does not exist (none backend)")

    def remove(self, folder: str, file_name: str) -> None:
        pass

    def exists(self, folder: str, file_name: str) -> bool:
        return False

    def clear_all(self) -> None:
        pass


class LocalColdStorageManager(ColdStorageManager):
    """Local filesystem backend.

    Files are stored under *storage_location* (defaults to the value of
    ``SiteConfiguration.coldstorage_local_storage_location``, or
    ``/tmp/cold_storage/`` when that field is blank).
    """

    def __init__(self, storage_location: Optional[str] = None) -> None:
        super().__init__()

        if storage_location is None:
            storage_location = (
                    SiteConfiguration.get_solo().coldstorage_local_storage_location
                    or None
            )

        if not storage_location:
            storage_location = "/tmp/cold_storage/"

        os.makedirs(storage_location, exist_ok=True)
        self._storage_location = storage_location

    def _path(self, folder: str, file_name: Optional[str] = None) -> str:
        base = os.path.join(self._storage_location, folder)
        if file_name is not None:
            return os.path.join(base, file_name)
        return base

    def list(self, folder: str):
        folder_path = self._path(folder)
        os.makedirs(folder_path, exist_ok=True)
        return os.listdir(folder_path)

    def store(self, folder: str, file_name: str, data: bytes) -> None:
        folder_path = self._path(folder)
        os.makedirs(folder_path, exist_ok=True)
        file_path = self._path(folder, file_name)
        with open(file_path, "wb") as fh:
            fh.write(data)

    def read(self, folder: str, file_name: str) -> io.BytesIO:
        file_path = self._path(folder, file_name)
        with open(file_path, "rb") as fh:
            return io.BytesIO(fh.read())

    def exists(self, folder: str, file_name: str) -> bool:
        return os.path.exists(self._path(folder, file_name))

    def remove(self, folder: str, file_name: str) -> None:
        os.remove(self._path(folder, file_name))

    def clear_all(self) -> None:
        shutil.rmtree(self._storage_location, ignore_errors=True)
        os.makedirs(self._storage_location, exist_ok=True)


class S3StorageManager(ColdStorageManager):
    """AWS S3 backend.

    Credentials and bucket name are read from ``SiteConfiguration``:

    * ``coldstorage_aws_access_key``
    * ``coldstorage_aws_secret_key``
    * ``coldstorage_aws_region``
    * ``coldstorage_aws_endpoint``
    * ``coldstorage_s3_bucket_name``
    """

    def __init__(self, bucket_name: Optional[str] = None) -> None:
        super().__init__()

        config = SiteConfiguration.get_solo()
        self._aws_region = config.coldstorage_aws_region or "us-east-1"

        if bucket_name is None:
            bucket_name = config.coldstorage_s3_bucket_name

        self._bucket_name = bucket_name

    def _create_s3_client(self):
        config = SiteConfiguration.get_solo()
        access_key = config.coldstorage_aws_access_key or None
        secret_key = config.coldstorage_aws_secret_key or None
        endpoint_url = config.coldstorage_aws_endpoint or None
        return boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=self._aws_region,
            endpoint_url=endpoint_url,
        )

    def list(self, folder: str):
        s3 = self._create_s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self._bucket_name, Prefix=folder)
        prefix = folder + "/"
        for page in pages:
            for item in page.get("Contents", []):
                key: str = item["Key"]
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    if key and not key.endswith("/"):
                        yield key

    def store(self, folder: str, file_name: str, data: bytes) -> None:
        object_key = f"{folder}/{file_name}"
        self._create_s3_client().put_object(
            Body=data, Key=object_key, Bucket=self._bucket_name
        )

    def _find_object(self, folder: str, file_name: str):
        if not file_name:
            raise Exception("File name is empty")
        object_key = f"{folder}/{file_name}"
        return self._create_s3_client().get_object(
            Bucket=self._bucket_name, Key=object_key
        )

    def read(self, folder: str, file_name: str) -> io.IOBase:
        return self._find_object(folder, file_name)["Body"]

    def exists(self, folder: str, file_name: str) -> bool:
        if not file_name:
            return False
        try:
            return self._find_object(folder, file_name) is not None
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return False
            raise

    def remove(self, folder: str, file_name: str) -> None:
        if not file_name:
            raise Exception("File name is empty")
        object_key = f"{folder}/{file_name}"
        self._create_s3_client().delete_object(
            Bucket=self._bucket_name, Key=object_key
        )

    def clear_all(self) -> None:
        s3 = self._create_s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self._bucket_name)
        for page in pages:
            for item in page.get("Contents", []):
                s3.delete_object(Bucket=self._bucket_name, Key=item["Key"])


# ---------------------------------------------------------------------------
# Helpers for AttachedFile integration
# ---------------------------------------------------------------------------


def store_attached_file(attached_file) -> bool:
    """Store the raw bytes for *attached_file* in cold storage.

    The raw bytes must be provided via the ``binary_content`` attribute on
    the instance **before** calling this function.  After a successful write the
    attribute is cleared on the instance to prevent accidental re-storage.

    This function is a no-op when:
    - ``binary_content`` is not set or is empty, or
    - cold storage is not enabled (type is ``none`` or offline flag is set).

    Args:
        attached_file: A saved ``AttachedFile`` instance (must have a PK).
    """
    raw: Optional[bytes] = attached_file.binary_content
    if not raw:
        return False

    if not ColdStorageManager.is_enabled():
        return False

    try:
        manager = ColdStorageManager.get_instance()
        manager.store(ATTACHED_FILES_FOLDER, str(attached_file.pk), raw)
        # Clear the transient attribute now that data is safely persisted.
        attached_file.binary_content = None
        logger.debug(
            "cold_storage.store_attached_file | id=%s | stored %d bytes",
            attached_file.pk,
            len(raw),
        )
        return True
    except Exception:
        logger.exception(
            "cold_storage.store_attached_file | id=%s | failed to store to cold storage",
            attached_file.pk,
        )
        return False


def clear_attached_file(attached_file) -> bool:
    """Clears the *attached_file* data from cold storage.

    This function is a no-op when:
    - cold storage is not enabled (type is ``none`` or offline flag is set).

    Args:
        attached_file: A saved ``AttachedFile`` instance (must have a PK).
    """
    if not ColdStorageManager.is_enabled():
        return False

    try:
        manager = ColdStorageManager.get_instance()
        if manager.exists(ATTACHED_FILES_FOLDER, str(attached_file.pk)):
            manager.remove(ATTACHED_FILES_FOLDER, str(attached_file.pk))

            logger.debug(
                "cold_storage.clear_attached_file | id=%s | cleared attached file data from cold storage",
                attached_file.pk,
            )
        return True
    except Exception:
        logger.exception(
            "cold_storage.clear_attached_file | id=%s | failed to clear data from cold storage",
            attached_file.pk,
        )
        return False


def get_attached_file_data(attached_file) -> Optional[bytes]:
    """Return the raw binary data for *attached_file* from cold storage.

    Returns:
        Raw bytes, or ``None`` if the data is not available in cold storage.
    """
    if not ColdStorageManager.is_enabled():
        return None

    try:
        manager = ColdStorageManager.get_instance()
        if manager.exists(ATTACHED_FILES_FOLDER, str(attached_file.pk)):
            return manager.read(ATTACHED_FILES_FOLDER, str(attached_file.pk)).read()
    except Exception:
        logger.exception(
            "cold_storage.get_attached_file_data | id=%s | failed to read from cold storage",
            attached_file.pk,
        )
    return None
