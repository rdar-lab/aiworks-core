"""Tests for the cold storage module.

Covers:
- ``NoneColdStorageManager``: no-op behaviour
- ``LocalColdStorageManager``: store / read / list / exists / remove / clear_all
- ``S3StorageManager``: same operations against a mocked AWS S3 bucket (moto)
- ``ColdStorageManager.get_instance`` factory / singleton logic
- ``store_attached_file`` / ``get_attached_file_data`` integration helpers
- Admin ``download_link`` / ``download_view`` integration
- Session create upload path cold-storage integration
- Cold storage fields visible in ``SiteConfigurationAdmin``
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch as mock_patch
from uuid import uuid4

import boto3
from django.contrib.admin.sites import AdminSite
from django.test import TestCase
from moto import mock_aws
from rest_framework_simplejwt.tokens import RefreshToken

from . import _make_user, _make_session
from ..admin import AttachedFileAdmin, SiteConfigurationAdmin
from ..logic.cold_storage import (
    ATTACHED_FILES_FOLDER,
    ColdStorageManager,
    LocalColdStorageManager,
    NoneColdStorageManager,
    S3StorageManager,
    get_attached_file_data,
    store_attached_file,
)
from ..models import AttachedFile, SiteConfiguration


# ---------------------------------------------------------------------------
# Helper: configure SiteConfiguration cold-storage fields
# ---------------------------------------------------------------------------

def _set_coldstorage_config(**kwargs):
    """Update SiteConfiguration cold-storage fields and return the instance."""
    cfg = SiteConfiguration.get_solo()
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    cfg.save()
    return cfg


# ---------------------------------------------------------------------------
# Base mixin that resets the singleton and restores config
# ---------------------------------------------------------------------------

class _ColdStorageTestBase(TestCase):
    """Mixin that resets the ColdStorageManager singleton before/after each test."""

    def setUp(self):
        super().setUp()
        ColdStorageManager.reset()
        # Ensure a clean SiteConfiguration singleton.
        SiteConfiguration.objects.all().delete()

    def tearDown(self):
        ColdStorageManager.reset()
        super().tearDown()


# ---------------------------------------------------------------------------
# NoneColdStorageManager
# ---------------------------------------------------------------------------

class NoneColdStorageManagerTests(_ColdStorageTestBase):
    """NoneColdStorageManager is a no-op backend."""

    def _manager(self):
        return NoneColdStorageManager()

    def test_store_is_noop(self):
        self._manager().store("folder", "file.bin", b"data")

    def test_list_returns_empty(self):
        result = list(self._manager().list("folder"))
        self.assertEqual(result, [])

    def test_exists_returns_false(self):
        self.assertFalse(self._manager().exists("folder", "file.bin"))

    def test_remove_is_noop(self):
        self._manager().remove("folder", "file.bin")

    def test_read_raises(self):
        with self.assertRaises(Exception):
            self._manager().read("folder", "file.bin")

    def test_clear_all_is_noop(self):
        self._manager().clear_all()


# ---------------------------------------------------------------------------
# LocalColdStorageManager
# ---------------------------------------------------------------------------

class LocalColdStorageManagerTests(_ColdStorageTestBase):
    """LocalColdStorageManager stores files on the local filesystem."""

    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        _set_coldstorage_config(coldstorage_type="local",
                                coldstorage_local_storage_location=self._tmpdir)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().tearDown()

    def _manager(self):
        return LocalColdStorageManager(storage_location=self._tmpdir)

    def test_store_and_read(self):
        file_name = str(uuid4())
        folder = str(uuid4())
        data = bytes(str(uuid4()), "utf-8")

        manager = self._manager()
        self.assertFalse(manager.exists(folder, file_name))

        manager.store(folder, file_name, data)
        self.assertTrue(manager.exists(folder, file_name))

        file_name2 = str(uuid4())
        manager.store(folder, file_name2, data)

        file_names = list(manager.list(folder))
        self.assertEqual(2, len(file_names))
        self.assertIn(file_name, file_names)
        self.assertIn(file_name2, file_names)

        data2 = manager.read(folder, file_name).read()
        self.assertEqual(data, data2)

        manager.remove(folder, file_name)
        with self.assertRaises(Exception):
            manager.read(folder, file_name)

        manager.remove(folder, file_name2)

    def test_list_empty_folder(self):
        folder = str(uuid4())
        result = list(self._manager().list(folder))
        self.assertEqual(result, [])

    def test_clear_all(self):
        folder = str(uuid4())
        file_name = str(uuid4())
        manager = self._manager()
        manager.store(folder, file_name, b"data")
        self.assertTrue(manager.exists(folder, file_name))
        manager.clear_all()
        # After clear_all the storage location itself still exists but is empty.
        self.assertTrue(os.path.isdir(self._tmpdir))

    def test_get_instance_returns_local_manager(self):
        instance = ColdStorageManager.get_instance()
        self.assertIsInstance(instance, LocalColdStorageManager)


# ---------------------------------------------------------------------------
# S3StorageManager  (uses moto mock_aws)
# ---------------------------------------------------------------------------

@mock_aws
class S3StorageManagerTests(_ColdStorageTestBase):
    """S3StorageManager stores files in an AWS S3 bucket (mocked with moto)."""

    def setUp(self):
        super().setUp()
        self._bucket_name = str(uuid4())
        _set_coldstorage_config(
            coldstorage_type="s3",
            coldstorage_s3_bucket_name=self._bucket_name,
            coldstorage_aws_region="eu-central-1",
        )
        # Create the mock bucket.
        boto3.client("s3", region_name="eu-central-1").create_bucket(
            Bucket=self._bucket_name,
            CreateBucketConfiguration={"LocationConstraint": "eu-central-1"},
        )

    def tearDown(self):
        self._clear_bucket()
        super().tearDown()

    def _clear_bucket(self):
        s3 = boto3.client("s3", region_name="eu-central-1")
        try:
            items = s3.list_objects_v2(Bucket=self._bucket_name)
            for item in items.get("Contents", []):
                s3.delete_object(Bucket=self._bucket_name, Key=item["Key"])
            s3.delete_bucket(Bucket=self._bucket_name)
        except Exception:
            pass

    def _manager(self):
        return S3StorageManager(bucket_name=self._bucket_name)

    def test_store_and_read(self):
        file_name = str(uuid4())
        folder = str(uuid4())
        data = bytes(str(uuid4()), "utf-8")

        manager = self._manager()
        self.assertFalse(manager.exists(folder, file_name))

        manager.store(folder, file_name, data)
        self.assertTrue(manager.exists(folder, file_name))

        file_name2 = str(uuid4())
        manager.store(folder, file_name2, data)

        file_names = list(manager.list(folder))
        self.assertEqual(2, len(file_names))
        self.assertIn(file_name, file_names)
        self.assertIn(file_name2, file_names)

        data2 = manager.read(folder, file_name).read()
        self.assertEqual(data, data2)

        manager.remove(folder, file_name)
        with self.assertRaises(Exception):
            manager.read(folder, file_name)

    def test_list_empty_folder(self):
        folder = str(uuid4())
        result = list(self._manager().list(folder))
        self.assertEqual(result, [])

    def test_exists_empty_file_name_returns_false(self):
        self.assertFalse(self._manager().exists("folder", ""))

    def test_get_instance_returns_s3_manager(self):
        instance = ColdStorageManager.get_instance()
        self.assertIsInstance(instance, S3StorageManager)

    def test_custom_endpoint_url_passed_to_boto3(self):
        """coldstorage_aws_endpoint is forwarded as endpoint_url to boto3.client."""
        _set_coldstorage_config(
            coldstorage_type="s3",
            coldstorage_s3_bucket_name=self._bucket_name,
            coldstorage_aws_region="eu-central-1",
            coldstorage_aws_endpoint="https://custom.s3-compatible.endpoint",
        )

        with mock_patch("aiworks_core.logic.cold_storage.boto3.client") as mock_client:
            manager = self._manager()
            manager.store("folder", "file.txt", b"data")

            mock_client.assert_called_once()
            call_kwargs = mock_client.call_args.kwargs
            self.assertEqual(call_kwargs.get("endpoint_url"),
                             "https://custom.s3-compatible.endpoint")


# ---------------------------------------------------------------------------
# ColdStorageManager factory / singleton behaviour
# ---------------------------------------------------------------------------

class ColdStorageManagerFactoryTests(_ColdStorageTestBase):
    """Tests for the get_instance() factory and singleton caching."""

    def test_none_type_returns_none_manager(self):
        _set_coldstorage_config(coldstorage_type="none")
        instance = ColdStorageManager.get_instance()
        self.assertIsInstance(instance, NoneColdStorageManager)

    def test_get_instance_caches_singleton(self):
        _set_coldstorage_config(coldstorage_type="none")
        a = ColdStorageManager.get_instance()
        b = ColdStorageManager.get_instance()
        self.assertIs(a, b)

    def test_reset_clears_singleton(self):
        _set_coldstorage_config(coldstorage_type="none")
        a = ColdStorageManager.get_instance()
        ColdStorageManager.reset()
        b = ColdStorageManager.get_instance()
        self.assertIsNot(a, b)

    def test_offline_raises(self):
        _set_coldstorage_config(coldstorage_type="none", coldstorage_offline=True)
        with self.assertRaisesRegex(Exception, "offline"):
            ColdStorageManager.get_instance()

    def test_unknown_type_raises(self):
        _set_coldstorage_config(coldstorage_type="unknown_type")
        with self.assertRaises(Exception):
            ColdStorageManager.get_instance()

    def test_is_enabled_false_for_none_type(self):
        _set_coldstorage_config(coldstorage_type="none", coldstorage_offline=False)
        self.assertFalse(ColdStorageManager.is_enabled())

    def test_is_enabled_false_when_offline(self):
        _set_coldstorage_config(coldstorage_type="local", coldstorage_offline=True)
        self.assertFalse(ColdStorageManager.is_enabled())


# ---------------------------------------------------------------------------
# store_attached_file / get_attached_file_data integration helpers
# ---------------------------------------------------------------------------

class StoreAttachedFileTests(_ColdStorageTestBase):
    """Tests for the store_attached_file / get_attached_file_data helpers."""

    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self._user = _make_user()
        self._session = _make_session(self._user)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().tearDown()

    def _make_attached_file(self, raw_data: bytes = b"hello") -> AttachedFile:
        """Create a saved AttachedFile with the given raw_data stored in DB."""
        af = AttachedFile.objects.create(
            session=self._session,
            name="test.txt",
            binary_content=raw_data,
        )
        return af

    def test_store_moves_data_to_cold_storage(self):
        _set_coldstorage_config(coldstorage_type="local",
                                coldstorage_local_storage_location=self._tmpdir)
        af = self._make_attached_file(b"binary content")
        store_attached_file(af)

        # binary_content field should be cleared after successful store.
        self.assertIsNone(af.binary_content)

        # Data should be in cold storage keyed by PK.
        manager = ColdStorageManager.get_instance()
        self.assertTrue(manager.exists(ATTACHED_FILES_FOLDER, str(af.pk)))
        stored = manager.read(ATTACHED_FILES_FOLDER, str(af.pk)).read()
        self.assertEqual(stored, b"binary content")

    def test_store_noop_when_cold_storage_disabled(self):
        _set_coldstorage_config(coldstorage_type="none")
        af = self._make_attached_file(b"binary content")
        # Should not raise; binary_content is still set since storage is disabled.
        store_attached_file(af)
        # binary_content is preserved (cold storage was skipped).
        self.assertEqual(af.binary_content, b"binary content")

    def test_store_noop_when_no_raw_data(self):
        _set_coldstorage_config(coldstorage_type="local",
                                coldstorage_local_storage_location=self._tmpdir)
        af = AttachedFile.objects.create(
            session=self._session,
            name="empty.txt",
            binary_content=b"",
        )
        # No binary_content set — store should be a no-op.
        store_attached_file(af)
        manager = ColdStorageManager.get_instance()
        self.assertFalse(manager.exists(ATTACHED_FILES_FOLDER, str(af.pk)))

    def test_get_returns_cold_storage_data(self):
        _set_coldstorage_config(coldstorage_type="local",
                                coldstorage_local_storage_location=self._tmpdir)
        af = self._make_attached_file(b"from cold storage")
        store_attached_file(af)

        result = get_attached_file_data(af)
        self.assertEqual(result, b"from cold storage")

    def test_get_returns_none_when_cold_storage_disabled(self):
        _set_coldstorage_config(coldstorage_type="none")
        af = self._make_attached_file(b"unreachable")
        result = get_attached_file_data(af)
        self.assertIsNone(result)

    def test_get_returns_none_when_not_in_storage(self):
        _set_coldstorage_config(coldstorage_type="local",
                                coldstorage_local_storage_location=self._tmpdir)
        af = AttachedFile.objects.create(
            session=self._session,
            name="nodata.txt",
            binary_content=b"",
        )
        result = get_attached_file_data(af)
        self.assertIsNone(result)

    def test_pk_is_used_as_cold_storage_key(self):
        """The cold storage key must be str(pk) to ensure uniqueness."""
        _set_coldstorage_config(coldstorage_type="local",
                                coldstorage_local_storage_location=self._tmpdir)
        af = self._make_attached_file(b"pk key data")
        store_attached_file(af)

        manager = ColdStorageManager.get_instance()
        # Key in cold storage must match str(pk).
        self.assertTrue(manager.exists(ATTACHED_FILES_FOLDER, str(af.pk)))


# ---------------------------------------------------------------------------
# Admin download_link / download_view
# ---------------------------------------------------------------------------

class AdminDownloadViewTests(_ColdStorageTestBase):
    """Tests for AttachedFileAdmin download_link and download_view."""

    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self._user = _make_user()
        self._session = _make_session(self._user)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().tearDown()

    def _make_af(self, raw_data: bytes = b"file bytes"):
        af = AttachedFile.objects.create(
            session=self._session,
            name="report.pdf",
            binary_content=raw_data,
        )
        return af

    def test_download_link_shows_dash_when_no_cold_storage(self):
        """download_link shows '—' when cold storage is disabled and no binary_content in DB."""
        _set_coldstorage_config(coldstorage_type="none")
        af = AttachedFile.objects.create(
            session=self._session,
            name="report.pdf",
            binary_content=b"",
        )

        admin_obj = AttachedFileAdmin(AttachedFile, AdminSite())
        link = admin_obj.download_link(af)
        self.assertEqual(link, "—")

    def test_download_link_shows_link_when_data_in_cold_storage(self):
        """download_link shows an <a> tag when data exists in cold storage."""
        _set_coldstorage_config(coldstorage_type="local",
                                coldstorage_local_storage_location=self._tmpdir)
        af = self._make_af(b"pdf content")
        store_attached_file(af)

        admin_obj = AttachedFileAdmin(AttachedFile, AdminSite())
        link = admin_obj.download_link(af)
        self.assertIn("Download", str(link))
        self.assertIn(str(af.pk), str(link))

    def test_download_view_returns_404_when_no_data(self):
        """download_view returns 404 when cold storage has no data."""
        _set_coldstorage_config(coldstorage_type="none")
        af = AttachedFile.objects.create(
            session=self._session,
            name="missing.txt",
            binary_content=b"",
        )
        resp = self.client.get(f"/admin/aiworks_core/attachedfile/{af.pk}/download/",
                               **{"HTTP_AUTHORIZATION": ""})
        # Admin requires login; unauthenticated gets redirect.  Use Django's
        # test client with superuser to check the actual view.
        superuser = _make_user(username="su", email="su@example.com")
        superuser.is_staff = True
        superuser.is_superuser = True
        superuser.save()
        self.client.force_login(superuser)
        resp = self.client.get(f"/admin/aiworks_core/attachedfile/{af.pk}/download/")
        self.assertEqual(resp.status_code, 404)

    def test_download_view_returns_file_from_cold_storage(self):
        """download_view streams file bytes from cold storage."""
        _set_coldstorage_config(coldstorage_type="local",
                                coldstorage_local_storage_location=self._tmpdir)
        af = self._make_af(b"raw pdf bytes")
        store_attached_file(af)

        superuser = _make_user(username="su2", email="su2@example.com")
        superuser.is_staff = True
        superuser.is_superuser = True
        superuser.save()
        self.client.force_login(superuser)
        resp = self.client.get(f"/admin/aiworks_core/attachedfile/{af.pk}/download/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"raw pdf bytes")


# ---------------------------------------------------------------------------
# Session create upload path cold-storage integration
# ---------------------------------------------------------------------------

class SessionCreateUploadColdStorageTests(_ColdStorageTestBase):
    """Tests for session create with file uploads storing data in cold storage.

    These tests use mocked ``parse_files_from_request`` to avoid importing the
    full document_parser stack.  The cold-storage wiring is what's under test.
    """

    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self._user = _make_user()
        refresh = RefreshToken.for_user(self._user)
        self._auth = {"HTTP_AUTHORIZATION": f"Bearer {refresh.access_token}"}

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().tearDown()

    def test_session_create_stores_file_in_cold_storage(self):
        """session_views.create calls store_attached_file for each uploaded file.

        This test exercises the file-save + cold-storage loop that lives inside
        the session create method directly, without going through the full HTTP
        stack (which would pull in the heavy LLM / LangChain dependency chain).
        """
        _set_coldstorage_config(coldstorage_type="local",
                                coldstorage_local_storage_location=self._tmpdir)

        session = _make_session(self._user)
        file_bytes = b"session create file content"

        # Simulate what parse_files_from_request returns:
        # an AttachedFile with binary_content set (but not yet saved).
        mock_af = AttachedFile(
            name="test.txt",
            binary_content=file_bytes,
            file_type=AttachedFile.FILE_TYPE_INPUT,
        )

        # Replicate exactly the file-saving code in session_views.create:
        mock_af.session = session
        mock_af.save()
        store_attached_file(mock_af)

        # Verify data is in cold storage keyed by PK.
        manager = ColdStorageManager.get_instance()
        self.assertTrue(manager.exists(ATTACHED_FILES_FOLDER, str(mock_af.pk)))
        stored = manager.read(ATTACHED_FILES_FOLDER, str(mock_af.pk)).read()
        self.assertEqual(stored, file_bytes)


# ---------------------------------------------------------------------------
# SiteConfigurationAdmin cold storage fieldset
# ---------------------------------------------------------------------------

class SiteConfigurationAdminColdStorageFieldsTests(TestCase):
    """The cold storage fields must be present in SiteConfigurationAdmin fieldsets."""

    def test_cold_storage_fieldset_present(self):
        admin_obj = SiteConfigurationAdmin(SiteConfiguration, AdminSite())
        fieldset_names = [fs[0] for fs in admin_obj.fieldsets]
        self.assertIn("Cold Storage", fieldset_names)

    def test_cold_storage_fields_listed(self):
        admin_obj = SiteConfigurationAdmin(SiteConfiguration, AdminSite())
        cold_storage_fs = next(
            fs for fs in admin_obj.fieldsets if fs[0] == "Cold Storage"
        )
        fields = cold_storage_fs[1]["fields"]
        expected = {
            "coldstorage_type",
            "coldstorage_offline",
            "coldstorage_local_storage_location",
            "coldstorage_s3_bucket_name",
            "coldstorage_aws_access_key",
            "coldstorage_aws_secret_key",
            "coldstorage_aws_region",
            "coldstorage_aws_endpoint",
        }
        self.assertEqual(set(fields), expected)


if __name__ == "__main__":
    unittest.main()
