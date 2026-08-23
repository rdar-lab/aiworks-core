import base64
import shutil
import tempfile
from unittest.mock import MagicMock, patch

from ..models import AttachedFile, SiteConfiguration, TextAttachedFile
from ..logic.cold_storage import ColdStorageManager
from ..logic.files import (
    can_extract_text,
    get_binary_data,
    get_text_content,
    offload_to_cold_storage,
    set_binary_content,
)
from django.test import TestCase
from . import _make_user, _make_session
from .test_cold_storage import _set_coldstorage_config


class CanExtractTextTestCase(TestCase):
    def test_extractable_types(self):
        extractable = [
            "document.pdf", "document.docx", "document.doc", "document.rtf",
            "document.xlsx", "document.xls", "document.ods",
            "document.pptx", "document.ppt", "document.odp",
            "document.odt", "document.txt", "document.md", "document.csv",
        ]
        for f in extractable:
            with self.subTest(filename=f):
                self.assertTrue(can_extract_text(f))

    def test_non_extractable_types(self):
        non_extractable = [
            "photo.png", "photo.jpg", "photo.jpeg", "photo.gif", "photo.webp",
            "audio.mp3", "audio.wav", "audio.flac",
            "video.mp4", "video.avi", "video.mov",
            "archive.zip", "archive.tar", "archive.gz",
        ]
        for f in non_extractable:
            with self.subTest(filename=f):
                self.assertFalse(can_extract_text(f))


class GetBinaryDataTestCase(TestCase):
    def test_db_binary_content_takes_precedence(self):
        af = MagicMock()
        af.pk = 1
        af.binary_content = b"db bytes"

        result = get_binary_data(af)
        self.assertEqual(result, b"db bytes")

    def test_cold_storage_fallback_when_db_empty(self):
        af = MagicMock()
        af.pk = 42
        af.binary_content = None

        mock_manager = MagicMock()
        mock_manager.exists.return_value = True
        mock_manager.read.return_value.read.return_value = b"cold storage bytes"

        with patch("aiworks_core.logic.files.ColdStorageManager.is_enabled", return_value=True):
            with patch("aiworks_core.logic.files.ColdStorageManager.get_instance", return_value=mock_manager):
                with patch("aiworks_core.logic.cold_storage.ATTACHED_FILES_FOLDER", "attached_files"):
                    result = get_binary_data(af)

        self.assertEqual(result, b"cold storage bytes")
        mock_manager.exists.assert_called_once_with("attached_files", "42")

    def test_returns_none_when_nowhere(self):
        af = MagicMock()
        af.pk = 99
        af.binary_content = None

        with patch("aiworks_core.logic.files.ColdStorageManager.is_enabled", return_value=False):
            result = get_binary_data(af)

        self.assertIsNone(result)


class SetBinaryContentTestCase(TestCase):
    def test_sets_binary_and_clears_text_attached_file(self):
        af = MagicMock()
        af.pk = 1
        af.binary_content = b"old"
        af.text_attached_file.delete = MagicMock()
        af.save = MagicMock()

        set_binary_content(af, b"new")

        af.text_attached_file.delete.assert_called_once()
        af.save.assert_called_once_with(update_fields=["binary_content"])
        self.assertEqual(af.binary_content, b"new")

    def test_works_when_no_text_attached_file(self):
        class FakeAF:
            pk = 1
            binary_content = b"old"
            _deleted_taf = False

            @property
            def text_attached_file(self):
                raise TextAttachedFile.DoesNotExist("no text_attached_file")

            def save(self, **kwargs):
                pass

        af = FakeAF()
        set_binary_content(af, b"new")
        self.assertEqual(af.binary_content, b"new")


class GetTextContentTestCase(TestCase):
    def test_returns_cached_text_attached_file(self):
        class FakeTAF:
            content = "cached text"

        class FakeAF:
            pk = 1

            @property
            def text_attached_file(self):
                return FakeTAF()

        af = FakeAF()
        with patch("aiworks_core.logic.files.get_binary_data", return_value=b"binary"):
            result = get_text_content(af)
        self.assertEqual(result, "cached text")

    def test_non_extractable_returns_base64(self):
        class FakeAF:
            pk = 1
            name = "test.png"
            binary_content = b"\x89PNG\r\n\x1a\n"

            @property
            def text_attached_file(self):
                raise TextAttachedFile.DoesNotExist("no text_attached_file")

        af = FakeAF()
        with patch("aiworks_core.logic.files.get_binary_data", return_value=b"\x89PNG\r\n\x1a\n"):
            with patch("aiworks_core.logic.files.can_extract_text", return_value=False):
                result = get_text_content(af)

        self.assertIsInstance(result, str)
        self.assertEqual(base64.b64decode(result), b"\x89PNG\r\n\x1a\n")


class OffloadToColdStorageTestCase(TestCase):
    def setUp(self):
        super().setUp()
        ColdStorageManager.reset()
        SiteConfiguration.objects.all().delete()
        self._tmpdir = tempfile.mkdtemp()
        _set_coldstorage_config(
            coldstorage_type="local",
            coldstorage_local_storage_location=self._tmpdir,
        )

    def tearDown(self):
        ColdStorageManager.reset()
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().tearDown()

    def test_disabled_cold_storage_returns_empty_result(self):
        _set_coldstorage_config(coldstorage_type="none")
        ColdStorageManager.reset()
        result = offload_to_cold_storage()
        self.assertEqual(result, {"offloaded": 0, "failed": 0, "skipped": 0})

    def test_offloads_files_and_clears_binary_content(self):
        user = _make_user()
        session = _make_session(user, session_id="sess-offload-1")
        af1 = AttachedFile.objects.create(session=session, name="file1.bin", binary_content=b"data1", file_type="input")
        af2 = AttachedFile.objects.create(session=session, name="file2.bin", binary_content=b"data2", file_type="input")

        result = offload_to_cold_storage()

        self.assertEqual(result["offloaded"], 2)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["skipped"], 0)

        af1.refresh_from_db()
        af2.refresh_from_db()
        self.assertIsNone(af1.binary_content)
        self.assertIsNone(af2.binary_content)

        manager = ColdStorageManager.get_instance()
        self.assertTrue(manager.exists("attached_files", str(af1.pk)))
        self.assertTrue(manager.exists("attached_files", str(af2.pk)))
        self.assertEqual(manager.read("attached_files", str(af1.pk)).read(), b"data1")
        self.assertEqual(manager.read("attached_files", str(af2.pk)).read(), b"data2")

    def test_skips_files_with_empty_binary_content(self):
        user = _make_user()
        session = _make_session(user, session_id="sess-offload-2")
        af = AttachedFile.objects.create(session=session, name="empty.bin", binary_content=b"", file_type="input")

        result = offload_to_cold_storage()

        self.assertEqual(result["offloaded"], 0)
        self.assertEqual(result["skipped"], 0)

        af.refresh_from_db()
        self.assertEqual(af.binary_content, b"")

    def test_counts_failed_offloads(self):
        user = _make_user()
        session = _make_session(user, session_id="sess-offload-3")
        _af = AttachedFile.objects.create(session=session, name="fail.bin", binary_content=b"data", file_type="input")

        with patch("aiworks_core.logic.files.store_attached_file", side_effect=Exception("store error")):
            result = offload_to_cold_storage()

        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["offloaded"], 0)