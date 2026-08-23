"""Tests for purge_unused_images logic."""
import json
import uuid

from django.test import TestCase

from aiworks_core.logic.purge_unused_images import (
    _extract_uuid_png_refs,
    _is_uuid_png_file,
    purge_unused_images,
)
from aiworks_core.models import AttachedFile, Session
from . import _make_user


class TestIsUuidPngFile(TestCase):
    """Unit tests for _is_uuid_png_file helper."""

    def test_valid_uuid_png(self):
        valid = f"{uuid.uuid4()}.png"
        self.assertTrue(_is_uuid_png_file(valid))

    def test_valid_uuid_png_uppercase(self):
        valid = f"{uuid.uuid4()}.PNG"
        self.assertTrue(_is_uuid_png_file(valid))

    def test_invalid_not_uuid(self):
        self.assertFalse(_is_uuid_png_file("chart.png"))
        self.assertFalse(_is_uuid_png_file("image.jpg"))
        self.assertFalse(_is_uuid_png_file("photo.png"))

    def test_invalid_random_string(self):
        self.assertFalse(_is_uuid_png_file("abc123def456.png"))
        self.assertFalse(_is_uuid_png_file("not-a-uuid.png"))

    def test_invalid_missing_extension(self):
        invalid_uuid = str(uuid.uuid4())
        self.assertFalse(_is_uuid_png_file(invalid_uuid))

    def test_invalid_partial_uuid(self):
        self.assertFalse(_is_uuid_png_file("a1b2c3d4.png"))


class TestExtractUuidPngRefs(TestCase):
    """Unit tests for _extract_uuid_png_refs helper."""

    def test_extracts_attachment_reference(self):
        uuid_str = str(uuid.uuid4())
        text = f'{{"image_url": "attachment://{uuid_str}.png"}}'
        refs = _extract_uuid_png_refs(text)
        self.assertIn(f"{uuid_str}.png", refs)

    def test_extracts_image_file_name_double_quoted(self):
        uuid_str = str(uuid.uuid4())
        text = f'{{"image_file_name": "{uuid_str}.png"}}'
        refs = _extract_uuid_png_refs(text)
        self.assertIn(f"{uuid_str}.png", refs)

    def test_extracts_image_file_name_single_quoted(self):
        uuid_str = str(uuid.uuid4())
        text = f"{{'image_file_name': '{uuid_str}.png'}}"
        refs = _extract_uuid_png_refs(text)
        self.assertIn(f"{uuid_str}.png", refs)

    def test_extracts_from_html(self):
        uuid_str = str(uuid.uuid4())
        text = f'<img src="attachment://{uuid_str}.png" />'
        refs = _extract_uuid_png_refs(text)
        self.assertIn(f"{uuid_str}.png", refs)

    def test_extracts_multiple_references(self):
        uuid1 = str(uuid.uuid4())
        uuid2 = str(uuid.uuid4())
        text = f'{{"a": "attachment://{uuid1}.png", "b": "attachment://{uuid2}.png"}}'
        refs = _extract_uuid_png_refs(text)
        self.assertIn(f"{uuid1}.png", refs)
        self.assertIn(f"{uuid2}.png", refs)

    def test_case_insensitive(self):
        uuid_str = str(uuid.uuid4()).upper()
        text = f'{{"image_url": "attachment://{uuid_str}.PNG"}}'
        refs = _extract_uuid_png_refs(text)
        self.assertIn(f"{uuid_str.lower()}.png", refs)

    def test_no_references(self):
        text = '{"posts": []}'
        refs = _extract_uuid_png_refs(text)
        self.assertEqual(refs, set())


class TestPurgeUnusedImages(TestCase):
    """Integration tests for purge_unused_images."""

    def setUp(self):
        self.user = _make_user()
        self.session = Session.objects.create(
            id="purge_unused_imgs",
            user=self.user,
            session_type="test",
            session_title="Test",
        )

    def _make_output_file(self, name: str, content: str = "") -> AttachedFile:
        return AttachedFile.objects.create(
            session=self.session,
            file_type=AttachedFile.FILE_TYPE_OUTPUT,
            name=name,
            binary_content=content.encode("utf-8") if content else None,
        )

    def _make_uuid_png_file(self) -> tuple[AttachedFile, str]:
        image_name = f"{uuid.uuid4()}.png"
        af = AttachedFile.objects.create(
            session=self.session,
            file_type=AttachedFile.FILE_TYPE_OUTPUT,
            name=image_name,
            binary_content=b"fake png data",
        )
        return af, image_name

    def test_no_files_returns_zero(self):
        count = purge_unused_images(self.session)
        self.assertEqual(count, 0)

    def test_no_uuid_png_files_returns_zero(self):
        self._make_output_file("index.html", "<html></html>")
        count = purge_unused_images(self.session)
        self.assertEqual(count, 0)

    def test_referenced_image_is_not_deleted(self):
        used_image_name = f"{uuid.uuid4()}.png"
        self._make_output_file(used_image_name)
        posts_json = f'{{"posts": [{{"title": "Test", "image_url": "attachment://{used_image_name}"}}]}}'
        self._make_output_file("posts.json", posts_json)

        count = purge_unused_images(self.session)
        self.assertEqual(count, 0)
        self.assertTrue(
            AttachedFile.objects.filter(name=used_image_name).exists()
        )

    def test_unreferenced_uuid_png_is_deleted(self):
        orphaned_af, orphaned_name = self._make_uuid_png_file()
        posts_json = '{"posts": []}'
        self._make_output_file("posts.json", posts_json)

        count = purge_unused_images(self.session)
        self.assertEqual(count, 1)
        self.assertFalse(
            AttachedFile.objects.filter(name=orphaned_name).exists()
        )

    def test_mixed_used_and_unused_images(self):
        used_name = f"{uuid.uuid4()}.png"
        orphaned_af, orphaned_name = self._make_uuid_png_file()
        posts_json = f'{{"posts": [{{"title": "Test", "image_url": "attachment://{used_name}"}}]}}'
        self._make_output_file("posts.json", posts_json)
        self._make_output_file(used_name, "used image data")

        count = purge_unused_images(self.session)
        self.assertEqual(count, 1)
        self.assertFalse(
            AttachedFile.objects.filter(name=orphaned_name).exists()
        )
        self.assertTrue(
            AttachedFile.objects.filter(name=used_name).exists()
        )

    def test_presentation_json_image_references(self):
        orphaned_af, orphaned_name = self._make_uuid_png_file()
        presentation_json = json.dumps({
            "slides": [{
                "elements": [
                    {"type": "image", "image_file_name": orphaned_name}
                ]
            }]
        })
        self._make_output_file("presentation.json", presentation_json)

        count = purge_unused_images(self.session)
        self.assertEqual(count, 0)

    def test_multiple_unreferenced_images_deleted(self):
        _, name1 = self._make_uuid_png_file()
        _, name2 = self._make_uuid_png_file()
        self._make_output_file("posts.json", '{"posts": []}')

        count = purge_unused_images(self.session)
        self.assertEqual(count, 2)
        self.assertFalse(AttachedFile.objects.filter(name=name1).exists())
        self.assertFalse(AttachedFile.objects.filter(name=name2).exists())

    def test_session_without_pk_returns_zero(self):
        class FakeSession:
            pk = None
        count = purge_unused_images(FakeSession())
        self.assertEqual(count, 0)

    def test_research_report_reference(self):
        orphaned_af, orphaned_name = self._make_uuid_png_file()
        self.session.agent_result = f'Generated images: attachment://{orphaned_name}'
        self.session.save()

        count = purge_unused_images(self.session)
        self.assertEqual(count, 0)

    def test_html_file_reference(self):
        orphaned_af, orphaned_name = self._make_uuid_png_file()
        html_content = f'<html><body><img src="attachment://{orphaned_name}" /></body></html>'
        self._make_output_file("index.html", html_content)

        count = purge_unused_images(self.session)
        self.assertEqual(count, 0)

    def test_arbitrary_json_file_reference(self):
        orphaned_af, orphaned_name = self._make_uuid_png_file()
        arbitrary_json = json.dumps({
            "hero_image": f"attachment://{orphaned_name}",
            "slides": []
        })
        self._make_output_file("content.json", arbitrary_json)

        count = purge_unused_images(self.session)
        self.assertEqual(count, 0)

    def test_case_insensitive_reference(self):
        orphaned_af, orphaned_name = self._make_uuid_png_file()
        posts_json = f'{{"posts": [{{"title": "Test", "image_url": "attachment://{orphaned_name.upper()}"}}]}}'
        self._make_output_file("posts.json", posts_json)

        count = purge_unused_images(self.session)
        self.assertEqual(count, 0)
