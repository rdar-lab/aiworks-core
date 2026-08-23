"""
Knowledge Base views tests for the Ai-Works Core API.
"""

import io
from unittest.mock import patch

from rest_framework import status
from rest_framework.test import APITestCase

from . import (
    _make_user,
    _auth_header,
)
from ..models import (
    KnowledgeBase,
    AttachedFile,
)


class KnowledgeBaseAPITests(APITestCase):
    def setUp(self):
        self.user = _make_user()
        self.other_user = _make_user(username="other", email="other@example.com")
        self.auth = _auth_header(self.user)

    def _create_kb(self, name="Test KB", user=None):
        return KnowledgeBase.objects.create(user=user or self.user, name=name)

    def test_list_returns_only_own_kbs(self):
        self._create_kb("My KB")
        self._create_kb("Other KB", user=self.other_user)
        response = self.client.get("/api/knowledge-bases/", **self.auth)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        results = (
            response.data
            if isinstance(response.data, list)
            else response.data.get("results", response.data)
        )
        names = [kb["name"] for kb in results]
        self.assertIn("My KB", names)
        self.assertNotIn("Other KB", names)

    def test_create_kb(self):
        response = self.client.post(
            "/api/knowledge-bases/",
            {"name": "New KB"},
            format="json",
            **self.auth,
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["name"], "New KB")
        self.assertEqual(response.data["filesCount"], 0)

    def test_update_kb_name(self):
        kb = self._create_kb("Old Name")
        response = self.client.patch(
            f"/api/knowledge-bases/{kb.id}/",
            {"name": "New Name"},
            format="json",
            **self.auth,
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["name"], "New Name")

    def test_cannot_update_other_users_kb(self):
        kb = self._create_kb("Other KB", user=self.other_user)
        response = self.client.patch(
            f"/api/knowledge-bases/{kb.id}/",
            {"name": "Hacked"},
            format="json",
            **self.auth,
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_delete_kb(self):
        kb = self._create_kb()
        response = self.client.delete(f"/api/knowledge-bases/{kb.id}/", **self.auth)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(KnowledgeBase.objects.filter(id=kb.id).exists())

    def test_upload_file_to_kb(self):
        kb = self._create_kb()
        file_data = io.BytesIO(b"Hello knowledge base")
        file_data.name = "notes.txt"
        response = self.client.post(
            f"/api/knowledge-bases/{kb.id}/upload_files/",
            {"files": file_data},
            format="multipart",
            **self.auth,
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(len(response.data["files"]), 1)
        self.assertEqual(response.data["files"][0]["name"], "notes.txt")

    def test_upload_file_to_kb_creates_attached_file(self):
        kb = self._create_kb()
        file_content = b"KB binary content"
        file_data = io.BytesIO(file_content)
        file_data.name = "kb_doc.txt"
        response = self.client.post(
            f"/api/knowledge-bases/{kb.id}/upload_files/",
            {"files": file_data},
            format="multipart",
            **self.auth,
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        af = AttachedFile.objects.get(knowledge_base=kb)
        self.assertEqual(af.name, "kb_doc.txt")

    def test_upload_disallows_invalid_extension(self):
        kb = self._create_kb()
        file_data = io.BytesIO(b"exec")
        file_data.name = "evil.exe"
        response = self.client.post(
            f"/api/knowledge-bases/{kb.id}/upload_files/",
            {"files": file_data},
            format="multipart",
            **self.auth,
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_upload_file_overrides_existing_kb_file(self):
        kb = self._create_kb()
        file1 = io.BytesIO(b"first content")
        file1.name = "dup.txt"
        resp1 = self.client.post(
            f"/api/knowledge-bases/{kb.id}/upload_files/",
            {"files": file1},
            format="multipart",
            **self.auth,
        )
        self.assertEqual(resp1.status_code, status.HTTP_201_CREATED)
        af = AttachedFile.objects.get(knowledge_base=kb, name="dup.txt")
        self.assertEqual(af.name, "dup.txt")

        file2 = io.BytesIO(b"second content")
        file2.name = "dup.txt"
        resp2 = self.client.post(
            f"/api/knowledge-bases/{kb.id}/upload_files/",
            {"files": file2},
            format="multipart",
            **self.auth,
        )
        self.assertEqual(resp2.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            AttachedFile.objects.filter(knowledge_base=kb, name="dup.txt").count(), 1
        )

    def test_upload_too_large_file_returns_clear_error(self):
        kb = self._create_kb()
        with patch("aiworks_core.logic.document_parser.MAX_FILE_SIZE_BYTES", 1):
            file_data = io.BytesIO(b"abcdef")
            file_data.name = "big.txt"
            resp = self.client.post(
                f"/api/knowledge-bases/{kb.id}/upload_files/",
                {"files": file_data},
                format="multipart",
                **self.auth,
            )
        self.assertEqual(resp.status_code, status.HTTP_207_MULTI_STATUS)
        self.assertIn("parse_errors", resp.data)
        self.assertEqual(len(resp.data["parse_errors"]), 1)
        self.assertIn("too large", resp.data["parse_errors"][0]["error"])

    def test_remove_file_from_kb(self):
        kb = self._create_kb()
        f = AttachedFile.objects.create(knowledge_base=kb, name="doc.txt", binary_content=b"x")
        response = self.client.delete(
            f"/api/knowledge-bases/{kb.id}/files/{f.id}/",
            **self.auth,
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(AttachedFile.objects.filter(id=f.id).exists())

    def test_remove_file_from_other_users_kb_returns_404(self):
        kb = self._create_kb("Other KB", user=self.other_user)
        f = AttachedFile.objects.create(knowledge_base=kb, name="doc.txt", binary_content=b"x")
        response = self.client.delete(
            f"/api/knowledge-bases/{kb.id}/files/{f.id}/",
            **self.auth,
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_create_duplicate_kb_name_returns_400_with_message(self):
        self._create_kb("My KB")
        response = self.client.post(
            "/api/knowledge-bases/",
            {"name": "My KB"},
            format="json",
            **self.auth,
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(
            response.data["error"], "A knowledge base with this name already exists."
        )

    def test_rename_to_existing_kb_name_returns_400_with_message(self):
        self._create_kb("Existing KB")
        kb = self._create_kb("Other KB")
        response = self.client.patch(
            f"/api/knowledge-bases/{kb.id}/",
            {"name": "Existing KB"},
            format="json",
            **self.auth,
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(
            response.data["error"], "A knowledge base with this name already exists."
        )

    def test_different_users_can_create_kb_with_same_name(self):
        self._create_kb("Shared Name")
        other_auth = _auth_header(self.other_user)
        response = self.client.post(
            "/api/knowledge-bases/",
            {"name": "Shared Name"},
            format="json",
            **other_auth,
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_unauthenticated_cannot_access_kbs(self):
        response = self.client.get("/api/knowledge-bases/")
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
