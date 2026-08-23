"""Tests for memory generation feature."""

from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import patch

from django.test import TestCase
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient
from rest_framework.test import APITestCase

from . import _auth_header, _make_session, _make_user
from ..logic.memory import run_memory_generation_job
from ..logic.session_helper import get_effective_context, get_effective_user_background
from ..models import MemoryEntry, Session, SiteConfiguration


class MemoryEntryModelTests(TestCase):
    """Tests for the MemoryEntry model."""

    def setUp(self):
        self.user = _make_user()

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        Session.objects.all().delete()
        try:
            self.user.delete()
        except ValueError:
            pass

    def test_create_memory_entry(self):
        entry = MemoryEntry.objects.create(
            user=self.user, content="User prefers bold action."
        )
        self.assertEqual(entry.user, self.user)
        self.assertEqual(entry.content, "User prefers bold action.")
        self.assertIsNotNone(entry.created_at)

    def test_memory_entry_str(self):
        entry = MemoryEntry.objects.create(user=self.user, content="A test memory.")
        self.assertIn("A test memory", str(entry))

    def test_memory_entry_ordering(self):
        e1 = MemoryEntry.objects.create(user=self.user, content="First memory.")
        e2 = MemoryEntry.objects.create(user=self.user, content="Second memory.")
        entries = list(MemoryEntry.objects.filter(user=self.user))
        # Most recent first
        self.assertEqual(entries[0], e2)
        self.assertEqual(entries[1], e1)

    def test_memory_deleted_with_user(self):
        MemoryEntry.objects.create(user=self.user, content="Will be deleted.")
        self.assertEqual(MemoryEntry.objects.filter(user=self.user).count(), 1)
        self.user.delete()
        self.assertEqual(MemoryEntry.objects.count(), 0)


class MemoryAPIViewTests(TestCase):
    """Tests for the /api/memory/ endpoints."""

    def setUp(self):
        self.user = _make_user()
        self.other_user = _make_user(username="other", email="other@example.com")
        self.client = APIClient()
        self.client.credentials(**_auth_header(self.user))

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        self.user.delete()
        self.other_user.delete()

    def test_list_memories_empty(self):
        response = self.client.get("/api/memory/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_list_memories_returns_user_memories(self):
        MemoryEntry.objects.create(user=self.user, content="Memory A")
        MemoryEntry.objects.create(user=self.user, content="Memory B")
        MemoryEntry.objects.create(user=self.other_user, content="Other user memory")

        response = self.client.get("/api/memory/")
        self.assertEqual(response.status_code, 200)
        contents = [m["content"] for m in response.json()]
        self.assertIn("Memory A", contents)
        self.assertIn("Memory B", contents)
        self.assertNotIn("Other user memory", contents)

    def test_delete_memory(self):
        entry = MemoryEntry.objects.create(user=self.user, content="To delete")
        response = self.client.delete(f"/api/memory/{entry.id}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})
        self.assertFalse(MemoryEntry.objects.filter(pk=entry.id).exists())

    def test_delete_other_users_memory_returns_404(self):
        entry = MemoryEntry.objects.create(user=self.other_user, content="Not mine")
        response = self.client.delete(f"/api/memory/{entry.id}/")
        self.assertEqual(response.status_code, 404)
        self.assertTrue(MemoryEntry.objects.filter(pk=entry.id).exists())

    def test_memory_list_requires_authentication(self):
        anon_client = APIClient()
        response = anon_client.get("/api/memory/")
        self.assertEqual(response.status_code, 401)


class GetEffectiveContextWithMemoriesTests(TestCase):
    """Tests that get_effective_context / get_effective_user_background
    use llm_generated_background (lazily generated via full LLM call)."""

    def setUp(self):
        self.user = _make_user()
        self.session = _make_session(self.user)
        self.session.include_user_context = False
        self.session.save()

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        Session.objects.all().delete()
        self.user.delete()

    # ------------------------------------------------------------------
    # get_effective_user_background — called directly
    # ------------------------------------------------------------------

    def test_get_effective_user_background_returns_empty_when_no_data(self):
        """Nothing to synthesise → empty string, no LLM call."""
        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            result = get_effective_user_background(self.user)
        mock_llm.assert_not_called()
        self.assertEqual(result, "")

    def test_get_effective_user_background_returns_cached_without_llm(self):
        """Cached background is returned without calling the LLM."""
        self.user.llm_generated_background = "Cached background."
        self.user.save(update_fields=["llm_generated_background"])
        MemoryEntry.objects.create(user=self.user, content="Some memory.")
        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            result = get_effective_user_background(self.user)
        mock_llm.assert_not_called()
        self.assertEqual(result, "Cached background.")

    def test_get_effective_user_background_calls_llm_when_background_empty(self):
        """Empty cache → LLM is called end-to-end, result persisted."""
        self.user.profile_context = "Senior executive."
        self.user.llm_generated_background = ""
        self.user.save(update_fields=["profile_context", "llm_generated_background"])
        MemoryEntry.objects.create(user=self.user, content="Prefers bold action.")

        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            mock_llm.return_value = "LLM synthesised background."
            result = get_effective_user_background(self.user)

        mock_llm.assert_called_once()
        self.assertEqual(result, "LLM synthesised background.")
        self.user.refresh_from_db()
        self.assertEqual(
            self.user.llm_generated_background, "LLM synthesised background."
        )

    # ------------------------------------------------------------------
    # get_effective_context — exercises the full chain via session
    # ------------------------------------------------------------------

    def test_no_memories_no_context(self):
        self.session.include_user_context = False
        result = get_effective_context(self.session)
        self.assertEqual(result, "")

    def test_pregenerated_background_used_when_include_user_context(self):
        self.user.llm_generated_background = "Pregenerated background text."
        self.user.save(update_fields=["llm_generated_background"])
        MemoryEntry.objects.create(user=self.user, content="User is a CTO.")
        self.session.include_user_context = True
        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            result = get_effective_context(self.session)
        mock_llm.assert_not_called()
        self.assertIn("Pregenerated background text.", result)
        self.assertIn("User Background", result)

    def test_background_not_included_when_not_include_user_context(self):
        self.user.llm_generated_background = "Should not appear."
        self.user.save(update_fields=["llm_generated_background"])
        MemoryEntry.objects.create(user=self.user, content="User is a CTO.")
        self.session.include_user_context = False
        result = get_effective_context(self.session)
        self.assertNotIn("Should not appear.", result)

    def test_effective_context_calls_llm_lazily_when_background_empty(self):
        """Full chain: empty cache → LLM called → background included in context."""
        self.user.profile_context = "Senior executive at a tech company."
        self.user.llm_generated_background = ""
        self.user.save(update_fields=["profile_context", "llm_generated_background"])
        self.session.include_user_context = True

        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            mock_llm.return_value = "Synthesised background."
            result = get_effective_context(self.session)

        mock_llm.assert_called_once()
        self.assertIn("Synthesised background.", result)
        self.assertIn("User Background", result)

    def test_no_background_when_no_profile_and_no_memories(self):
        """When there is nothing to synthesise, get_effective_context returns empty."""
        self.user.profile_context = ""
        self.user.llm_generated_background = ""
        self.user.save(update_fields=["profile_context", "llm_generated_background"])
        self.session.include_user_context = True
        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            result = get_effective_context(self.session)
        mock_llm.assert_not_called()
        self.assertEqual(result.strip(), "")


class MemoryGenerationLogicTests(TestCase):
    """Tests for the memory generation functions."""

    def setUp(self):
        SiteConfiguration.objects.all().delete()
        self.user = _make_user()
        self.session = _make_session(self.user)

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        Session.objects.all().delete()
        self.user.delete()
        SiteConfiguration.objects.all().delete()

    def test_run_memory_generation_job_disabled(self):
        """When memory_generation_enabled=False, the job should do nothing (no sessions in window)."""
        # run_memory_generation_job no longer checks the enabled flag — that's apps.py's job.
        # Pass a future since_dt so no sessions qualify, simulating a no-op.
        run_memory_generation_job(timezone.now() + timedelta(days=1))
        self.assertEqual(MemoryEntry.objects.count(), 0)

    def test_run_memory_generation_job_no_sessions(self):
        """Job should not fail when there are no sessions since last run."""
        # Delete session created in setUp so no sessions qualify
        Session.objects.all().delete()

        run_memory_generation_job(timezone.now() - timedelta(hours=1))

        self.assertEqual(MemoryEntry.objects.count(), 0)

    def test_run_memory_generation_job_creates_memories(self):
        """Job should extract and store new memories from sessions."""
        with patch(
            "aiworks_core.logic.memory.invoke_llm", new_callable=AsyncMock
        ) as mock_invoke:
            mock_invoke.return_value = [
                "User is a CTO.",
                "Prefers data-driven decisions.",
            ]

            run_memory_generation_job(timezone.now() - timedelta(days=1))

        memories = list(
            MemoryEntry.objects.filter(user=self.user).values_list("content", flat=True)
        )
        self.assertIn("User is a CTO.", memories)
        self.assertIn("Prefers data-driven decisions.", memories)

    def test_run_memory_generation_triggers_compression_when_too_many(self):
        """Job should compress memories when count exceeds memory_max_entries."""
        cfg = SiteConfiguration.get_solo()
        cfg.memory_max_entries = 5
        cfg.save()

        # Pre-create 5 existing memories (at max)
        for i in range(5):
            MemoryEntry.objects.create(user=self.user, content=f"Existing memory {i}")

        llm_responses = [
            ["New memory that exceeds limit."],  # extraction call
            ["Compressed memory 1.", "Compressed memory 2."],  # compression call
        ]
        call_idx = [0]

        async def _multi_response(*args, **kwargs):
            idx = min(call_idx[0], len(llm_responses) - 1)
            call_idx[0] += 1
            return llm_responses[idx]

        with patch("aiworks_core.logic.memory.invoke_llm", side_effect=_multi_response):
            run_memory_generation_job(timezone.now() - timedelta(days=1))

        # After compression, memories should be the compressed set
        memories = list(
            MemoryEntry.objects.filter(user=self.user).values_list("content", flat=True)
        )
        self.assertIn("Compressed memory 1.", memories)
        self.assertIn("Compressed memory 2.", memories)
        # Old memories should be gone
        self.assertNotIn("Existing memory 0", memories)


class SiteConfigurationMemoryFieldsTests(TestCase):
    """Tests for the memory generation fields on SiteConfiguration."""

    def setUp(self):
        SiteConfiguration.objects.all().delete()

    def test_memory_generation_disabled_by_default(self):
        cfg = SiteConfiguration.get_solo()
        self.assertFalse(cfg.memory_generation_enabled)

    def test_memory_last_run_null_by_default(self):
        cfg = SiteConfiguration.get_solo()
        self.assertIsNone(cfg.memory_last_run)

    def test_memory_max_entries_default(self):
        cfg = SiteConfiguration.get_solo()
        self.assertEqual(cfg.memory_max_entries, 50)


class MemoryListViewTests(APITestCase):
    """Tests for GET /api/memory/."""

    def setUp(self):
        self.user = _make_user(username="memview_user", email="memview@example.com")
        self.other_user = _make_user(
            username="memview_other", email="memview_other@example.com"
        )

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        self.user.delete()
        self.other_user.delete()

    def test_list_returns_empty_when_no_memories(self):
        resp = self.client.get("/api/memory/", **_auth_header(self.user))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json(), [])

    def test_list_returns_only_current_users_memories(self):
        MemoryEntry.objects.create(user=self.user, content="My memory A")
        MemoryEntry.objects.create(user=self.user, content="My memory B")
        MemoryEntry.objects.create(user=self.other_user, content="Other user memory")

        resp = self.client.get("/api/memory/", **_auth_header(self.user))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        contents = [m["content"] for m in resp.json()]
        self.assertIn("My memory A", contents)
        self.assertIn("My memory B", contents)
        self.assertNotIn("Other user memory", contents)
        self.assertEqual(len(contents), 2)

    def test_list_returns_id_content_created_at_fields(self):
        entry = MemoryEntry.objects.create(user=self.user, content="Field check")
        resp = self.client.get("/api/memory/", **_auth_header(self.user))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        item = resp.json()[0]
        self.assertIn("id", item)
        self.assertIn("content", item)
        self.assertIn("created_at", item)
        self.assertEqual(item["id"], entry.id)

    def test_list_requires_authentication(self):
        resp = self.client.get("/api/memory/")
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)


class MemoryDeleteViewTests(APITestCase):
    """Tests for DELETE /api/memory/{id}/."""

    def setUp(self):
        self.user = _make_user(username="memdel_user", email="memdel@example.com")
        self.other_user = _make_user(
            username="memdel_other", email="memdel_other@example.com"
        )

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        self.user.delete()
        self.other_user.delete()

    def test_delete_own_memory_returns_200(self):
        entry = MemoryEntry.objects.create(user=self.user, content="To delete")
        resp = self.client.delete(f"/api/memory/{entry.id}/", **_auth_header(self.user))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json(), {"status": "OK"})
        self.assertFalse(MemoryEntry.objects.filter(pk=entry.id).exists())

    def test_delete_other_users_memory_returns_404(self):
        entry = MemoryEntry.objects.create(user=self.other_user, content="Not mine")
        resp = self.client.delete(f"/api/memory/{entry.id}/", **_auth_header(self.user))
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)
        self.assertTrue(MemoryEntry.objects.filter(pk=entry.id).exists())

    def test_delete_nonexistent_memory_returns_404(self):
        resp = self.client.delete("/api/memory/999999/", **_auth_header(self.user))
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_delete_requires_authentication(self):
        entry = MemoryEntry.objects.create(user=self.user, content="Auth check")
        resp = self.client.delete(f"/api/memory/{entry.id}/")
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertTrue(MemoryEntry.objects.filter(pk=entry.id).exists())


class MemoryImportViewTests(APITestCase):
    """Tests for POST /api/memory/import/."""

    def setUp(self):
        self.user = _make_user(username="memimp_user", email="memimp@example.com")
        self.free_user = _make_user(username="memimp_free", email="memimp_free@example.com")
        self.free_user.tier = "free"
        self.free_user.save(update_fields=["tier"])

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        self.user.delete()
        self.free_user.delete()

    def test_import_requires_authentication(self):
        resp = self.client.post("/api/memory/import/", {"text": "test"})
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_import_forbidden_for_free_user(self):
        resp = self.client.post(
            "/api/memory/import/", {"text": "test"}, **_auth_header(self.free_user)
        )
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)

    def test_import_empty_text_returns_400(self):
        resp = self.client.post("/api/memory/import/", {"text": ""}, **_auth_header(self.user))
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_import_no_text_returns_400(self):
        resp = self.client.post("/api/memory/import/", {}, **_auth_header(self.user))
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_import_replaces_existing_memories(self):
        MemoryEntry.objects.create(user=self.user, content="Old memory A")
        MemoryEntry.objects.create(user=self.user, content="Old memory B")
        self.assertEqual(MemoryEntry.objects.filter(user=self.user).count(), 2)

        with patch("aiworks_core.logic.memory.invoke_llm") as mock_invoke:
            mock_invoke.return_value = [
                "New imported memory 1.",
                "New imported memory 2.",
            ]
            resp = self.client.post(
                "/api/memory/import/",
                {"text": "## Instructions\nAlways use markdown.\n\n## Identity\nJohn, 35, London."},
                **_auth_header(self.user),
            )

        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json()["count"], 2)
        remaining = list(
            MemoryEntry.objects.filter(user=self.user).values_list("content", flat=True)
        )
        self.assertNotIn("Old memory A", remaining)
        self.assertNotIn("Old memory B", remaining)
        self.assertIn("New imported memory 1.", remaining)
        self.assertIn("New imported memory 2.", remaining)

    def test_import_with_no_existing_memories(self):
        self.assertEqual(MemoryEntry.objects.filter(user=self.user).count(), 0)

        with patch("aiworks_core.logic.memory.invoke_llm") as mock_invoke:
            mock_invoke.return_value = ["Fresh imported memory."]
            resp = self.client.post(
                "/api/memory/import/",
                {"text": "Some exported text."},
                **_auth_header(self.user),
            )

        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json()["count"], 1)
        self.assertTrue(
            MemoryEntry.objects.filter(user=self.user, content="Fresh imported memory.").exists()
        )

    def test_import_llm_returns_non_list_logs_warning(self):
        with patch("aiworks_core.logic.memory.invoke_llm") as mock_invoke:
            mock_invoke.return_value = "not a list"
            resp = self.client.post(
                "/api/memory/import/", {"text": "Test"}, **_auth_header(self.user)
            )

        self.assertEqual(resp.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertEqual(resp.json()["error"], "Failed to process memories")

    def test_import_llm_returns_empty_list_ignored(self):
        MemoryEntry.objects.create(user=self.user, content="Will stay")

        with patch("aiworks_core.logic.memory.invoke_llm") as mock_invoke:
            mock_invoke.return_value = []
            resp = self.client.post(
                "/api/memory/import/", {"text": "Test"}, **_auth_header(self.user)
            )

        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json()["count"], 1)
        self.assertEqual(MemoryEntry.objects.filter(user=self.user).count(), 1)
