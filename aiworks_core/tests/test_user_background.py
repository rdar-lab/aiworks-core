"""
Tests for user background synthesis (generate_user_background, get_effective_user_background)
and the Django signals that auto-clear llm_generated_background.
"""

from unittest.mock import patch

from django.test import TestCase

from . import _make_user
from ..logic.session_helper import get_effective_user_background
from ..logic.user_background import generate_user_background
from ..models import MemoryEntry


class GenerateUserBackgroundTests(TestCase):
    """Tests for generate_user_background()."""

    def setUp(self):
        self.user = _make_user(username="bg_user", email="bguser@example.com")

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        self.user.delete()

    def test_returns_empty_string_when_no_profile_and_no_memories(self):
        """No LLM call when there is nothing to synthesise."""
        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            result = generate_user_background(self.user)
        mock_llm.assert_not_called()
        self.assertEqual(result, "")

    def test_calls_llm_and_persists_background(self):
        """Happy path: LLM returns text → persisted and returned."""
        self.user.profile_context = "CEO of a tech startup."
        self.user.save(update_fields=["profile_context"])
        MemoryEntry.objects.create(user=self.user, content="Prefers bold action.")

        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            mock_llm.return_value = "Synthesised background paragraph."
            result = generate_user_background(self.user)

        self.assertEqual(result, "Synthesised background paragraph.")
        self.user.refresh_from_db()
        self.assertEqual(
            self.user.llm_generated_background, "Synthesised background paragraph."
        )

    def test_raises_on_llm_error(self):
        """LLM exception must be re-raised — not silenced."""
        self.user.profile_context = "Some context."
        self.user.save(update_fields=["profile_context"])

        with patch(
            "aiworks_core.logic.user_background.invoke_llm",
            side_effect=RuntimeError("LLM error"),
        ):
            with self.assertRaises(RuntimeError):
                generate_user_background(self.user)

        # Background must remain empty in DB
        self.user.refresh_from_db()
        self.assertEqual(self.user.llm_generated_background, "")

    def test_raises_on_empty_llm_response(self):
        """If the LLM returns an empty string a ValueError must be raised."""
        self.user.profile_context = "Some context."
        self.user.save(update_fields=["profile_context"])

        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            mock_llm.return_value = ""
            with self.assertRaises(ValueError):
                generate_user_background(self.user)

        self.user.refresh_from_db()
        self.assertEqual(self.user.llm_generated_background, "")


class GetEffectiveUserBackgroundTests(TestCase):
    """Tests for get_effective_user_background() with full LLM patching."""

    def setUp(self):
        self.user = _make_user(username="ubg_user", email="ubguser@example.com")

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        self.user.delete()

    def test_returns_empty_when_no_profile_and_no_memories(self):
        """Nothing to synthesise → empty string returned, no LLM call."""
        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            result = get_effective_user_background(self.user)
        mock_llm.assert_not_called()
        self.assertEqual(result, "")

    def test_returns_cached_background_without_llm_call(self):
        """Populated llm_generated_background is returned without calling the LLM."""
        self.user.llm_generated_background = "Cached background."
        self.user.save(update_fields=["llm_generated_background"])
        MemoryEntry.objects.create(user=self.user, content="Some memory.")

        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            result = get_effective_user_background(self.user)
        mock_llm.assert_not_called()
        self.assertEqual(result, "Cached background.")

    def test_generates_via_llm_when_background_empty(self):
        """When llm_generated_background is empty the LLM is called end-to-end."""
        self.user.profile_context = "Senior executive."
        self.user.llm_generated_background = ""
        self.user.save(update_fields=["profile_context", "llm_generated_background"])
        MemoryEntry.objects.create(user=self.user, content="Prefers bold action.")

        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            mock_llm.return_value = "Full synthesised paragraph."
            result = get_effective_user_background(self.user)

        mock_llm.assert_called_once()
        self.assertEqual(result, "Full synthesised paragraph.")
        # Should be persisted to DB
        self.user.refresh_from_db()
        self.assertEqual(self.user.llm_generated_background, "Full synthesised paragraph.")

    def test_propagates_llm_exception(self):
        """LLM errors propagate out of get_effective_user_background."""
        self.user.profile_context = "Some context."
        self.user.llm_generated_background = ""
        self.user.save(update_fields=["profile_context", "llm_generated_background"])

        with patch(
            "aiworks_core.logic.user_background.invoke_llm",
            side_effect=RuntimeError("LLM down"),
        ):
            with self.assertRaises(RuntimeError):
                get_effective_user_background(self.user)


class LlmGeneratedBackgroundSignalTests(TestCase):
    """Tests that llm_generated_background is cleared by signals."""

    def setUp(self):
        self.user = _make_user(username="sig_user", email="siguser@example.com")
        self.user.llm_generated_background = "Existing background."
        self.user.save(update_fields=["llm_generated_background"])

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        self.user.delete()

    def test_background_cleared_when_memory_entry_created(self):
        MemoryEntry.objects.create(user=self.user, content="New memory.")
        self.user.refresh_from_db()
        self.assertEqual(self.user.llm_generated_background, "")

    def test_background_cleared_when_memory_entry_deleted(self):
        entry = MemoryEntry.objects.create(user=self.user, content="Memory to delete.")
        # Reset background to simulate a populated cache
        self.user.llm_generated_background = "Background."
        self.user.save(update_fields=["llm_generated_background"])
        entry.delete()
        self.user.refresh_from_db()
        self.assertEqual(self.user.llm_generated_background, "")

    def test_background_cleared_when_profile_context_changes(self):
        self.user.profile_context = "Updated profile context."
        self.user.save()
        self.user.refresh_from_db()
        self.assertEqual(self.user.llm_generated_background, "")

    def test_background_not_cleared_when_profile_context_unchanged(self):
        original_ctx = self.user.profile_context
        self.user.profile_context = original_ctx
        self.user.save()
        self.user.refresh_from_db()
        self.assertEqual(self.user.llm_generated_background, "Existing background.")

    def test_background_not_cleared_when_update_fields_excludes_profile_context(self):
        """Signal should not clear background when saving unrelated fields."""
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.user.refresh_from_db()
        self.assertEqual(self.user.llm_generated_background, "Existing background.")
