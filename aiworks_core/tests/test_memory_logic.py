"""
Memory generation logic tests for the Ai-Works Core API.
"""

import datetime
import json
from unittest.mock import patch

from django.test import TestCase
from django.utils import timezone

from . import _make_user, _make_session
from ..logic.memory import (
    generate_memories_for_user,
    run_memory_generation_job,
    run_user_background_calculation,
)
from ..logic.session_helper import get_effective_context
from ..logic.schema_validation_utils import validate_json_with_schema_file
from ..models import MemoryEntry, Session, SiteConfiguration


class GetEffectiveContextWithMemoriesTests(TestCase):
    def setUp(self):
        self.user = _make_user(username="ctx_mem_user", email="ctxmem@example.com")
        self.session = _make_session(self.user, session_id="ctx_mem_sess")

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        Session.objects.filter(pk=self.session.pk).delete()
        self.user.delete()

    def _ctx(self):
        return get_effective_context(self.session)

    def test_no_memories_no_context_returns_empty(self):
        self.session.include_user_context = False
        result = self._ctx()
        self.assertEqual(result.strip(), "")

    def test_include_user_context_false_ignores_background(self):
        self.user.llm_generated_background = "Should not appear"
        self.user.save(update_fields=["llm_generated_background"])
        MemoryEntry.objects.create(user=self.user, content="Should not appear")
        self.session.include_user_context = False
        result = self._ctx()
        self.assertNotIn("Should not appear", result)

    def test_pregenerated_background_used_when_include_user_context_true(self):
        self.user.llm_generated_background = "User is a CTO"
        self.user.save(update_fields=["llm_generated_background"])
        MemoryEntry.objects.create(user=self.user, content="User is a CTO")
        self.session.include_user_context = True
        result = self._ctx()
        self.assertIn("User is a CTO", result)
        self.assertIn("User Background", result)

    def test_generate_called_lazily_when_background_empty(self):
        self.user.profile_context = "Senior executive."
        self.user.llm_generated_background = ""
        self.user.save(update_fields=["profile_context", "llm_generated_background"])
        MemoryEntry.objects.create(user=self.user, content="Prefers bold action")
        self.session.include_user_context = True
        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            mock_llm.return_value = (
                "Synthesised: Senior executive. Prefers bold action."
            )
            result = self._ctx()
        mock_llm.assert_called_once()
        self.assertIn("Synthesised:", result)
        self.assertIn("User Background", result)

    def test_profile_context_without_memories_triggers_background_generation(self):
        self.user.profile_context = "Just a profile."
        self.user.llm_generated_background = ""
        self.user.save(update_fields=["profile_context", "llm_generated_background"])
        self.session.include_user_context = True
        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            mock_llm.return_value = "Synthesised profile."
            result = self._ctx()
        self.assertIn("Synthesised profile.", result)

    def test_no_generation_when_no_profile_and_no_memories(self):
        self.user.profile_context = ""
        self.user.llm_generated_background = ""
        self.user.save(update_fields=["profile_context", "llm_generated_background"])
        self.session.include_user_context = True
        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            result = self._ctx()
        mock_llm.assert_not_called()
        self.assertEqual(result.strip(), "")


class GenerateMemoriesForUserTests(TestCase):
    def setUp(self):
        SiteConfiguration.objects.all().delete()
        self.user = _make_user(username="gen_mem_user", email="genmem@example.com")
        self.session = _make_session(self.user, session_id="gen_mem_sess")
        Session.objects.filter(pk=self.session.pk).update(
            created_at=timezone.now() - datetime.timedelta(hours=2)
        )
        self.since_dt = timezone.now() - datetime.timedelta(days=1)

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        Session.objects.filter(pk=self.session.pk).delete()
        self.user.delete()
        SiteConfiguration.objects.all().delete()

    def test_extracts_and_stores_new_memories(self):
        with patch("aiworks_core.logic.memory.invoke_llm") as mock_invoke:
            mock_invoke.return_value = [
                "User values data-driven decisions.",
                "Operates in B2B SaaS.",
            ]
            count = generate_memories_for_user(self.user, self.since_dt)

        self.assertEqual(count, 2)
        memories = list(
            MemoryEntry.objects.filter(user=self.user).values_list("content", flat=True)
        )
        self.assertIn("User values data-driven decisions.", memories)
        self.assertIn("Operates in B2B SaaS.", memories)

    def test_returns_zero_when_no_sessions(self):
        future_since = timezone.now() + datetime.timedelta(days=1)

        with patch("aiworks_core.logic.memory.invoke_llm") as mock_invoke:
            count = generate_memories_for_user(self.user, future_since)

        self.assertEqual(count, 0)
        mock_invoke.assert_not_called()

    def test_returns_zero_on_non_list_llm_output(self):
        with patch("aiworks_core.logic.memory.invoke_llm") as mock_invoke:
            mock_invoke.return_value = {"unexpected": "dict"}
            count = generate_memories_for_user(self.user, self.since_dt)

        self.assertEqual(count, 0)
        self.assertEqual(MemoryEntry.objects.filter(user=self.user).count(), 0)

    def test_triggers_compression_when_over_limit(self):
        cfg = SiteConfiguration.get_solo()
        cfg.memory_max_entries = 3
        cfg.save()

        for i in range(3):
            MemoryEntry.objects.create(user=self.user, content=f"Old memory {i}")

        responses = [
            ["New memory that exceeds the limit."],
            ["Compressed A.", "Compressed B."],
        ]
        call_idx = [0]

        def _multi(*args, **kwargs):
            idx = min(call_idx[0], len(responses) - 1)
            call_idx[0] += 1
            return responses[idx]

        with patch("aiworks_core.logic.memory.invoke_llm", side_effect=_multi):
            generate_memories_for_user(self.user, self.since_dt)

        memories = list(
            MemoryEntry.objects.filter(user=self.user).values_list("content", flat=True)
        )
        self.assertIn("Compressed A.", memories)
        self.assertIn("Compressed B.", memories)
        self.assertNotIn("Old memory 0", memories)


class RunMemoryGenerationJobTests(TestCase):
    def setUp(self):
        SiteConfiguration.objects.all().delete()
        self.user = _make_user(username="job_mem_user", email="jobmem@example.com")
        self.session = _make_session(self.user, session_id="job_mem_sess")
        self.since_dt = timezone.now() - datetime.timedelta(days=1)

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        Session.objects.filter(pk=self.session.pk).delete()
        self.user.delete()
        SiteConfiguration.objects.all().delete()

    def test_does_nothing_when_no_sessions_in_window(self):
        future_since = timezone.now() + datetime.timedelta(days=1)
        run_memory_generation_job(future_since)
        self.assertEqual(MemoryEntry.objects.count(), 0)

    def test_does_not_raise_when_no_sessions(self):
        run_memory_generation_job(timezone.now() + datetime.timedelta(days=1))

    def test_does_not_create_memories_for_free_user(self):
        self.user.tier = self.user.TIER_FREE
        self.user.save()

        with patch("aiworks_core.logic.memory.invoke_llm") as mock_invoke:
            mock_invoke.return_value = ["Should not be saved."]
            run_memory_generation_job(self.since_dt)

        self.assertEqual(MemoryEntry.objects.filter(user=self.user).count(), 0)
        mock_invoke.assert_not_called()

    def test_creates_memories_for_pro_user(self):
        with patch("aiworks_core.logic.memory.invoke_llm") as mock_invoke:
            mock_invoke.return_value = ["Job-created memory."]
            run_memory_generation_job(self.since_dt)

        memories = list(
            MemoryEntry.objects.filter(user=self.user).values_list("content", flat=True)
        )
        self.assertIn("Job-created memory.", memories)

    def test_handles_per_user_error_gracefully(self):
        with patch("aiworks_core.logic.memory.generate_memories_for_user") as mock_gen:
            mock_gen.side_effect = RuntimeError("LLM exploded")
            run_memory_generation_job(self.since_dt)

        self.assertEqual(MemoryEntry.objects.filter(user=self.user).count(), 0)


class RunUserBackgroundCalculationTests(TestCase):
    def setUp(self):
        self.user = _make_user(username="bg_calc_user", email="bgcalc@example.com")

    def tearDown(self):
        MemoryEntry.objects.all().delete()
        self.user.delete()

    def test_skips_user_with_no_memories_and_no_profile_context(self):
        self.user.profile_context = ""
        self.user.llm_generated_background = ""
        self.user.save(update_fields=["profile_context", "llm_generated_background"])
        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            run_user_background_calculation()
        mock_llm.assert_not_called()
        self.user.refresh_from_db()
        self.assertEqual(self.user.llm_generated_background, "")

    def test_recalculates_user_with_memories_and_empty_string_background(self):
        self.user.profile_context = ""
        self.user.llm_generated_background = ""
        self.user.save(update_fields=["profile_context", "llm_generated_background"])
        MemoryEntry.objects.create(user=self.user, content="Memory content")
        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            mock_llm.return_value = "Synthesised background from memories."
            run_user_background_calculation()
        self.user.refresh_from_db()
        self.assertEqual(self.user.llm_generated_background, "Synthesised background from memories.")

    def test_recalculates_user_with_profile_context_and_empty_string_background(self):
        self.user.profile_context = "Executive in fintech."
        self.user.llm_generated_background = ""
        self.user.save(update_fields=["profile_context", "llm_generated_background"])
        with patch("aiworks_core.logic.user_background.invoke_llm") as mock_llm:
            mock_llm.return_value = "Synthesised background from profile."
            run_user_background_calculation()
        self.user.refresh_from_db()
        self.assertEqual(self.user.llm_generated_background, "Synthesised background from profile.")

    def test_skips_user_with_existing_background(self):
        self.user.profile_context = "Executive in fintech."
        self.user.llm_generated_background = "Already set background."
        self.user.save(update_fields=["profile_context", "llm_generated_background"])
        MemoryEntry.objects.create(user=self.user, content="Memory content")
        with patch("aiworks_core.logic.user_background.generate_user_background") as mock_gen:
            run_user_background_calculation()
        mock_gen.assert_not_called()

    def test_handles_per_user_error_gracefully(self):
        self.user.profile_context = "Executive in fintech."
        self.user.llm_generated_background = ""
        self.user.save(update_fields=["profile_context", "llm_generated_background"])
        MemoryEntry.objects.create(user=self.user, content="Memory content")
        with patch("aiworks_core.logic.user_background.generate_user_background") as mock_gen:
            mock_gen.side_effect = RuntimeError("LLM exploded")
            run_user_background_calculation()
        self.user.refresh_from_db()
        self.assertEqual(self.user.llm_generated_background, "")


class MemoryGenerationSchemaTests(TestCase):
    """Schema validation for memory generation via the real invoke_llm pipeline."""

    def test_memory_generation_schema_accepts_array_of_strings(self):
        """memory_generation_schema.json accepts array of strings for memory entries."""
        valid_json = json.dumps([
            "User prefers data-driven decisions",
            "Recent focus on cost optimization",
            "Prefers bold strategic moves",
        ])
        is_valid, err = validate_json_with_schema_file(valid_json, "api/resources/schema/memory_generation_schema.json")
        self.assertTrue(is_valid, f"Schema should accept array of strings: {err}")

    def test_memory_generation_schema_rejects_non_array(self):
        """memory_generation_schema.json rejects non-array responses."""
        invalid_json = json.dumps({
            "patterns": ["not an array structure"],
        })
        is_valid, err = validate_json_with_schema_file(invalid_json, "api/resources/schema/memory_generation_schema.json")
        self.assertFalse(is_valid)
        self.assertIsNotNone(err)
        self.assertIn("array", err.lower())


class MemoryCompressionSchemaTests(TestCase):
    """Schema validation for memory compression via the real invoke_llm pipeline."""

    def test_memory_compression_schema_accepts_array_of_strings(self):
        """memory_compression_schema.json accepts array of strings for compressed memories."""
        valid_json = json.dumps([
            "Compressed: Focus on strategic decisions",
            "Compressed: Risk-aware approach",
        ])
        is_valid, err = validate_json_with_schema_file(valid_json, "api/resources/schema/memory_compression_schema.json")
        self.assertTrue(is_valid, f"Schema should accept array of strings: {err}")

    def test_memory_compression_schema_rejects_non_array(self):
        """memory_compression_schema.json rejects non-array responses."""
        invalid_json = json.dumps({"data": "not an array"})
        is_valid, err = validate_json_with_schema_file(invalid_json, "api/resources/schema/memory_compression_schema.json")
        self.assertFalse(is_valid)
        self.assertIsNotNone(err)
        self.assertIn("array", err.lower())
