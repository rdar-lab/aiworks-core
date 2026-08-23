"""
Unit tests for the quality-check gate and recovery cycle in deepagent_utils.

Tests cover:
- _perform_quality_check behaviour (approval, rejection, LLM error handling)
- run_deep_agent quality gate integration
- recover_deep_agent_run quality gate on recovery output
- Max-recovery-attempt limit enforcement
- recovery_reason injection into recovery prompt template
- CustomLLMResponseValidator integration in quality check flow
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from ..utils import async_to_sync
from django.test import TestCase

from ..logic.deepagent_utils import (
    run_deep_agent,
    recover_deep_agent_run,
    _perform_quality_check,
    _RECOVERY_MAX_ATTEMPTS,
)
from ..logic.schema_validation_utils import validate_json_with_schema_file
from ..logic.llm import CustomLLMResponseValidator


class PerformQualityCheckTests(TestCase):
    """Unit tests for _perform_quality_check."""

    @pytest.mark.asyncio
    async def test_qa_returns_approved_true(self):
        """QA check returns is_approved=true → (True, feedback)."""
        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_invoke.return_value = {
                "is_approved": True,
                "feedback": "Looks good",
            }

            passed, _, feedback = await _perform_quality_check(
                "LLM Response",
                "some_quality_check",
                "/report.md",
                output_files={"/report.md": "# Report content here"},
            )

            assert passed is True
            assert feedback == "Looks good"
            mock_invoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_qa_returns_approved_false(self):
        """QA check returns is_approved=false → (False, feedback)."""
        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_invoke.return_value = {
                "is_approved": False,
                "feedback": "Missing citations",
            }

            passed, _, feedback = await _perform_quality_check(
                "LLM Response",
                "some_quality_check",
                "/report.md",
                output_files={"/report.md": "# Report content here"},
            )

            assert passed is False
            assert feedback == "Missing citations"

    @pytest.mark.asyncio
    async def test_qa_llm_throws_returns_false_not_crash(self):
        """invoke_llm throws → returns (False, error_message), no exception propagates."""
        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_invoke.side_effect = RuntimeError("LLM provider unavailable")

            passed, _, feedback = await _perform_quality_check(
                "LLM Response",
                "some_quality_check",
                "/report.md",
                output_files={"/report.md": "# Report content here"},
            )

            assert passed is False
            assert "LLM provider unavailable" in feedback
            assert "Quality check prompt failed with error" in feedback

    @pytest.mark.asyncio
    async def test_qa_no_quality_check_key_returns_true(self):
        """quality_check_prompt_key=None → returns (True, ''), no invoke_llm call."""
        passed, _, feedback = await _perform_quality_check(
            "LLM Response",
            None,
            "/report.md",
            output_files={"/report.md": "# Report content here"},
        )

        assert passed is True
        assert feedback == ""

    async def test_qa_no_quality_check_rejects_no_file(self):
        """quality_check_prompt_key=None → returns (True, ''), no invoke_llm call."""
        passed, fatal, feedback = await _perform_quality_check(
            "LLM Response",
            None,
            "/report.md",
            output_files={},
        )

        assert passed is False
        assert fatal is True
        assert feedback == "Agent did not generate the necessary /report.md"


    async def test_qa_no_quality_check_rejects_no_response(self):
        """quality_check_prompt_key=None → returns (True, ''), no invoke_llm call."""
        passed, fatal, feedback = await _perform_quality_check(
            "",
            None,
            "/report.md",
            output_files={"/report.md": "# Report content here"},
        )

        assert passed is False
        assert fatal is True
        assert feedback == "Detected LLM Early-Stop or crash. LLM did not send a final response"

    @pytest.mark.asyncio
    async def test_qa_feedback_carries_specific_failure_reason(self):
        """QA feedback string is propagated back verbatim."""
        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_invoke.return_value = {
                "is_approved": False,
                "feedback": "index.html uses hardcoded data instead of fetch()",
            }

            passed, _, feedback = await _perform_quality_check(
                "LLM Response",
                "dashboard_agent_quality_check",
                "/report.md",
                output_files={"/report.md": "<html>hardcoded chart data</html>"},
            )

            assert passed is False
            assert "hardcoded data" in feedback
            assert "fetch()" in feedback


class RunDeepAgentQualityGateTests(TestCase):
    """Tests for quality gate integration in run_deep_agent.

    NOTE: run_deep_agent() no longer accepts dilemma/context/qa_text kwargs.
    These tests use the old API and are skipped.
    """

    def _mock_invoke_llm_success_with_report(self, report_content="# Report"):
        """Return a mock that simulates successful agent run with report file written."""
        mock = AsyncMock()
        mock.return_value = True
        return mock

    def test_qa_approved_no_recovery(self):
        """QA returns is_approved=true → run_deep_agent returns successfully, no recovery called."""
        call_log = []

        async def fake_invoke_llm(*args, **kwargs):
            call_log.append(kwargs.get("prompt_key", args[0] if args else None))
            if "quality_check" in (call_log[-1] or ""):
                return {"is_approved": True, "feedback": "OK"}
            output_files = kwargs.get("output_files", {})
            output_files["/report.md"] = "# Valid report content"
            return True

        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            side_effect=fake_invoke_llm,
        ):
            report, files = async_to_sync(run_deep_agent)(
                prompt_key="test_agent",
                file_content="",
                kb_manifest="",
                report_filename="/report.md",
                recovery_prompt_key="test_recovery",
                quality_check_prompt_key="test_quality_check",
            )

        assert report == "# Valid report content"
        assert "/report.md" in files
        assert "test_quality_check" in call_log
        assert "test_recovery" not in call_log

    def test_qa_rejected_triggers_recovery(self):
        """QA returns is_approved=false with recovery available → recover_deep_agent_run is called."""
        call_log = []

        async def fake_invoke_llm(*args, **kwargs):
            prompt_key = kwargs.get("prompt_key", args[0] if args else None)
            call_log.append(prompt_key)
            if "quality_check" in (prompt_key or ""):
                return {"is_approved": False, "feedback": "Report too short"}
            output_files = kwargs.get("output_files", {})
            output_files["/report.md"] = "# Valid report content"
            return True

        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            side_effect=fake_invoke_llm,
        ):
            with patch(
                "aiworks_core.logic.deepagent_utils.recover_deep_agent_run",
                new_callable=AsyncMock,
            ) as mock_recover:
                mock_recover.return_value = ("# Recovered report", {})

                report, files = async_to_sync(run_deep_agent)(
                    prompt_key="test_agent",
                    file_content="",
                    kb_manifest="",
                    report_filename="/report.md",
                    recovery_prompt_key="test_recovery",
                    quality_check_prompt_key="test_quality_check",
                )

        mock_recover.assert_awaited_once()
        call_kwargs = mock_recover.call_args.kwargs
        assert call_kwargs["recovery_reason"] == "Report too short"
        assert call_kwargs["recovery_prompt_key"] == "test_recovery"
        assert call_kwargs["quality_check_prompt_key"] == "test_quality_check"

    def test_qa_rejected_no_recovery_raises(self):
        """QA returns is_approved=false with no recovery → Exception raised."""
        async def fake_invoke_llm(*args, **kwargs):
            if "quality_check" in (kwargs.get("prompt_key", "") or ""):
                return {"is_approved": False, "feedback": "Missing sections"}
            output_files = kwargs.get("output_files", {})
            output_files["/report.md"] = "# Valid report content"
            return True

        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            side_effect=fake_invoke_llm,
        ):
            with pytest.raises(Exception) as exc_info:
                async_to_sync(run_deep_agent)(
                    prompt_key="test_agent",
                    file_content="",
                    kb_manifest="",
                    report_filename="/report.md",
                    recovery_prompt_key=None,
                    quality_check_prompt_key="test_quality_check",
                )

        assert "quality check" in str(exc_info.value)

    def test_qa_feedback_propagates_to_recovery(self):
        """QA feedback string becomes recovery_reason in the recovery call."""
        call_log = []

        async def fake_invoke_llm(*args, **kwargs):
            prompt_key = kwargs.get("prompt_key", args[0] if args else None)
            if prompt_key and "quality_check" in prompt_key:
                call_log.append(("qa", kwargs.get("template_params", {}).get("agent_result", "")))
                return {"is_approved": False, "feedback": "Invalid JSON structure"}
            output_files = kwargs.get("output_files", {})
            output_files["/report.md"] = "# Report content"
            return True

        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            side_effect=fake_invoke_llm,
        ):
            with patch(
                "aiworks_core.logic.deepagent_utils.recover_deep_agent_run",
                new_callable=AsyncMock,
            ) as mock_recover:
                mock_recover.return_value = ("# Recovered", {})

                async_to_sync(run_deep_agent)(
                    prompt_key="test_agent",
                    file_content="",
                    kb_manifest="",
                    report_filename="/report.md",
                    recovery_prompt_key="test_recovery",
                    quality_check_prompt_key="test_quality_check",
                )

        call_kwargs = mock_recover.call_args.kwargs
        assert call_kwargs["recovery_reason"] == "Invalid JSON structure"


class RecoverDeepAgentRunQualityGateTests(TestCase):
    """Tests for quality gate on recovery output."""

    def test_recovery_runs_qa_on_recovery_output_pass(self):
        """Recovery output passes QA → returns successfully without further recovery."""
        call_log = []

        async def fake_invoke_llm(*args, **kwargs):
            prompt_key = kwargs.get("prompt_key", args[0] if args else None)
            call_log.append(prompt_key)
            if "quality_check" in (prompt_key or ""):
                return {"is_approved": True, "feedback": "OK"}
            output_files = kwargs.get("output_files", {})
            output_files["/report.md"] = "# Recovered report"
            return True

        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            side_effect=fake_invoke_llm,
        ):
            report, files = async_to_sync(recover_deep_agent_run)(
                recovery_prompt_key="test_recovery",
                report_filename="/report.md",
                recovery_reason="Initial failure reason",
                quality_check_prompt_key="test_quality_check",
            )

        assert report == "# Recovered report"
        assert "/report.md" in files
        qa_calls = [k for k in call_log if "quality_check" in k]
        assert len(qa_calls) == 1

    def test_recovery_qa_fails_exhausts_all_attempts_then_returns_result(self):
        """Recovery output fails QA on every attempt → returns result after _RECOVERY_MAX_ATTEMPTS since report was found."""
        call_log = []

        async def fake_invoke_llm(*args, **kwargs):
            prompt_key = kwargs.get("prompt_key", args[0] if args else None)
            call_log.append(prompt_key)
            if "quality_check" in (prompt_key or ""):
                return {"is_approved": False, "feedback": "Still insufficient"}
            output_files = kwargs.get("output_files", {})
            output_files["/report.md"] = "# Attempt content"
            return True

        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            side_effect=fake_invoke_llm,
        ):
            report, files = async_to_sync(recover_deep_agent_run)(
                recovery_prompt_key="test_recovery",
                report_filename="/report.md",
                recovery_reason="First failure",
                quality_check_prompt_key="test_quality_check",
            )

        assert report == "# Attempt content"
        assert "/report.md" in files
        qa_calls = [k for k in call_log if "quality_check" in k]
        assert len(qa_calls) == _RECOVERY_MAX_ATTEMPTS
        recovery_calls = [k for k in call_log if k and "recovery" in k and "quality_check" not in k]
        assert len(recovery_calls) == _RECOVERY_MAX_ATTEMPTS

    def test_recovery_fails_after_max_attempts_returns_result(self):
        """All recovery attempts fail QA but report was found → returns result after _RECOVERY_MAX_ATTEMPTS."""
        async def fake_invoke_llm(*args, **kwargs):
            if "quality_check" in (kwargs.get("prompt_key", "") or ""):
                return {"is_approved": False, "feedback": "Always failing"}
            output_files = kwargs.get("output_files", {})
            output_files["/report.md"] = "# Attempt content"
            return True

        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            side_effect=fake_invoke_llm,
        ):
            report, files = async_to_sync(recover_deep_agent_run)(
                recovery_prompt_key="test_recovery",
                report_filename="/report.md",
                recovery_reason="Always fails",
                quality_check_prompt_key="test_quality_check",
            )

        assert report == "# Attempt content"
        assert "/report.md" in files

    def test_recovery_injects_reason_into_template(self):
        """recovery_reason is passed into the recovery LLM call template_params."""
        captured_params = {}

        async def fake_invoke_llm(*args, **kwargs):
            prompt_key = kwargs.get("prompt_key", args[0] if args else None)
            if prompt_key and prompt_key == "test_recovery":
                captured_params.update(kwargs.get("template_params", {}))
            output_files = kwargs.get("output_files", {})
            output_files["/report.md"] = "# Report"
            return True

        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            side_effect=fake_invoke_llm,
        ):
            async_to_sync(recover_deep_agent_run)(
                recovery_prompt_key="test_recovery",
                report_filename="/report.md",
                recovery_reason="Dashboard data.json missing _meta key",
                quality_check_prompt_key=None,
            )

        assert captured_params.get("recovery_reason") == "Dashboard data.json missing _meta key"
        assert "current_date" in captured_params

    def test_recovery_qa_failure_feeds_into_next_recovery_reason(self):
        """QA failure on recovery output becomes the recovery_reason for the next attempt."""
        call_log = []

        async def fake_invoke_llm(*args, **kwargs):
            prompt_key = kwargs.get("prompt_key", args[0] if args else None)
            if "quality_check" in (prompt_key or ""):
                call_log.append(("qa", kwargs.get("template_params", {}).get("agent_result", "")))
                return {"is_approved": False, "feedback": "Content too brief"}
            output_files = kwargs.get("output_files", {})
            output_files["/report.md"] = "# Attempt content"
            return True

        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            side_effect=fake_invoke_llm,
        ):
            report, files = async_to_sync(recover_deep_agent_run)(
                recovery_prompt_key="test_recovery",
                report_filename="/report.md",
                recovery_reason="Original failure",
                quality_check_prompt_key="test_quality_check",
            )

        assert report == "# Attempt content"
        second_qa_call = [v for k, v in call_log if k == "qa"]
        assert len(second_qa_call) == _RECOVERY_MAX_ATTEMPTS


class RecoveryMaxAttemptsTests(TestCase):
    """Tests that _RECOVERY_MAX_ATTEMPTS is respected."""

    def test_max_attempts_constant_is_three(self):
        """_RECOVERY_MAX_ATTEMPTS must be 3 for the test assertions to be valid."""
        assert _RECOVERY_MAX_ATTEMPTS == 3


class QualityCheckSchemaTests(TestCase):
    """Schema validation for quality check responses via the real invoke_llm pipeline."""

    def test_quality_check_schema_accepts_approval(self):
        """quality_check_schema.json accepts is_approved=true with feedback."""
        valid_json = json.dumps({
            "is_approved": True,
            "feedback": "Output meets quality standards",
        })
        is_valid, err = validate_json_with_schema_file(valid_json, "api/resources/schema/quality_check_schema.json")
        self.assertTrue(is_valid, f"Schema should accept approval: {err}")

    def test_quality_check_schema_accepts_rejection(self):
        """quality_check_schema.json accepts is_approved=false with feedback."""
        valid_json = json.dumps({
            "is_approved": False,
            "feedback": "Missing key sections",
        })
        is_valid, err = validate_json_with_schema_file(valid_json, "api/resources/schema/quality_check_schema.json")
        self.assertTrue(is_valid, f"Schema should accept rejection: {err}")

    def test_quality_check_schema_rejects_missing_is_approved(self):
        """quality_check_schema.json rejects response missing is_approved field."""
        invalid_json = json.dumps({
            "feedback": "Only feedback provided",
        })
        is_valid, err = validate_json_with_schema_file(invalid_json, "api/resources/schema/quality_check_schema.json")
        self.assertFalse(is_valid)
        self.assertIn("is_approved", err)

    def test_quality_check_schema_rejects_wrong_type(self):
        """quality_check_schema.json rejects is_approved when it's a string instead of boolean."""
        invalid_json = json.dumps({
            "is_approved": "yes",
            "feedback": "Some feedback",
        })
        is_valid, err = validate_json_with_schema_file(invalid_json, "api/resources/schema/quality_check_schema.json")
        self.assertFalse(is_valid)


class CustomValidatorQualityCheckTests(TestCase):
    """Tests for CustomLLMResponseValidator integration in _perform_quality_check, run_deep_agent, and recover_deep_agent_run."""

    @pytest.mark.asyncio
    async def test_custom_validator_called_in_quality_check(self):
        """When custom_validator is provided, validate_response is called before QA LLM call."""
        validated_args = {}

        class TestValidator(CustomLLMResponseValidator):
            def validate_response(self, response, **kwargs):
                validated_args.update(response=response, **kwargs)

        validator = TestValidator()
        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_invoke.return_value = {"is_approved": True, "feedback": "OK"}
            await _perform_quality_check(
                "LLM Response",
                "some_quality_check",
                "/report.md",
                output_files={"/report.md": "# Report content"},
                custom_validator=validator,
            )

        self.assertEqual(validated_args.get("response"), "# Report content")

    @pytest.mark.asyncio
    async def test_custom_validator_exception_fails_quality_check(self):
        """When validator raises, _perform_quality_check returns (False, error_message)."""
        exc = RuntimeError("LLM response validation failed: missing required section")

        class FailingValidator(CustomLLMResponseValidator):
            def validate_response(self, response, **kwargs):
                raise exc

        validator = FailingValidator()
        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_invoke.return_value = {"is_approved": True, "feedback": "OK"}
            passed, _, feedback = await _perform_quality_check(
                "LLM Response",
                "some_quality_check",
                "/report.md",
                output_files={"/report.md": "# Report content"},
                custom_validator=validator,
            )

        self.assertFalse(passed)
        self.assertIn("LLM response validation failed:", feedback)
        self.assertIn("missing required section", feedback)
        mock_invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_validator_pass_proceeds_to_qa(self):
        """When validator passes (no exception), quality check proceeds to LLM QA call."""
        class PassingValidator(CustomLLMResponseValidator):
            def validate_response(self, response, **kwargs):
                pass

        validator = PassingValidator()
        with patch(
            "aiworks_core.logic.deepagent_utils.invoke_llm",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_invoke.return_value = {"is_approved": True, "feedback": "OK"}
            passed, _, feedback = await _perform_quality_check(
                "LLM Response",
                "some_quality_check",
                "/report.md",
                output_files={"/report.md": "# Report content"},
                custom_validator=validator,
            )

        self.assertTrue(passed)
        mock_invoke.assert_awaited_once()

    def test_custom_validator_propagates_through_run_deep_agent(self):
        """run_deep_agent with custom_validator threads it through to _perform_quality_check."""
        validated_args = {}

        class TestValidator(CustomLLMResponseValidator):
            def validate_response(self, response, **kwargs):
                validated_args.update(response=response, **kwargs)

        async def fake_invoke_llm(*args, **kwargs):
            prompt_key = kwargs.get("prompt_key", args[0] if args else None)
            if "quality_check" in (prompt_key or ""):
                return {"is_approved": True, "feedback": "OK"}
            output_files = kwargs.get("output_files", {})
            output_files["/report.md"] = "# Valid report content"
            return True

        with patch("aiworks_core.logic.deepagent_utils.invoke_llm", side_effect=fake_invoke_llm):
            report, files = async_to_sync(run_deep_agent)(
                prompt_key="test_agent",
                file_content="",
                kb_manifest="",
                report_filename="/report.md",
                quality_check_prompt_key="test_quality_check",
                custom_validator=TestValidator(),
            )

        self.assertEqual(validated_args.get("response"), "# Valid report content")
        self.assertEqual(report, "# Valid report content")

    def test_custom_validator_propagates_through_recover_deep_agent_run(self):
        """recover_deep_agent_run with custom_validator threads it through to _perform_quality_check."""
        validated_args = {}

        class TestValidator(CustomLLMResponseValidator):
            def validate_response(self, response, **kwargs):
                validated_args.update(response=response, **kwargs)

        async def fake_invoke_llm(*args, **kwargs):
            prompt_key = kwargs.get("prompt_key", args[0] if args else None)
            if "quality_check" in (prompt_key or ""):
                return {"is_approved": True, "feedback": "OK"}
            output_files = kwargs.get("output_files", {})
            output_files["/report.md"] = "# Recovered report"
            return True

        with patch("aiworks_core.logic.deepagent_utils.invoke_llm", side_effect=fake_invoke_llm):
            report, files = async_to_sync(recover_deep_agent_run)(
                recovery_prompt_key="test_recovery",
                report_filename="/report.md",
                recovery_reason="Initial failure",
                quality_check_prompt_key="test_quality_check",
                custom_validator=TestValidator(),
            )

        self.assertEqual(validated_args.get("response"), "# Recovered report")
        self.assertEqual(report, "# Recovered report")
