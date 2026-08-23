"""
Tests for LLMOperationConfig model and per-operation LLM resolution in invoke_llm.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from django.core.exceptions import ValidationError
from django.test import TestCase

from ..logic.llm import invoke_llm
from ..models import LLMConfiguration, LLMOperationConfig


class LLMOperationConfigModelTests(TestCase):
    """Model validation and CRUD for LLMOperationConfig."""

    def test_operation_name_unique(self):
        """operation_name must be unique."""
        LLMOperationConfig.objects.create(
            operation_name="test_op",
            description="Test",
        )
        with self.assertRaises(Exception):
            LLMOperationConfig.objects.create(
                operation_name="test_op",
                description="Duplicate",
            )

    def test_clean_rejects_override_model_with_selected_llm_type(self):
        """override_model and selected_llm_type are mutually exclusive."""
        op = LLMOperationConfig(
            operation_name="bad_op",
            selected_llm_type="fast",
            override_model="gpt-4o",
            override_provider="openai",
        )
        with self.assertRaises(ValidationError) as ctx:
            op.clean()
        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_clean_rejects_override_model_without_provider(self):
        """override_model requires override_provider to be set."""
        op = LLMOperationConfig(
            operation_name="bad_op2",
            override_model="gpt-4o",
            override_provider="",
        )
        with self.assertRaises(ValidationError) as ctx:
            op.clean()
        self.assertIn("override_provider", str(ctx.exception))

    def test_clean_allows_override_model_with_provider(self):
        """override_model with override_provider is valid."""
        op = LLMOperationConfig(
            operation_name="good_op",
            override_model="gpt-4o-mini",
            override_provider="openai",
        )
        op.clean()
        self.assertEqual(op.operation_name, "good_op")

    def test_clean_allows_selected_llm_type_only(self):
        """selected_llm_type alone is valid."""
        op = LLMOperationConfig(
            operation_name="good_op2",
            selected_llm_type="fast",
        )
        op.clean()

    def test_str_returns_operation_name(self):
        op = LLMOperationConfig.objects.create(
            operation_name="my_llm_op",
            description="Test",
        )
        self.assertEqual(str(op), "my_llm_op")


class InvokeLlmOperationResolutionTests(TestCase):
    """Tests that invoke_llm correctly resolves per-operation config."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()
        self.cfg = LLMConfiguration.get_solo()
        self.cfg.provider = "openai"
        self.cfg.openai_api_key = "test-key"
        self.cfg.openai_model = "gpt-4o"
        self.cfg.openai_fast_model = "gpt-4o-mini"
        self.cfg.save()

    def _make_mock_op_config(self, **kwargs):
        """Create a mock LLMOperationConfig with the given field values."""
        defaults = dict(
            operation_name="test_op",
            description="",
            override_provider="",
            selected_llm_type="",
            override_model="",
            is_enabled=True,
        )
        defaults.update(kwargs)
        mock = MagicMock(spec=LLMOperationConfig)
        for k, v in defaults.items():
            setattr(mock, k, v)
        return mock

    def test_no_config_falls_back_to_smart(self):
        """When no LLMOperationConfig exists, defaults to smart model."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="result"))

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create"):
            with patch(
                    "aiworks_core.logic.llm.LLMConfiguration.get_solo", return_value=self.cfg
            ):
                with patch(
                        "aiworks_core.logic.llm.get_chat_model", return_value=mock_llm
                ) as mock_gcm:
                    with patch(
                            "aiworks_core.logic.llm.LLMOperationConfig.objects.get"
                    ) as mock_get:
                        from ..models import LLMOperationConfig

                        mock_get.side_effect = LLMOperationConfig.DoesNotExist("test")
                        asyncio.run(
                            invoke_llm(
                                "unknown_operation",
                                messages=[],
                                parse_json=False,
                            )
                        )
                    mock_gcm.assert_called_once()
                    _, kwargs = mock_gcm.call_args
                    self.assertEqual(kwargs.get("llm_type"), "smart")

    def test_selected_llm_type_resolved_from_config(self):
        """When config has selected_llm_type, it is used instead of default."""
        mock_config = self._make_mock_op_config(
            operation_name="fast_op",
            selected_llm_type="fast",
        )
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="result"))

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create"):
            with patch(
                    "aiworks_core.logic.llm.LLMConfiguration.get_solo", return_value=self.cfg
            ):
                with patch(
                        "aiworks_core.logic.llm.get_chat_model", return_value=mock_llm
                ) as mock_gcm:
                    with patch(
                            "aiworks_core.logic.llm.LLMOperationConfig.objects.get",
                            return_value=mock_config,
                    ):
                        asyncio.run(
                            invoke_llm(
                                "fast_op",
                                messages=[],
                                parse_json=False,
                            )
                        )
                    mock_gcm.assert_called_once()
                    _, kwargs = mock_gcm.call_args
                    self.assertEqual(kwargs.get("llm_type"), "fast")
                    self.assertIsNone(kwargs.get("override_provider"))
                    self.assertIsNone(kwargs.get("override_model"))

    def test_disabled_config_falls_back_to_default(self):
        """When config exists but is_enabled=False, falls back to defaults."""
        mock_config = self._make_mock_op_config(
            operation_name="disabled_op",
            selected_llm_type="fast",
            is_enabled=False,
        )
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="result"))

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create"):
            with patch(
                    "aiworks_core.logic.llm.LLMConfiguration.get_solo", return_value=self.cfg
            ):
                with patch(
                        "aiworks_core.logic.llm.get_chat_model", return_value=mock_llm
                ) as mock_gcm:
                    with patch(
                            "aiworks_core.logic.llm.LLMOperationConfig.objects.get",
                            return_value=mock_config,
                    ):
                        asyncio.run(
                            invoke_llm(
                                "disabled_op",
                                messages=[],
                                parse_json=False,
                            )
                        )
                    mock_gcm.assert_called_once()
                    _, kwargs = mock_gcm.call_args
                    self.assertEqual(kwargs.get("llm_type"), "smart")

    def test_override_provider_only(self):
        """When config has only override_provider (no model override), falls back to smart model of default provider.

        override_provider takes effect only when override_model is also set.
        Without override_model, the system falls back to the configured llm_type resolution.
        """
        mock_config = self._make_mock_op_config(
            operation_name="custom_provider_op",
            override_provider="anthropic",
            override_model="",
        )
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="result"))

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create"):
            with patch(
                    "aiworks_core.logic.llm.LLMConfiguration.get_solo", return_value=self.cfg
            ):
                with patch(
                        "aiworks_core.logic.llm.get_chat_model", return_value=mock_llm
                ) as mock_gcm:
                    with patch(
                            "aiworks_core.logic.llm.LLMOperationConfig.objects.get",
                            return_value=mock_config,
                    ):
                        asyncio.run(
                            invoke_llm(
                                "custom_provider_op",
                                messages=[],
                                parse_json=False,
                            )
                        )
                    mock_gcm.assert_called_once()
                    _, kwargs = mock_gcm.call_args
                    self.assertEqual(kwargs["provider"], "openai")
                    self.assertEqual(kwargs["llm_type"], "smart")

    def test_override_model_bypasses_llm_type_resolution(self):
        """When config has both override_provider and override_model, both are used directly.

        override_provider + override_model together bypass the llm_type resolution.
        """
        mock_config = self._make_mock_op_config(
            operation_name="explicit_model_op",
            override_provider="openai",
            override_model="gpt-4o-mini",
        )
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="result"))

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create"):
            with patch(
                    "aiworks_core.logic.llm.LLMConfiguration.get_solo", return_value=self.cfg
            ):
                with patch(
                        "aiworks_core.logic.llm.get_chat_model", return_value=mock_llm
                ) as mock_gcm:
                    with patch(
                            "aiworks_core.logic.llm.LLMOperationConfig.objects.get",
                            return_value=mock_config,
                    ):
                        asyncio.run(
                            invoke_llm(
                                "explicit_model_op",
                                messages=[],
                                parse_json=False,
                            )
                        )
                    mock_gcm.assert_called_once()
                    _, kwargs = mock_gcm.call_args
                    self.assertEqual(kwargs["provider"], "openai")
                    self.assertEqual(kwargs["model"], "gpt-4o-mini")

    def test_ultra_fast_llm_type_resolved(self):
        """selected_llm_type='ultra-fast' is correctly passed to get_chat_model."""
        mock_config = self._make_mock_op_config(
            operation_name="ultra_fast_op",
            selected_llm_type="ultra-fast",
        )
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="result"))

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create"):
            with patch(
                    "aiworks_core.logic.llm.LLMConfiguration.get_solo", return_value=self.cfg
            ):
                with patch(
                        "aiworks_core.logic.llm.get_chat_model", return_value=mock_llm
                ) as mock_gcm:
                    with patch(
                            "aiworks_core.logic.llm.LLMOperationConfig.objects.get",
                            return_value=mock_config,
                    ):
                        asyncio.run(
                            invoke_llm(
                                "ultra_fast_op",
                                messages=[],
                                parse_json=False,
                            )
                        )
                    mock_gcm.assert_called_once()
                    _, kwargs = mock_gcm.call_args
                    self.assertEqual(kwargs.get("llm_type"), "ultra-fast")
