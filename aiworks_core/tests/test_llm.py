"""
Llm tests for the Ai-Works Core API.
"""
import asyncio
import inspect
import json
import logging
import queue
import uuid
from types import SimpleNamespace
from typing import Optional
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch
from uuid import uuid4

import deepagents.graph as graph_module
import deepagents.middleware.summarization as summarization_module
from ..utils import async_to_sync
from django.test import TestCase
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from langchain_core.tools import BaseTool, StructuredTool
from tenacity import before_sleep_log
# noinspection PyProtectedMember
from tenacity._utils import LoggerProtocol
from ..logic import prompt_compression as pc
# noinspection PyProtectedMember
from ..logic.llm import (
    _extract_json,
    _extract_llm_response,
    _navigate_prompts,
    get_chat_model,
    get_agent,
    _AgentLoggingCallbackHandler,
    _AgentCallbacksHandler,
    AgentCallbacks,
    invoke_llm,
    _MAX_AGENT_ITERATIONS,
    _audit_llm_call,
    _patch_deepagents_summarization,
    _DebugDataCollector,
    _extract_todos,
    _extract_thoughts,
    _coerce_tool_output,
    _SafeToolWrapper,
    _SafeStructuredToolWrapper,
    _make_safe_tool,
    _build_memory_md,
    _resolve_provider_model,
    CustomLLMResponseValidator,
)
from ..logic.logic_utils import compress_if_needed
# noinspection PyProtectedMember
from ..logic.prompt_compression import compress_prompt, _get_llmlingua_compressor_pool, _COMPRESS_CHUNK_CHARS
from ..models import (
    LLMConfiguration,
    LLMOperationConfig,
)
from ..logic.image_generator import ImageGeneratorModel
from ..logic.video_generator import VideoGeneratorModel
from ..logic.audio_generator import AudioGeneratorModel


class ExtractJsonTests(TestCase):
    def test_plain_json_string(self):
        data = '{"key": "value"}'
        self.assertEqual(_extract_json(data), '{"key": "value"}')

    def test_strips_leading_trailing_whitespace(self):
        self.assertEqual(_extract_json("  [1,2,3]  "), "[1,2,3]")

    def test_json_inside_markdown_json_fence(self):
        text = '```json\n{"a": 1}\n```'
        self.assertEqual(json.loads(_extract_json(text)), {"a": 1})

    def test_json_inside_plain_code_fence(self):
        text = "```\n[1, 2]\n```"
        self.assertEqual(json.loads(_extract_json(text)), [1, 2])

    def test_prefers_json_fence_over_plain_fence(self):
        text = '```json\n{"json": true}\n``` some text ```\n{"plain": true}\n```'
        result = json.loads(_extract_json(text))
        self.assertTrue(result.get("json"))


class ExtractLlmResponseTests(TestCase):
    def test_string_returned_as_is(self):
        self.assertEqual(_extract_llm_response("hello"), "hello")

    def test_think_removed(self):
        self.assertEqual(_extract_llm_response("<think>thought</think>hello"), "hello")

    def test_empty_list_returns_empty_string(self):
        self.assertEqual(_extract_llm_response([]), "")

    def test_list_with_string_returns_last(self):
        self.assertEqual(_extract_llm_response(["a", "b", "c"]), "c")

    def test_dict_with_empty_messages_returns_empty_string(self):
        self.assertEqual(_extract_llm_response({"messages": []}), "")

    def test_dict_with_messages_returns_last(self):
        self.assertEqual(_extract_llm_response({"messages": ["first", "last"]}), "last")

    def test_strips_thinking_tags_from_string(self):
        """<think> tags are removed and inner text is returned as clean string."""
        input_str = "<think>My reasoning step</think>Final answer"
        self.assertEqual(_extract_llm_response(input_str), "Final answer")

    def test_strips_multiple_thinking_tags_from_string(self):
        """Multiple <think> blocks are all stripped from string."""
        input_str = "<think>First step</think>Middle<think>Second step</think>End"
        self.assertEqual(_extract_llm_response(input_str), "MiddleEnd")

    def test_strips_reasoning_tags_and_recurse(self):
        """Tags stripped from string, then result is re-parsed as dict."""
        input_str = "<think>reasoning</think>some text"
        self.assertEqual(_extract_llm_response(input_str), "some text")


class NavigatePromptsTests(TestCase):
    @patch("aiworks_core.logic.llm._prompts", {"level1": {"level2": "leaf value"}})
    def test_single_level_key(self):
        self.assertEqual(_navigate_prompts("level1"), {"level2": "leaf value"})

    @patch("aiworks_core.logic.llm._prompts", {"level1": {"level2": "leaf value"}})
    def test_nested_key(self):
        self.assertEqual(_navigate_prompts("level1.level2"), "leaf value")

    @patch("aiworks_core.logic.llm._prompts", {"a": {"b": {"c": "deep"}}})
    def test_three_levels_deep(self):
        self.assertEqual(_navigate_prompts("a.b.c"), "deep")


class CompressPromptLLMTests(TestCase):
    """Tests for the LLM-based compression path inside compress_prompt."""

    @staticmethod
    def _mock_cfg(**kwargs):
        """Return a MagicMock LLMConfiguration with sensible defaults."""
        defaults = dict(
            compress_prompts=True,
            compress_prompts_use_llm=False,
            compress_prompts_min_length=5,
            compress_prompts_llm_prompt="Compress: {text}",
            compress_prompts_force_tokens=None,
            compress_prompts_chunk_end_tokens=None,
            compress_prompts_target_token_rate=0.6,
            compress_prompts_cache_size=256,
            compress_prompts_pool_size=1,
        )
        defaults.update(kwargs)
        cfg = MagicMock(spec=LLMConfiguration)
        for k, v in defaults.items():
            setattr(cfg, k, v)
        return cfg

    @patch("aiworks_core.logic.prompt_compression._COMPRESS_CACHE_FUNC", None)
    @patch("aiworks_core.logic.llm.invoke_llm", new_callable=AsyncMock)
    @patch("aiworks_core.logic.prompt_compression.LLMConfiguration.get_solo")
    def test_llm_path_called_when_use_llm_enabled(self, mock_get_solo, mock_invoke):
        mock_get_solo.return_value = self._mock_cfg(compress_prompts_use_llm=True)
        mock_invoke.return_value = "short text"
        result = asyncio.run(compress_prompt("a long text that exceeds the minimum"))
        self.assertEqual(result, "short text")
        mock_invoke.assert_called_once()
        args, kwargs = mock_invoke.call_args
        self.assertEqual(kwargs["parse_json"], False)
        self.assertEqual(args[0], "prompt_compression")

    @patch("aiworks_core.logic.prompt_compression._COMPRESS_CACHE_FUNC", None)
    @patch("aiworks_core.logic.llm.invoke_llm", new_callable=AsyncMock)
    @patch("aiworks_core.logic.prompt_compression.LLMConfiguration.get_solo")
    def test_llm_result_is_cached(self, mock_get_solo, mock_invoke):
        mock_get_solo.return_value = self._mock_cfg(compress_prompts_use_llm=True)
        mock_invoke.return_value = "compressed"
        text = "a long text that exceeds the minimum"
        asyncio.run(compress_prompt(text))
        asyncio.run(compress_prompt(text))
        # invoke_llm should only be called once due to caching
        mock_invoke.assert_called_once()

    @patch("aiworks_core.logic.llm.invoke_llm", new_callable=AsyncMock)
    @patch("aiworks_core.logic.prompt_compression.LLMConfiguration.get_solo")
    def test_llm_path_skipped_when_use_llm_disabled(self, mock_get_solo, mock_invoke):
        """When compress_prompts_use_llm is False, LLMLingua path is used (mocked to avoid loading model)."""
        mock_get_solo.return_value = self._mock_cfg(compress_prompts_use_llm=False)
        with patch(
                "aiworks_core.logic.prompt_compression._get_compress_cache_func"
        ) as mock_cache_factory:
            mock_cache_func = MagicMock(return_value="llmlingua result")
            mock_cache_factory.return_value = mock_cache_func
            result = asyncio.run(
                compress_prompt("a long text that exceeds the minimum")
            )
        self.assertEqual(result, "llmlingua result")
        mock_invoke.assert_not_called()

    @patch("aiworks_core.logic.prompt_compression.LLMConfiguration.get_solo")
    def test_returns_original_when_compression_disabled(self, mock_get_solo):
        mock_get_solo.return_value = self._mock_cfg(compress_prompts=False)
        result = asyncio.run(compress_prompt("any text"))
        self.assertEqual(result, "any text")

    @patch("aiworks_core.logic.prompt_compression.LLMConfiguration.get_solo")
    def test_returns_original_when_below_min_length(self, mock_get_solo):
        mock_get_solo.return_value = self._mock_cfg(
            compress_prompts_use_llm=True,
            compress_prompts_min_length=1000,
        )
        short = "short"
        result = asyncio.run(compress_prompt(short))
        self.assertEqual(result, short)


class GetChatModelTests(TestCase):
    """Tests for the generic chat model factory (llm.get_chat_model)."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()

    @staticmethod
    def _cfg(**kwargs):
        cfg = LLMConfiguration.get_solo()
        for attr, value in kwargs.items():
            setattr(cfg, attr, value)
        cfg.save()
        return cfg

    @staticmethod
    def _resolve(llm_type=None):
        if llm_type is not None:
            op_name = f"__test_{uuid.uuid4().hex}__"
            LLMOperationConfig.objects.create(
                operation_name=op_name, selected_llm_type=llm_type
            )
            try:
                return _resolve_provider_model(op_name)
            finally:
                LLMOperationConfig.objects.filter(operation_name=op_name).delete()
        else:
            return _resolve_provider_model("test_op")

    # ------------------------------------------------------------------
    # Google / Gemini
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.ChatGoogleGenerativeAI")
    def test_google_provider_returns_gemini_llm(self, mock_cls):
        self._cfg(
            provider="google",
            google_model="gemini-2.0-flash",
            google_api_key="test-key",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["api_key"], "test-key")

    def test_google_provider_raises_when_no_api_key(self):
        self._cfg(provider="google", google_model="gemini-2.0-flash", google_api_key="")
        provider, model, llm_type = self._resolve()
        with self.assertRaises(ValueError):
            get_chat_model(provider, model, llm_type)

    def test_raises_when_model_not_configured(self):
        """get_chat_model() must raise a clear ValueError when no model is set for the provider."""
        self._cfg(provider="google", google_model="", google_api_key="")
        with self.assertRaises(ValueError, msg="No model configured"):
            get_chat_model("google", "", "smart")

    @patch("aiworks_core.logic.llm.ChatGoogleGenerativeAI")
    def test_google_provider_respects_model_field(self, mock_cls):
        self._cfg(provider="google", google_model="gemini-1.5-pro", google_api_key="k")
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["model"], "gemini-1.5-pro")

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.ChatOpenAI")
    def test_openai_provider(self, mock_cls):
        self._cfg(provider="openai", openai_model="gpt-4o", openai_api_key="sk-test")
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["api_key"], "sk-test")
        self.assertEqual(kwargs["model"], "gpt-4o")

    @patch("aiworks_core.logic.llm.ChatOpenAI", MagicMock())
    def test_openai_provider_raises_when_no_api_key(self):
        self._cfg(provider="openai", openai_model="gpt-4o", openai_api_key="")
        provider, model, llm_type = self._resolve()
        with self.assertRaises(ValueError):
            get_chat_model(provider, model, llm_type)

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.ChatAnthropic")
    def test_anthropic_provider(self, mock_cls):
        self._cfg(
            provider="anthropic",
            anthropic_model="claude-3-5-sonnet-20241022",
            anthropic_api_key="ant-key",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["api_key"], "ant-key")

    @patch("aiworks_core.logic.llm.ChatAnthropic", MagicMock())
    def test_anthropic_provider_raises_when_no_api_key(self):
        self._cfg(
            provider="anthropic",
            anthropic_model="claude-3-5-sonnet-20241022",
            anthropic_api_key="",
        )
        provider, model, llm_type = self._resolve()
        with self.assertRaises(ValueError):
            get_chat_model(provider, model, llm_type)

    @patch("aiworks_core.logic.llm.ChatOpenAI")
    def test_openai_provider_with_custom_base_url(self, mock_cls):
        self._cfg(
            provider="openai",
            openai_model="gpt-4o",
            openai_api_key="sk-test",
            openai_base_url="https://api.minimax.chat/v1",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["api_key"], "sk-test")
        self.assertEqual(kwargs["model"], "gpt-4o")
        self.assertEqual(kwargs["base_url"], "https://api.minimax.chat/v1")

    @patch("aiworks_core.logic.llm.ChatOpenAI")
    def test_openai_provider_without_base_url(self, mock_cls):
        self._cfg(provider="openai", openai_model="gpt-4o", openai_api_key="sk-test", openai_base_url="")
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["api_key"], "sk-test")
        self.assertEqual(kwargs["model"], "gpt-4o")
        self.assertNotIn("base_url", kwargs)

    @patch("aiworks_core.logic.llm.ChatAnthropic")
    def test_anthropic_provider_with_custom_base_url(self, mock_cls):
        self._cfg(
            provider="anthropic",
            anthropic_model="claude-3-5-sonnet-20241022",
            anthropic_api_key="ant-key",
            anthropic_base_url="https://api.minimax.chat/v1",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["api_key"], "ant-key")
        self.assertEqual(kwargs["base_url"], "https://api.minimax.chat/v1")

    @patch("aiworks_core.logic.llm.ChatAnthropic")
    def test_anthropic_provider_without_base_url(self, mock_cls):
        self._cfg(
            provider="anthropic",
            anthropic_model="claude-3-5-sonnet-20241022",
            anthropic_api_key="ant-key",
            anthropic_base_url="",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["api_key"], "ant-key")
        self.assertNotIn("base_url", kwargs)

    # ------------------------------------------------------------------
    # Azure
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.AzureChatOpenAI")
    def test_azure_provider(self, mock_cls):
        self._cfg(
            provider="azure",
            azure_api_key="az-key",
            azure_deployment="dep1",
            azure_endpoint="https://example.openai.azure.com",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["api_key"], "az-key")
        self.assertEqual(kwargs["azure_deployment"], "dep1")

    def test_azure_provider_raises_when_no_api_key(self):
        self._cfg(
            provider="azure",
            azure_api_key="",
            azure_deployment="dep1",
            azure_endpoint="https://example.openai.azure.com",
        )
        provider, model, llm_type = self._resolve()
        with self.assertRaises(ValueError):
            get_chat_model(provider, model, llm_type)

    def test_azure_provider_raises_when_no_deployment(self):
        self._cfg(
            provider="azure",
            azure_api_key="az-key",
            azure_deployment="",
            azure_endpoint="https://example.openai.azure.com",
        )
        with self.assertRaises(ValueError, msg="No model configured"):
            provider, model, llm_type = self._resolve()
            get_chat_model(provider, model, llm_type)

    def test_azure_provider_raises_when_no_endpoint(self):
        self._cfg(
            provider="azure",
            azure_api_key="az-key",
            azure_deployment="dep1",
            azure_endpoint="",
        )
        provider, model, llm_type = self._resolve()
        with self.assertRaises(ValueError):
            get_chat_model(provider, model, llm_type)

    @patch("aiworks_core.logic.llm.AzureChatOpenAI")
    def test_azure_fast_provider_uses_fast_deployment(self, mock_cls):
        self._cfg(
            provider="azure",
            azure_api_key="az-key",
            azure_deployment="smart-dep",
            azure_fast_deployment="fast-dep",
            azure_endpoint="https://example.openai.azure.com",
        )
        provider, model, llm_type = self._resolve("fast")
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["azure_deployment"], "fast-dep")

    @patch("aiworks_core.logic.llm.AzureChatOpenAI")
    def test_azure_reasoning_provider_uses_reasoning_deployment(self, mock_cls):
        self._cfg(
            provider="azure",
            azure_api_key="az-key",
            azure_deployment="smart-dep",
            azure_reasoning_deployment="reason-dep",
            azure_endpoint="https://example.openai.azure.com",
        )
        provider, model, llm_type = self._resolve("reasoning")
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["azure_deployment"], "reason-dep")

    @patch("aiworks_core.logic.llm.AzureChatOpenAI")
    def test_azure_fast_fallback_to_smart_deployment(self, mock_cls):
        """When azure_fast_deployment is blank, fast falls back to azure_deployment."""
        self._cfg(
            provider="azure",
            azure_api_key="az-key",
            azure_deployment="smart-dep",
            azure_fast_deployment="",
            azure_endpoint="https://example.openai.azure.com",
        )
        provider, model, llm_type = self._resolve("fast")
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["azure_deployment"], "smart-dep")

    @patch("aiworks_core.logic.llm.AzureChatOpenAI")
    def test_azure_fast_only_no_smart_deployment_succeeds(self, mock_cls):
        """fast_deployment set but azure_deployment blank must NOT raise (bug fix)."""
        self._cfg(
            provider="azure",
            azure_api_key="az-key",
            azure_deployment="",
            azure_fast_deployment="fast-only-dep",
            azure_endpoint="https://example.openai.azure.com",
        )
        provider, model, llm_type = self._resolve("fast")
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["azure_deployment"], "fast-only-dep")

    # ------------------------------------------------------------------
    # AWS Bedrock
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.ChatBedrockConverse")
    def test_aws_provider(self, mock_cls):
        self._cfg(
            provider="aws",
            aws_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            aws_region="eu-west-1",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type, temperature=0.5)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["region_name"], "eu-west-1")
        self.assertEqual(kwargs["temperature"], 0.5)

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.ChatOllama")
    def test_ollama_provider(self, mock_cls):
        self._cfg(
            provider="ollama",
            ollama_model="llama3.2",
            ollama_base_url="http://localhost:11434",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["base_url"], "http://localhost:11434")
        self.assertEqual(kwargs["model"], "llama3.2")

    # ------------------------------------------------------------------
    # HuggingFace
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.ChatHuggingFace")
    @patch("aiworks_core.logic.llm.HuggingFaceEndpoint")
    def test_huggingface_provider(self, mock_endpoint_cls, mock_chat_cls):
        self._cfg(
            provider="huggingface",
            huggingface_model="mistralai/Mistral-7B-Instruct-v0.3",
            huggingface_api_key="hf-key",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        mock_endpoint_cls.assert_called_once()
        mock_chat_cls.assert_called_once()

    @patch("aiworks_core.logic.llm.ChatHuggingFace", MagicMock())
    @patch("aiworks_core.logic.llm.HuggingFaceEndpoint", MagicMock())
    def test_huggingface_provider_raises_when_no_api_key(self):
        self._cfg(
            provider="huggingface",
            huggingface_model="mistralai/Mistral-7B-Instruct-v0.3",
            huggingface_api_key="",
        )
        provider, model, llm_type = self._resolve()
        with self.assertRaises(ValueError):
            get_chat_model(provider, model, llm_type)

    # ------------------------------------------------------------------
    # DeepSeek
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.ChatDeepSeek")
    def test_deepseek_provider(self, mock_cls):
        self._cfg(
            provider="deepseek",
            deepseek_model="deepseek-chat",
            deepseek_api_key="ds-key",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["model"], "deepseek-chat")

    @patch("aiworks_core.logic.llm.ChatDeepSeek", MagicMock())
    def test_deepseek_provider_raises_when_no_api_key(self):
        self._cfg(
            provider="deepseek", deepseek_model="deepseek-chat", deepseek_api_key=""
        )
        provider, model, llm_type = self._resolve()
        with self.assertRaises(ValueError):
            get_chat_model(provider, model, llm_type)

    # ------------------------------------------------------------------
    # OpenRouter
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended")
    def test_openrouter_provider(self, mock_cls):
        self._cfg(
            provider="openrouter",
            openrouter_model="openai/gpt-4o",
            openrouter_api_key="or-key",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["model"], "openai/gpt-4o")
        self.assertEqual(kwargs["api_key"], "or-key")

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended", MagicMock())
    def test_openrouter_provider_raises_when_no_api_key(self):
        self._cfg(
            provider="openrouter",
            openrouter_model="openai/gpt-4o",
            openrouter_api_key="",
        )
        provider, model, llm_type = self._resolve()
        with self.assertRaises(ValueError):
            get_chat_model(provider, model, llm_type)

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended")
    def test_openrouter_fast_model(self, mock_cls):
        self._cfg(
            provider="openrouter",
            openrouter_model="openai/gpt-4o",
            openrouter_fast_model="openai/gpt-4o-mini",
            openrouter_api_key="or-key",
        )
        provider, model, llm_type = self._resolve("fast")
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended")
    def test_openrouter_reasoning_model(self, mock_cls):
        self._cfg(
            provider="openrouter",
            openrouter_model="openai/gpt-4o",
            openrouter_reasoning_model="anthropic/claude-opus-4-6",
            openrouter_api_key="or-key",
        )
        provider, model, llm_type = self._resolve("reasoning")
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["model"], "anthropic/claude-opus-4-6")

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended")
    def test_openrouter_reasoning_injects_effort_when_configured(self, mock_cls):
        """llm_type='reasoning' + effort configured passes reasoning directly to ChatOpenRouter."""
        self._cfg(
            provider="openrouter",
            openrouter_model="openai/gpt-4o",
            openrouter_reasoning_model="anthropic/claude-opus-4-6",
            openrouter_reasoning_effort="high",
            openrouter_api_key="or-key",
        )
        provider, model, llm_type = self._resolve("reasoning")
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs.get("reasoning"), {"effort": "high", "include_reasoning": True})

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended")
    def test_openrouter_reasoning_no_kwarg_when_effort_blank(self, mock_cls):
        """llm_type='reasoning' with blank effort does not pass reasoning to ChatOpenRouter."""
        self._cfg(
            provider="openrouter",
            openrouter_model="openai/gpt-4o",
            openrouter_reasoning_model="anthropic/claude-opus-4-6",
            openrouter_reasoning_effort="",
            openrouter_api_key="or-key",
        )
        provider, model, llm_type = self._resolve("reasoning")
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertNotIn("reasoning", kwargs)

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended")
    def test_openrouter_smart_call_no_reasoning(self, mock_cls):
        """Smart calls (llm_type=None) never pass reasoning even when effort is configured."""
        self._cfg(
            provider="openrouter",
            openrouter_model="openai/gpt-4o",
            openrouter_reasoning_effort="high",
            openrouter_api_key="or-key",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertNotIn("reasoning", kwargs)

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended")
    def test_openrouter_fast_call_no_reasoning(self, mock_cls):
        """Fast calls (llm_type='fast') never pass reasoning even when effort is configured."""
        self._cfg(
            provider="openrouter",
            openrouter_model="openai/gpt-4o",
            openrouter_fast_model="openai/gpt-4o-mini",
            openrouter_reasoning_effort="high",
            openrouter_api_key="or-key",
        )
        provider, model, llm_type = self._resolve("fast")
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertNotIn("reasoning", kwargs)

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended")
    def test_openrouter_provider_suffix_injects_only(self, mock_cls):
        """Provider suffix <PROVIDER> injects only into model_kwargs."""
        self._cfg(
            provider="openrouter",
            openrouter_model="minimax_27<MYPROVIDER>",
            openrouter_api_key="or-key",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["model"], "minimax_27")
        self.assertEqual(kwargs["model_kwargs"]["provider"]["order"], ["MYPROVIDER"])
        self.assertEqual(kwargs["model_kwargs"]["provider"]["allow_fallbacks"], False)

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended")
    def test_openrouter_provider_suffix_injects_only_and_fallbacks(self, mock_cls):
        """Provider suffix <PROVIDER> injects only + allow_fallbacks into model_kwargs."""
        self._cfg(
            provider="openrouter",
            openrouter_model="minimax_27<MYPROVIDER+>",
            openrouter_api_key="or-key",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["model"], "minimax_27")
        self.assertEqual(kwargs["model_kwargs"]["provider"]["order"], ["MYPROVIDER"])
        self.assertEqual(kwargs["model_kwargs"]["provider"]["allow_fallbacks"], True)

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended")
    def test_openrouter_provider_suffix_injects_multi_providers_support(self, mock_cls):
        """Provider suffix <MYPROVIDER1,MYPROVIDER2> multi support into model_kwargs."""
        self._cfg(
            provider="openrouter",
            openrouter_model="minimax_27<MYPROVIDER1,MYPROVIDER2>",
            openrouter_api_key="or-key",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["model"], "minimax_27")
        self.assertEqual(kwargs["model_kwargs"]["provider"]["order"], ["MYPROVIDER1","MYPROVIDER2"])
        self.assertEqual(kwargs["model_kwargs"]["provider"]["allow_fallbacks"], False)

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended")
    def test_openrouter_provider_suffix_no_suffix_unchanged(self, mock_cls):
        """No suffix means no provider routing dict injected."""
        self._cfg(
            provider="openrouter",
            openrouter_model="openai/gpt-4o",
            openrouter_api_key="or-key",
        )
        provider, model, llm_type = self._resolve()
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["model"], "openai/gpt-4o")
        self.assertNotIn("model_kwargs", kwargs)

    @patch("aiworks_core.logic.llm.ChatOpenRouterExtended")
    def test_openrouter_provider_suffix_reasoning_merges_both(self, mock_cls):
        """Provider suffix + reasoning effort: provider routing merged with reasoning."""
        self._cfg(
            provider="openrouter",
            openrouter_model="openai/gpt-4o",
            openrouter_reasoning_model="anthropic/claude-opus-4-6<MYPROVIDER>",
            openrouter_reasoning_effort="high",
            openrouter_api_key="or-key",
        )
        provider, model, llm_type = self._resolve("reasoning")
        get_chat_model(provider, model, llm_type)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["model"], "anthropic/claude-opus-4-6")
        self.assertEqual(kwargs["reasoning"], {"effort": "high", "include_reasoning": True})
        self.assertEqual(kwargs["model_kwargs"]["provider"]["order"], ["MYPROVIDER"])
        self.assertEqual(kwargs["model_kwargs"]["provider"]["allow_fallbacks"], False)
        self.assertEqual(kwargs["model_kwargs"]["provider"]["require_reasoning"], True)
        self.assertEqual(kwargs["model_kwargs"]["provider"]["sort"], "throughput")

    # ------------------------------------------------------------------
    # Unknown provider
    # ------------------------------------------------------------------

    def test_unknown_provider_raises(self):
        self._cfg(provider="unknown_provider")
        with self.assertRaises(ValueError):
            get_chat_model("unknown_provider", "my_model", "smart")


class ResolveProviderModelModelSelectionTests(TestCase):
    """Tests for model selection (fast/smart/reasoning) in _resolve_provider_model."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()

    @staticmethod
    def _cfg(**kwargs):
        cfg = LLMConfiguration.get_solo()
        for attr, value in kwargs.items():
            setattr(cfg, attr, value)
        cfg.save()
        return cfg

    @staticmethod
    def _resolve(llm_type=None):
        if llm_type is not None:
            op_name = f"__test_{uuid.uuid4().hex}__"
            LLMOperationConfig.objects.create(
                operation_name=op_name, selected_llm_type=llm_type
            )
            try:
                return _resolve_provider_model(op_name)
            finally:
                LLMOperationConfig.objects.filter(operation_name=op_name).delete()
        else:
            return _resolve_provider_model("test_op")

    def test_fast_type_uses_fast_model_when_set(self):
        """When llm_type='fast' and google_fast_model is set, it should use it."""
        self._cfg(
            provider="google",
            google_api_key="k",
            google_model="gemini-pro",
            google_fast_model="gemini-flash",
        )
        provider, model, llm_type = self._resolve("fast")
        self.assertEqual(llm_type, "fast")
        self.assertEqual(provider, "google")
        self.assertEqual(model, "gemini-flash")

    def test_fast_type_falls_back_to_smart_model_when_fast_model_blank(self):
        """When llm_type='fast' but google_fast_model is blank, it should fall back to google_model."""
        self._cfg(
            provider="google",
            google_api_key="k",
            google_model="gemini-pro",
            google_fast_model="",
        )
        provider, model, llm_type = self._resolve("fast")
        self.assertEqual(llm_type, "fast")
        self.assertEqual(provider, "google")
        self.assertEqual(model, "gemini-pro")

    def test_smart_type_uses_smart_model(self):
        """When llm_type is None (default), it should use the smart model."""
        self._cfg(
            provider="google",
            google_api_key="k",
            google_model="gemini-pro",
            google_fast_model="gemini-flash",
        )
        provider, model, llm_type = self._resolve()
        self.assertEqual(llm_type, "smart")
        self.assertEqual(provider, "google")
        self.assertEqual(model, "gemini-pro")

    def test_explicit_smart_type_uses_smart_model(self):
        """When llm_type='smart', it should use the smart model."""
        self._cfg(
            provider="google",
            google_api_key="k",
            google_model="gemini-pro",
            google_fast_model="gemini-flash",
        )
        provider, model, llm_type = self._resolve()
        self.assertEqual(llm_type, "smart")
        self.assertEqual(provider, "google")
        self.assertEqual(model, "gemini-pro")


class ResolveProviderModelReasoningTests(TestCase):
    """Tests for llm_type='reasoning' model selection in _resolve_provider_model."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()

    @staticmethod
    def _cfg(**kwargs):
        cfg = LLMConfiguration.get_solo()
        for attr, value in kwargs.items():
            setattr(cfg, attr, value)
        cfg.save()
        return cfg

    @staticmethod
    def _resolve(llm_type=None):
        if llm_type is not None:
            op_name = f"__test_{uuid.uuid4().hex}__"
            LLMOperationConfig.objects.create(
                operation_name=op_name, selected_llm_type=llm_type
            )
            try:
                return _resolve_provider_model(op_name)
            finally:
                LLMOperationConfig.objects.filter(operation_name=op_name).delete()
        else:
            return _resolve_provider_model("test_op")

    def test_reasoning_type_uses_reasoning_model_when_set(self):
        """When llm_type='reasoning' and google_reasoning_model is set, it should use it."""
        self._cfg(
            provider="google",
            google_api_key="k",
            google_model="gemini-pro",
            google_reasoning_model="gemini-think",
        )
        provider, model, llm_type = self._resolve("reasoning")
        self.assertEqual(llm_type, "reasoning")
        self.assertEqual(provider, "google")
        self.assertEqual(model, "gemini-think")

    def test_reasoning_type_falls_back_to_smart_model_when_reasoning_model_blank(self):
        """When llm_type='reasoning' but google_reasoning_model is blank, it should fall back to google_model."""
        self._cfg(
            provider="google",
            google_api_key="k",
            google_model="gemini-pro",
            google_reasoning_model="",
        )
        provider, model, llm_type = self._resolve("reasoning")
        self.assertEqual(llm_type, "reasoning")
        self.assertEqual(provider, "google")
        self.assertEqual(model, "gemini-pro")

    def test_none_type_unaffected_by_reasoning_model(self):
        """When llm_type is None, google_reasoning_model has no effect — smart model is used."""
        self._cfg(
            provider="google",
            google_api_key="k",
            google_model="gemini-pro",
            google_reasoning_model="gemini-think",
        )
        provider, model, llm_type = self._resolve()
        self.assertEqual(llm_type, "smart")
        self.assertEqual(model, "gemini-pro")

    def test_fast_type_unaffected_by_reasoning_model(self):
        """When llm_type='fast', google_reasoning_model has no effect — google_fast_model is used."""
        self._cfg(
            provider="google",
            google_api_key="k",
            google_model="gemini-pro",
            google_fast_model="gemini-flash",
            google_reasoning_model="gemini-think",
        )
        provider, model, llm_type = self._resolve("fast")
        self.assertEqual(llm_type, "fast")
        self.assertEqual(model, "gemini-flash")


class ResolveProviderModelProviderSelectionTests(TestCase):
    """Tests for per-task-type provider selection via fast_provider / reasoning_provider in _resolve_provider_model."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()

    @staticmethod
    def _cfg(**kwargs):
        cfg = LLMConfiguration.get_solo()
        for attr, value in kwargs.items():
            setattr(cfg, attr, value)
        cfg.save()
        return cfg

    @staticmethod
    def _resolve(llm_type=None):
        if llm_type is not None:
            op_name = f"__test_{uuid.uuid4().hex}__"
            LLMOperationConfig.objects.create(
                operation_name=op_name, selected_llm_type=llm_type
            )
            try:
                return _resolve_provider_model(op_name)
            finally:
                LLMOperationConfig.objects.filter(operation_name=op_name).delete()
        else:
            return _resolve_provider_model("test_op")

    def test_fast_provider_overrides_main_provider_for_fast_tasks(self):
        """When fast_provider='openai', llm_type='fast' uses openai, not google."""
        self._cfg(
            provider="google",
            google_api_key="gk",
            google_model="gemini-pro",
            fast_provider="openai",
            openai_api_key="sk",
            openai_fast_model="gpt-4o-mini",
        )
        provider, model, llm_type = self._resolve("fast")
        self.assertEqual(llm_type, "fast")
        self.assertEqual(provider, "openai")
        self.assertEqual(model, "gpt-4o-mini")

    def test_reasoning_provider_overrides_main_provider_for_reasoning_tasks(self):
        """When reasoning_provider='anthropic', llm_type='reasoning' uses anthropic."""
        self._cfg(
            provider="google",
            google_api_key="gk",
            google_model="gemini-pro",
            reasoning_provider="anthropic",
            anthropic_api_key="ant",
            anthropic_reasoning_model="claude-opus-4-6",
        )
        provider, model, llm_type = self._resolve("reasoning")
        self.assertEqual(llm_type, "reasoning")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(model, "claude-opus-4-6")

    def test_fast_provider_blank_falls_back_to_main_provider(self):
        """When fast_provider is blank, llm_type='fast' falls back to the main provider."""
        self._cfg(
            provider="google",
            google_api_key="gk",
            google_model="gemini-pro",
            google_fast_model="gemini-flash",
            fast_provider="",
        )
        provider, model, llm_type = self._resolve("fast")
        self.assertEqual(llm_type, "fast")
        self.assertEqual(provider, "google")
        self.assertEqual(model, "gemini-flash")

    def test_reasoning_provider_blank_falls_back_to_main_provider(self):
        """When reasoning_provider is blank, llm_type='reasoning' falls back to the main provider."""
        self._cfg(
            provider="google",
            google_api_key="gk",
            google_model="gemini-pro",
            google_reasoning_model="gemini-think",
            reasoning_provider="",
        )
        provider, model, llm_type = self._resolve("reasoning")
        self.assertEqual(llm_type, "reasoning")
        self.assertEqual(provider, "google")
        self.assertEqual(model, "gemini-think")

    def test_cross_provider_all_three_types(self):
        """smart=google, fast=openai, reasoning=anthropic — each task type hits the right provider."""
        self._cfg(
            provider="google",
            google_api_key="gk",
            google_model="gemini-pro",
            fast_provider="openai",
            openai_api_key="sk",
            openai_fast_model="gpt-4o-mini",
            reasoning_provider="anthropic",
            anthropic_api_key="ant",
            anthropic_reasoning_model="claude-opus-4-6",
        )
        provider, model, llm_type = self._resolve()
        self.assertEqual(llm_type, "smart")
        self.assertEqual(provider, "google")
        self.assertEqual(model, "gemini-pro")

        provider, model, llm_type = self._resolve("fast")
        self.assertEqual(llm_type, "fast")
        self.assertEqual(provider, "openai")
        self.assertEqual(model, "gpt-4o-mini")

        provider, model, llm_type = self._resolve("reasoning")
        self.assertEqual(llm_type, "reasoning")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(model, "claude-opus-4-6")


class GetAgentTests(TestCase):
    """Tests for the agent factory (llm.get_agent)."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()
        cfg = LLMConfiguration.get_solo()
        cfg.provider = "google"
        cfg.google_model = "gemini-2.0-flash"
        cfg.google_api_key = "k"
        cfg.save()

    @patch("aiworks_core.logic.llm.create_agent")
    @patch("aiworks_core.logic.llm.ChatGoogleGenerativeAI")
    def test_get_agent_react_default(self, _mock_llm_cls, mock_agent_fn):
        """get_agent() without use_deep_agent uses create_agent."""
        provider, model, llm_type = _resolve_provider_model("test_op")
        get_agent(provider, model, llm_type, tools=[])
        mock_agent_fn.assert_called_once()

    @patch("aiworks_core.logic.llm.create_deep_agent")
    @patch("aiworks_core.logic.llm.ChatGoogleGenerativeAI")
    def test_get_agent_deep(self, _mock_llm_cls, mock_deep_fn):
        """get_agent(use_deep_agent=True) uses create_deep_agent."""
        provider, model, llm_type = _resolve_provider_model("test_op")
        get_agent(provider, model, llm_type, tools=[], use_deep_agent=True)
        mock_deep_fn.assert_called_once()

    def test_get_agent_has_no_system_prompt_parameter(self):
        """get_agent() should not accept a system_prompt parameter."""
        sig = inspect.signature(get_agent)
        self.assertNotIn("system_prompt", sig.parameters)


class InvokeLlmAgentRecursionLimitTests(TestCase):
    """Tests that invoke_llm passes recursion_limit via config when using an agent."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()
        cfg = LLMConfiguration.get_solo()
        cfg.provider = "google"
        cfg.google_model = "gemini-2.0-flash"
        cfg.google_api_key = "k"
        cfg.save()

    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.get_agent")
    @patch("aiworks_core.logic.llm.get_chat_model")
    def test_agent_invoke_passes_recursion_limit_in_config(
            self, _mock_get_chat_model, mock_get_agent, _mock_get_solo, _mock_debug_create
    ):
        """invoke_llm with is_agent=True should pass recursion_limit in config, not in input."""

        mock_agent = MagicMock()
        captured = {}

        async def capturing_ainvoke(inputs, config=None):
            captured["inputs"] = inputs
            captured["config"] = config
            return {"messages": [AIMessage(content="ok")]}

        mock_agent.ainvoke = capturing_ainvoke
        mock_get_agent.return_value = mock_agent

        asyncio.run(
            invoke_llm(
                "test_operation",
                messages=[SystemMessage(content="sys"), HumanMessage(content="user")],
                is_agent=True,
                tools=[],
                parse_json=False,
            )
        )

        self.assertIn("config", captured)
        self.assertIn("recursion_limit", captured["config"])
        self.assertEqual(
            captured["config"]["recursion_limit"], _MAX_AGENT_ITERATIONS * 3
        )

        # Ensure ignored keys are NOT in the input dict
        self.assertNotIn("max_iterations", captured.get("inputs", {}))
        self.assertNotIn("max_steps", captured.get("inputs", {}))
        self.assertNotIn("max_hops", captured.get("inputs", {}))
        self.assertNotIn("max_calls", captured.get("inputs", {}))


class InitAgentBackendContextVarTests(TestCase):
    """Tests that _agent_backend_var is properly set and cleared around agent.ainvoke().

    This verifies the ContextVar-based approach works: init_agent_backend() is called
    in the main async thread before agent.ainvoke(), and tools running in thread pool
    workers (via run_in_executor) can read the value via copy_context().
    """

    def setUp(self):
        LLMConfiguration.objects.all().delete()
        cfg = LLMConfiguration.get_solo()
        cfg.provider = "google"
        cfg.google_model = "gemini-2.0-flash"
        cfg.google_api_key = "k"
        cfg.save()

    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.get_agent")
    @patch("aiworks_core.logic.llm.get_chat_model")
    @patch("aiworks_core.logic.llm.OverridingStateBackend")
    def test_agent_backend_visible_to_tools_via_context_var(
            self, _mock_backend_cls, _mock_get_chat_model, mock_get_agent, _mock_get_solo, _mock_debug_create
    ):
        """Tools running in thread pool see the backend set in the main thread via ContextVar."""
        from aiworks_core.logic.llm import get_agent_backend

        captured_tool_backend = {}

        async def capturing_ainvoke(inputs, config=None):
            """Simulates agent.ainvoke: runs a tool via run_in_executor to test ContextVar propagation."""
            from langchain_core.runnables.config import run_in_executor

            def tool_that_reads_backend():
                captured_tool_backend["backend"] = get_agent_backend()

            # Simulate what deep agent does: run tool in thread pool via run_in_executor
            await run_in_executor(
                None,
                tool_that_reads_backend,
            )
            return {"messages": [AIMessage(content="ok")]}

        mock_agent = MagicMock()
        mock_agent.ainvoke = capturing_ainvoke
        mock_get_agent.return_value = mock_agent

        asyncio.run(
            invoke_llm(
                "test_operation",
                messages=[SystemMessage(content="sys"), HumanMessage(content="user")],
                is_agent=True,
                is_deep_agent=True,
                tools=[],
                parse_json=False,
            )
        )

        self.assertIsNotNone(
            captured_tool_backend.get("backend"),
            "Backend should be visible to tools via ContextVar copy_context()"
        )


class AgentLoggingCallbackHandlerTests(TestCase):
    """Tests for the _AgentLoggingCallbackHandler callback."""

    def setUp(self):
        self.handler = _AgentLoggingCallbackHandler()

    def test_on_tool_start_logs_tool_name_and_input(self):
        """on_tool_start logs the tool name and truncated input."""
        with self.assertLogs("aiworks_core.logic.llm", level="INFO") as cm:
            self.handler.on_tool_start({"name": "search"}, "test query")
        self.assertTrue(
            any("AGENT TOOL START" in line and "search" in line for line in cm.output)
        )

    def test_on_tool_start_truncates_long_input(self):
        """on_tool_start truncates input to 500 characters."""
        long_input = "x" * 1000
        with self.assertLogs("aiworks_core.logic.llm", level="INFO") as cm:
            self.handler.on_tool_start({"name": "search"}, long_input)
        logged = next(line for line in cm.output if "AGENT TOOL START" in line)
        self.assertIn("x" * 500, logged)
        self.assertNotIn("x" * 501, logged)

    def test_on_tool_start_handles_non_string_input(self):
        """on_tool_start converts non-string input to string before truncating."""
        with self.assertLogs("aiworks_core.logic.llm", level="INFO") as cm:
            self.handler.on_tool_start({"name": "search"}, "TEST")
        self.assertTrue(any("AGENT TOOL START" in line for line in cm.output))

    def test_on_tool_end_logs_output_length(self):
        """on_tool_end logs the output length."""
        with self.assertLogs("aiworks_core.logic.llm", level="INFO") as cm:
            self.handler.on_tool_end("some result")
        self.assertTrue(any("AGENT TOOL END" in line for line in cm.output))

    def test_on_tool_error_logs_warning(self):
        """on_tool_error logs a warning with the error."""
        with self.assertLogs("aiworks_core.logic.llm", level="WARNING") as cm:
            self.handler.on_tool_error(RuntimeError("boom"))
        self.assertTrue(
            any("AGENT TOOL ERROR" in line and "boom" in line for line in cm.output)
        )

    def test_on_llm_start_logs_model_name(self):
        """on_llm_start logs the model name and prompts count."""
        with self.assertLogs("aiworks_core.logic.llm", level="INFO") as cm:
            self.handler.on_llm_start({"name": "gpt-4"}, ["hello"])
        self.assertTrue(
            any("AGENT LLM ITERATION" in line and "gpt-4" in line for line in cm.output)
        )

    def test_on_chat_model_start_logs_model_and_message_count(self):
        """on_chat_model_start logs the model name and total message count."""
        with self.assertLogs("aiworks_core.logic.llm", level="INFO") as cm:
            self.handler.on_chat_model_start({"name": "gemini"}, [["msg1", "msg2"]])
        self.assertTrue(
            any(
                "AGENT LLM ITERATION" in line and "gemini" in line for line in cm.output
            )
        )


class AgentCallbacksHandlerTests(TestCase):
    """Tests for the _AgentCallbacksHandler callback."""

    @staticmethod
    def _make_callbacks(**handlers):
        return AgentCallbacks(**handlers)

    def test_on_tool_call_fires_callback(self):
        """on_tool_start calls on_tool_call with tool name and input."""
        received = []
        callbacks = self._make_callbacks(on_tool_call=received.append)
        handler = _AgentCallbacksHandler(callbacks)
        handler.on_tool_start({"name": "search"}, '{"query": "test"}')
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["tool"], "search")
        self.assertEqual(received[0]["input"], '{"query": "test"}')

    def test_on_tool_call_non_write_todos_still_fires(self):
        """on_tool_call is triggered for ALL tools, not just write_todos."""
        received = []
        callbacks = self._make_callbacks(on_tool_call=received.append)
        handler = _AgentCallbacksHandler(callbacks)
        handler.on_tool_start({"name": "extract"}, "https://example.com")
        self.assertEqual(received[0]["tool"], "extract")

    def test_on_tool_call_error_is_swallowed(self):
        """Errors in on_tool_call are caught and logged, not propagated."""

        def bad_callback(_):
            raise RuntimeError("callback error")

        callbacks = self._make_callbacks(on_tool_call=bad_callback)
        handler = _AgentCallbacksHandler(callbacks)
        # Must not raise
        with self.assertLogs("aiworks_core.logic.llm", level="WARNING"):
            handler.on_tool_start({"name": "search"}, "query")

    def test_write_todos_invokes_on_todos_update(self):
        """on_tool_start for write_todos calls on_todos_update."""
        received = []
        callbacks = self._make_callbacks(on_todos_update=received.append)
        handler = _AgentCallbacksHandler(callbacks)
        todos = [{"content": "Task A", "status": "completed"}]
        handler.on_tool_start(
            {"name": "write_todos"},
            json.dumps({"todos": todos}),
            inputs={"todos": todos},
        )
        self.assertEqual(received, [todos])

    def test_write_todos_with_inputs_dict(self):
        """write_todos prefers the structured inputs dict over raw JSON."""
        received = []
        callbacks = self._make_callbacks(on_todos_update=received.append)
        handler = _AgentCallbacksHandler(callbacks)
        todos = [{"content": "Do X", "status": "pending"}]
        handler.on_tool_start(
            {"name": "write_todos"},
            "bad json",
            inputs={"todos": todos},
        )
        self.assertEqual(received, [todos])

    def test_write_todos_callback_error_is_swallowed(self):
        """Errors in on_todos_update are caught and logged, not propagated."""

        def bad_callback(_):
            raise RuntimeError("DB down")

        callbacks = self._make_callbacks(on_todos_update=bad_callback)
        handler = _AgentCallbacksHandler(callbacks)
        with self.assertLogs("aiworks_core.logic.llm", level="WARNING"):
            handler.on_tool_start(
                {"name": "write_todos"},
                json.dumps({"todos": [{"content": "X", "status": "pending"}]}),
            )

    def test_non_write_todos_does_not_invoke_on_todos_update(self):
        """Non-write_todos tools do not trigger on_todos_update."""
        received = []
        callbacks = self._make_callbacks(on_todos_update=received.append)
        handler = _AgentCallbacksHandler(callbacks)
        handler.on_tool_start({"name": "search"}, "query")
        self.assertEqual(received, [])

    def test_on_llm_end_fires_on_thinking_update(self):
        """on_llm_end extracts thinking blocks and calls on_thinking_update."""
        received = []
        callbacks = self._make_callbacks(on_thinking_update=received.append)
        handler = _AgentCallbacksHandler(callbacks)

        msg = AIMessage(content=[{"type": "thinking", "thinking": "My reasoning"}])
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])
        handler.on_llm_end(result)

        self.assertEqual(received, ["My reasoning"])

    def test_on_llm_end_multiple_thinking_blocks(self):
        """Multiple thinking blocks in one response are all reported."""
        received = []
        callbacks = self._make_callbacks(on_thinking_update=received.append)
        handler = _AgentCallbacksHandler(callbacks)

        msg = AIMessage(content=[
            {"type": "thinking", "thinking": "First"},
            {"type": "thinking", "thinking": "Second"},
        ])
        gen = ChatGeneration(message=msg, text="")
        handler.on_llm_end(LLMResult(generations=[[gen]]))

        self.assertEqual(received, ["First", "Second"])

    def test_on_llm_end_error_is_swallowed(self):
        """Errors in on_thinking_update are caught and logged, not propagated."""

        def bad_callback(_):
            raise RuntimeError("callback error")

        callbacks = self._make_callbacks(on_thinking_update=bad_callback)
        handler = _AgentCallbacksHandler(callbacks)

        msg = AIMessage(content=[{"type": "thinking", "thinking": "X"}])
        gen = ChatGeneration(message=msg, text="")
        with self.assertLogs("aiworks_core.logic.llm", level="WARNING"):
            handler.on_llm_end(LLMResult(generations=[[gen]]))

    def test_on_llm_end_no_op_when_no_thinking_callback(self):
        """on_llm_end does nothing when on_thinking_update is None."""
        callbacks = self._make_callbacks(on_thinking_update=None)
        handler = _AgentCallbacksHandler(callbacks)
        # Must not raise
        msg = AIMessage(content=[{"type": "thinking", "thinking": "X"}])
        gen = ChatGeneration(message=msg, text="")
        handler.on_llm_end(LLMResult(generations=[[gen]]))


class AgentCallbacksDataclassTests(TestCase):
    """Tests for the AgentCallbacks dataclass."""

    def test_all_fields_optional(self):
        """AgentCallbacks can be instantiated with no arguments."""
        c = AgentCallbacks()
        self.assertIsNone(c.on_todos_update)
        self.assertIsNone(c.on_thinking_update)
        self.assertIsNone(c.on_tool_call)

    def test_fields_set_via_constructor(self):
        """All three callbacks can be set via constructor kwargs."""
        cb1 = MagicMock()
        cb2 = MagicMock()
        cb3 = MagicMock()
        c = AgentCallbacks(
            on_todos_update=cb1,
            on_thinking_update=cb2,
            on_tool_call=cb3,
        )
        self.assertIs(c.on_todos_update, cb1)
        self.assertIs(c.on_thinking_update, cb2)
        self.assertIs(c.on_tool_call, cb3)


class ExtractTodosTests(TestCase):
    """Tests for the _extract_todos utility function."""

    def test_extract_todos_from_inputs_dict(self):
        """_extract_todos prefers the structured inputs dict."""
        todos = [{"content": "Do X", "status": "pending"}]
        result = _extract_todos("{}", {"todos": todos})
        self.assertEqual(result, todos)

    def test_extract_todos_from_input_str(self):
        """_extract_todos falls back to parsing the raw JSON string."""
        todos = [{"content": "Do Y", "status": "completed"}]
        result = _extract_todos(json.dumps({"todos": todos}), None)
        self.assertEqual(result, todos)

    def test_extract_todos_returns_none_on_bad_json(self):
        """_extract_todos returns None when both sources fail."""
        result = _extract_todos("not json", None)
        self.assertIsNone(result)


class RetryLoggingTests(TestCase):
    """Tests that tenacity logs the exception on retry."""

    def test_invoke_llm_has_before_sleep_configured(self):
        """invoke_llm's @retry decorator should have before_sleep set."""
        # tenacity stores the retry instance on the wrapped function
        self.assertTrue(hasattr(invoke_llm, "retry"))
        self.assertIsNotNone(invoke_llm.retry.before_sleep)

    def test_before_sleep_logs_warning_to_llm_logger(self):
        """The before_sleep callback logs a WARNING to the 'llm' logger on retry."""

        callback = before_sleep_log(
            cast(LoggerProtocol, cast(object, logging.getLogger("aiworks_core.logic.llm"))),
            logging.WARNING,
        )

        retry_state = MagicMock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = RuntimeError("transient error")
        retry_state.fn = "invoke_llm"
        retry_state.next_action.sleep = 2

        with self.assertLogs("aiworks_core.logic.llm", level="WARNING") as cm:
            callback(retry_state)

        self.assertTrue(any("Retrying" in line for line in cm.output))


class AuditLlmCallTests(TestCase):
    """Tests for the _audit_llm_call debug-logging helper.

    All database interactions are mocked to avoid SQLite table-lock errors
    that arise when sync_to_async dispatches DB work to a worker thread while
    the Django TestCase transaction is held on the main thread.
    """

    @staticmethod
    def _messages():
        return [SystemMessage(content="sys"), HumanMessage(content="user")]

    @staticmethod
    def _enabled_cfg():
        mock_cfg = MagicMock()
        mock_cfg.llm_debug = True
        return mock_cfg

    @staticmethod
    def _disabled_cfg():
        mock_cfg = MagicMock()
        mock_cfg.llm_debug = False
        return mock_cfg

    # ------------------------------------------------------------------
    # debug disabled
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_creates_metadata_row_when_debug_disabled(self, mock_get_solo):
        """A LLMDebugLog row is always created; raw payloads are omitted when debug=False."""
        mock_get_solo.return_value = self._disabled_cfg()

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create") as mock_create:
            asyncio.run(
                _audit_llm_call(
                    "test_op",
                    "openai",
                    "gpt-4o",
                    "label",
                    self._messages(),
                    "output",
                    None,
                )
            )
            mock_create.assert_called_once()
            kwargs = mock_create.call_args[1]
            # Raw payload fields must be empty when debug is off and no error
            self.assertEqual(kwargs["input_messages"], "")
            self.assertEqual(kwargs["raw_output"], "")
            self.assertEqual(kwargs["agent_history"], "")
            self.assertEqual(kwargs["agent_tasks"], "")
            # Metadata fields must still be populated
            self.assertEqual(kwargs["call_label"], "label")
            self.assertEqual(kwargs["error"], "")

    # ------------------------------------------------------------------
    # debug enabled – success path
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_creates_log_on_success(self, mock_get_solo):
        """LLMDebugLog.objects.create is called with correct fields on a successful call."""
        mock_get_solo.return_value = self._enabled_cfg()

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create") as mock_create:
            asyncio.run(
                _audit_llm_call(
                    "test_op",
                    "openai",
                    "gpt-4o",
                    "my_template",
                    self._messages(),
                    "raw response text",
                    None,
                )
            )
            mock_create.assert_called_once()
            kwargs = mock_create.call_args[1]
            self.assertEqual(kwargs["call_label"], "my_template")
            self.assertEqual(kwargs["raw_output"], "raw response text")
            self.assertEqual(kwargs["error"], "")

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_success_log_input_messages_is_pretty_printed_json(self, mock_get_solo):
        """input_messages is stored as pretty-printed JSON (with real newlines)."""
        mock_get_solo.return_value = self._enabled_cfg()

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create") as mock_create:
            asyncio.run(
                _audit_llm_call(
                    "test_op",
                    "openai",
                    "gpt-4o",
                    "tpl",
                    [SystemMessage(content="sys"), HumanMessage(content="user")],
                    "out",
                    None,
                )
            )
            kwargs = mock_create.call_args[1]
            serialized = kwargs["input_messages"]
            parsed = json.loads(serialized)
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0]["type"], "SystemMessage")
            self.assertEqual(parsed[0]["content"], "sys")
            self.assertEqual(parsed[1]["type"], "HumanMessage")
            self.assertEqual(parsed[1]["content"], "user")
            # Must be indented (pretty-printed) so newlines render in the admin textarea
            self.assertIn("\n", serialized)

    # ------------------------------------------------------------------
    # debug enabled – error path
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_creates_log_on_exception(self, mock_get_solo):
        """LLMDebugLog.objects.create is called with the error message when exp is set."""
        mock_get_solo.return_value = self._enabled_cfg()

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create") as mock_create:
            asyncio.run(
                _audit_llm_call(
                    "test_op",
                    "openai",
                    "gpt-4o",
                    "tpl",
                    self._messages(),
                    "",
                    RuntimeError("boom"),
                )
            )
            mock_create.assert_called_once()
            kwargs = mock_create.call_args[1]
            self.assertEqual(kwargs["error"], "boom")
            self.assertEqual(kwargs["raw_output"], "")

    # ------------------------------------------------------------------
    # debug disabled – error path
    # ------------------------------------------------------------------

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_creates_log_when_debug_disabled_but_error_present(self, mock_get_solo):
        """LLMDebugLog row must be written with raw payloads even when debug=False if an error occurred."""
        mock_get_solo.return_value = self._disabled_cfg()

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create") as mock_create:
            asyncio.run(
                _audit_llm_call(
                    "test_op",
                    "openai",
                    "gpt-4o",
                    "tpl",
                    self._messages(),
                    "partial output",
                    RuntimeError("boom"),
                )
            )
            mock_create.assert_called_once()
            kwargs = mock_create.call_args[1]
            self.assertEqual(kwargs["error"], "boom")
            self.assertEqual(kwargs["raw_output"], "partial output")

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_metadata_row_written_when_debug_disabled_and_no_error(self, mock_get_solo):
        """A metadata-only LLMDebugLog row is always written even when debug=False and no error."""
        mock_get_solo.return_value = self._disabled_cfg()

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create") as mock_create:
            asyncio.run(
                _audit_llm_call(
                    "test_op",
                    "openai",
                    "gpt-4o",
                    "tpl",
                    self._messages(),
                    "output",
                    None,
                )
            )
            mock_create.assert_called_once()
            kwargs = mock_create.call_args[1]
            # Raw payload omitted
            self.assertEqual(kwargs["input_messages"], "")
            self.assertEqual(kwargs["raw_output"], "")

    # ------------------------------------------------------------------
    # silent failure
    # ------------------------------------------------------------------

    @patch(
        "aiworks_core.logic.llm.LLMConfiguration.get_solo", side_effect=Exception("db is down")
    )
    def test_swallows_exception_silently(self, _mock):
        """Exceptions inside _audit_llm_call never propagate to the caller."""
        # Should complete without raising
        asyncio.run(
            _audit_llm_call(
                "test_op", "openai", "gpt-4o", "tpl", self._messages(), "out", None
            )
        )

    @patch(
        "aiworks_core.logic.llm.LLMConfiguration.get_solo", side_effect=Exception("db is down")
    )
    def test_swallows_exception_logs_error(self, _mock):
        """Exceptions inside _audit_llm_call are logged at ERROR level."""
        with self.assertLogs("aiworks_core.logic.llm", level="ERROR") as cm:
            asyncio.run(
                _audit_llm_call(
                    "test_op", "openai", "gpt-4o", "tpl", self._messages(), "out", None
                )
            )
        self.assertTrue(any("AUDIT LLM CALL" in line for line in cm.output))


class ExtractThoughtsTests(TestCase):
    """Tests for the _extract_thoughts utility function."""

    @staticmethod
    def _make_result(content):
        msg = AIMessage(content=content)
        gen = ChatGeneration(message=msg, text="")
        return LLMResult(generations=[[gen]])

    def test_extracts_single_thinking_block(self):
        """Returns the thinking text from a single thinking block."""
        result = self._make_result(
            [
                {"type": "thinking", "thinking": "My reasoning"},
                {"type": "text", "text": "Final answer"},
            ]
        )
        self.assertEqual(_extract_thoughts(result), ["My reasoning"])

    def test_extracts_multiple_thinking_blocks(self):
        """Returns all thinking texts when multiple thinking blocks are present."""
        result = self._make_result(
            [
                {"type": "thinking", "thinking": "First"},
                {"type": "thinking", "thinking": "Second"},
            ]
        )
        self.assertEqual(_extract_thoughts(result), ["First", "Second"])

    def test_ignores_non_thinking_blocks(self):
        """Returns empty list when no thinking blocks are present."""
        result = self._make_result([{"type": "text", "text": "No thoughts"}])
        self.assertEqual(_extract_thoughts(result), [])

    def test_ignores_empty_thinking_text(self):
        """Skips thinking blocks where the 'thinking' value is empty."""
        result = self._make_result([{"type": "thinking", "thinking": ""}])
        self.assertEqual(_extract_thoughts(result), [])

    def test_handles_string_content(self):
        """Returns empty list gracefully when message content is a plain string."""
        result = self._make_result("plain text")
        self.assertEqual(_extract_thoughts(result), [])

    def test_returns_only_current_turn_thoughts(self):
        """Each call returns only thoughts from the provided response, not accumulated history."""
        result1 = self._make_result([{"type": "thinking", "thinking": "Turn 1"}])
        result2 = self._make_result([{"type": "thinking", "thinking": "Turn 2"}])
        self.assertEqual(_extract_thoughts(result1), ["Turn 1"])
        self.assertEqual(_extract_thoughts(result2), ["Turn 2"])

    def test_accumulation_via_extend(self):
        """Callers accumulating results via extend capture thoughts across turns."""
        all_thoughts: list[str] = []
        for thought_text in ("Turn A", "Turn B"):
            msg = AIMessage(content=[{"type": "thinking", "thinking": thought_text}])
            gen = ChatGeneration(message=msg, text="")
            all_thoughts.extend(_extract_thoughts(LLMResult(generations=[[gen]])))
        self.assertEqual(all_thoughts, ["Turn A", "Turn B"])

    # ------------------------------------------------------------------
    # DeepSeek Reasoner (reasoning_content in additional_kwargs)
    # ------------------------------------------------------------------

    def test_extracts_deepseek_reasoning_content(self):
        """Returns reasoning_content from additional_kwargs (DeepSeek Reasoner format)."""
        msg = AIMessage(content="Final answer")
        msg.additional_kwargs["reasoning_content"] = "DeepSeek thought here"
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])
        self.assertEqual(_extract_thoughts(result), ["DeepSeek thought here"])

    def test_ignores_empty_deepseek_reasoning_content(self):
        """Skips reasoning_content when it is an empty string."""
        msg = AIMessage(content="Answer")
        msg.additional_kwargs["reasoning_content"] = ""
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])
        self.assertEqual(_extract_thoughts(result), [])

    def test_captures_both_anthropic_and_deepseek_in_same_response(self):
        """If both content blocks and reasoning_content are present, both are returned."""
        msg = AIMessage(content=[{"type": "thinking", "thinking": "Anthropic thought"}])
        msg.additional_kwargs["reasoning_content"] = "DeepSeek thought"
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])
        thoughts = _extract_thoughts(result)
        self.assertIn("Anthropic thought", thoughts)
        self.assertIn("DeepSeek thought", thoughts)

    # ------------------------------------------------------------------
    # Google Gemini thinking models (same type="thinking" format as Anthropic)
    # ------------------------------------------------------------------

    def test_extracts_gemini_thinking_block(self):
        """Returns thinking text from Gemini's type='thinking' content block."""
        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "Gemini reasoning here"},
                {"type": "text", "text": "Final answer"},
            ]
        )
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])
        self.assertEqual(_extract_thoughts(result), ["Gemini reasoning here"])

    # ------------------------------------------------------------------
    # AWS Bedrock (Claude extended thinking: type="reasoning_content")
    # ------------------------------------------------------------------

    def test_extracts_bedrock_reasoning_content_block(self):
        """Returns text from AWS Bedrock's type='reasoning_content' content block."""
        msg = AIMessage(
            content=[
                {
                    "type": "reasoning_content",
                    "reasoning_content": {
                        "text": "Bedrock Claude reasoning here",
                        "signature": "abc123",
                    },
                },
                {"type": "text", "text": "Final answer"},
            ]
        )
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])
        self.assertEqual(_extract_thoughts(result), ["Bedrock Claude reasoning here"])

    def test_ignores_bedrock_reasoning_content_block_without_text(self):
        """Skips Bedrock reasoning_content block when text is empty."""
        msg = AIMessage(
            content=[
                {
                    "type": "reasoning_content",
                    "reasoning_content": {"signature": "abc123"},
                },
            ]
        )
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])
        self.assertEqual(_extract_thoughts(result), [])

    # ------------------------------------------------------------------
    # MiniMax / OpenAI-compatible providers: <think> tags in raw string
    # ------------------------------------------------------------------

    def test_extracts_thinking_tags_from_raw_string_content(self):
        """Returns the full <think>...</think> block from plain string content."""
        msg = AIMessage(content="<think>My reasoning here</think>Final answer")
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])
        self.assertEqual(_extract_thoughts(result), ["My reasoning here"])

    def test_extracts_multiple_thinking_tags_from_raw_string_content(self):
        """Multiple <think> blocks are all captured from raw string content."""
        msg = AIMessage(content="<think>First</think>Hello<think>Second</think>")
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])
        self.assertEqual(
            _extract_thoughts(result),
            ["First", "Second"],
        )

    def test_deduplicates_thinking_tags_across_generations(self):
        """Same tag appearing in multiple generations is not duplicated."""
        msg1 = AIMessage(content="<think>A</think>")
        msg2 = AIMessage(content="<think>A</think>")
        gen1 = ChatGeneration(message=msg1, text="")
        gen2 = ChatGeneration(message=msg2, text="")
        result = LLMResult(generations=[[gen1], [gen2]])
        thoughts = _extract_thoughts(result)
        self.assertEqual(thoughts.count("A"), 1)

    def test_case_insensitive_thinking_tags(self):
        """<think> and <think> variants are also captured."""
        msg = AIMessage(content="<THINK>Uppercase</THINK>Result")
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])
        self.assertEqual(_extract_thoughts(result), ["Uppercase"])


def _add_thought(collector: _DebugDataCollector, thought: str):
    response = SimpleNamespace(
        generations=
        [
            [
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=
                        [
                            {
                                "type": "thinking",
                                "thinking": thought,
                            }
                        ],
                    )
                )
            ]
        ]
    )

    collector.on_llm_end(response)


def _add_tool_run(collector: _DebugDataCollector, tool_name: str, tool_input: str, run_id: Optional[str] = None, output: Optional[str] = None):
    if not run_id:
        run_id = uuid4()

    collector.on_tool_start({"name": tool_name}, tool_input, run_id=run_id)
    if output:
        collector.on_tool_end(output, run_id=run_id)

def _add_tasks(collector: _DebugDataCollector, tasks: list[dict]):
    _add_tool_run(collector, "write_todos", json.dumps({"todos": tasks}))

class DebugDataCollectorTests(TestCase):
    """Tests for the _DebugDataCollector callback handler and accumulator."""

    def test_initial_state_is_empty(self):
        """A new collector has no history or tasks."""
        c = _DebugDataCollector()
        self.assertEqual(c.history, [])
        self.assertEqual(c.tasks, [])

    def test_add_thought(self):
        """add_thought appends a thought entry to history."""
        c = _DebugDataCollector()
        _add_thought(c, "First thought")
        _add_thought(c, "Second thought")
        self.assertEqual(len(c.history), 2)
        self.assertEqual(c.history[0], {"type": "thought", "content": "First thought"})
        self.assertEqual(c.history[1], {"type": "thought", "content": "Second thought"})

    def test_add_tool_run(self):
        """add_tool_run appends a tool_run entry to history."""
        c = _DebugDataCollector()
        _add_tool_run(c, "search", '{"query": "test"}')
        self.assertEqual(len(c.history), 1)
        self.assertEqual(c.history[0]["type"], "tool_run")
        self.assertEqual(c.history[0]["tool"], "search")
        self.assertEqual(c.history[0]["input"], '{"query": "test"}')

    def test_set_tasks_replaces_existing(self):
        """set_tasks replaces any previously stored tasks."""
        c = _DebugDataCollector()
        _add_tasks(c, [{"content": "old", "status": "pending"}])
        _add_tasks(c, [{"content": "new", "status": "completed"}])
        self.assertEqual(len(c.tasks), 1)
        self.assertEqual(c.tasks[0]["content"], "new")

    def test_history_preserves_interleaved_order(self):
        """Thoughts and tool runs are stored in insertion order."""
        c = _DebugDataCollector()
        _add_thought(c, "reasoning before search")
        _add_tool_run(c, "search", "query1")
        _add_thought(c, "reasoning before extract")
        _add_tool_run(c, "extract", "url1")
        types = [e["type"] for e in c.history]
        self.assertEqual(types, ["thought", "tool_run", "thought", "tool_run"])

    # ------------------------------------------------------------------
    # on_tool_start
    # ------------------------------------------------------------------

    def test_on_tool_start_records_all_tool_runs(self):
        """on_tool_start records every tool invocation regardless of name."""
        c = _DebugDataCollector()
        c.on_tool_start({"name": "search"}, '{"query": "climate"}')
        self.assertEqual(len(c.history), 1)
        self.assertEqual(c.history[0]["type"], "tool_run")
        self.assertEqual(c.history[0]["tool"], "search")
        self.assertEqual(c.history[0]["input"], '{"query": "climate"}')

    def test_on_tool_start_multiple_tools_appended(self):
        """Multiple tool calls are all appended to history in order."""
        c = _DebugDataCollector()
        c.on_tool_start({"name": "search"}, "query1")
        c.on_tool_start({"name": "extract"}, "url1")
        self.assertEqual(len(c.history), 2)
        self.assertEqual(c.history[0]["tool"], "search")
        self.assertEqual(c.history[1]["tool"], "extract")

    def test_on_tool_start_write_todos_records_tasks(self):
        """on_tool_start for write_todos also captures the task list."""
        c = _DebugDataCollector()
        todos = [{"content": "Do research", "status": "in_progress"}]
        c.on_tool_start({"name": "write_todos"}, json.dumps({"todos": todos}))
        self.assertEqual(c.tasks, todos)
        # Also recorded in history
        self.assertEqual(len(c.history), 1)
        self.assertEqual(c.history[0]["tool"], "write_todos")

    def test_on_tool_start_write_todos_replaces_tasks_on_second_call(self):
        """Subsequent write_todos calls overwrite the captured tasks."""
        c = _DebugDataCollector()
        first = [{"content": "Step 1", "status": "completed"}]
        second = [{"content": "Step 2", "status": "in_progress"}]
        c.on_tool_start({"name": "write_todos"}, json.dumps({"todos": first}))
        c.on_tool_start({"name": "write_todos"}, json.dumps({"todos": second}))
        self.assertEqual(c.tasks, second)

    def test_on_tool_start_non_write_todos_does_not_set_tasks(self):
        """Non-write_todos tool calls should not populate the tasks list."""
        c = _DebugDataCollector()
        c.on_tool_start({"name": "search"}, "query")
        self.assertEqual(c.tasks, [])

    # ------------------------------------------------------------------
    # on_tool_end
    # ------------------------------------------------------------------

    def test_on_tool_end_uses_run_id_when_available(self):
        """on_tool_end matches by run_id, not by tool name."""
        c = _DebugDataCollector()
        c.on_tool_start({"name": "search"}, "query1", run_id="run_1")
        c.on_tool_start({"name": "search"}, "query2", run_id="run_2")
        c.on_tool_end("second result", run_id="run_2")
        self.assertEqual(c.history[0].get("output"), None)
        self.assertEqual(c.history[1].get("output"), "second result")

    def test_on_tool_end_run_id_not_in_history_ignored(self):
        """on_tool_end skips when run_id not found in history."""
        c = _DebugDataCollector()
        c.on_tool_start({"name": "search"}, "query", run_id="run_1")
        c.on_tool_end("result", run_id="nonexistent")
        self.assertNotIn("output", c.history[0])

    # ------------------------------------------------------------------
    # on_llm_end
    # ------------------------------------------------------------------

    def test_on_llm_end_captures_thinking_blocks(self):
        """on_llm_end extracts thinking-type content blocks."""
        c = _DebugDataCollector()
        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "My reasoning here"},
                {"type": "text", "text": "Final answer"},
            ]
        )
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])

        c.on_llm_end(result)
        self.assertEqual(len(c.history), 1)
        self.assertEqual(
            c.history[0], {"type": "thought", "content": "My reasoning here"}
        )

    def test_on_llm_end_multiple_thoughts_captured(self):
        """Thoughts from multiple turns are all appended in order."""
        c = _DebugDataCollector()
        for thought_text in ("Think A", "Think B"):
            msg = AIMessage(content=[{"type": "thinking", "thinking": thought_text}])
            gen = ChatGeneration(message=msg, text="")
            c.on_llm_end(LLMResult(generations=[[gen]]))
        thoughts_in_history = [
            e["content"] for e in c.history if e["type"] == "thought"
        ]
        self.assertEqual(thoughts_in_history, ["Think A", "Think B"])

    def test_on_llm_end_ignores_non_thinking_blocks(self):
        """on_llm_end ignores blocks that are not type 'thinking'."""
        c = _DebugDataCollector()
        msg = AIMessage(content=[{"type": "text", "text": "No thoughts here"}])
        gen = ChatGeneration(message=msg, text="")
        c.on_llm_end(LLMResult(generations=[[gen]]))
        self.assertEqual(c.history, [])

    def test_on_llm_end_handles_string_content(self):
        """on_llm_end gracefully handles messages with plain-string content."""
        c = _DebugDataCollector()
        msg = AIMessage(content="plain text response")
        gen = ChatGeneration(message=msg, text="")
        c.on_llm_end(LLMResult(generations=[[gen]]))  # should not raise
        self.assertEqual(c.history, [])


def _build_memory_md_forced(
        collector: "_DebugDataCollector", messages = None) -> str:
    result: Optional[str] = _build_memory_md(collector, messages)
    if result is None:
        raise Exception("Result returned none for _build_memory_md")
    return result

class BuildMemoryMdTests(TestCase):
    """Tests for _build_memory_md."""

    @staticmethod
    def _make_collector(thoughts=None, tool_runs=None, tasks=None):
        c = _DebugDataCollector()
        for t in thoughts or []:
            _add_thought(c, t)
        for tool_run in tool_runs or []:
            if len(tool_run) == 2:
                name, inp = tool_run
                out = None
            else:
                name, inp, out = tool_run
            _add_tool_run(c, name, inp, output=out)
        if tasks:
            _add_tasks(c, tasks)
        return c

    def test_returns_none_when_empty(self):
        """Returns None when collector has no history and no tasks."""
        c = _DebugDataCollector()
        self.assertIsNone(_build_memory_md(c))

    def test_returns_none_when_empty_with_messages(self):
        """Returns None even when messages are provided but collector is empty."""
        c = _DebugDataCollector()
        msgs = [SystemMessage(content="sys"), HumanMessage(content="user")]
        self.assertIsNone(_build_memory_md(c, messages=msgs))

    # noinspection PyTypeChecker
    def test_input_messages_section_included(self):
        """Input messages are rendered under ## Input Messages when provided."""
        c = self._make_collector(thoughts=["some thought"])
        msgs = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="What is 2+2?"),
        ]
        result = _build_memory_md(c, messages=msgs)
        self.assertIn("## Input Messages", result)
        self.assertIn("You are helpful.", result)
        self.assertIn("What is 2+2?", result)

    def test_input_messages_section_absent_without_messages(self):
        """## Input Messages section is absent when no messages are passed."""
        c = self._make_collector(thoughts=["thought"])
        result = _build_memory_md_forced(c)
        self.assertNotIn("## Input Messages", result)

    def test_input_messages_appear_before_tasks_and_history(self):
        """Input Messages section precedes Tasks and Execution History sections."""
        c = self._make_collector(
            thoughts=["reasoning"],
            tasks=[{"content": "Do X", "status": "completed"}],
        )
        msgs = [HumanMessage(content="original question")]
        result = _build_memory_md_forced(c, messages=msgs)
        idx_messages = result.index("## Input Messages")
        idx_tasks = result.index("## Tasks")
        idx_history = result.index("## Execution History")
        self.assertLess(idx_messages, idx_tasks)
        self.assertLess(idx_messages, idx_history)

    def test_system_and_human_role_labels(self):
        """System and Human message types are labelled correctly."""
        c = self._make_collector(thoughts=["t"])
        msgs = [
            SystemMessage(content="sys content"),
            HumanMessage(content="human content"),
        ]
        result = _build_memory_md_forced(c, messages=msgs)
        self.assertIn("**System:**", result)
        self.assertIn("**Human:**", result)

    def test_tasks_and_history_still_rendered(self):
        """Tasks and history are still present when messages are included."""
        c = self._make_collector(
            thoughts=["deep thought"],
            tool_runs=[("search", "query1")],
            tasks=[{"content": "Research task", "status": "in_progress"}],
        )
        msgs = [HumanMessage(content="hello")]
        result = _build_memory_md_forced(c, messages=msgs)
        self.assertIn("## Tasks", result)
        self.assertIn("Research task", result)
        self.assertIn("## Execution History", result)
        self.assertIn("deep thought", result)
        self.assertIn("search", result)

    def test_tool_output_included_in_execution_history(self):
        """Tool output is included in the Execution History section."""
        c = self._make_collector(
            tool_runs=[("search", "query1", "relevant article content")],
        )
        result = _build_memory_md_forced(c)
        self.assertIn("**Output:** relevant article content", result)
        self.assertIn("**Input:** query1", result)

    def test_tool_output_absent_when_no_output_recorded(self):
        """When a tool has no output, only input is shown."""
        c = self._make_collector(tool_runs=[("search", "query1")])
        result = _build_memory_md_forced(c)
        self.assertIn("**Input:** query1", result)
        self.assertNotIn("**Output:**", result)

    def test_empty_messages_list_treated_as_absent(self):
        """An empty messages list does not render the Input Messages section."""
        c = self._make_collector(thoughts=["t"])
        result = _build_memory_md_forced(c, messages=[])
        self.assertNotIn("## Input Messages", result)


class AgentLoggingCallbackHandlerThoughtLoggingTests(TestCase):
    """Tests that _AgentLoggingCallbackHandler logs thinking blocks."""

    def test_on_llm_end_logs_thinking_blocks(self):
        """on_llm_end logs a thinking block at INFO level."""
        handler = _AgentLoggingCallbackHandler()
        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "My reasoning here"},
                {"type": "text", "text": "Final answer"},
            ]
        )
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])

        with self.assertLogs("aiworks_core.logic.llm", level="INFO") as cm:
            handler.on_llm_end(result)
        self.assertTrue(any("AGENT THOUGHT" in line for line in cm.output))

    def test_on_llm_end_ignores_non_thinking_blocks(self):
        """on_llm_end produces no log output for non-thinking blocks."""
        handler = _AgentLoggingCallbackHandler()
        msg = AIMessage(content=[{"type": "text", "text": "No thoughts"}])
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])
        # Should not emit any AGENT THOUGHT log line
        with self.assertLogs("aiworks_core.logic.llm", level="DEBUG") as cm:
            # Force at least one log so assertLogs doesn't fail on empty
            logging.getLogger("aiworks_core.logic.llm").debug("_sentinel_")
            handler.on_llm_end(result)
        self.assertFalse(any("AGENT THOUGHT" in line for line in cm.output))

    @staticmethod
    def test_on_llm_end_handles_string_content():
        """on_llm_end does not raise on plain-string message content."""
        handler = _AgentLoggingCallbackHandler()
        msg = AIMessage(content="plain text")
        gen = ChatGeneration(message=msg, text="")
        result = LLMResult(generations=[[gen]])
        handler.on_llm_end(result)  # should not raise


class AuditLlmCallCollectorTests(TestCase):
    """Tests that _audit_llm_call persists collector data to LLMDebugLog."""

    @staticmethod
    def _enabled_cfg():
        mock_cfg = MagicMock()
        mock_cfg.llm_debug = True
        return mock_cfg

    @staticmethod
    def _messages():
        return [SystemMessage(content="sys"), HumanMessage(content="user")]

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_saves_agent_fields_when_collector_provided(self, mock_get_solo):
        """LLMDebugLog.objects.create receives agent_history and agent_tasks."""
        mock_get_solo.return_value = self._enabled_cfg()
        collector = _DebugDataCollector()
        _add_thought(collector, "Some reasoning")
        _add_tool_run(collector, "search", '{"q": "test"}')
        _add_tasks(collector, [{"content": "Task A", "status": "completed"}])

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create") as mock_create:
            asyncio.run(
                _audit_llm_call(
                    "test_op",
                    "openai",
                    "gpt-4o",
                    "tpl",
                    self._messages(),
                    "out",
                    None,
                    collector,
                )
            )
            mock_create.assert_called_once()
            kwargs = mock_create.call_args[1]

        history = json.loads(kwargs["agent_history"])
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0], {"type": "thought", "content": "Some reasoning"})
        self.assertEqual(history[1]["type"], "tool_run")
        self.assertEqual(history[1]["tool"], "search")
        self.assertEqual(history[2]["type"], "tool_run")
        self.assertEqual(history[2]["tool"], "write_todos")

        tasks = json.loads(kwargs["agent_tasks"])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["content"], "Task A")

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_saves_agent_fields_on_failure(self, mock_get_solo):
        """agent_history/tasks are persisted even when exp is set (failure case)."""
        mock_get_solo.return_value = self._enabled_cfg()
        collector = _DebugDataCollector()
        _add_thought(collector, "Partial thought before crash")
        _add_tool_run(collector, "extract", "some url")

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create") as mock_create:
            asyncio.run(
                _audit_llm_call(
                    "test_op",
                    "openai",
                    "gpt-4o",
                    "tpl",
                    self._messages(),
                    "",
                    RuntimeError("boom"),
                    collector,
                )
            )
            mock_create.assert_called_once()
            kwargs = mock_create.call_args[1]

        self.assertEqual(kwargs["error"], "boom")
        self.assertEqual(kwargs["raw_output"], "")
        history = json.loads(kwargs["agent_history"])
        thought_contents = [e["content"] for e in history if e["type"] == "thought"]
        self.assertIn("Partial thought before crash", thought_contents)

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_saves_empty_agent_fields_when_no_collector(self, mock_get_solo):
        """When no collector is supplied, agent_* fields default to empty strings."""
        mock_get_solo.return_value = self._enabled_cfg()

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create") as mock_create:
            asyncio.run(
                _audit_llm_call(
                    "test_op",
                    "openai",
                    "gpt-4o",
                    "tpl",
                    self._messages(),
                    "out",
                    None,
                    None,
                )
            )
            kwargs = mock_create.call_args[1]

        self.assertEqual(kwargs["agent_history"], "")
        self.assertEqual(kwargs["agent_tasks"], "")

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_new_metadata_fields_passed_through(self, mock_get_solo):
        """operation_name, provider, model, is_agent, is_deep_agent, token counts are persisted."""
        mock_get_solo.return_value = self._enabled_cfg()

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create") as mock_create:
            asyncio.run(
                _audit_llm_call(
                    "my_llm_op",
                    "openai",
                    "gpt-4o",
                    "my_template",
                    self._messages(),
                    "output",
                    None,
                    is_agent=True,
                    is_deep_agent=False,
                    input_tokens=100,
                    output_tokens=50,
                )
            )
            kwargs = mock_create.call_args[1]

        self.assertEqual(kwargs["operation_name"], "my_llm_op")
        self.assertEqual(kwargs["provider"], "openai")
        self.assertEqual(kwargs["model"], "gpt-4o")
        self.assertTrue(kwargs["is_agent"])
        self.assertFalse(kwargs["is_deep_agent"])
        self.assertEqual(kwargs["input_tokens"], 100)
        self.assertEqual(kwargs["output_tokens"], 50)

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_token_counts_merged_from_collector(self, mock_get_solo):
        """Token counts accumulated by a collector are merged into the final totals."""
        mock_get_solo.return_value = self._enabled_cfg()
        collector = _DebugDataCollector()
        collector.input_tokens = 200
        collector.output_tokens = 80

        with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create") as mock_create:
            asyncio.run(
                _audit_llm_call(
                    "test_op",
                    "openai",
                    "gpt-4o",
                    "tpl",
                    self._messages(),
                    "out",
                    None,
                    collector,
                    input_tokens=50,
                    output_tokens=10,
                )
            )
            kwargs = mock_create.call_args[1]

        self.assertEqual(kwargs["input_tokens"], 250)
        self.assertEqual(kwargs["output_tokens"], 90)


class DebugDataCollectorTokenTests(TestCase):
    """Tests for _DebugDataCollector token accumulation via on_llm_end."""

    @staticmethod
    def _make_response(input_tokens: int, output_tokens: int):
        """Build a minimal LLMResult-like object with usage_metadata."""
        msg = MagicMock()
        msg.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        gen = MagicMock()
        gen.message = msg
        response = MagicMock()
        response.generations = [[gen]]
        # _extract_thoughts also reads .generations; make content non-list to skip thoughts
        msg.content = "some text"
        msg.additional_kwargs = {}
        return response

    def test_token_counts_accumulated_across_turns(self):
        """input_tokens and output_tokens are summed over multiple on_llm_end calls."""
        collector = _DebugDataCollector()
        collector.on_llm_end(self._make_response(100, 40))
        collector.on_llm_end(self._make_response(80, 30))
        self.assertEqual(collector.input_tokens, 180)
        self.assertEqual(collector.output_tokens, 70)

    def test_no_usage_metadata_leaves_counts_none(self):
        """If the response has no usage_metadata, counts remain None."""
        msg = MagicMock()
        msg.usage_metadata = None
        msg.content = "text"
        msg.additional_kwargs = {}
        gen = MagicMock()
        gen.message = msg
        response = MagicMock()
        response.generations = [[gen]]
        collector = _DebugDataCollector()
        collector.on_llm_end(response)
        self.assertIsNone(collector.input_tokens)
        self.assertIsNone(collector.output_tokens)


class ResolveProviderModelTests(TestCase):
    """Tests for the _resolve_provider_model helper."""

    @staticmethod
    def _make_cfg(**kwargs):
        cfg = MagicMock()
        cfg.provider = kwargs.get("provider", "openai")
        cfg.fast_provider = kwargs.get("fast_provider", "")
        cfg.reasoning_provider = kwargs.get("reasoning_provider", "")
        cfg.ultra_fast_provider = kwargs.get("ultra_fast_provider", "")
        cfg.ultra_smart_provider = kwargs.get("ultra_smart_provider", "")
        cfg.openai_model = kwargs.get("openai_model", "gpt-4o")
        cfg.openai_fast_model = kwargs.get("openai_fast_model", "gpt-4o-mini")
        cfg.openai_reasoning_model = kwargs.get("openai_reasoning_model", "o1")
        cfg.openai_ultra_fast_model = kwargs.get("openai_ultra_fast_model", "")
        cfg.openai_ultra_smart_model = kwargs.get("openai_ultra_smart_model", "")
        for key, value in kwargs.items():
            setattr(cfg, key, value)
        return cfg

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_resolves_smart_model(self, mock_get_solo):
        """Default llm_type resolves to the smart model."""
        mock_get_solo.return_value = self._make_cfg()
        provider, model, llm_type = _resolve_provider_model("test_op")
        self.assertEqual(provider, "openai")
        self.assertEqual(model, "gpt-4o")

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_resolves_fast_model(self, mock_get_solo):
        """llm_type='fast' resolves to the fast model when set in LLMOperationConfig."""
        mock_get_solo.return_value = self._make_cfg()

        op_name = f"__test_{uuid.uuid4().hex}__"
        LLMOperationConfig.objects.create(
            operation_name=op_name, selected_llm_type="fast"
        )
        try:
            provider, model, llm_type = _resolve_provider_model(op_name)
            self.assertEqual(provider, "openai")
            self.assertEqual(model, "gpt-4o-mini")
        finally:
            LLMOperationConfig.objects.filter(operation_name=op_name).delete()

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_override_provider_and_model(self, mock_get_solo):
        """Explicit override_provider and override_model take precedence via LLMOperationConfig."""
        mock_get_solo.return_value = self._make_cfg()

        op_name = f"__test_{uuid.uuid4().hex}__"
        LLMOperationConfig.objects.create(
            operation_name=op_name,
            override_provider="anthropic",
            override_model="claude-opus-4",
        )
        try:
            provider, model, llm_type = _resolve_provider_model(op_name)
            self.assertEqual(provider, "anthropic")
            self.assertEqual(model, "claude-opus-4")
        finally:
            LLMOperationConfig.objects.filter(operation_name=op_name).delete()

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo", side_effect=Exception("db down"))
    def test_returns_empty_strings_on_exception(self, _mock):
        """Any exception from get_solo propagates without being caught."""
        with self.assertRaises(Exception):
            _resolve_provider_model("test_op")


class PatchDeepagentsSummarizationTests(TestCase):
    """_patch_deepagents_summarization must update both module bindings."""

    def setUp(self):
        self._summarization_module = summarization_module
        self._patch = _patch_deepagents_summarization

        # Save originals and clear any prior patch flag so each test starts clean.
        self._original_sum_fn = summarization_module.compute_summarization_defaults
        for mod in (graph_module, summarization_module):
            # noinspection PyUnresolvedReferences
            mod.__dict__.pop("_deepagents_patched", None)

    def tearDown(self):
        # Restore originals so other tests are not affected.
        self._summarization_module._compute_summarization_defaults = (
            self._original_sum_fn
        )
        # noinspection PyUnresolvedReferences
        self._summarization_module.__dict__.pop("_deepagents_patched", None)

    def test_patch_updates_graph_module_reference(self):
        """
        deepagents.graph imports _compute_summarization_defaults via a direct
        'from … import', creating an independent local binding.  The patch must
        update that binding too, otherwise create_deep_agent continues to use
        the library default (170 k-token trigger) and summarization never fires.
        """
        self._patch()

        # Both module-level names must now point to the patched function.
        self.assertIsNot(
            self._summarization_module.compute_summarization_defaults,
            self._original_sum_fn,
            "summarization module reference was not updated by the patch",
        )

    def test_patched_defaults_use_low_trigger_and_small_keep(self):
        """The patched summarization defaults must use a low token trigger and few kept messages."""
        self._patch()

        mock_model = MagicMock()
        defaults = self._summarization_module.compute_summarization_defaults(mock_model)

        trigger_type, trigger_value = defaults["trigger"]
        self.assertEqual(trigger_type, "tokens")
        self.assertLessEqual(
            trigger_value,
            60000,
            "Trigger must be low enough to prevent context overflow",
        )

        keep_type, keep_value = defaults["keep"]
        self.assertEqual(keep_type, "messages")
        self.assertLessEqual(
            keep_value,
            50,
            "Keep must be small to prevent rapid context re-accumulation",
        )

    def test_idempotent_patch(self):
        """Calling _patch_deepagents_summarization twice must not raise and must remain patched."""
        self._patch()
        patched_fn = self._summarization_module.compute_summarization_defaults
        self._patch()  # second call

        # Must still be the patched function (no double-patch regression).
        self.assertIs(self._summarization_module.compute_summarization_defaults, patched_fn)


class SafeToolTests(TestCase):
    """Tests for _coerce_tool_output and _SafeToolWrapper."""

    # ------------------------------------------------------------------
    # _coerce_tool_output
    # ------------------------------------------------------------------

    def test_coerce_tool_output_string(self):
        self.assertEqual(_coerce_tool_output("hello"), "hello")

    def test_coerce_tool_output_list_of_blocks(self):
        blocks = [
            {"type": "text", "text": "line one"},
            {"type": "text", "text": "line two"},
        ]
        self.assertEqual(_coerce_tool_output(blocks), json.dumps(blocks))

    def test_coerce_tool_output_tuple_content_artifact(self):
        content = [{"type": "text", "text": "result"}]
        result = _coerce_tool_output((content, "artifact"))
        self.assertEqual(result, json.dumps([content, "artifact"]))

    # ------------------------------------------------------------------
    # _SafeToolWrapper
    # ------------------------------------------------------------------

    @staticmethod
    def _make_inner(return_value=None, raises=None):
        inner = MagicMock(spec=BaseTool)
        inner.name = "test_tool"
        inner.description = "A test tool"
        inner.args_schema = None
        inner.return_direct = False
        if raises:
            inner._run.side_effect = raises
            inner._arun = AsyncMock(side_effect=raises)
        else:
            inner._run.return_value = return_value
            inner._arun = AsyncMock(return_value=return_value)
        return inner

    def test_safe_tool_wrapper_returns_string(self):
        payload = [{"type": "text", "text": "ok"}]
        inner = self._make_inner(return_value=payload)
        wrapped = _SafeToolWrapper.wrap(inner)
        result = wrapped._run()
        self.assertIsInstance(result, str)
        self.assertEqual(result, json.dumps(payload))

    def test_safe_tool_wrapper_catches_exception(self):
        inner = self._make_inner(raises=RuntimeError("network failure"))
        wrapped = _SafeToolWrapper.wrap(inner)
        result = wrapped._run()
        self.assertIsInstance(result, str)
        self.assertIn("network failure", result)

    def test_safe_tool_wrapper_async_catches_exception(self):
        inner = self._make_inner(raises=RuntimeError("async failure"))
        wrapped = _SafeToolWrapper.wrap(inner)
        result = async_to_sync(wrapped._arun)()
        self.assertIn("async failure", result)

    def test_get_agent_wraps_tools(self):
        inner = self._make_inner(return_value="x")
        wrapped = _make_safe_tool(inner)
        self.assertIsInstance(wrapped, _SafeToolWrapper)

    # ------------------------------------------------------------------
    # _SafeStructuredToolWrapper
    # ------------------------------------------------------------------

    @staticmethod
    def _make_structured_inner(return_value=None, raises=None):
        inner = MagicMock(spec=StructuredTool)
        inner.name = "structured_tool"
        inner.description = "A structured test tool"
        inner.args_schema = None
        inner.return_direct = False
        if raises:
            inner._run.side_effect = raises
            inner._arun = AsyncMock(side_effect=raises)
        else:
            inner._run.return_value = return_value
            inner._arun = AsyncMock(return_value=return_value)
        return inner

    def test_safe_structured_tool_wrapper_calls_arun_with_config(self):
        inner = self._make_structured_inner(return_value="structured result")
        wrapped = _SafeStructuredToolWrapper.wrap(inner)
        result = async_to_sync(wrapped._arun)()
        inner._arun.assert_called_once_with(config=None, run_manager=None)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "structured result")

    def test_make_safe_tool_routes_structured_tool(self):
        inner = self._make_structured_inner(return_value="x")
        wrapped = _make_safe_tool(inner)
        self.assertIsInstance(wrapped, _SafeStructuredToolWrapper)

    def test_make_safe_tool_routes_base_tool(self):
        inner = self._make_inner(return_value="x")
        wrapped = _make_safe_tool(inner)
        self.assertIsInstance(wrapped, _SafeToolWrapper)

    # ------------------------------------------------------------------
    # Timeout tests — _SafeToolWrapper
    # ------------------------------------------------------------------

    def test_safe_tool_wrapper_arun_timeout(self):
        async def hanging_arun(*args, **kwargs):
            await asyncio.sleep(10)
            return "done"
        inner = MagicMock(spec=BaseTool)
        inner.name = "hanging_tool"
        inner.description = ""
        inner.args_schema = None
        inner.return_direct = False
        inner._arun = hanging_arun
        wrapped = _SafeToolWrapper.wrap(inner, timeout=0.01)
        with patch("aiworks_core.logic.llm.close_connections"):
            result = async_to_sync(wrapped._arun)()
        self.assertIn("Timeout after", result)
        self.assertIn("0.01", result)

    def test_safe_tool_wrapper_arun_no_timeout(self):
        inner = self._make_inner(return_value="fast result")
        wrapped = _SafeToolWrapper.wrap(inner, timeout=600)
        with patch("aiworks_core.logic.llm.close_connections"):
            result = async_to_sync(wrapped._arun)()
        self.assertEqual(result, "fast result")

    def test_safe_tool_wrapper_run_timeout(self):
        async def hanging_arun(*args, **kwargs):
            await asyncio.sleep(10)
            return "done"
        inner = MagicMock(spec=BaseTool)
        inner.name = "hanging_tool"
        inner.description = ""
        inner.args_schema = None
        inner.return_direct = False
        inner._arun = hanging_arun
        wrapped = _SafeToolWrapper.wrap(inner, timeout=0.01)
        with patch("aiworks_core.logic.llm.close_connections"):
            result = wrapped._run()
        self.assertIn("Timeout after", result)
        self.assertIn("0.01", result)

    def test_safe_tool_wrapper_run_no_timeout(self):
        inner = self._make_inner(return_value="fast sync result")
        wrapped = _SafeToolWrapper.wrap(inner, timeout=600)
        with patch("aiworks_core.logic.llm.close_connections"):
            result = wrapped._run()
        self.assertEqual(result, "fast sync result")

    # ------------------------------------------------------------------
    # Timeout tests — _SafeStructuredToolWrapper
    # ------------------------------------------------------------------

    def test_safe_structured_tool_wrapper_arun_timeout(self):
        async def hanging_arun(*args, config=None, run_manager=None, **kwargs):
            await asyncio.sleep(10)
            return "done"
        inner = MagicMock(spec=StructuredTool)
        inner.name = "hanging_structured_tool"
        inner.description = ""
        inner.args_schema = None
        inner.return_direct = False
        inner._arun = hanging_arun
        wrapped = _SafeStructuredToolWrapper.wrap(inner, timeout=0.01)
        with patch("aiworks_core.logic.llm.close_connections"):
            result = async_to_sync(wrapped._arun)()
        self.assertIn("Timeout after", result)
        self.assertIn("0.01", result)

    def test_safe_structured_tool_wrapper_arun_no_timeout(self):
        inner = self._make_structured_inner(return_value="fast structured result")
        wrapped = _SafeStructuredToolWrapper.wrap(inner, timeout=600)
        with patch("aiworks_core.logic.llm.close_connections"):
            result = async_to_sync(wrapped._arun)()
        self.assertEqual(result, "fast structured result")

    def test_safe_structured_tool_wrapper_run_timeout(self):
        async def hanging_arun(*args, config=None, run_manager=None, **kwargs):
            await asyncio.sleep(10)
            return "done"
        inner = MagicMock(spec=StructuredTool)
        inner.name = "hanging_structured_tool"
        inner.description = ""
        inner.args_schema = None
        inner.return_direct = False
        inner._arun = hanging_arun
        wrapped = _SafeStructuredToolWrapper.wrap(inner, timeout=0.01)
        with patch("aiworks_core.logic.llm.close_connections"):
            result = wrapped._run()
        self.assertIn("Timeout after", result)
        self.assertIn("0.01", result)

    def test_safe_structured_tool_wrapper_run_no_timeout(self):
        inner = self._make_structured_inner(return_value="fast structured sync result")
        wrapped = _SafeStructuredToolWrapper.wrap(inner, timeout=600)
        with patch("aiworks_core.logic.llm.close_connections"):
            result = wrapped._run()
        self.assertEqual(result, "fast structured sync result")


class InvokeLlmJsonParseErrorTests(TestCase):
    """Tests that invoke_llm logs and re-raises errors during JSON parsing."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()
        cfg = LLMConfiguration.get_solo()
        cfg.provider = "google"
        cfg.google_model = "gemini-2.0-flash"
        cfg.google_api_key = "k"
        cfg.save()

    @staticmethod
    def _run(raw_content, **kwargs):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=raw_content))
        with patch("aiworks_core.logic.llm.get_chat_model", return_value=mock_llm):
            with patch("aiworks_core.logic.llm.LLMOperationConfig.objects.get") as mock_op_get:
                mock_op_get.side_effect = LLMOperationConfig.DoesNotExist("test")
                asyncio.run(
                    invoke_llm(
                        "test_operation",
                        messages=[
                            SystemMessage(content="sys"),
                            HumanMessage(content="user"),
                        ],
                        parse_json=True,
                        **kwargs,
                    )
                )

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    def test_json_decode_error_is_reraised(self, _mock_create, _mock_get_solo):
        """A truncated/invalid JSON response must propagate as JSONDecodeError."""
        with self.assertRaises(json.JSONDecodeError):
            self._run('{"unterminated": "string')

    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_json_decode_error_logs_raw_content(self, _mock_create, _mock_get_solo):
        """The raw LLM response must appear in the exception log."""
        raw = '{"unterminated": "string'
        with self.assertRaises(Exception):
            with self.assertLogs("aiworks_core.logic.llm", level="ERROR") as cm:
                self._run(raw)
        self.assertTrue(any(raw in line for line in cm.output))

    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_extract_json_error_is_reraised(self, _mock_create, _mock_get_solo):
        """Any error from _extract_json (e.g. IndexError on malformed fences) must propagate."""
        with patch(
                "aiworks_core.logic.llm._extract_json", side_effect=IndexError("fence error")
        ):
            with self.assertRaises(IndexError):
                self._run("```json```")

    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_extract_json_error_logs_raw_content(self, _mock_create, _mock_get_solo):
        """Raw content must be logged even when the error originates in _extract_json."""
        raw = "```json```"
        with patch(
                "aiworks_core.logic.llm._extract_json", side_effect=IndexError("fence error")
        ):
            with self.assertRaises(Exception):
                with self.assertLogs("aiworks_core.logic.llm", level="ERROR") as cm:
                    self._run(raw)
        self.assertTrue(any(raw in line for line in cm.output))

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_audit_record_written_on_json_parse_error(self, mock_get_solo):
        """LLMDebugLog must be written with raw content and error when JSON parsing fails."""
        mock_cfg = MagicMock()
        mock_cfg.llm_debug = False  # debug off — error should still force a write
        mock_get_solo.return_value = mock_cfg

        raw = '{"unterminated": "string'
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=raw))

        with patch("aiworks_core.logic.llm.get_chat_model", return_value=mock_llm):
            with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create") as mock_create:
                with patch(
                        "aiworks_core.logic.llm.LLMOperationConfig.objects.get"
                ) as mock_op_get:
                    mock_op_get.side_effect = LLMOperationConfig.DoesNotExist("test")
                    with self.assertRaises(json.JSONDecodeError):
                        asyncio.run(
                            invoke_llm(
                                "test_operation",
                                messages=[
                                    SystemMessage(content="sys"),
                                    HumanMessage(content="user"),
                                ],
                                parse_json=True,
                            )
                        )
                self.assertGreater(mock_create.call_count, 0)
                # All retry attempts produce the same content/error — check the first call
                kwargs = mock_create.call_args_list[0][1]
                self.assertEqual(kwargs["raw_output"], raw)
                self.assertIn("Unterminated", kwargs["error"])


class InvokeLlmEmptyResponseTests(TestCase):
    """Tests that invoke_llm raises ValueError when LLM returns empty/blank content."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()
        cfg = LLMConfiguration.get_solo()
        cfg.provider = "google"
        cfg.google_model = "gemini-2.0-flash"
        cfg.google_api_key = "k"
        cfg.save()

    @staticmethod
    def _run(content, **kwargs):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
        with patch("aiworks_core.logic.llm.get_chat_model", return_value=mock_llm):
            with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create"):
                with patch(
                        "aiworks_core.logic.llm.LLMOperationConfig.objects.get"
                ) as mock_op_get:
                    mock_op_get.side_effect = LLMOperationConfig.DoesNotExist("test")
                    return asyncio.run(
                        invoke_llm(
                            "test_operation",
                            messages=[
                                SystemMessage(content="sys"),
                                HumanMessage(content="user"),
                            ],
                            parse_json=False,
                            **kwargs,
                        )
                    )

    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_empty_string_raises_exception(self, _mock_create, _mock_get_solo):
        with self.assertRaises(Exception) as cm:
            self._run("")
        self.assertIn("empty response", str(cm.exception))

    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_whitespace_only_raises_exception(self, _mock_create, _mock_get_solo):
        with self.assertRaises(Exception) as cm:
            self._run("   \n\t  ")
        self.assertIn("empty response", str(cm.exception))

    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_non_empty_response_succeeds(self, _mock_create, _mock_get_solo):
        result = self._run("Hello world")
        self.assertEqual(result, "Hello world")


class InvokeLlmSchemaValidationTests(TestCase):
    """Tests that invoke_llm validates LLM JSON responses against a schema file when provided."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()
        cfg = LLMConfiguration.get_solo()
        cfg.provider = "google"
        cfg.google_model = "gemini-2.0-flash"
        cfg.google_api_key = "k"
        cfg.save()

    @staticmethod
    def _run(raw_content, **kwargs):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=raw_content))
        with patch("aiworks_core.logic.llm.get_chat_model", return_value=mock_llm):
            with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create"):
                with patch(
                        "aiworks_core.logic.llm.LLMOperationConfig.objects.get"
                ) as mock_op_get:
                    mock_op_get.side_effect = LLMOperationConfig.DoesNotExist("test")
                    return asyncio.run(
                        invoke_llm(
                            "test_operation",
                            messages=[
                                SystemMessage(content="sys"),
                                HumanMessage(content="user"),
                            ],
                            parse_json=True,
                            **kwargs,
                        )
                    )

    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_valid_content_passes_schema_validation(self, _mock_create, _mock_get_solo):
        """When schema_file_path is provided and content is valid, no exception is raised."""
        with patch(
                "aiworks_core.logic.llm.validate_json_with_schema_file",
                return_value=(True, ""),
        ) as mock_validate:
            result = self._run('{"name": "Alice", "age": 30}', schema_file_path="/path/to/schema.json")
            self.assertEqual(result, {"name": "Alice", "age": 30})
            mock_validate.assert_called_once_with('{"name": "Alice", "age": 30}', "/path/to/schema.json")

    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_invalid_content_raises_exception(self, _mock_create, _mock_get_solo):
        """When schema validation fails, invoke_llm raises Exception with the error message."""
        with patch(
                "aiworks_core.logic.llm.validate_json_with_schema_file",
                return_value=(False, "properties.age.type mismatch"),
        ):
            with self.assertRaises(Exception) as cm:
                self._run('{"name": "Alice", "age": "not-a-number"}', schema_file_path="/path/to/schema.json")
            self.assertIn("Schema validation failed:", str(cm.exception))
            self.assertIn("properties.age.type mismatch", str(cm.exception))

    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_schema_validation_skipped_when_not_provided(self, _mock_create, _mock_get_solo):
        """When schema_file_path is None, validation is skipped and behavior is unchanged."""
        with patch(
                "aiworks_core.logic.llm.validate_json_with_schema_file",
        ) as mock_validate:
            result = self._run('{"key": "value"}')
            self.assertEqual(result, {"key": "value"})
            mock_validate.assert_not_called()

    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_schema_validation_operates_on_raw_response(self, _mock_create, _mock_get_solo):
        """Schema validation receives the raw response string, not parsed JSON."""
        with patch(
                "aiworks_core.logic.llm.validate_json_with_schema_file",
                return_value=(True, ""),
        ) as mock_validate:
            self._run('  {"name": "Bob"}  ', schema_file_path="/path/to/schema.json")
            # The raw content (with whitespace) is passed to validation before JSON parsing
            call_arg = mock_validate.call_args[0][0]
            self.assertIn('"name": "Bob"', call_arg)


# ---------------------------------------------------------------------------
# CompressIfNeededTests — unit tests for logic_utils.compress_if_needed()
# ---------------------------------------------------------------------------


class CompressIfNeededTests(TestCase):
    """Unit tests for compress_if_needed() in logic_utils.

    compress_if_needed is a thin wrapper around compress_prompt that adds:
    - short-circuit when compression is disabled in config
    - cache hit: return cached_compressed without re-compressing
    - cache miss: compress and invoke save_compressed_callback
    - fallback to original when compression returns empty or identical text
    """

    @staticmethod
    def _mock_cfg(compress_prompts=True):
        cfg = MagicMock(spec=LLMConfiguration)
        cfg.compress_prompts = compress_prompts
        return cfg

    def test_returns_original_when_compression_disabled(self):
        """compress_prompts=False → return original immediately, no compression called."""
        with patch(
                "aiworks_core.logic.logic_utils.LLMConfiguration.get_solo",
                return_value=self._mock_cfg(compress_prompts=False),
        ):
            with patch("aiworks_core.logic.logic_utils.compress_prompt") as mock_compress:
                result = compress_if_needed("long original text", None, lambda x: None)

        self.assertEqual(result, "long original text")
        mock_compress.assert_not_called()

    def test_returns_cached_when_cached_compressed_is_truthy(self):
        """When cached_compressed is provided, return it without re-compressing."""
        with patch(
                "aiworks_core.logic.logic_utils.LLMConfiguration.get_solo",
                return_value=self._mock_cfg(compress_prompts=True),
        ):
            with patch("aiworks_core.logic.logic_utils.compress_prompt") as mock_compress:
                result = compress_if_needed(
                    "original", "already compressed", lambda x: None
                )

        self.assertEqual(result, "already compressed")
        mock_compress.assert_not_called()

    def test_compresses_and_invokes_callback_on_cache_miss(self):
        """When there is no cache, compress_prompt is called and save_callback is invoked."""
        saved = []

        async def _mock_compress(_text):
            return "compressed text"

        with patch(
                "aiworks_core.logic.logic_utils.LLMConfiguration.get_solo",
                return_value=self._mock_cfg(compress_prompts=True),
        ):
            with patch(
                    "aiworks_core.logic.logic_utils.compress_prompt", side_effect=_mock_compress
            ):
                result = compress_if_needed("long original text", None, saved.append)

        self.assertEqual(result, "compressed text")
        self.assertEqual(saved, ["compressed text"])

    def test_callback_not_called_when_compression_returns_original(self):
        """When compression returns the same text, the save callback must not be invoked."""
        saved = []

        async def _mock_compress(text):
            return text  # no-op compression

        with patch(
                "aiworks_core.logic.logic_utils.LLMConfiguration.get_solo",
                return_value=self._mock_cfg(compress_prompts=True),
        ):
            with patch(
                    "aiworks_core.logic.logic_utils.compress_prompt", side_effect=_mock_compress
            ):
                result = compress_if_needed("long original text", None, saved.append)

        self.assertEqual(result, "long original text")
        self.assertEqual(
            saved, [], "Callback should not be called when compression is a no-op"
        )

    def test_falls_back_to_original_when_compression_returns_empty(self):
        """When compression returns empty string, original text must be returned."""

        async def _mock_compress(_text):
            return ""

        with patch(
                "aiworks_core.logic.logic_utils.LLMConfiguration.get_solo",
                return_value=self._mock_cfg(compress_prompts=True),
        ):
            with patch(
                    "aiworks_core.logic.logic_utils.compress_prompt", side_effect=_mock_compress
            ):
                result = compress_if_needed("original text", None, lambda x: None)

        self.assertEqual(result, "original text")

    def test_falls_back_to_original_when_compression_returns_none(self):
        """When compression returns None, original text must be returned."""

        async def _mock_compress(_text):
            return None

        with patch(
                "aiworks_core.logic.logic_utils.LLMConfiguration.get_solo",
                return_value=self._mock_cfg(compress_prompts=True),
        ):
            with patch(
                    "aiworks_core.logic.logic_utils.compress_prompt", side_effect=_mock_compress
            ):
                result = compress_if_needed("original text", None, lambda x: None)

        self.assertEqual(result, "original text")

# ---------------------------------------------------------------------------
# CompressorPoolManagementTests — unit tests for _get_llmlingua_compressor_pool()
# ---------------------------------------------------------------------------


class CompressorPoolManagementTests(TestCase):
    """Unit tests for the LLMLingua compressor pool in prompt_compression.py.

    The pool is a process-level singleton backed by a bounded queue.
    These tests verify: pool size, singleton behaviour, and instance return.
    """

    @staticmethod
    def _mock_cfg(pool_size=2, model_name="llmlingua-2-bert-base-uncased"):
        cfg = MagicMock()
        cfg.compress_prompts_pool_size = pool_size
        cfg.compress_prompts_model = model_name
        return cfg

    def test_pool_size_matches_config(self):
        """Pool must be created with exactly compress_prompts_pool_size instances."""
        mock_compressor = MagicMock()
        cfg = self._mock_cfg(pool_size=3)

        original_pool = pc._COMPRESSOR_POOL
        pc._COMPRESSOR_POOL = None
        try:
            with patch(
                    "aiworks_core.logic.prompt_compression.PromptCompressor",
                    return_value=mock_compressor,
            ):
                pool = _get_llmlingua_compressor_pool(cfg)
            self.assertEqual(pool.qsize(), 3)
        finally:
            pc._COMPRESSOR_POOL = original_pool

    def test_pool_is_singleton(self):
        """Calling _get_llmlingua_compressor_pool twice returns the same object."""
        mock_compressor = MagicMock()
        cfg = self._mock_cfg(pool_size=1)

        original_pool = pc._COMPRESSOR_POOL
        pc._COMPRESSOR_POOL = None
        try:
            with patch(
                    "aiworks_core.logic.prompt_compression.PromptCompressor",
                    return_value=mock_compressor,
            ):
                pool1 = _get_llmlingua_compressor_pool(cfg)
                pool2 = _get_llmlingua_compressor_pool(cfg)
            self.assertIs(pool1, pool2)
        finally:
            pc._COMPRESSOR_POOL = original_pool

    def test_existing_pool_is_returned_without_reinitialisation(self):
        """If _COMPRESSOR_POOL is already set, PromptCompressor is never constructed again."""

        existing_pool = queue.Queue()
        existing_pool.put(MagicMock())

        original_pool = pc._COMPRESSOR_POOL
        pc._COMPRESSOR_POOL = existing_pool
        try:
            cfg = self._mock_cfg(pool_size=5)
            with patch("aiworks_core.logic.prompt_compression.PromptCompressor") as mock_cls:
                returned = _get_llmlingua_compressor_pool(cfg)
            mock_cls.assert_not_called()
            self.assertIs(returned, existing_pool)
        finally:
            pc._COMPRESSOR_POOL = original_pool


# ---------------------------------------------------------------------------
# Image Generation Tests
# ---------------------------------------------------------------------------


class ResolveProviderModelImageTests(TestCase):
    """Tests for _resolve_provider_model with llm_type='image'."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()

    @staticmethod
    def _make_cfg(**kwargs):
        cfg = MagicMock()
        cfg.provider = kwargs.get("provider", "openrouter")
        cfg.image_provider = kwargs.get("image_provider", "")
        cfg.openrouter_api_key = kwargs.get("openrouter_api_key", "sk-test")
        cfg.openrouter_model = kwargs.get("openrouter_model", "x-ai/grok-4.1-fast")
        cfg.openrouter_image_model = kwargs.get("openrouter_image_model", "google/gemini-2.5-flash-image")
        cfg.google_api_key = kwargs.get("google_api_key", "")
        cfg.google_image_model = kwargs.get("google_image_model", "gemini-pro-image")
        for key, value in kwargs.items():
            setattr(cfg, key, value)
        return cfg

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_image_llm_type_uses_image_provider(self, mock_get_solo):
        """When image_provider is set, it is used for llm_type='image'."""
        mock_get_solo.return_value = self._make_cfg(
            provider="openai",
            image_provider="google",
            google_api_key="k",
            google_image_model="gemini-pro-image",
        )
        op_name = f"__test_{uuid.uuid4().hex}__"
        LLMOperationConfig.objects.create(operation_name=op_name, selected_llm_type="image")
        try:
            provider, model, llm_type = _resolve_provider_model(op_name)
            self.assertEqual(provider, "google")
            self.assertEqual(model, "gemini-pro-image")
            self.assertEqual(llm_type, "image")
        finally:
            LLMOperationConfig.objects.filter(operation_name=op_name).delete()

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_image_llm_type_falls_back_to_main_provider(self, mock_get_solo):
        """When image_provider is blank, main provider is used for llm_type='image'."""
        mock_get_solo.return_value = self._make_cfg(
            provider="openrouter",
            image_provider="",
        )
        op_name = f"__test_{uuid.uuid4().hex}__"
        LLMOperationConfig.objects.create(operation_name=op_name, selected_llm_type="image")
        try:
            provider, model, llm_type = _resolve_provider_model(op_name)
            self.assertEqual(provider, "openrouter")
            self.assertEqual(model, "google/gemini-2.5-flash-image")
            self.assertEqual(llm_type, "image")
        finally:
            LLMOperationConfig.objects.filter(operation_name=op_name).delete()

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_image_llm_type_uses_openrouter_image_model_field(self, mock_get_solo):
        """llm_type='image' resolves to {provider}_image_model field."""
        mock_get_solo.return_value = self._make_cfg(
            provider="openrouter",
            image_provider="",
            openrouter_image_model="black-forest-labs/flux.2-pro",
        )
        op_name = f"__test_{uuid.uuid4().hex}__"
        LLMOperationConfig.objects.create(operation_name=op_name, selected_llm_type="image")
        try:
            provider, model, llm_type = _resolve_provider_model(op_name)
            self.assertEqual(model, "black-forest-labs/flux.2-pro")
        finally:
            LLMOperationConfig.objects.filter(operation_name=op_name).delete()

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_image_llm_type_raises_when_no_model_configured(self, mock_get_solo):
        """_resolve_provider_model raises when no image model AND no base model is configured."""
        mock_get_solo.return_value = self._make_cfg(
            provider="openrouter",
            image_provider="",
            openrouter_image_model="",
            openrouter_model="",  # No base model either
        )
        op_name = f"__test_{uuid.uuid4().hex}__"
        LLMOperationConfig.objects.create(operation_name=op_name, selected_llm_type="image")
        try:
            with self.assertRaises(ValueError) as cm:
                _resolve_provider_model(op_name)
            self.assertIn("No model configured", str(cm.exception))
        finally:
            LLMOperationConfig.objects.filter(operation_name=op_name).delete()

    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    def test_image_llm_type_falls_back_to_base_model_when_image_model_blank(self, mock_get_solo):
        """When *_image_model is blank, falls back to base *_model."""
        mock_get_solo.return_value = self._make_cfg(
            provider="openrouter",
            image_provider="",
            openrouter_image_model="",  # blank - should fall back
            openrouter_model="x-ai/grok-4.1-fast",  # base model
        )
        op_name = f"__test_{uuid.uuid4().hex}__"
        LLMOperationConfig.objects.create(operation_name=op_name, selected_llm_type="image")
        try:
            provider, model, llm_type = _resolve_provider_model(op_name)
            self.assertEqual(model, "x-ai/grok-4.1-fast")
        finally:
            LLMOperationConfig.objects.filter(operation_name=op_name).delete()


class InvokeLlmRenderOutputImageTests(TestCase):
    """Tests for invoke_llm with render_output_image=True."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()
        cfg = LLMConfiguration.get_solo()
        cfg.provider = "openrouter"
        cfg.openrouter_api_key = "sk-test"
        cfg.openrouter_model = "x-ai/grok-4.1-fast"
        cfg.openrouter_image_model = "google/gemini-2.5-flash-image"
        cfg.save()

    def _make_mock_image_llm(self, response):
        """Create a mock LLM that passes isinstance and has generate_image mocked."""
        class MockImageGeneratorModel(ImageGeneratorModel):
            def __init__(self, resp):
                self._response = resp

            async def generate_image(self, messages, **kwargs):
                return self._response

        return MockImageGeneratorModel(response)

    @patch("aiworks_core.logic.llm.get_chat_model")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    def test_render_output_image_returns_image_url_from_additional_kwargs(
        self, mock_log, mock_cfg_get_solo, mock_get_chat_model
    ):
        """When images are in additional_kwargs, the URL is extracted and returned."""
        mock_cfg_get_solo.return_value = LLMConfiguration.get_solo()

        mock_response = AIMessage(
            content="Here is your image",
            additional_kwargs={
                "images": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC123"}}
                ]
            },
        )
        mock_llm = self._make_mock_image_llm(mock_response)
        mock_get_chat_model.return_value = mock_llm

        result = asyncio.run(
            invoke_llm(
                "test_image_op",
                messages=[HumanMessage(content="Generate an image of a cat")],
                render_output_image=True,
            )
        )
        self.assertEqual(result, "data:image/png;base64,ABC123")

    @patch("aiworks_core.logic.llm.get_chat_model")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    def test_render_output_image_raises_when_is_agent_true(
        self, mock_log, mock_cfg_get_solo, mock_get_chat_model
    ):
        """render_output_image=True with is_agent=True raises ValueError."""
        mock_cfg_get_solo.return_value = LLMConfiguration.get_solo()

        with self.assertRaises(ValueError) as cm:
            asyncio.run(
                invoke_llm(
                    "test_image_op",
                    messages=[HumanMessage(content="Generate an image")],
                    render_output_image=True,
                    is_agent=True,
                )
            )
        self.assertIn("not supported in agent mode", str(cm.exception))

    @patch("aiworks_core.logic.llm.get_chat_model")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    def test_render_output_image_raises_when_no_images_in_response(
        self, mock_log, mock_cfg_get_solo, mock_get_chat_model
    ):
        """When response has no images, Exception is raised."""
        mock_cfg_get_solo.return_value = LLMConfiguration.get_solo()

        mock_response = AIMessage(content="No image generated", additional_kwargs={})
        mock_llm = self._make_mock_image_llm(mock_response)
        mock_get_chat_model.return_value = mock_llm

        with self.assertRaises(Exception) as cm:
            asyncio.run(
                invoke_llm(
                    "test_image_op",
                    messages=[HumanMessage(content="Generate an image of a cat")],
                    render_output_image=True,
                )
            )
        self.assertIn("did not include any images", str(cm.exception))

    @patch("aiworks_core.logic.llm.get_chat_model")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    def test_render_output_image_raises_when_images_list_empty(
        self, mock_log, mock_cfg_get_solo, mock_get_chat_model
    ):
        """When images list is empty, Exception is raised."""
        mock_cfg_get_solo.return_value = LLMConfiguration.get_solo()

        mock_response = AIMessage(
            content="No image",
            additional_kwargs={"images": []},
        )
        mock_llm = self._make_mock_image_llm(mock_response)
        mock_get_chat_model.return_value = mock_llm

        with self.assertRaises(Exception) as cm:
            asyncio.run(
                invoke_llm(
                    "test_image_op",
                    messages=[HumanMessage(content="Generate an image")],
                    render_output_image=True,
                )
            )
        self.assertIn("did not include any images", str(cm.exception))

    @patch("aiworks_core.logic.llm.get_chat_model")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    def test_render_output_image_with_template_messages(
        self, mock_log, mock_cfg_get_solo, mock_get_chat_model
    ):
        """render_output_image=True works with template-based messages."""
        mock_cfg_get_solo.return_value = LLMConfiguration.get_solo()

        mock_response = AIMessage(
            content="Image generated",
            additional_kwargs={
                "images": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,XYZ789"}}
                ]
            },
        )
        mock_llm = self._make_mock_image_llm(mock_response)
        mock_get_chat_model.return_value = mock_llm

        with patch("aiworks_core.logic.llm._navigate_prompts") as mock_nav:
            mock_nav.side_effect = lambda k: {
                "image_generation.system_message": "You are an image generator.",
                "image_generation.prompt_template": "Generate an image: {prompt}",
            }.get(k, "")

            result = asyncio.run(
                invoke_llm(
                    "test_image_op",
                    user_message_template_name="image_generation.prompt_template",
                    template_params={"prompt": "a sunset"},
                    render_output_image=True,
                )
            )

        self.assertEqual(result, "data:image/png;base64,XYZ789")


class TestExtractTodos(TestCase):
    """Tests for _extract_todos and _validate_todos."""

    def test_valid_list(self):
        """Valid list with content/status returns as-is."""
        from ..logic.llm import _extract_todos
        todos = [
            {"content": "Phase 1", "status": "completed"},
            {"content": "Phase 2", "status": "in_progress"},
        ]
        result = _extract_todos('{"todos": []}', {"todos": todos})
        self.assertEqual(result, todos)

    def test_valid_string_json(self):
        """String that parses to valid list returns parsed list."""
        from ..logic.llm import _extract_todos
        todos_str = '[{"content": "Phase 1", "status": "completed"}]'
        result = _extract_todos('{"todos": []}', {"todos": todos_str})
        self.assertEqual(result, [{"content": "Phase 1", "status": "completed"}])

    def test_string_not_json(self):
        """String that is not valid JSON returns None."""
        from ..logic.llm import _extract_todos
        result = _extract_todos('{"todos": []}', {"todos": "not json"})
        self.assertIsNone(result)

    def test_not_list(self):
        """Non-list, non-string todos returns None."""
        from ..logic.llm import _extract_todos
        result = _extract_todos('{"todos": []}', {"todos": 123})
        self.assertIsNone(result)

    def test_item_not_dict(self):
        """List containing a non-dict item returns None."""
        from ..logic.llm import _extract_todos
        todos = [
            {"content": "Phase 1", "status": "completed"},
            "string_item",
        ]
        result = _extract_todos('{"todos": []}', {"todos": todos})
        self.assertIsNone(result)

    def test_item_missing_content(self):
        """List with item missing content returns None."""
        from ..logic.llm import _extract_todos
        todos = [{"status": "completed"}]
        result = _extract_todos('{"todos": []}', {"todos": todos})
        self.assertIsNone(result)

    def test_item_missing_status(self):
        """List with item missing status returns None."""
        from ..logic.llm import _extract_todos
        todos = [{"content": "Phase 1"}]
        result = _extract_todos('{"todos": []}', {"todos": todos})
        self.assertIsNone(result)

    def test_empty_list(self):
        """Empty list passes validation as a valid list."""
        from ..logic.llm import _extract_todos
        result = _extract_todos('{"todos": []}', {"todos": []})
        self.assertEqual(result, [])

    def test_fallback_to_input_str(self):
        """When inputs['todos'] is None, falls back to parsing input_str."""
        from ..logic.llm import _extract_todos
        result = _extract_todos(
            '[{"content": "Phase 1", "status": "completed"}]',
            {},
        )
        self.assertEqual(result, [{"content": "Phase 1", "status": "completed"}])

    def test_string_json_in_input_str_fallback(self):
        """Fallback also handles string-encoded JSON."""
        from ..logic.llm import _extract_todos
        todos_str = '[{"content": "Phase 1", "status": "completed"}]'
        result = _extract_todos(todos_str, {})
        self.assertEqual(result, [{"content": "Phase 1", "status": "completed"}])

    def test_validate_todos_valid(self):
        """_validate_todos returns valid list unchanged."""
        from ..logic.llm import _validate_todos
        todos = [{"content": "x", "status": "completed"}]
        self.assertEqual(_validate_todos(todos), todos)

    def test_validate_todos_none(self):
        """_validate_todos returns None for None input (no warning)."""
        from ..logic.llm import _validate_todos
        self.assertIsNone(_validate_todos(None))

    def test_validate_todos_string_invalid_json(self):
        """_validate_todos returns None for unparseable string."""
        from ..logic.llm import _validate_todos
        self.assertIsNone(_validate_todos("not json"))

    def test_validate_todos_not_list(self):
        """_validate_todos returns None for non-list."""
        from ..logic.llm import _validate_todos
        self.assertIsNone(_validate_todos(123))


class CompressChunkedTests(TestCase):
    """Unit tests for _compress_chunked() — large text is split before compression."""

    @staticmethod
    def _mock_cfg(compress_prompts=True, compress_prompts_min_length=5):
        cfg = MagicMock(spec=LLMConfiguration)
        cfg.compress_prompts = compress_prompts
        cfg.compress_prompts_min_length = compress_prompts_min_length
        return cfg

    def test_small_text_not_chunked(self):
        """Text shorter than _COMPRESS_CHUNK_CHARS is compressed in one call."""
        calls = []

        def _mock_compression(text, **kwargs):
            calls.append(text)
            return f"compressed:{text[:10]}"

        with patch(
                "aiworks_core.logic.prompt_compression.LLMConfiguration.get_solo",
                new_callable=AsyncMock,
                return_value=self._mock_cfg(compress_prompts=True),
        ):
            from aiworks_core.logic.prompt_compression import _compress_chunked

            result = _compress_chunked(_mock_compression, "short text")

        self.assertEqual(len(calls), 1)
        self.assertEqual(result, "compressed:short text")

    def test_large_text_is_split(self):
        """Text exceeding _COMPRESS_CHUNK_CHARS is split and each chunk compressed."""
        chunk_calls = []

        def _mock_compression(text, **kwargs):
            chunk_calls.append(text)
            return f"compressed:{text[:10]}"

        with patch(
                "aiworks_core.logic.prompt_compression.LLMConfiguration.get_solo",
                new_callable=AsyncMock,
                return_value=self._mock_cfg(compress_prompts=True),
        ):
            from aiworks_core.logic.prompt_compression import _compress_chunked

            large_text = "x" * (_COMPRESS_CHUNK_CHARS + 100)
            result = _compress_chunked(_mock_compression, large_text)

        self.assertEqual(len(chunk_calls), 2)
        self.assertEqual(len(chunk_calls[0]), _COMPRESS_CHUNK_CHARS)
        self.assertEqual(len(chunk_calls[1]), 100)
        self.assertEqual(result, "compressed:" + "x" * 10 + "\n" + "compressed:" + "x" * 10)


    def test_large_text_is_compressed_chunked(self):
        """Text exceeding _COMPRESS_CHUNK_CHARS is split into chunks before compression."""
        async def _mock_compress(text):
            return f"c:{text[:5]}"

        with patch(
                "aiworks_core.logic.logic_utils.LLMConfiguration.get_solo",
                return_value=self._mock_cfg(compress_prompts=True),
        ):
            with patch(
                    "aiworks_core.logic.logic_utils.compress_prompt",
                    side_effect=_mock_compress,
            ):
                large_text = "y" * (50_000 + 50)
                result = compress_if_needed(large_text, None, lambda x: None)

        self.assertEqual(result, f"c:{'y' * 5}")


class CustomLLMResponseValidatorTests(TestCase):
    """Tests for CustomLLMResponseValidator ABC and its integration in invoke_llm."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()
        cfg = LLMConfiguration.get_solo()
        cfg.provider = "google"
        cfg.google_model = "gemini-2.0-flash"
        cfg.google_api_key = "k"
        cfg.save()

    @staticmethod
    def _run_via_direct_llm(raw_content, **kwargs):
        """Run invoke_llm in non-agent (direct LLM) mode."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=raw_content))
        with patch("aiworks_core.logic.llm.get_chat_model", return_value=mock_llm):
            with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create"):
                with patch("aiworks_core.logic.llm.LLMConfiguration.get_solo"):
                    with patch(
                            "aiworks_core.logic.llm.LLMOperationConfig.objects.get"
                    ) as mock_op_get:
                        mock_op_get.side_effect = LLMOperationConfig.DoesNotExist("test")
                        return asyncio.run(
                            invoke_llm(
                                "test_operation",
                                messages=[
                                    SystemMessage(content="sys"),
                                    HumanMessage(content="user"),
                                ],
                                parse_json=False,
                                **kwargs,
                            )
                        )

    @staticmethod
    def _run_via_agent(raw_content, **kwargs):
        """Run invoke_llm in agent mode."""
        mock_agent = MagicMock()

        async def fake_ainvoke(inputs, config=None):
            return {"messages": [AIMessage(content=raw_content)]}

        mock_agent.ainvoke = fake_ainvoke
        with patch("aiworks_core.logic.llm.get_agent", return_value=mock_agent):
            with patch("aiworks_core.logic.llm.LLMDebugLog.objects.create"):
                with patch("aiworks_core.logic.llm.LLMConfiguration.get_solo"):
                    return asyncio.run(
                        invoke_llm(
                            "test_operation",
                            messages=[
                                SystemMessage(content="sys"),
                                HumanMessage(content="user"),
                            ],
                            is_agent=True,
                            parse_json=False,
                            **kwargs,
                        )
                    )

    def test_validator_is_abstract_class(self):
        """CustomLLMResponseValidator is an ABC with validate_response as abstract method."""
        self.assertTrue(hasattr(CustomLLMResponseValidator, "validate_response"))
        self.assertTrue(
            getattr(CustomLLMResponseValidator.validate_response, "__isabstractmethod__", False)
        )
        with self.assertRaises(TypeError):
            # noinspection PyAbstractClass
            CustomLLMResponseValidator()

    def test_validator_validate_response_called_in_direct_llm_mode(self):
        """When custom_validator is provided, validate_response is called after LLM returns."""
        validated_args = {}

        class TestValidator(CustomLLMResponseValidator):
            def validate_response(self, response, **kwargs):
                validated_args.update(
                    response=response,
                    **{k: v for k, v in kwargs.items() if k != "self"},
                )

        validator = TestValidator()
        self._run_via_direct_llm("LLM output content", custom_validator=validator)

        self.assertEqual(validated_args.get("response"), "LLM output content")

    def test_validator_validate_response_called_in_agent_mode(self):
        """validate_response is called after agent run completes in is_agent=True mode."""
        validated_args = {}

        class TestValidator(CustomLLMResponseValidator):
            def validate_response(self, response, **kwargs):
                validated_args.update(
                    response=response,
                    **{k: v for k, v in kwargs.items() if k != "self"},
                )

        validator = TestValidator()
        self._run_via_agent("agent output content", custom_validator=validator)

        self.assertEqual(validated_args.get("response"), "agent output content")

    def test_validator_exception_propagates_from_direct_llm_mode(self):
        """When validator raises, invoke_llm propagates the exception in direct LLM mode."""
        exc = RuntimeError("validation failed: missing required field")

        class FailingValidator(CustomLLMResponseValidator):
            def validate_response(self, response, **kwargs):
                raise exc

        with self.assertRaises(RuntimeError) as cm:
            self._run_via_direct_llm("some output", custom_validator=FailingValidator())
        self.assertEqual(str(cm.exception), str(exc))

    def test_validator_exception_propagates_from_agent_mode(self):
        """When validator raises, invoke_llm propagates the exception in agent mode."""
        exc = RuntimeError("validation failed: content too short")

        class FailingValidator(CustomLLMResponseValidator):
            def validate_response(self, response, **kwargs):
                raise exc

        with self.assertRaises(RuntimeError) as cm:
            self._run_via_agent("some output", custom_validator=FailingValidator())
        self.assertEqual(str(cm.exception), str(exc))

    def test_validator_receives_correct_kwargs_in_direct_llm_mode(self):
        """Validator receives response and agent_callbacks in direct LLM mode."""
        received_args = {}
        callbacks = AgentCallbacks()

        class InspectingValidator(CustomLLMResponseValidator):
            def validate_response(self, response, **kwargs):
                received_args["response"] = response
                received_args.update(kwargs)

        self._run_via_direct_llm(
            "output",
            custom_validator=InspectingValidator(),
            agent_callbacks=callbacks,
        )

        self.assertEqual(received_args.get("response"), "output")
        self.assertIs(received_args.get("agent_callbacks"), callbacks)

    def test_validator_receives_correct_kwargs_in_agent_mode(self):
        """Validator receives response, tools, attached_files, output_files, agent_callbacks in agent mode."""
        received_args = {}

        class InspectingValidator(CustomLLMResponseValidator):
            def validate_response(self, response, **kwargs):
                received_args["response"] = response
                received_args.update(kwargs)

        tools = [MagicMock()]
        attached_files = {"file1.txt": "content"}
        output_files = {"out.txt": "output"}
        callbacks = AgentCallbacks()

        self._run_via_agent(
            "output",
            custom_validator=InspectingValidator(),
            tools=tools,
            attached_files=attached_files,
            output_files=output_files,
            agent_callbacks=callbacks,
        )

        self.assertEqual(received_args.get("response"), "output")
        self.assertIs(received_args.get("tools"), tools)
        self.assertIs(received_args.get("attached_files"), attached_files)
        self.assertIs(received_args.get("output_files"), output_files)
        self.assertIs(received_args.get("agent_callbacks"), callbacks)

    def test_validator_not_called_when_none(self):
        """When custom_validator=None, invoke_llm succeeds without calling any validator."""
        result = self._run_via_direct_llm("normal output", custom_validator=None)
        self.assertEqual(result, "normal output")


class InvokeLlmRenderOutputVideoTests(TestCase):
    """Tests for invoke_llm with render_output_video=True."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()
        cfg = LLMConfiguration.get_solo()
        cfg.provider = "openrouter"
        cfg.openrouter_api_key = "sk-test"
        cfg.openrouter_model = "x-ai/grok-4.1-fast"
        cfg.openrouter_video_model = "google/veo-3.1-fast"
        cfg.save()

    @patch("aiworks_core.logic.llm.get_chat_model")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    def test_render_output_video_raises_for_non_video_provider(
        self, mock_log, mock_cfg_get_solo, mock_get_chat_model
    ):
        """render_output_video=True with non-VideoGeneratorModel raises ValueError."""
        mock_cfg_get_solo.return_value = LLMConfiguration.get_solo()

        mock_llm = MagicMock()
        mock_get_chat_model.return_value = mock_llm

        with self.assertRaises(ValueError) as cm:
            asyncio.run(
                invoke_llm(
                    "test_video_op",
                    messages=[HumanMessage(content="Generate a video of a cat")],
                    render_output_video=True,
                )
            )
        self.assertIn("VideoGeneratorModel", str(cm.exception))
        self.assertIn("render_output_video=True", str(cm.exception))

    @patch("aiworks_core.logic.llm.get_chat_model")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    def test_render_output_video_calls_generate_video_with_correct_args(
        self, mock_log, mock_cfg_get_solo, mock_get_chat_model
    ):
        """render_output_video=True calls generate_video with messages and destination."""
        mock_cfg_get_solo.return_value = LLMConfiguration.get_solo()

        class MockVideoModel(VideoGeneratorModel):
            def __init__(self):
                self._mock = AsyncMock(return_value=AIMessage(
                    content="",
                    additional_kwargs={
                        "videos": [{"destination_file_path": {"path": "/tmp/video.mp4"}}]
                    }
                ))

            async def generate_video(self, messages, **kwargs):
                return await self._mock(messages, **kwargs)

        mock_llm = MockVideoModel()
        mock_get_chat_model.return_value = mock_llm

        asyncio.run(
            invoke_llm(
                "test_video_op",
                messages=[HumanMessage(content="Generate a video of a cat")],
                render_output_video=True,
                video_config={"duration": 10},
            )
        )

        mock_llm._mock.assert_called_once()
        call_args = mock_llm._mock.call_args
        self.assertEqual(call_args.kwargs["duration"], 10)

    @patch("aiworks_core.logic.llm.get_chat_model")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    def test_render_output_video_returns_destination_file_path(
        self, mock_log, mock_cfg_get_solo, mock_get_chat_model
    ):
        """render_output_video=True returns the destination_file_path from generate_video."""
        mock_cfg_get_solo.return_value = LLMConfiguration.get_solo()

        class MockVideoModel(VideoGeneratorModel):
            def __init__(self):
                self._mock = AsyncMock(return_value=AIMessage(
                    content="",
                    additional_kwargs={
                        "videos": [{"destination_file_path": {"path": "/tmp/video.mp4"}}]
                    }
                ))

            async def generate_video(self, messages, **kwargs):
                return await self._mock(messages, **kwargs)

        mock_llm = MockVideoModel()
        mock_get_chat_model.return_value = mock_llm

        result = asyncio.run(
            invoke_llm(
                "test_video_op",
                messages=[HumanMessage(content="Generate a video of a cat")],
                render_output_video=True,
            )
        )

        self.assertEqual(result, "/tmp/video.mp4")


class RenderOutputAudioTests(TestCase):
    """Tests for invoke_llm with render_output_audio=True."""

    def setUp(self):
        LLMConfiguration.objects.all().delete()
        cfg = LLMConfiguration.get_solo()
        cfg.provider = "openrouter"
        cfg.openrouter_api_key = "sk-test"
        cfg.openrouter_model = "x-ai/grok-4.1-fast"
        cfg.openrouter_audio_model = "gpt-4o-mini-tts"
        cfg.save()

    @patch("aiworks_core.logic.llm.get_chat_model")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    def test_render_output_audio_raises_for_non_audio_provider(
        self, mock_log, mock_cfg_get_solo, mock_get_chat_model
    ):
        """render_output_audio=True with non-AudioGeneratorModel raises ValueError."""
        mock_cfg_get_solo.return_value = LLMConfiguration.get_solo()

        mock_llm = MagicMock()
        mock_get_chat_model.return_value = mock_llm

        with self.assertRaises(ValueError) as cm:
            asyncio.run(
                invoke_llm(
                    "test_audio_op",
                    messages=[HumanMessage(content="Say hello")],
                    render_output_audio=True,
                )
            )
        self.assertIn("AudioGeneratorModel", str(cm.exception))
        self.assertIn("render_output_audio=True", str(cm.exception))

    @patch("aiworks_core.logic.llm.get_chat_model")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    def test_render_output_audio_calls_generate_audio_with_correct_args(
        self, mock_log, mock_cfg_get_solo, mock_get_chat_model
    ):
        """render_output_audio=True calls generate_audio with messages and audio_config."""
        mock_cfg_get_solo.return_value = LLMConfiguration.get_solo()

        class MockAudioModel(AudioGeneratorModel):
            def __init__(self):
                self._mock = AsyncMock(return_value=AIMessage(
                    content="",
                    additional_kwargs={
                        "audio": [{"destination_file_path": {"path": "/tmp/audio.mp3"}}]
                    }
                ))

            async def generate_audio(self, messages, **kwargs):
                return await self._mock(messages, **kwargs)

        mock_llm = MockAudioModel()
        mock_get_chat_model.return_value = mock_llm

        asyncio.run(
            invoke_llm(
                "test_audio_op",
                messages=[HumanMessage(content="Say hello")],
                render_output_audio=True,
                audio_config={"voice": "nova", "input": "Hello world"},
            )
        )

        mock_llm._mock.assert_called_once()
        call_args = mock_llm._mock.call_args
        self.assertEqual(call_args.kwargs["voice"], "nova")
        self.assertEqual(call_args.kwargs["input"], "Hello world")

    @patch("aiworks_core.logic.llm.get_chat_model")
    @patch("aiworks_core.logic.llm.LLMConfiguration.get_solo")
    @patch("aiworks_core.logic.llm.LLMDebugLog.objects.create")
    def test_render_output_audio_returns_destination_file_path(
        self, mock_log, mock_cfg_get_solo, mock_get_chat_model
    ):
        """render_output_audio=True returns the destination_file_path from generate_audio."""
        mock_cfg_get_solo.return_value = LLMConfiguration.get_solo()

        class MockAudioModel(AudioGeneratorModel):
            def __init__(self):
                self._mock = AsyncMock(return_value=AIMessage(
                    content="",
                    additional_kwargs={
                        "audio": [{"destination_file_path": {"path": "/tmp/audio.mp3"}}]
                    }
                ))

            async def generate_audio(self, messages, **kwargs):
                return await self._mock(messages, **kwargs)

        mock_llm = MockAudioModel()
        mock_get_chat_model.return_value = mock_llm

        result = asyncio.run(
            invoke_llm(
                "test_audio_op",
                messages=[HumanMessage(content="Say hello")],
                render_output_audio=True,
                audio_config={"voice": "nova", "input": "Hello world"},
            )
        )

        self.assertEqual(result, "/tmp/audio.mp3")

