"""
Generic LLM and Agent factory supporting multiple providers via LangChain.

Public API:
  get_chat_model(llm_type, temperature)  – create a chat model instance from the DB configuration
  get_agent(llm_type, tools, temperature, use_deep_agent) – create a LangGraph/Deep Agent
  invoke_llm(operation_name, ...)         – invoke LLM/agent with per-operation config lookup,
                                           template resolution, automatic retries, and JSON extraction
"""

import asyncio
import contextvars
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml
from deepagents.backends import StateBackend
from deepagents.backends.protocol import WriteResult
from deepagents.backends.utils import create_file_data, file_data_to_string
from deepagents.graph import create_deep_agent
from deepagents.middleware.summarization import SummarizationDefaults
from django.conf import settings
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools import tool as lang_tool
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_tavily import TavilySearch, TavilyExtract
from pydantic import PrivateAttr
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    retry_if_not_exception_type,
)
from .audio_generator import AudioGeneratorModel
from .image_generator import ImageGeneratorModel
from .minimax import ChatMiniMax
from .openrouter import ChatOpenRouterExtended, parse_openrouter_model
from .schema_validation_utils import validate_json_with_schema_file
from .video_generator import VideoGeneratorModel
from ..models import LLMConfiguration, LLMDebugLog, LLMOperationConfig
from ..utils import async_to_sync, sync_to_async
from ..utils import close_connections

logger = logging.getLogger(__name__)

DEFAULT_TOOL_TIMEOUT = 600

_REASONING_THINK_TAG_RE = re.compile(r'<think[^>]*>.*?</think>', re.DOTALL | re.IGNORECASE)
_THINK_TAGS_RE = re.compile(r'|<think>|</think>', re.DOTALL | re.IGNORECASE)

_agent_backend_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "agent_backend", default=None
)

_active_tools_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "active_tools", default=None
)


def init_agent_backend(backend) -> None:
    _agent_backend_var.set(backend)


def get_agent_backend():
    return _agent_backend_var.get()


def init_active_tools(tools) -> None:
    _active_tools_var.set(tools)


def get_active_tools():
    return _active_tools_var.get()


class CustomLLMResponseValidator(ABC):
    @abstractmethod
    def validate_response(self,
                          response: Any,
                          tools: Optional[List] = None,
                          attached_files: Optional[Dict[str, str]] = None,
                          output_files: Optional[Dict[str, str]] = None,
                          agent_callbacks=None):
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# Sub-agent Support
# ---------------------------------------------------------------------------

@lang_tool
def spawn_subagent(custom_prompt: str) -> str:
    """
    Spawn a subagent to perform work on your behalf.

    The subagent executes with access to all tools available to you
    and can read and write files in the filesystem.
    The subagent has access to the same tools as the main agent, and is running sandboxed like the main agent.
    A subagent cannot execute code or run shell scripts.

    Args:
        custom_prompt: Instructions for the subagent describing what it should do,
                       its role, and any context or data it needs.
    """
    backend = get_agent_backend()
    if backend is None:
        return "Error: Subagent context not available. spawn_subagent must be called from within a running deep agent."
    messages = [SystemMessage(content=custom_prompt)]
    return async_to_sync(invoke_llm)(
        operation_name="subagent",
        messages=messages,
        is_agent=True,
        is_deep_agent=True,
        is_sub_agent=True,
        parse_json=False,
    )


# ---------------------------------------------------------------------------
# Agent callbacks
# ---------------------------------------------------------------------------


@dataclass
class AgentCallbacks:
    """Encapsulates real-time callbacks for agent execution progress.

    All fields are optional.  Pass an instance to ``invoke_llm`` (or through
    ``run_deep_agent`` / ``run_deep_agent_on_session``) to receive live updates
    as the agent runs.

    on_todos_update:
        Called whenever the agent calls ``write_todos`` with the current
        task list.  Use this to persist step tasks to the database for the
        SSE stream.
    on_thinking_update:
        Called on every LLM response that contains thinking/reasoning blocks
        (e.g. Anthropic extended-thinking, DeepSeek Reasoner, Google Gemini).
        The string argument is the new thinking text to append / replace.
    on_tool_call:
        Called whenever the agent makes a tool call.  The dict argument has
        keys ``tool`` (str) and ``input`` (str).  Use this to surface the last
        tool call to the UI in real time.
    """

    on_todos_update: Optional[Callable[[List[Dict]], None]] = None
    on_thinking_update: Optional[Callable[[str], None]] = None
    on_tool_call: Optional[Callable[[Dict[str, str]], None]] = None


# ---------------------------------------------------------------------------
# Retry support
# ---------------------------------------------------------------------------

_RETRY_ATTEMPTS = 3
_RETRY_MIN_WAIT = 2
_RETRY_MAX_WAIT = 10

_MAX_AGENT_ITERATIONS = 100


# ---------------------------------------------------------------------------
# Patching support
# ---------------------------------------------------------------------------


def _patch_deepagents_summarization():
    import deepagents.middleware.summarization as summarization_module

    # Guard against double-patching using the graph module's flag, since that is the
    # module whose local reference actually needs replacing (see below).
    if getattr(summarization_module, "_deepagents_patched", False):
        return

    # noinspection PyUnusedLocal
    def _patched_compute_summarization_defaults(
            model: BaseChatModel,
    ) -> SummarizationDefaults:
        """
        Prevents Deep Agents from overflowing the LLM context window during
        web-research tasks while retaining enough message history for the agent
        to remember what it has already read and avoid repeating work.

        The token trigger is set to 30 000 so that summarization only fires once
        the accumulated context is genuinely large (typically after 3-5 articles).
        Keeping 20 messages gives the agent enough conversational memory to track
        which subjects and URLs it has already covered across the whole research run.
        The truncate_args_settings mirror those values for tool-call argument history.
        """
        return {
            "trigger": ("tokens", 60000),
            "keep": ("messages", 50),
            "truncate_args_settings": {
                "trigger": ("messages", 70),
                "keep": ("messages", 50),
            },
        }

    summarization_module.compute_summarization_defaults = (
        _patched_compute_summarization_defaults
    )
    summarization_module._deepagents_patched = True


class OverridingStateBackend(StateBackend):

    def write(self, file_path: str, content: str) -> WriteResult:
        new_file_data = create_file_data(content)
        self._send_files_update({file_path: self._prepare_for_storage(new_file_data)})
        return WriteResult(path=file_path)


# ---------------------------------------------------------------------------
# Chat model factory
# ---------------------------------------------------------------------------


def get_chat_model(
        provider: str,
        model: str,
        llm_type: str,
        temperature: float = 0.7,
        timeout_sec: float = 60.0 * 10
) -> BaseChatModel:
    """
    Create and return a LangChain chat model based on the LLM configuration
    stored in the database (``LLMConfiguration`` singleton).

    Supported providers and their required fields:
        google      – google_api_key
        openai      – openai_api_key
        minimax     – minimax_api_key
        anthropic   – anthropic_api_key
        azure       – azure_api_key, azure_deployment, azure_endpoint
        aws         – aws_access_key_id / aws_secret_access_key / aws_region
        ollama      – ollama_base_url  (optional)
        huggingface – huggingface_api_key
        deepseek    – deepseek_api_key
        openrouter  – openrouter_api_key

    Args:
        provider: the provider to user
        model:    The model to use
        llm_type: THE llm type str
        temperature: Sampling temperature.
        timeout_sec: The timeout in seconds
    """
    cfg = LLMConfiguration.get_solo()

    if provider == "google":
        api_key = cfg.google_api_key
        if not api_key:
            raise ValueError("google_api_key is not configured in LLM Configuration")
        return ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout_sec,
        )

    if provider == "openai":
        api_key = cfg.openai_api_key
        if not api_key:
            raise ValueError("openai_api_key is not configured in LLM Configuration")
        openai_kwargs = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
            "timeout": timeout_sec,
        }
        if cfg.openai_base_url:
            openai_kwargs["base_url"] = cfg.openai_base_url
        return ChatOpenAI(**openai_kwargs)

    if provider == "minimax":
        api_key = cfg.minimax_api_key
        if not api_key:
            raise ValueError("minimax_api_key is not configured in LLM Configuration")
        minimax_kwargs = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
            "timeout": timeout_sec
        }

        return ChatMiniMax(**minimax_kwargs)

    if provider == "anthropic":
        api_key = cfg.anthropic_api_key
        if not api_key:
            raise ValueError("anthropic_api_key is not configured in LLM Configuration")

        anthropic_kwargs = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
            "timeout": timeout_sec,
        }
        if cfg.anthropic_base_url:
            anthropic_kwargs["base_url"] = cfg.anthropic_base_url

        # noinspection PyArgumentList
        return ChatAnthropic(**anthropic_kwargs)

    if provider == "azure":
        azure_api_key = cfg.azure_api_key
        if not azure_api_key:
            raise ValueError("azure_api_key is not configured in LLM Configuration")
        azure_endpoint = cfg.azure_endpoint
        if not azure_endpoint:
            raise ValueError("azure_endpoint is not configured in LLM Configuration")

        return AzureChatOpenAI(
            azure_deployment=model,
            temperature=temperature,
            timeout=timeout_sec,
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
        )

    if provider == "aws":
        creds_keys = {
            "aws_access_key_id": cfg.aws_access_key_id,
            "aws_secret_access_key": cfg.aws_secret_access_key,
            "aws_session_token": cfg.aws_session_token,
        }
        credentials = {k: v for k, v in creds_keys.items() if v}

        # Use profile_name only when no explicit credentials are provided
        # This allows fallback to AWS credentials file (~/.aws/credentials)
        # noinspection PyArgumentList
        return ChatBedrockConverse(
            model_id=model,
            region_name=cfg.aws_region or "us-east-1",
            temperature=temperature,
            timeout=timeout_sec,
            credentials_profile_name=cfg.aws_profile_name if not credentials else None,
            **credentials,
        )

    if provider == "ollama":
        base_url = cfg.ollama_base_url or "http://localhost:11434"
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

    if provider == "huggingface":
        api_key = cfg.huggingface_api_key
        if not api_key:
            raise ValueError(
                "huggingface_api_key is not configured in LLM Configuration"
            )
        # noinspection PyArgumentList
        endpoint = HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=api_key,
            temperature=temperature,
            timeout=timeout_sec,
        )
        return ChatHuggingFace(llm=endpoint)

    if provider == "deepseek":
        api_key = cfg.deepseek_api_key
        if not api_key:
            raise ValueError("deepseek_api_key is not configured in LLM Configuration")
        return ChatDeepSeek(
            model=model,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout_sec,
        )

    if provider == "openrouter":
        api_key = cfg.openrouter_api_key
        if not api_key:
            raise ValueError(
                "openrouter_api_key is not configured in LLM Configuration"
            )
        clean_model, openrouter_providers, allow_fallback = parse_openrouter_model(model)

        openrouter_kwargs = {}

        if openrouter_providers:
            openrouter_kwargs["model_kwargs"] = {
                "provider": {
                    "order": openrouter_providers,
                    "allow_fallbacks": allow_fallback,
                }
            }

        if llm_type == "reasoning" and cfg.openrouter_reasoning_effort:
            openrouter_kwargs["reasoning"] = {
                "effort": cfg.openrouter_reasoning_effort,
                "include_reasoning": True
            }
            openrouter_kwargs.setdefault("model_kwargs", {})
            openrouter_kwargs["model_kwargs"].setdefault("provider", {})
            openrouter_kwargs["model_kwargs"]["provider"].update({
                "require_reasoning": True,
                "sort": "throughput"
            })

        return ChatOpenRouterExtended(
            model=clean_model,
            api_key=api_key,
            temperature=temperature,
            timeout=int(timeout_sec * 1000),
            **openrouter_kwargs,
        )

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. "
        "Supported providers: google, openai, anthropic, azure, aws, ollama, huggingface, deepseek, openrouter"
    )


def _resolve_provider_model(
        operation_name: str,
) -> tuple[str, str, str]:
    """Return ``(active_provider, model_name, resolved_llm_type)``.

    Raises ``ValueError`` if the provider or model cannot be resolved from
    ``LLMConfiguration`` / ``LLMOperationConfig``.
    """

    # Resolve operation config
    try:
        op_config = LLMOperationConfig.objects.get(operation_name=operation_name)
    except LLMOperationConfig.DoesNotExist:
        op_config = None

    # Treat disabled config as non-existent
    if op_config is not None and not op_config.is_enabled:
        op_config = None

    override_provider = (op_config.override_provider or None) if op_config else None
    override_model = (op_config.override_model or None) if op_config else None
    llm_type = (op_config.selected_llm_type or None) if op_config else None

    # Default llm_type is smart
    if not llm_type:
        llm_type = "smart"

    cfg = LLMConfiguration.get_solo()

    if override_provider and override_model:
        active_provider = override_provider.lower()
        model = override_model
    else:
        # Determine which provider to use for this task type.
        if llm_type == "fast":
            active_provider = (cfg.fast_provider or cfg.provider).lower()
        elif llm_type == "reasoning":
            active_provider = (cfg.reasoning_provider or cfg.provider).lower()
        elif llm_type == "ultra-fast":
            active_provider = (cfg.ultra_fast_provider or cfg.provider).lower()
        elif llm_type == "ultra-smart":
            active_provider = (cfg.ultra_smart_provider or cfg.provider).lower()
        elif llm_type == "image":
            active_provider = (cfg.image_provider or cfg.provider).lower()
        elif llm_type == "video":
            active_provider = (cfg.video_provider or cfg.provider).lower()
        elif llm_type == "audio":
            active_provider = (cfg.audio_provider or cfg.provider).lower()
        else:
            active_provider = cfg.provider.lower()

        # If not configured
        if not active_provider:
            active_provider = "not_configured"

        # Raise immediately if not configured — no point retrying
        if active_provider == "not_configured":
            raise ValueError(
                "LLM provider not configured. Set the provider in LLMConfiguration in the Django admin."
            )

        # Resolve model from the active provider's per-task field.
        # Azure uses "deployment" terminology; all other providers use *_model fields.
        if active_provider == "azure":
            model = cfg.azure_deployment or ""
            if llm_type == "fast":
                model = cfg.azure_fast_deployment or model
            elif llm_type == "reasoning":
                model = cfg.azure_reasoning_deployment or model
            elif llm_type == "ultra-fast":
                model = cfg.azure_ultra_fast_deployment or model
            elif llm_type == "ultra-smart":
                model = cfg.azure_ultra_smart_deployment or model
            elif llm_type == "image":
                model = cfg.azure_image_deployment or model
            elif llm_type == "video":
                model = cfg.azure_video_deployment or model
            elif llm_type == "audio":
                model = cfg.azure_audio_deployment or model
        else:
            model = getattr(cfg, f"{active_provider}_model", "") or ""
            if llm_type == "fast":
                model = getattr(cfg, f"{active_provider}_fast_model", "") or model
            elif llm_type == "reasoning":
                model = getattr(cfg, f"{active_provider}_reasoning_model", "") or model
            elif llm_type == "ultra-fast":
                model = getattr(cfg, f"{active_provider}_ultra_fast_model", "") or model
            elif llm_type == "ultra-smart":
                model = (
                        getattr(cfg, f"{active_provider}_ultra_smart_model", "") or model
                )
            elif llm_type == "image":
                model = getattr(cfg, f"{active_provider}_image_model", "") or model
            elif llm_type == "video":
                model = getattr(cfg, f"{active_provider}_video_model", "") or model
            elif llm_type == "audio":
                model = getattr(cfg, f"{active_provider}_audio_model", "") or model

        if not model:
            raise ValueError(
                f'No model configured for provider "{active_provider}" with llm_type="{llm_type}". '
                "Set the model in LLMConfiguration in the Django admin."
            )

    return active_provider, model, llm_type


# ---------------------------------------------------------------------------
# Tool output normalization and safe wrapping
# ---------------------------------------------------------------------------


def _coerce_tool_output(result: Any) -> str:
    """Coerce any tool output to a plain string safe for ToolMessage.content."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result)
    except Exception as exp:
        logger.warning(
            f"Tool output coercion failed for result: {result} | error: {exp}"
        )
        return str(result)


class _SafeToolWrapper(BaseTool):
    """Wraps a BaseTool to coerce output to string and catch all exceptions."""

    description: str = ""
    args_schema: Optional[Any] = None
    return_direct: bool = False
    _inner: BaseTool = PrivateAttr()
    _timeout: float = PrivateAttr()

    @classmethod
    def wrap(cls, tool: BaseTool, timeout: Optional[float] = None) -> "_SafeToolWrapper":
        instance = cls(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            return_direct=tool.return_direct,
        )
        instance._inner = tool
        instance._timeout = timeout if timeout is not None else DEFAULT_TOOL_TIMEOUT
        return instance

    def _run(self, *args: Any, **kwargs: Any) -> str:
        try:
            logger.info(
                f"Invoking tool: {self._inner.name} with args: {args} and kwargs: {kwargs}"
            )
            result = _coerce_tool_output(
                async_to_sync(asyncio.wait_for)(
                    self._inner._arun(*args, **kwargs),
                    timeout=self._timeout
                )
            )
            logger.info(f"Tool {self._inner.name} returned result: {result[:1000]}")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Tool {self._inner.name} timed out after {self._timeout}s")
            return f"Tool error: Timeout after {self._timeout}s"
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Tool {self._inner.name} - error: {exc}")
            return f"Tool error: {repr(exc)}"
        finally:
            close_connections()

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        try:
            logger.info(
                f"Invoking tool: {self._inner.name} with args: {args} and kwargs: {kwargs}"
            )
            result = _coerce_tool_output(
                await asyncio.wait_for(
                    self._inner._arun(*args, **kwargs),
                    timeout=self._timeout
                )
            )
            logger.info(f"Tool {self._inner.name} returned result: {result[:1000]}")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Tool {self._inner.name} timed out after {self._timeout}s")
            return f"Tool error: Timeout after {self._timeout}s"
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Tool {self._inner.name} - error: {exc}")
            return f"Tool error: {repr(exc)}"
        finally:
            close_connections()


class _SafeStructuredToolWrapper(BaseTool):
    """Safe wrapper for StructuredTool — forwards config and run_manager explicitly."""

    description: str = ""
    args_schema: Optional[Any] = None
    return_direct: bool = False
    _inner: StructuredTool = PrivateAttr()
    _timeout: float = PrivateAttr()

    @classmethod
    def wrap(cls, tool: StructuredTool, timeout: Optional[float] = None) -> "_SafeStructuredToolWrapper":
        instance = cls(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            return_direct=tool.return_direct,
        )
        instance._inner = tool
        instance._timeout = timeout if timeout is not None else DEFAULT_TOOL_TIMEOUT
        return instance

    def _run(self, *args: Any, run_manager=None, **kwargs: Any) -> str:
        try:
            logger.info(f"Invoking tool: {self._inner.name} | input: {args or kwargs}")
            # noinspection PyProtectedMember
            result = _coerce_tool_output(
                async_to_sync(asyncio.wait_for)(
                    self._inner._arun(*args, run_manager=run_manager, **kwargs),
                    timeout=self._timeout
                )
            )
            logger.info(f"Tool {self._inner.name} returned: {result[:1000]}")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Tool {self._inner.name} timed out after {self._timeout}s")
            return f"Tool error: Timeout after {self._timeout}s"
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Tool {self._inner.name} - error: {exc}")
            return f"Tool error: {repr(exc)}"
        finally:
            close_connections()

    async def _arun(
            self, *args: Any, config=None, run_manager=None, **kwargs: Any
    ) -> str:
        try:
            logger.info(f"Invoking tool: {self._inner.name} | input: {args or kwargs}")
            # noinspection PyProtectedMember
            result = _coerce_tool_output(
                await asyncio.wait_for(
                    self._inner._arun(
                        *args, config=config, run_manager=run_manager, **kwargs
                    ),
                    timeout=self._timeout
                )
            )
            logger.info(f"Tool {self._inner.name} returned: {result[:1000]}")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Tool {self._inner.name} timed out after {self._timeout}s")
            return f"Tool error: Timeout after {self._timeout}s"
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Tool {self._inner.name} - error: {exc}")
            return f"Tool error: {repr(exc)}"
        finally:
            close_connections()


def _make_safe_tool(tool: BaseTool) -> BaseTool:
    """Return a safe-wrapped version of *tool*."""
    if isinstance(tool, StructuredTool):
        return _SafeStructuredToolWrapper.wrap(tool)
    return _SafeToolWrapper.wrap(tool)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def get_agent(
        provider: str,
        model: str,
        llm_type: Optional[str] = None,
        tools: Optional[Sequence] = None,
        temperature: float = 0.7,
        use_deep_agent: bool = False,
        backend: StateBackend | None = None
):
    """
    Create and return a LangGraph agent using the configured chat model.

    Args:
        llm_type:       Passed through to ``get_chat_model`` to select fast or
                        smart model (see ``get_chat_model`` for details).
        tools:          List of LangChain tools to give the agent.
        temperature:    Sampling temperature forwarded to the underlying model.
        use_deep_agent: When True, create a Deep Agent (requires deepagents).
                        When False (default), create a standard ReAct agent
                        (requires langgraph).
        provider:       Passed to ``get_chat_model``.
        model:          Passed to ``get_chat_model``.
        backend:        Backend to use or None

    Returns:
        A compiled LangGraph ``CompiledStateGraph``.

    Note:
        System prompt / instructions should be passed at invocation time via
        the ``messages`` argument (e.g. include a SystemMessage as the first
        message when calling ``agent.ainvoke``).
    """
    llm = get_chat_model(
        provider=provider,
        model=model,
        llm_type=llm_type or 'smart',
        temperature=temperature,
    )
    tools = [_make_safe_tool(t) for t in (tools or [])]

    if use_deep_agent:
        _patch_deepagents_summarization()
        return create_deep_agent(model=llm, tools=tools, backend=backend)
    else:
        return create_agent(llm, tools=tools)


# ---------------------------------------------------------------------------
# Prompts helpers
# ---------------------------------------------------------------------------

_prompts: Optional[Dict] = None


def get_prompts() -> Dict:
    """Lazy-load the application prompt templates (cached after first load).

    The host application must set ``AIWORKS_CORE_PROMPTS_FILE`` in settings
    to point to its prompts.yaml.  The library does not ship a default file.
    """
    global _prompts
    if _prompts is None:
        prompts_path = getattr(settings, "AIWORKS_CORE_PROMPTS_FILE", None)
        if prompts_path is None:
            raise ValueError(
                "AIWORKS_CORE_PROMPTS_FILE is not set in Django settings. "
                "aiworks-core requires the host application to provide a prompts.yaml "
                "file and set AIWORKS_CORE_PROMPTS_FILE to its path."
            )
        with open(prompts_path, "r") as f:
            _prompts = yaml.safe_load(f)

        if _prompts is None or not isinstance(_prompts, dict):
            raise ValueError(f"Prompts file at {prompts_path} is empty or not a dict")

    return _prompts


def _navigate_prompts(path: str) -> str:
    """Navigate a dot-separated key path within the prompts dict."""
    node: Any = get_prompts()
    for part in path.split("."):
        node = node[part]
    return node


def _extract_llm_response(result: Any) -> Any:
    # If it is STR return it
    if isinstance(result, str):
        return _REASONING_THINK_TAG_RE.sub('', result)

    # If it is base message - get the content out of it and parse again
    if isinstance(result, BaseMessage):
        return _extract_llm_response(result.content)

    if isinstance(result, dict):
        # If it contains a structured_response, return it
        if "structured_response" in result:
            return result["structured_response"]

        # If it is a list of messages - grab the last one and parse again
        if "messages" in result:
            if not result["messages"]:
                return ""
            return _extract_llm_response(result["messages"][-1])

        if "output" in result:
            return _extract_llm_response(result["output"])

        if "content" in result:
            return _extract_llm_response(result["content"])

        if "text" in result:
            return _extract_llm_response(result["text"])

    if isinstance(result, list):
        if not result:
            return ""
        return _extract_llm_response(result[-1])

    return str(result)


def _extract_json(text: str) -> str:
    """Strip markdown fences from an LLM response that wraps JSON."""
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Shared todos extraction utility
# ---------------------------------------------------------------------------


def _validate_todos(todos: Any) -> Optional[List[Dict]]:
    """Validate and normalize a todos value.

    Returns a list of dicts with 'content' (str) and 'status' (str), or None.
    """
    if todos is None:
        return None

    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "AGENT TODOS | received todos as malformed JSON string, dropping: %s",
                repr(todos),
            )
            return None

    if not isinstance(todos, list):
        logger.warning(
            "AGENT TODOS | todos is not a list, dropping: %s",
            repr(todos),
        )
        return None

    for item in todos:
        if not isinstance(item, dict):
            logger.warning(
                "AGENT TODOS | todo item is not a dict, dropping: %s",
                repr(todos),
            )
            return None
        if not isinstance(item.get("content"), str) or not isinstance(item.get("status"), str):
            logger.warning(
                "AGENT TODOS | todo item missing content or status, dropping: %s",
                repr(todos),
            )
            return None

    return todos


def _extract_todos(
        input_str: str, inputs: Optional[Dict[str, Any]]
) -> Optional[List[Dict]]:
    """Extract the todos list from a write_todos tool call.

    Prefers the structured ``inputs`` dict supplied by modern LangChain;
    falls back to parsing the raw ``input_str`` JSON.
    """
    todos_val = None

    if inputs and "todos" in inputs:
        todos_val = inputs["todos"]

    if todos_val is None:
        try:
            parsed = json.loads(input_str)
            if isinstance(parsed, dict) and "todos" in parsed:
                todos_val = parsed["todos"]
            elif isinstance(parsed, list):
                todos_val = parsed
        except (json.JSONDecodeError, TypeError):
            pass

    return _validate_todos(todos_val)


def _extract_thoughts(response: Any) -> List[str]:
    """Extract thinking/reasoning blocks from a LangChain ``LLMResult``.

    Supports multiple provider formats:

    * **Anthropic extended-thinking / Google Gemini thinking models** —
      ``type: "thinking"`` content blocks inside ``AIMessage.content`` (a list).
      Both providers produce the same block shape
      ``{"type": "thinking", "thinking": "<text>"}`` via their respective
      LangChain integrations (``langchain-anthropic`` and
      ``langchain-google-genai``).  Provider APIs may evolve independently, but
      both libraries currently map thinking output to this format.    * **AWS Bedrock** (Claude with extended thinking) — ``type: "reasoning_content"``
      blocks inside ``AIMessage.content``, shaped as
      ``{"type": "reasoning_content", "reasoning_content": {"text": "<text>"}}``.
    * **DeepSeek Reasoner** — ``reasoning_content`` stored in
      ``AIMessage.additional_kwargs`` by ``ChatDeepSeek`` natively.
    * **OpenRouter** — ``reasoning_content`` stored in
      ``AIMessage.additional_kwargs`` by ``ChatOpenRouter`` natively, in the same
      format as DeepSeek.  Both captured by the same branch below.
    * **OpenAI / Azure OpenAI** — standard and reasoning (o1/o3) models do not
      expose raw reasoning text in the API response; nothing to capture.

    This helper is called from ``on_llm_end`` which fires after every
    LLM turn — including turns that end with a tool call — so thoughts
    emitted before tool invocations are captured as well.  Each invocation
    returns only the thoughts present in the *current* response; callers are
    responsible for accumulating results across turns.
    """
    thoughts: List[str] = []
    for gen_list in getattr(response, "generations", []):
        for gen in gen_list:
            msg = getattr(gen, "message", None)
            if msg is None:
                continue

            content = getattr(msg, "content", None)
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue

                    # Anthropic / Google Gemini: {"type": "thinking", "thinking": "..."}
                    if block.get("type") == "thinking":
                        thought = block.get("thinking", "")
                        if thought:
                            thoughts.append(thought)

                    # AWS Bedrock (Claude): {"type": "reasoning_content", "reasoning_content": {"text": "..."}}
                    elif block.get("type") == "reasoning_content":
                        rc = block.get("reasoning_content") or {}
                        thought = rc.get("text", "")
                        if thought:
                            thoughts.append(thought)

            # MiniMax / OpenAI-compatible providers: <thinking>...</thinking> tags in raw string content
            elif "<think>" in str(content).lower():
                for match in _REASONING_THINK_TAG_RE.finditer(str(content)):
                    thought = match.group(0)
                    thought = _THINK_TAGS_RE.sub('', thought)
                    if thought and thought not in thoughts:
                        thoughts.append(thought)

            # DeepSeek Reasoner / OpenRouter: reasoning_content in additional_kwargs
            additional_kwargs = getattr(msg, "additional_kwargs", {}) or {}
            reasoning: Optional[str] = additional_kwargs.get("reasoning_content")
            if reasoning:
                thoughts.append(reasoning)

    return thoughts


# ---------------------------------------------------------------------------
# Agent debug data collector
# ---------------------------------------------------------------------------


class _DebugDataCollector(BaseCallbackHandler):
    """Accumulates agent debug data (history, tasks) during execution.

    Implements ``BaseCallbackHandler`` directly so it can be registered as a
    standalone callback alongside the logging handler.  It does **not** write
    to the application logger — all logging concerns are handled by the
    dedicated ``_AgentLoggingCallbackHandler``.

    ``history`` is a chronologically-ordered list of agent events.  Each
    entry has a ``type`` key:

    * ``{"type": "thought", "content": "<reasoning text>"}``
    * ``{"type": "tool_run", "tool": "<name>", "input": "<raw args>", "output": "<result>"}``

    Interleaving thoughts and tool runs in a single list lets the reader
    trace the full chain of reasoning and actions in execution order.

    An instance is created per ``invoke_llm`` call and passed to
    ``_audit_llm_call`` so the collected data is persisted to the
    ``LLMDebugLog`` row.
    """

    def __init__(self) -> None:
        super().__init__()
        self.history: List[Dict[str, Any]] = []
        self.tasks: List[Dict] = []
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None

    def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Record every tool invocation; capture write_todos task lists."""
        tool_name = serialized.get("name", "unknown")
        run_id = kwargs.get("run_id")
        self.history.append(
            {"type": "tool_run", "tool": tool_name, "input": input_str, "run_id": str(run_id) if run_id else "unknown"})

        if tool_name == "write_todos":
            todos = _extract_todos(input_str, kwargs.get("inputs"))
            if todos is not None:
                self.tasks = todos

    def on_tool_end(
            self, output: Any, **kwargs: Any
    ) -> None:
        """Record the output of every tool invocation."""
        run_id = kwargs.get("run_id")
        if run_id is not None:
            run_id = str(run_id)
            for entry in reversed(self.history):
                if entry.get("run_id") == run_id:
                    entry["output"] = str(output)
                    return

        logger.warning(f'Tool {run_id if run_id else "unknown"} output could not be captured in history. '
                       'Was unable to find relevant history record')

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Capture thinking/reasoning blocks and accumulate token usage across turns."""
        for thought in _extract_thoughts(response):
            self.history.append({"type": "thought", "content": thought})

        # Accumulate token counts reported by the LLM across all agent turns
        for gen_list in getattr(response, "generations", []):
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                usage = getattr(msg, "usage_metadata", None)
                if not usage:
                    continue
                in_tok = usage.get("input_tokens") or 0
                out_tok = usage.get("output_tokens") or 0
                if in_tok:
                    self.input_tokens = (self.input_tokens or 0) + in_tok
                if out_tok:
                    self.output_tokens = (self.output_tokens or 0) + out_tok


def _build_memory_md(
        collector: "_DebugDataCollector",
        messages: Optional[List["BaseMessage"]] = None,
        response_content: Optional[str] = None,
) -> Optional[str]:
    """Generate MEMORY.md content from agent execution history and tasks.

    The resulting file captures the original input messages, the agent's
    reasoning trace and task list so it can be injected into a future session
    to resume or inspect the run.
    """
    if len(collector.tasks) > 0 or len(collector.history) > 0:
        lines: list[str] = ["# Agent Execution Memory\n"]

        if messages:
            lines.append("## Input Messages\n")
            for msg in messages:
                role = type(msg).__name__.replace("Message", "")
                content_text = (
                    str(msg.content)
                    if not isinstance(msg.content, str)
                    else msg.content
                )
                lines.append(f"**{role}:**\n\n{content_text}\n")
            lines.append("")

        if collector.tasks:
            lines.append("## Tasks\n")
            for task in collector.tasks:
                status = task.get("status", "")
                content_text = task.get("content", "")
                lines.append(f"- [{status}] {content_text}")
            lines.append("")

        if collector.history:
            lines.append("## Execution History\n")
            for entry in collector.history:
                if entry.get("type") == "thought":
                    thought_text = str(entry.get("content", ""))
                    lines.append(f"**Thought:** {thought_text}\n")
                elif entry.get("type") == "tool_run":
                    tool_name = entry.get("tool", "")
                    tool_input = str(entry.get("input", ""))
                    tool_output = str(entry.get("output", ""))
                    if tool_output:
                        lines.append(f"**Tool:** {tool_name} | **Input:** {tool_input} | **Output:** {tool_output}\n")
                    else:
                        lines.append(f"**Tool:** {tool_name} | **Input:** {tool_input}\n")

            lines.append("")

        if response_content is not None:
            lines.append("## Final Response\n")
            lines.append(response_content)
        lines.append("")

        return "\n".join(lines)
    return None


# ---------------------------------------------------------------------------
# Agent logging callback
# ---------------------------------------------------------------------------


class _AgentLoggingCallbackHandler(BaseCallbackHandler):
    """Logs agent tool calls and LLM iterations for troubleshooting."""

    _STATUS_ICONS: Dict[str, str] = {
        "pending": "○",
        "in_progress": "⏳",
        "completed": "✓",
    }

    def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        # NOTE: this logs the LLM-requested arguments; instance-level overrides (e.g.
        # search_depth, extract_depth) are applied inside _run and may differ.
        logger.info(
            "AGENT TOOL START | tool=%s | llm_input=%s", tool_name, str(input_str)[:500]
        )

        if tool_name == "write_todos":
            todos = _extract_todos(input_str, kwargs.get("inputs"))
            if todos is not None:
                self._log_todos(todos)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        logger.info("AGENT TOOL END | output=%s", str(output)[:500])

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        logger.warning("AGENT TOOL ERROR | error=%s", error)

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        logger.info(
            "AGENT LLM ITERATION | model=%s | prompts_count=%d",
            serialized.get("name", "unknown"),
            len(prompts),
        )

    def on_chat_model_start(
            self, serialized: Dict[str, Any], messages: List[List], **kwargs: Any
    ) -> None:
        logger.info(
            "AGENT LLM ITERATION | model=%s | messages_count=%d",
            serialized.get("name", "unknown"),
            sum(len(m) for m in messages),
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Log thinking/reasoning blocks from extended-thinking LLM responses."""
        for thought in _extract_thoughts(response):
            logger.info("AGENT THOUGHT | %s", str(thought)[:500])

    def _log_todos(self, todos: List[Dict]) -> None:
        """Log a todos list with status indicators."""
        lines = ["AGENT TODOS | updated todo list:"]
        for item in todos:
            icon = self._STATUS_ICONS.get(item.get("status", ""), "?")
            lines.append(
                f"  {icon} [{item.get('status', '?')}] {item.get('content', '')}"
            )
        logger.info("\n".join(lines))


class _AgentCallbacksHandler(BaseCallbackHandler):
    """Unified handler for all agent progress callbacks.

    Receives an ``AgentCallbacks`` instance and fires the appropriate
    callback for each agent event:

    - ``write_todos`` tool call → ``on_todos_update``
    - LLM response with thinking blocks → ``on_thinking_update``
    - Any tool call → ``on_tool_call``
    """

    def __init__(self, callbacks: AgentCallbacks) -> None:
        super().__init__()
        self._callbacks = callbacks

    def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "unknown")

        # Fire on_tool_call for every tool invocation
        if self._callbacks.on_tool_call is not None:
            try:
                self._callbacks.on_tool_call({"tool": tool_name, "input": input_str})
            except Exception as exc:
                logger.warning("AGENT CALLBACKS | on_tool_call error: %s", exc)

        # Handle write_todos for on_todos_update
        if tool_name != "write_todos":
            return

        todos = _extract_todos(input_str, kwargs.get("inputs"))
        if todos is None:
            return

        if self._callbacks.on_todos_update is not None:
            try:
                self._callbacks.on_todos_update(todos)
            except Exception as exc:
                logger.warning("AGENT CALLBACKS | on_todos_update error: %s", exc)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Fire on_thinking_update for every LLM response that contains thinking."""
        if self._callbacks.on_thinking_update is None:
            return
        for thought in _extract_thoughts(response):
            try:
                self._callbacks.on_thinking_update(thought)
            except Exception as exc:
                logger.warning("AGENT CALLBACKS | on_thinking_update error: %s", exc)


# ---------------------------------------------------------------------------
# LLM audit / debug logging
# ---------------------------------------------------------------------------


async def _audit_llm_call(
        operation_name: str,
        provider: str,
        model: str,
        call_label: str,
        messages: List[BaseMessage],
        raw_output: str,
        exp: Optional[Exception],
        collector: Optional[_DebugDataCollector] = None,
        *,
        is_agent: bool = False,
        is_deep_agent: bool = False,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
) -> None:
    """Write a ``LLMDebugLog`` row for every LLM call.

    Metadata (operation_name, provider, model, is_agent, is_deep_agent, token
    counts, error) is always persisted regardless of the debug setting.  Raw
    payload fields (input_messages, raw_output, agent_history, agent_tasks) are
    only populated when ``LLMConfiguration.llm_debug`` is enabled **or** the
    call failed, so that storage stays lean in production while still providing
    full data for troubleshooting.

    * **Debug enabled, no error** — full row written.
    * **Debug enabled, error** — full row written with the error message.
    * **Debug disabled, error** — full row written with error message and raw payloads.
    * **Debug disabled, no error** — metadata-only row (no raw payloads).

    Audit failures never propagate to the application layer (caught internally
    and logged at ERROR level).

    When a ``_DebugDataCollector`` is supplied token counts accumulated across
    agent turns are merged into the totals, and agent-specific data (thoughts,
    tool runs, tasks) is included when raw payloads are enabled.
    """
    try:
        cfg = await sync_to_async(LLMConfiguration.get_solo)()
        include_raw = cfg.llm_debug or exp is not None

        serialized_input = ""
        if include_raw:
            serialized_input = json.dumps(
                [{"type": type(m).__name__, "content": m.content} for m in messages],
                indent=2,
                ensure_ascii=False,
            )

        agent_history = ""
        agent_tasks = ""

        # Merge token counts accumulated by the collector (agent calls)
        if collector is not None:
            if collector.input_tokens is not None:
                input_tokens = (input_tokens or 0) + collector.input_tokens
            if collector.output_tokens is not None:
                output_tokens = (output_tokens or 0) + collector.output_tokens

            if include_raw:
                agent_history = json.dumps(
                    collector.history, indent=2, ensure_ascii=False
                )
                agent_tasks = json.dumps(collector.tasks, indent=2, ensure_ascii=False)

        await sync_to_async(LLMDebugLog.objects.create)(
            operation_name=operation_name,
            call_label=call_label,
            provider=provider,
            model=model,
            is_agent=is_agent,
            is_deep_agent=is_deep_agent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_messages=serialized_input,
            raw_output=raw_output if include_raw else "",
            error=str(exp) if exp is not None else "",
            agent_history=agent_history,
            agent_tasks=agent_tasks,
        )
    except Exception as exc:
        logger.error(
            "AUDIT LLM CALL | failed to write debug log | error=%s", exc, exc_info=True
        )


# ---------------------------------------------------------------------------
# Generic LLM invocation
# ---------------------------------------------------------------------------


# noinspection PyTypeChecker
@retry(
    retry=retry_if_not_exception_type(ValueError),
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=2, min=_RETRY_MIN_WAIT, max=_RETRY_MAX_WAIT),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.ERROR, exc_info=True),
)
async def invoke_llm(
        operation_name: str,
        temperature: float = 0.7,
        system_message_template_name: Optional[str] = None,
        user_message_template_name: Optional[str] = None,
        template_params: Optional[Dict] = None,
        messages: Optional[List] = None,
        is_agent: bool = False,
        is_deep_agent: bool = False,
        is_sub_agent: bool = False,
        tools: Optional[list[BaseTool]] = None,
        parse_json: bool = True,
        agent_callbacks: Optional[AgentCallbacks] = None,
        output_files: Optional[dict] = None,
        attached_files: Optional[Dict[str, str]] = None,
        render_output_image: bool = False,
        image_config: Optional[dict] = None,
        render_output_video: bool = False,
        video_config: Optional[dict] = None,
        render_output_audio: bool = False,
        audio_config: Optional[dict] = None,
        fail_on_empty_response: bool = True,
        schema_file_path: Optional[str] = None,
        custom_validator: Optional[CustomLLMResponseValidator] = None
) -> Any:
    """
    Generic LLM invocation with per-operation config lookup, template resolution,
    automatic retries, and optional JSON extraction.

    Either ``messages`` (pre-built list of LangChain message objects) or
    the template names/params should be supplied.

    Steps performed:
      1. Look up ``LLMOperationConfig`` for ``operation_name`` to resolve
         ``llm_type``, ``override_provider``, and ``override_model``.
         Falls back to system defaults when no config exists.
      2. Resolve ``system_message_template_name`` and
         ``user_message_template_name`` from the prompts YAML and format
         them with the supplied params (skipped when ``messages`` is given).
      3. Obtain the chat model (or agent) via ``get_chat_model`` /
         ``get_agent`` using the resolved config.
      4. Invoke with automatic retries (tenacity, up to
         ``_RETRY_ATTEMPTS`` attempts).
      5. Optionally parse and return the JSON payload from the response.

    Args:
        operation_name:                Identifier for this LLM call (e.g. my_llm_op,
                                       synthesis).  Looked up in ``LLMOperationConfig``
                                       to resolve per-operation LLM settings.
        temperature:                   Sampling temperature.
        system_message_template_name:  Dot-path into prompts.yaml for the
                                       system message template.
        user_message_template_name:    Dot-path into prompts.yaml for the
                                       user message template.
        template_params:               Format kwargs for both templates.
        messages:                      Pre-built message list; overrides
                                       template resolution when supplied.
        is_agent:                      Use a ReAct agent instead of direct LLM.
        is_deep_agent:                 Use a Deep Agent (implies is_agent=True).
        is_sub_agent:                  Is this a sub-agent.
        tools:                         Tools for the agent.
        parse_json:                    When True (default), parse and return
                                       the JSON payload from the response.
                                       When False, return raw response content.
        agent_callbacks:               Optional ``AgentCallbacks`` instance
                                       containing ``on_todos_update``,
                                       ``on_thinking_update``, and ``on_tool_call``
                                       handlers for real-time agent progress
                                       (step tasks, thinking, last tool call).
        output_files:                  Output dictionary to return any generated file on
        attached_files:                Optional mapping of file path → content string to
                                       pre-seed the agent's virtual filesystem before
                                       execution starts.  Only meaningful when
                                       ``is_agent=True``.  The agent can read these files
                                       via its standard ``read_file`` tool.  Only files
                                       that the agent *writes* (new or changed entries)
                                       are captured in ``output_files``; the seed files
                                       themselves are not re-persisted unless the agent
                                       overwrites them.
        render_output_image:           When True, render and return the image output from
                                       the LLM response instead of text.  Only supported
                                       in non-agent mode (``is_agent=False``).
        image_config:                  When render_output_image=True allows specifying specific constraints for
                                       image generation
        render_output_video:           When True, render and return the video output from
                                       the LLM response instead of text.  Only supported
                                       in non-agent mode (``is_agent=False``).
        video_config:                  When render_output_video=True allows specifying specific constraints for
                                       video generation
        render_output_audio:           When True, render and return the audio output from
                                       the LLM response instead of text.  Only supported
                                       in non-agent mode (``is_agent=False``).
        audio_config:                  When render_output_audio=True allows specifying specific constraints for
                                       audio generation (e.g. voice, input).
        fail_on_empty_response:        When True (default), raise an exception if the LLM
                                       returns an empty or whitespace-only response.
                                       When False, return the empty response as-is.
        schema_file_path:              Optional file path to a JSON Schema that the LLM response should conform to.
        custom_validator:              Optional custom validator

    Returns:
        Parsed JSON (dict/list) when ``parse_json=True``, or the raw
        response content string when ``parse_json=False``.
    """
    # Resolve the actual provider and model name for audit purposes
    provider, model, resolved_llm_type = await sync_to_async(_resolve_provider_model)(
        operation_name=operation_name
    )

    # Build messages list from templates when not provided directly
    if messages is None:
        system_content = _navigate_prompts(system_message_template_name)
        user_content = _navigate_prompts(user_message_template_name)

        if template_params:
            system_content = system_content.format(**template_params)
            user_content = user_content.format(**template_params)

        messages: list[BaseMessage] = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]

    # Log the outgoing call
    call_label = system_message_template_name or "(custom messages)"
    display_llm_type = resolved_llm_type or "smart"
    logger.info(
        "LLM CALL | operation=%s | llm_type=%s | parse_json=%s | messages_count=%d",
        operation_name,
        display_llm_type,
        parse_json,
        len(messages),
    )
    for i, msg in enumerate(messages):
        logger.debug("  [msg %d] %s: %s", i, type(msg).__name__, msg.content)

    collector: Optional[_DebugDataCollector] = None
    content = ""
    audit_exception: Optional[Exception] = None
    # Token counts for non-agent calls (agent calls accumulate in collector)
    audit_input_tokens: Optional[int] = None
    audit_output_tokens: Optional[int] = None
    try:
        # --- LLM invocation ---
        try:
            if render_output_image and is_agent:
                raise ValueError(
                    "render_output_image=True is not supported in agent mode. "
                    "Use is_agent=False for image generation."
                )
            if render_output_video and is_agent:
                raise ValueError(
                    "render_output_video=True is not supported in agent mode. "
                    "Use is_agent=False for video generation."
                )
            if render_output_audio and is_agent:
                raise ValueError(
                    "render_output_audio=True is not supported in agent mode. "
                    "Use is_agent=False for audio generation."
                )
            if is_agent:
                if is_sub_agent:
                    backend = get_agent_backend()
                    tools = get_active_tools() if not tools else tools
                else:
                    backend = OverridingStateBackend() if is_deep_agent else None

                agent = await sync_to_async(get_agent)(
                    provider=provider,
                    model=model,
                    tools=tools,
                    temperature=temperature,
                    use_deep_agent=is_deep_agent,
                    llm_type=resolved_llm_type,
                    backend=backend
                )
                collector = _DebugDataCollector()
                callbacks = [_AgentLoggingCallbackHandler(), collector]
                if agent_callbacks is not None:
                    callbacks.append(_AgentCallbacksHandler(agent_callbacks))
                # Build initial agent input; optionally pre-seed the VFS with attached_files
                agent_input: dict = {"messages": messages}
                if attached_files:
                    # Normalise paths and convert string content to FileData dicts
                    seeded: dict = {}
                    for path, content in attached_files.items():
                        normalised = "/" + path.lstrip("/")
                        seeded[normalised] = create_file_data(content)
                    agent_input["files"] = seeded
                    logger.info(
                        "invoke_llm | seeding agent VFS with %d file(s): %s",
                        len(seeded),
                        list(seeded.keys()),
                    )

                try:
                    if not is_sub_agent:
                        init_agent_backend(backend)
                        init_active_tools(tools)
                    result = await agent.ainvoke(
                        agent_input,
                        config={
                            "callbacks": callbacks,
                            "recursion_limit": _MAX_AGENT_ITERATIONS * 3,
                        },
                    )
                finally:
                    if not is_sub_agent:
                        init_agent_backend(None)
                        init_active_tools(None)
                content = _extract_llm_response(result)
                if output_files is not None:
                    for path, file_data in result.get("files", {}).items():
                        # If path contains "large_tool_results" we ignore it
                        if "large_tool_results" in path:
                            continue

                        if "conversation_history" in path:
                            continue

                        # Only capture files that were written (new or changed) by the agent.
                        # Skip files that were seeded unchanged — compare by identity/value.
                        seed_data = (agent_input.get("files") or {}).get(path)

                        if not path.startswith("/") and not seed_data:
                            seed_data = (agent_input.get("files") or {}).get(f'/{path}')

                        if seed_data is not None and seed_data == file_data:
                            # File was seeded but not touched by the agent; skip.
                            continue
                        output_files[path] = file_data_to_string(file_data)
                        logger.info(f"Saved output file from agent: {path}")
                    # Generate MEMORY.md from the agent execution trace when using deep agent
                    if is_deep_agent and collector is not None:
                        memory_md = _build_memory_md(collector, messages=messages, response_content=content)
                        if memory_md:
                            output_files["/memory.md"] = memory_md
                            logger.info("Saved memory output file: /memory.md")

                if fail_on_empty_response and (not content or not content.strip()):
                    raise Exception(
                        f"LLM returned empty response for operation '{operation_name}'"
                    )

                if custom_validator:
                    custom_validator.validate_response(
                        content,
                        tools=tools,
                        attached_files=attached_files,
                        output_files=output_files,
                        agent_callbacks=agent_callbacks,
                    )

            else:
                llm = await sync_to_async(get_chat_model)(
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    llm_type=resolved_llm_type,
                )
                if render_output_image:
                    if not isinstance(llm, ImageGeneratorModel):
                        raise ValueError(
                            f"render_output_image=True requires a ImageGeneratorModel provider, "
                            f"but got {type(llm).__name__}."
                        )
                    response = await llm.generate_image(
                        messages,
                        **(image_config or {}),
                    )
                    content = _extract_llm_image_response(response)
                elif render_output_video:
                    if not isinstance(llm, VideoGeneratorModel):
                        raise ValueError(
                            f"render_output_video=True requires a VideoGeneratorModel provider, "
                            f"but got {type(llm).__name__}."
                        )
                    response = await llm.generate_video(
                        messages,
                        **(video_config or {}),
                    )
                    content = _extract_llm_video_response(response)
                elif render_output_audio:
                    if not isinstance(llm, AudioGeneratorModel):
                        raise ValueError(
                            f"render_output_audio=True requires an AudioGeneratorModel provider, "
                            f"but got {type(llm).__name__}."
                        )
                    response = await llm.generate_audio(
                        messages,
                        **(audio_config or {}),
                    )
                    content = _extract_llm_audio_response(response)
                else:
                    response = await llm.ainvoke(messages)
                    content = _extract_llm_response(response)

                # Extract token usage from the response when available
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    audit_input_tokens = usage.get("input_tokens")
                    audit_output_tokens = usage.get("output_tokens")

                if not content or not content.strip():
                    raise Exception(
                        f"LLM returned empty response for operation '{operation_name}'"
                    )

                if custom_validator:
                    custom_validator.validate_response(
                        content,
                        agent_callbacks=agent_callbacks,
                    )
        except Exception as exc:
            audit_exception = exc
            raise

        # Log the raw response before parsing
        logger.info(
            "LLM RESPONSE | template=%s | raw_length=%d", call_label, len(content)
        )
        logger.debug("LLM RESPONSE RAW:\n%s", content)

        # --- JSON parsing ---
        if parse_json and not render_output_image and not render_output_video and not render_output_audio:
            try:
                json_content = _extract_json(content)

                if schema_file_path:
                    valid, err = validate_json_with_schema_file(json_content, schema_file_path)
                    if not valid:
                        raise Exception(f"Schema validation failed: {err}")

                parsed = json.loads(json_content)
            except Exception as exc:
                logger.exception(
                    "LLM JSON PARSE ERROR | template=%s | raw_content=%s",
                    call_label,
                    content,
                )
                audit_exception = exc
                raise
            logger.debug(
                "LLM RESPONSE PARSED: %s",
                json.dumps(parsed, ensure_ascii=False, default=str),
            )
            return parsed
        return content
    finally:
        await _audit_llm_call(
            operation_name,
            provider,
            model,
            call_label,
            messages,
            content,
            audit_exception,
            collector,
            is_agent=is_agent,
            is_deep_agent=is_deep_agent,
            input_tokens=audit_input_tokens,
            output_tokens=audit_output_tokens,
        )


def _extract_llm_image_response(response: AIMessage) -> Any:
    images = None
    if hasattr(response, "additional_kwargs"):
        images = response.additional_kwargs.get("images")
    if not images:
        raise Exception(
            "Response did not include any images"
        )
    content = images[0]["image_url"]["url"]
    return content


def _extract_llm_video_response(response: AIMessage) -> Any:
    videos = None
    if hasattr(response, "additional_kwargs"):
        videos = response.additional_kwargs.get("videos")
    if not videos:
        raise Exception(
            "Response did not include any videos"
        )
    content = videos[0]["destination_file_path"]["path"]
    return content


def _extract_llm_audio_response(response: AIMessage) -> Any:
    audio = None
    if hasattr(response, "additional_kwargs"):
        audio = response.additional_kwargs.get("audio")
    if not audio:
        raise Exception(
            "Response did not include any audio"
        )
    content = audio[0]["destination_file_path"]["path"]
    return content


def get_web_tools() -> list[BaseTool]:
    cfg = LLMConfiguration.get_solo()
    tavily_api_key = cfg.tavily_api_key or None

    tools: list[BaseTool] = []

    if not tavily_api_key:
        logger.info("Using duckduckgo search tool as tavily_api_key is not configured")
        tools.append(DuckDuckGoSearchRun())
    else:
        logger.info("Using tavily search tool")
        tools.append(TavilySearch(
            max_results=5,
            topic="general",
            search_depth="basic",
            tavily_api_key=tavily_api_key,
        ))
        tools.append(TavilyExtract(
            extract_depth="basic",
            format="markdown",
            chunks_per_source=3,
            tavily_api_key=tavily_api_key,
        ))

    return tools


def get_playwright_tools() -> list[BaseTool]:
    is_playwright_enabled = settings.PLAYWRIGHT_MCP_ENABLED
    playwright_url = settings.PLAYWRIGHT_MCP_URL

    if is_playwright_enabled:
        try:
            client = MultiServerMCPClient({
                "playwright": {
                    "url": playwright_url,
                    "transport": "streamable_http",
                }
            })
            playwright_tools = async_to_sync(client.get_tools)()
            logger.info("Loaded %d Playwright MCP tools from %s", len(playwright_tools), playwright_url)
            return playwright_tools
        except Exception as exc:
            logger.warning("Failed to load Playwright MCP tools from %s: %s", playwright_url, exc)
    return []
