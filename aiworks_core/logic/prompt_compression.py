import asyncio
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache

from ..utils import sync_to_async
from langchain_core.messages import HumanMessage
from llmlingua import PromptCompressor
from ..models import LLMConfiguration
from ..utils import close_connections

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compressor support
# ---------------------------------------------------------------------------

_COMPRESSOR_POOL = None
_COMPRESSOR_INIT_LOCK = threading.Lock()
_COMPRESS_CACHE_FUNC = None
_COMPRESS_CACHE_LOCK = threading.Lock()
_LLM_CALL_EXECUTOR = None
_LLM_CALL_EXECUTOR_LOCK = threading.Lock()
_LLM_CALL_EXECUTOR_MAX_WORKERS = 4
_COMPRESS_CHUNK_CHARS = 50_000


def _get_llm_call_executor():
    global _LLM_CALL_EXECUTOR
    if _LLM_CALL_EXECUTOR is None:
        with _LLM_CALL_EXECUTOR_LOCK:
            if _LLM_CALL_EXECUTOR is None:
                _LLM_CALL_EXECUTOR = ThreadPoolExecutor(
                    max_workers=_LLM_CALL_EXECUTOR_MAX_WORKERS
                )
    return _LLM_CALL_EXECUTOR


def _get_llmlingua_compressor_pool(cfg):
    """Lazy-load the pool once per Gunicorn worker process."""
    global _COMPRESSOR_POOL
    if _COMPRESSOR_POOL is None:
        with _COMPRESSOR_INIT_LOCK:
            if _COMPRESSOR_POOL is None:
                pool_size = cfg.compress_prompts_pool_size
                model_name = cfg.compress_prompts_model
                # maxsize=pool_size makes .get() block when all instances are busy
                q = queue.Queue(maxsize=pool_size)
                for _ in range(pool_size):
                    inst = PromptCompressor(
                        model_name=model_name, use_llmlingua2=True, device_map="cpu"
                    )
                    q.put(inst)
                _COMPRESSOR_POOL = q
    return _COMPRESSOR_POOL


@contextmanager
def _get_llmlingua_compressor_from_pool(cfg):
    pool = _get_llmlingua_compressor_pool(cfg)
    # Wait up to 30 seconds for a compressor to become available
    instance = pool.get(block=True, timeout=30)
    try:
        yield instance
    finally:
        # This ALWAYS runs, preventing 'leaking' instances
        pool.put(instance)


def _get_compress_cache_func(cfg):
    """Lazily create the LRU-cached compression function, sized from config."""
    global _COMPRESS_CACHE_FUNC
    if _COMPRESS_CACHE_FUNC is None:
        with _COMPRESS_CACHE_LOCK:
            if _COMPRESS_CACHE_FUNC is None:
                max_cache_size = cfg.compress_prompts_cache_size or 256

                @lru_cache(maxsize=max_cache_size)
                def _cached(
                        text: str,
                        use_llm: bool,
                        llm_prompt: str,
                        rate: float,
                        force_tokens: tuple,
                        chunk_end_tokens: tuple,
                ):
                    logger.debug(f"Compressing prompt: {text}")
                    compressed_prompt = text

                    try:
                        if use_llm:
                            messages = [
                                HumanMessage(content=llm_prompt.format(text=text)),
                            ]
                            from .llm import invoke_llm

                            async def _run_inner():
                                try:
                                    return await invoke_llm(
                                        "prompt_compression",
                                        messages=messages,
                                        parse_json=False,
                                    )
                                finally:
                                    close_connections()

                            # Run in a fresh thread so asyncio.run() can create its own event loop.
                            # lru_cache ensures this only happens on cache misses.
                            result = (
                                _get_llm_call_executor()
                                .submit(asyncio.run, _run_inner())
                                .result()
                            )
                            compressed_prompt = (
                                result.strip()
                                if isinstance(result, str) and result.strip()
                                else text
                            )
                        else:
                            with _get_llmlingua_compressor_from_pool(cfg) as compressor:
                                result = compressor.compress_prompt(
                                    [text],
                                    rate=rate,
                                    force_tokens=list(force_tokens),
                                    force_reserve_digit=True,
                                    chunk_end_tokens=list(chunk_end_tokens),
                                    use_sentence_level_filter=True,
                                    keep_first_sentence=1,
                                    strict_preserve_uncompressed=True,
                                )
                                compressed_prompt = result.get(
                                    "compressed_prompt", text
                                )
                    except Exception as exp:
                        logger.warning(
                            f"Prompt compression failed, using original prompt. Error: {exp}"
                        )
                        compressed_prompt = text

                    logger.debug(f"Compressed prompt: {compressed_prompt}")
                    logger.info(
                        f"Prompt compressed from {len(text)} to {len(compressed_prompt)} characters"
                    )
                    return compressed_prompt

                _COMPRESS_CACHE_FUNC = _cached
    return _COMPRESS_CACHE_FUNC


async def compress_prompt(text: str) -> str:
    cfg = await sync_to_async(LLMConfiguration.get_solo)()
    if not cfg.compress_prompts:
        return text
    min_len = cfg.compress_prompts_min_length

    if len(text) < min_len:
        logger.debug(f"Skipping compression (length {len(text)} < min {min_len})")
        return text

    use_llm = cfg.compress_prompts_use_llm
    llm_prompt = cfg.compress_prompts_llm_prompt
    force_tokens = tuple(
        cfg.compress_prompts_force_tokens or [".", "\n", ":", "-", "$", "%"]
    )
    chunk_end_tokens = tuple(cfg.compress_prompts_chunk_end_tokens or [".", "\n"])
    compression_func = _get_compress_cache_func(cfg)
    compressed_prompt = _compress_chunked(
        compression_func,
        text=text,
        use_llm=use_llm,
        llm_prompt=llm_prompt,
        rate=cfg.compress_prompts_target_token_rate,
        force_tokens=force_tokens,
        chunk_end_tokens=chunk_end_tokens,
    )

    return compressed_prompt


def _compress_chunked(compression_func, text, **additional_params) -> str:
    if len(text) <= _COMPRESS_CHUNK_CHARS:
        return compression_func(text, **additional_params)
    chunks = []
    start = 0
    while start < len(text):
        end = start + _COMPRESS_CHUNK_CHARS
        logger.info("_compress_chunked | compressing chunk %d:%d of %d", start, end, len(text))
        chunk = text[start:end]
        compressed = compression_func(chunk, **additional_params)
        chunks.append(compressed)
        start = end
    return "\n".join(chunks)
