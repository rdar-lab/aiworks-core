"""OpenRouter-specific LangChain extensions."""

import asyncio
import logging
import tempfile
from typing import Any

import httpx
from httpx import AsyncClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter
from .audio_generator import AudioGeneratorModel
from .image_generator import ImageGeneratorModel
from .video_generator import VideoGeneratorModel
from ..models import LLMConfiguration
from ..utils import raise_for_status, sync_to_async

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenRouter provider suffix parsing
# ---------------------------------------------------------------------------


def parse_openrouter_model(model: str) -> tuple[str, list | None, bool]:
    """Strip <PROVIDER> suffix from model name for OpenRouter providers routing.

    E.g. 'minimax_27<MYPROVIDER>' -> ('minimax_27', 'MYPROVIDER')
         'minimax_27'             -> ('minimax_27', None)
    """
    if "<" in model and model.endswith(">"):
        providers_str = model[model.index("<") + 1: -1]
        allow_fallback = providers_str.endswith("+")
        if allow_fallback:
            providers_str = providers_str[:-1]
        providers = providers_str.split(",")

        clean_model = model[:model.index("<")]
        return clean_model, providers, allow_fallback
    return model, None, False


class ChatOpenRouterExtended(ChatOpenRouter, VideoGeneratorModel, ImageGeneratorModel, AudioGeneratorModel):
    """ChatOpenRouter that properly extracts images from assistant message responses.

    The base ChatOpenRouter._create_chat_result() calls _convert_dict_to_message() which
    is a module-level function that does not extract the 'images' field from OpenRouter
    API responses. This subclass overrides _create_chat_result to inject images into
    response.additional_kwargs["images"] so that downstream code can access them.

    implements VideoGeneratorModel for video generation via OpenRouter's video API.
    implements ImageGeneratorModel for image generation via OpenRouter's image API.
    """

    def _create_chat_result(self, response: Any) -> Any:
        result = super()._create_chat_result(response)

        if not isinstance(response, dict):
            response = response.model_dump(by_alias=True)

        choices = response.get("choices", [])
        for i, gen in enumerate(result.generations):
            if i < len(choices):
                raw_message = choices[i].get("message", {})
                images = raw_message.get("images", [])
                if images:
                    msg = gen.message
                    if isinstance(msg, AIMessage):
                        if not hasattr(msg, "additional_kwargs") or msg.additional_kwargs is None:
                            msg.additional_kwargs = {}
                        msg.additional_kwargs["images"] = images
                        logger.info("Extracted %d images from response and attached to additional_kwargs of message",
                                    len(images))
                    else:
                        logger.warning(
                            "Detected images, but result is not an AIMessage, cannot attach images to additional_kwargs")

        return result

    async def generate_image(
            self,
            messages: list[BaseMessage],
            **kwargs: Any,
    ) -> AIMessage:
        response = await self.ainvoke(
            messages,
            modalities=["text", "image"],
            **{
                "image_config": kwargs
            } if kwargs else {}
        )
        return response

    async def generate_video(
            self,
            messages: list[BaseMessage],
            **kwargs: Any,
    ) -> AIMessage:
        prompt = self._messages_to_prompt(messages)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            destination = f.name

        video_url = "https://openrouter.ai/api/v1/videos"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
        }

        if kwargs:
            for key, val in kwargs.items():
                payload[key] = kwargs[key]

        async with httpx.AsyncClient() as client:
            polling_url = await self._submit_video_job(client, video_url, payload, headers)
            video_download_urls = await self._poll_for_video_completion(client, polling_url, self.request_timeout,
                                                                        headers)
            return await self._download_video(client, video_download_urls, destination, headers)

    @staticmethod
    async def _submit_video_job(client: AsyncClient, video_url: str, payload: dict[str, str],
                                headers: dict[str, str]) -> Any:
        logger.info("OpenRouter - Initiating video generation...")
        response = await client.post(video_url, headers=headers, json=payload, timeout=30)
        raise_for_status(response)
        result = response.json()
        return result["polling_url"]

    @staticmethod
    async def _poll_for_video_completion(client: AsyncClient, polling_url, request_timeout, headers):
        start_time = asyncio.get_event_loop().time()
        timeout_sec = request_timeout / 1000.0 if request_timeout else 600

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if timeout_sec and elapsed >= timeout_sec:
                raise TimeoutError(
                    f"Video generation timed out after {elapsed:.1f}s"
                )

            poll_response = await client.get(polling_url, headers=headers, timeout=30)
            raise_for_status(poll_response)
            status_result = poll_response.json()
            status = status_result.get("status")
            logger.info(f"OpenRouter - Video Generation - Status is {status}")

            if status == "completed":
                unsigned_urls = status_result.get("unsigned_urls", [])
                if not unsigned_urls:
                    raise Exception("Video job completed but no unsigned_urls returned")
                return unsigned_urls
            if status == "failed":
                error = status_result.get("error", "Unknown error")
                raise Exception(f"Video generation failed: {error}")

            await asyncio.sleep(30)

    @staticmethod
    async def _download_video(client, video_download_urls, destination, headers):
        with open(destination, "wb") as f:
            for url in video_download_urls:
                logger.info(f"OpenRouter - Video Generation - Downloading result from URL {url}")
                async with client.stream("GET", url, headers=headers, timeout=60) as video_response:
                    raise_for_status(video_response)
                    async for chunk in video_response.aiter_bytes(chunk_size=65536):
                        f.write(chunk)

        return_result = AIMessage(content="")
        return_result.additional_kwargs = {
            "videos": [
                {
                    "destination_file_path": {
                        "path": destination
                    }
                }
            ]
        }

        return return_result

    async def generate_audio(
            self,
            messages: list[BaseMessage],
            **kwargs: Any,
    ) -> AIMessage:
        input_text = kwargs.get("input", "")
        voice = kwargs.get("voice", "")

        if not input_text:
            input_text = self._messages_to_prompt(messages)

        if not voice:
            voice = await sync_to_async(self._get_default_voice)()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            destination = f.name

        audio_url = "https://openrouter.ai/api/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            **kwargs,
            "model": self.model,
            "input": input_text,
            "voice": voice,
            "response_format": "mp3"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(audio_url, headers=headers, json=payload, timeout=60)
            raise_for_status(response)
            audio_bytes = response.content

        with open(destination, "wb") as f:
            f.write(audio_bytes)

        return_result = AIMessage(content="")
        return_result.additional_kwargs = {
            "audio": [
                {
                    "destination_file_path": {
                        "path": destination
                    }
                }
            ]
        }
        return return_result

    @staticmethod
    def _messages_to_prompt(messages: list[BaseMessage]) -> str:
        """Convert a list of LangChain messages to a single text prompt."""
        parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                parts.append(msg.content)
            elif isinstance(msg, AIMessage):
                if msg.content:
                    parts.append(msg.content)
            else:
                parts.append(str(msg.content))
        return "\n".join(parts).strip()

    @staticmethod
    def _get_default_voice():
        voices = LLMConfiguration.get_solo().audio_voices
        if not voices:
            raise Exception("No voices specified")
        return voices[0]
