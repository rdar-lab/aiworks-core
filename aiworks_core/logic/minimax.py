"""OpenRouter-specific LangChain extensions."""

import asyncio
import logging
import tempfile
from typing import Any

import httpx
from httpx import AsyncClient
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from .audio_generator import AudioGeneratorModel
from .image_generator import ImageGeneratorModel
from .video_generator import VideoGeneratorModel
from ..models import LLMConfiguration
from ..utils import raise_for_status, sync_to_async

logger = logging.getLogger(__name__)

_MINIMAX_BASE_URL = "https://api.minimax.io/v1"


class ChatMiniMax(ChatOpenAI, ImageGeneratorModel, AudioGeneratorModel, VideoGeneratorModel):
    """
    ChatMiniMax implementation
    """

    async def generate_video(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        prompt = self._messages_to_prompt(messages)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            destination = f.name

        video_url = "https://api.minimax.io/v1/video_generation"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key.get_secret_value()}",
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
            task_id = await self._submit_video_job(client, video_url, payload, headers)
            if not task_id:
                raise Exception("Failed to submit video job. Got empty task_id")
            file_id = await self._poll_for_video_completion(client, task_id, self.request_timeout, headers)
            if not file_id:
                raise Exception("Failed to get Job download file_id. Got empty file_id")
            return await self._download_video(client, file_id, destination, headers)

    @staticmethod
    async def _download_video(client, file_id, destination, headers):
        download_url = f"https://api.minimax.io/v1/files/retrieve?file_id={file_id}"

        with open(destination, "wb") as f:
            logger.info(f"Minimax - Video Generation - Downloading result from URL {download_url}")
            async with client.stream("GET", download_url, headers=headers, timeout=60) as video_response:
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

    @staticmethod
    async def _submit_video_job(client: AsyncClient, video_url: str, payload: dict[str, str],
                                headers: dict[str, str]) -> Any:
        logger.info("Minimax - Initiating video generation...")
        response = await client.post(video_url, headers=headers, json=payload, timeout=30)
        raise_for_status(response)
        result = response.json()
        logger.info(f"Minimax - Video generation - Submit video result: {result}")
        return result["task_id"]

    @staticmethod
    async def _poll_for_video_completion(client: AsyncClient, task_id, request_timeout, headers):
        start_time = asyncio.get_event_loop().time()
        timeout_sec = request_timeout if request_timeout else 600
        polling_url = f"https://api.minimax.io/v1/query/video_generation?task_id={task_id}"

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if timeout_sec and elapsed >= timeout_sec:
                raise TimeoutError(
                    f"Video generation timed out after {elapsed:.1f}s"
                )

            poll_response = await client.get(polling_url, headers=headers, timeout=30)
            raise_for_status(poll_response)
            status_result = poll_response.json()
            logger.info(f"Minimax - Video generation - Polling result: {status_result}")
            status = status_result.get("status")
            logger.info(f"MiniMax - Video Generation - Status is {status}")

            if status == "Success":
                return status_result.get("file_id")
            if status == "failed":
                error = status_result.get("error", "Unknown error")
                raise Exception(f"Video generation failed: {error}")

            await asyncio.sleep(30)

    @staticmethod
    def _get_default_voice():
        voices = LLMConfiguration.get_solo().audio_voices
        if not voices:
            raise Exception("No voices specified")
        return voices[0]

    async def generate_audio(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        input_text = kwargs.get("input", "")
        voice = kwargs.get("voice", "")

        if not input_text:
            input_text = self._messages_to_prompt(messages)

        if not voice:
            voice = await sync_to_async(self._get_default_voice)()

        audio_url = "https://api.minimax.io/v1/t2a_v2"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "text": input_text,
            "stream": False,
            "voice_setting": {
                "voice_id": voice,
            },
            "language_boost": "auto",
            "output_format": "hex"
        }

        if kwargs:
            for key, val in kwargs.items():
                payload[key] = kwargs[key]

        async with httpx.AsyncClient() as client:
            response = await client.post(audio_url, headers=headers, json=payload, timeout=60)
            raise_for_status(response)
            audio_hex = response.json()["data"]["audio"]

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            destination = f.name

        with open(destination, "wb") as f:
            f.write(bytes.fromhex(audio_hex))

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

    async def generate_image(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        prompt = self._messages_to_prompt(messages)
        image_url = "https://api.minimax.io/v1/image_generation"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "response_format": "base64",
        }
        if kwargs:
            for key, val in kwargs.items():
                payload[key] = kwargs[key]

        async with httpx.AsyncClient() as client:
            response = await client.post(image_url, headers=headers, json=payload, timeout=60)
            raise_for_status(response)
            images = response.json()["data"]["image_base64"]

        return_result = AIMessage(content="")
        return_result.additional_kwargs = {
            "images": [
                {
                    "image_url": {
                        "url": "data:image/jpeg;base64," + image
                    }
                }
                for image in images
            ]
        }

        return return_result

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            **{
                **kwargs,
                "base_url": _MINIMAX_BASE_URL,
            }
        )

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
