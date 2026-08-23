from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage, AIMessage


class VideoGeneratorModel(ABC):
    @abstractmethod
    async def generate_video(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        """Generate a video, write it to the destination path, and return a dict.

        Args:
            messages: List of LangChain messages to convert to a video prompt.
            **kwargs: Provider-specific video generation options
                (e.g., duration, resolution, aspect_ratio, etc.).

        Returns:
            An AIMessage containing the generated video as  additional_kwargs/videos[:]/destination_file_path/path

        Raises:
            TimeoutError — if the generation exceeds the provider's timeout.
            Exception — on job failure or other provider errors.
        """
        raise NotImplementedError()
