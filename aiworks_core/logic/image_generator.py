from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage, AIMessage


class ImageGeneratorModel(ABC):
    @abstractmethod
    async def generate_image(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        """Generate an image, and returns the image embedded within the result AIMessage.

        Args:
            messages: List of LangChain messages to convert to a video prompt.
            **kwargs: Provider-specific video generation options
                (e.g., duration, resolution, aspect_ratio, etc.).

        Returns:
            An AIMessage containing the image as additional_kwargs/images[:]/image_url/url
        """
        raise NotImplementedError()
