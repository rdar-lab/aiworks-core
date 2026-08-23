from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


class AudioGeneratorModel(ABC):
    @abstractmethod
    async def generate_audio(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        """Generate audio from text, write it to a destination path, and return an AIMessage.

        Args:
            messages: List of LangChain messages (prompt context).
            **kwargs: Provider-specific audio generation options
                (e.g., voice, input, model, etc.).

        Returns:
            An AIMessage containing the generated audio as
            additional_kwargs/audio[:]/destination_file_path/path

        Raises:
            TimeoutError — if the generation exceeds the provider's timeout.
            Exception — on job failure or other provider errors.
        """
        raise NotImplementedError()
