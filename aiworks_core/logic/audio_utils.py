import logging
import os

from langchain_core.messages import HumanMessage
from moviepy import AudioFileClip

from .llm import invoke_llm
from ..utils import async_to_sync

logger = logging.getLogger(__name__)


def fetch_audio(
        text: str,
        voice: str,
) -> tuple[bytes, float]:
    """Generate TTS audio from text using audio/speech API.

    Args:
        text: The text to convert to speech.
        voice: The voice ID to use (e.g., 'alloy', 'onyx', 'nova').

    Returns:
        Tuple of (raw audio bytes, duration in seconds).
    """
    try:
        temp_path = async_to_sync(invoke_llm)(
            "generate_audio",
            messages=[HumanMessage(content=text)],
            render_output_audio=True,
            audio_config={"voice": voice},
        )
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
        try:
            audio_clip = AudioFileClip(temp_path)
            duration = audio_clip.duration
            audio_clip.close()
        except Exception as exp:
            logger.exception("fetch_audio | error: %s", exp)
            duration = 0.0
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        logger.info("fetch_audio | voice=%s | text_len=%d | duration=%.1fs", voice, len(text), duration)
        return audio_bytes, duration
    except Exception as error:
        logger.exception("fetch_audio | error: %s", error)
        raise
