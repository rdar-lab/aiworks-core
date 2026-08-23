import base64
import logging
import uuid

from ..utils import async_to_sync
from langchain_core.tools import tool
from .files import set_binary_content
from .image_utils import resize_image, calculate_aspect_ratio
from .llm import invoke_llm
from ..models import Session, AttachedFile

logger = logging.getLogger(__name__)


def _resize_image(image_embedded_url: str, new_requested_width: int, new_requested_height: int) -> str:
    """Resize a base64 PNG image to fit within max bounds while maintaining aspect ratio.

    Args:
        image_embedded_url: Full data URL (data:image/png;base64,...)
        new_requested_width: Max width in pixels
        new_requested_height: Max height in pixels

    Returns:
        Resized image as data URL, or original URL on failure
    """
    try:
        header, b64_data = image_embedded_url.split(",", 1)
        image_bytes = base64.b64decode(b64_data)
        new_image_bytes = resize_image(image_bytes, new_requested_width, new_requested_height, result_format="PNG")
        new_b64 = base64.b64encode(new_image_bytes).decode("ascii")
        return f"{header},{new_b64}"
    except Exception as exc:
        logger.exception(f"_resize_image | error resizing image: {exc}")
        return image_embedded_url


def _save_image_to_session_file(session: Session, image_embedded_url: str):
    header, b64_data = image_embedded_url.split(",", 1)
    image_bytes = base64.b64decode(b64_data)

    image_name = str(uuid.uuid4()) + ".png"

    af = AttachedFile.objects.create(
        session=session,
        file_type=AttachedFile.FILE_TYPE_OUTPUT,
        name=image_name
    )
    set_binary_content(af, image_bytes, offload_immediately=True)
    logger.info(f"Stored image under name {image_name} for session {session.id}")
    return image_name


def _generate_image_inner(description: str, width: int = 800, height: int = 450) -> str:
    if not description or not str(description).strip():
        raise ValueError("Error: description is required")

    if not isinstance(width, int):
        width = 800

    if not isinstance(height, int):
        height = 450

    requested_aspect_ratio = calculate_aspect_ratio(width, height)
    logger.info(
        f"generate_image | requested aspect ratio: {requested_aspect_ratio} for width={width} and height={height}")

    try:
        image_url = async_to_sync(invoke_llm)(
            'generate_image',
            system_message_template_name='image_tool.system_message',
            user_message_template_name='image_tool.prompt_template',
            template_params={
                "description": str(description),
                "width": f'{width}px',
                "height": f'{height}px'
            },
            render_output_image=True,
            image_config={
                "aspect_ratio": requested_aspect_ratio
            }
        )
        if not image_url:
            raise ValueError("Error: image generation returned empty result")

        image_url = _resize_image(image_url, width, height)
        return image_url
    except Exception as exc:
        logger.exception("generate_image | error generating image: %s", exc)
        raise ValueError(f"Error: {repr(exc)}")


@tool
def generate_image(description: str, width: int = 800, height: int = 450) -> str:
    """Generate an image based on a text description.

    Args:
        description: Str. A detailed description of the image you want to generate,
            including style, colors, subjects, mood, and any specific elements.
        width: Int. Display width in pixels (default 800).
        height: Int. Display height in pixels (default 450).

    Returns:
        An image encoded in URL if generation is successful,
        or an error message if generation failed.
    """
    return _generate_image_inner(description, width, height)


def create_generate_image_tool_for_session(session: Session):
    # noinspection PyShadowingNames
    @tool
    def generate_image(description: str, width: int = 800, height: int = 450) -> str:
        """Generate an image based on a text description.

        Args:
            description: Str. A detailed description of the image you want to generate,
                including style, colors, subjects, mood, and any specific elements.
            width: Int. Display width in pixels (default 800).
            height: Int. Display height in pixels (default 450).

        Returns:
            The image name (this image will be available in the site running context),
            or an error message if generation failed.
        """
        image_url = _generate_image_inner(description, width, height)
        return _save_image_to_session_file(session, image_url)

    return generate_image
