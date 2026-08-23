import logging
from io import BytesIO

from PIL import Image

logger = logging.getLogger(__name__)

_POSSIBLE_ASPECT_RATIOS: dict[str, float] = {
    "1:1": 1/1,
    "2:3": 2/3,
    "3:2": 3/2,
    "3:4": 3/4,
    "4:3": 4/3,
    "4:5": 4/5,
    "5:4": 5/4,
    "9:16": 9/16,
    "16:9": 16/9,
    "21:9": 21/9
}

def resize_image(image_bytes: bytes, new_requested_width: int, new_requested_height: int, result_format="PNG") -> bytes:
    """Resize a base64 PNG image to fit within max bounds while maintaining aspect ratio.

    Args:
        image_bytes: Full data bytes
        new_requested_width: Max width in pixels
        new_requested_height: Max height in pixels
        result_format: Output image format (e.g., "PNG", "JPEG")

    Returns:
        Resized image as data bytes
    """
    try:
        original_size = len(image_bytes)

        image = Image.open(BytesIO(image_bytes))
        orig_w, orig_h = image.size

        scale_w = new_requested_width / orig_w
        scale_h = new_requested_height / orig_h
        scale = min(scale_w, scale_h)

        if scale >= 1:
            logger.info(f"resize_image: Image resize skipped (already smaller than requested): {original_size}")
            return image_bytes

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        output = BytesIO()
        image.save(output, format=result_format, optimize=True)
        output.seek(0)
        output_bytes = output.read()
        new_size = len(output_bytes)
        logger.info(f"resize_image: Image compressed from {original_size} to {new_size}")
        return output_bytes
    except Exception as exc:
        logger.exception(f"resize_image: error resizing image: {exc}")
        raise


def calculate_aspect_ratio(width: int, height: int) -> str:
    """ Calculate the closest aspect ratio out of a supported list of possible values (_POSSIBLE_ASPECT_RATIOS) """
    real_aspect_ratio: float = width/height
    closest_match = ""
    closest_match_distance: float | None = None

    for match_str, match_ratio in _POSSIBLE_ASPECT_RATIOS.items():
        distance = abs(real_aspect_ratio - match_ratio)
        if not closest_match_distance or distance < closest_match_distance:
            closest_match_distance = distance
            closest_match = match_str

    if not closest_match:
        closest_match = "1:1"

    return closest_match