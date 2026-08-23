"""
Tests for the generate_image_tool module.
"""

import base64
from io import BytesIO
from unittest.mock import patch, AsyncMock

from django.test import TestCase
from PIL import Image

from . import _make_user, _make_session
from ..logic.generate_image_tool import generate_image, _resize_image, create_generate_image_tool_for_session
from ..models import AttachedFile


def _make_test_png(width: int, height: int, color: tuple = (255, 0, 0)) -> str:
    """Generate a solid-color PNG as base64 data URL for testing."""
    img = Image.new("RGB", (width, height), color)
    output = BytesIO()
    img.save(output, format="PNG")
    b64 = base64.b64encode(output.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


class GenerateImageToolTests(TestCase):
    def setUp(self):
        self.user = _make_user(tier="pro")


    def test_generate_image_empty_description_returns_error(self):
        """generate_image returns error when description is empty string."""
        with self.assertRaises(Exception) as err_context:
            generate_image.run({"description": ""})
        self.assertIn("required", repr(err_context.exception))

    def test_generate_image_missing_description_raises_validation_error(self):
        """generate_image raises ValidationError when description is missing from args."""
        with self.assertRaises(Exception):
            generate_image.run({})

    def test_generate_image_returns_direct_url(self):
        """generate_image calls invoke_llm with correct parameters."""
        mock_image_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        with patch("aiworks_core.logic.generate_image_tool.invoke_llm", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_image_url
            result = generate_image.run({
                "description": "A beautiful sunset over mountains",
            })

        self.assertEqual(result, mock_image_url)
        mock_invoke.assert_called_once()
        call_args = mock_invoke.call_args
        self.assertEqual(
            call_args.kwargs["system_message_template_name"],
            "image_tool.system_message",
        )
        self.assertEqual(
            call_args.kwargs["user_message_template_name"],
            "image_tool.prompt_template",
        )
        self.assertEqual(call_args[0][0], 'generate_image')
        self.assertEqual(call_args.kwargs['render_output_image'], True)
        self.assertIn("sunset", call_args.kwargs['template_params']['description'])

    def test_generate_image_exception_raises(self):
        """generate_image returns error string on exception."""
        with patch("aiworks_core.logic.generate_image_tool.invoke_llm", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = RuntimeError("LLM failed")
            with self.assertRaises(Exception) as err_context:
                generate_image.run({
                    "description": "Test image"
                })
            self.assertIn("LLM failed", repr(err_context.exception))


class ResizeImageTests(TestCase):
    def setUp(self):
        self.user = _make_user(tier="pro")

    def test_resize_image_actually_resizes(self):
        """_resize_image resizes a large image to fit within max bounds."""
        original = _make_test_png(1920, 800, (255, 0, 0))
        original_size = len(original.encode("ascii"))

        result = _resize_image(original, 800, 450)
        result_size = len(result.encode("ascii"))

        self.assertLess(result_size, original_size)

        header, b64_data = result.split(",", 1)
        image_bytes = base64.b64decode(b64_data)
        image = Image.open(BytesIO(image_bytes))
        w, h = image.size

        self.assertLessEqual(w, 800)
        self.assertLessEqual(h, 450)

        aspect_ratio = 1920 / 800
        expected_h = 800 / aspect_ratio
        self.assertAlmostEqual(h, expected_h, delta=5)

    def test_resize_image_no_upscale(self):
        """_resize_image returns original URL when image is already smaller than bounds."""
        original = _make_test_png(400, 300, (0, 255, 0))
        original_encoded = original.encode("ascii")

        result = _resize_image(original, 800, 450)

        self.assertEqual(result, original)
        self.assertEqual(result.encode("ascii"), original_encoded)

    def test_resize_image_corrupt_returns_original(self):
        """_resize_image returns original URL on decode failure."""
        original = "data:image/png;base64,INVALID_BASE64_!!!"

        with patch("aiworks_core.logic.generate_image_tool.logger") as mock_logger:
            result = _resize_image(original, 800, 450)

        self.assertEqual(result, original)
        mock_logger.exception.assert_called_once()

    def test_resize_image_tall_image(self):
        """_resize_image handles tall images (height > width) correctly."""
        original = _make_test_png(800, 1920, (128, 128, 128))

        result = _resize_image(original, 800, 450)

        header, b64_data = result.split(",", 1)
        image_bytes = base64.b64decode(b64_data)
        image = Image.open(BytesIO(image_bytes))
        w, h = image.size

        self.assertLessEqual(w, 800)
        self.assertLessEqual(h, 450)

        scale_h = 450 / 1920
        expected_w = int(800 * scale_h)
        self.assertAlmostEqual(w, expected_w, delta=5)
        self.assertAlmostEqual(h, 450, delta=5)


class CreateGenerateImageToolForSessionTests(TestCase):
    def setUp(self):
        self.user = _make_user(tier="pro")
        self.session = _make_session(self.user, session_id="img_tool_session")

    def test_create_generate_image_tool_for_session_returns_tool(self):
        """create_generate_image_tool_for_session returns a LangChain tool."""
        tool_fn = create_generate_image_tool_for_session(self.session)
        self.assertTrue(hasattr(tool_fn, "invoke"))

    def test_session_tool_saves_image_and_returns_filename(self):
        """Session-bound tool calls invoke_llm, saves image as AttachedFile, returns filename."""
        mock_image_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        with patch("aiworks_core.logic.generate_image_tool.invoke_llm", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_image_url
            tool_fn = create_generate_image_tool_for_session(self.session)
            result = tool_fn.invoke({
                "description": "A test hero image",
                "width": 800,
                "height": 450,
            })

        self.assertTrue(result.endswith(".png"))
        mock_invoke.assert_called_once()
        call_args = mock_invoke.call_args
        self.assertEqual(call_args.kwargs["render_output_image"], True)

        af = AttachedFile.objects.get(session=self.session, name=result)
        self.assertEqual(af.file_type, AttachedFile.FILE_TYPE_OUTPUT)

    def test_session_tool_invalid_description_raises(self):
        """Session-bound tool raises when description is empty."""
        tool_fn = create_generate_image_tool_for_session(self.session)
        with self.assertRaises(Exception) as err_context:
            tool_fn.invoke({"description": ""})
        self.assertIn("required", repr(err_context.exception))