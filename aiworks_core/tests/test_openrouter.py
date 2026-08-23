"""Tests for the ChatOpenRouterExtended class."""

from aiworks_core.logic.openrouter import ChatOpenRouterExtended, parse_openrouter_model
from django.test import TestCase


class ChatOpenRouterExtendedTests(TestCase):
    """Unit tests for ChatOpenRouterExtended._create_chat_result."""

    def test_images_extracted_from_response(self):
        """When response contains images, they appear in message.additional_kwargs."""
        llm = ChatOpenRouterExtended(model="test", api_key="test")

        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Here is your image",
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,ABC123"
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "model": "google/gemini-2.5-flash-image",
        }

        result = llm._create_chat_result(mock_response)

        self.assertEqual(len(result.generations), 1)
        msg = result.generations[0].message
        self.assertTrue(hasattr(msg, "additional_kwargs"))
        self.assertIn("images", msg.additional_kwargs)
        self.assertEqual(
            msg.additional_kwargs["images"][0]["image_url"]["url"],
            "data:image/png;base64,ABC123",
        )

    def test_no_images_when_not_in_response(self):
        """When response has no images, additional_kwargs does not contain images."""
        llm = ChatOpenRouterExtended(model="test", api_key="test")

        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Just text response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }

        result = llm._create_chat_result(mock_response)

        msg = result.generations[0].message
        self.assertFalse(hasattr(msg, "additional_kwargs") and "images" in msg.additional_kwargs)

    def test_multiple_images(self):
        """When response contains multiple images, all are extracted."""
        llm = ChatOpenRouterExtended(model="test", api_key="test")

        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Two images",
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,IMG1"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,IMG2"},
                            },
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "model": "test-model",
        }

        result = llm._create_chat_result(mock_response)

        msg = result.generations[0].message
        self.assertEqual(len(msg.additional_kwargs["images"]), 2)
        self.assertEqual(msg.additional_kwargs["images"][0]["image_url"]["url"], "data:image/png;base64,IMG1")
        self.assertEqual(msg.additional_kwargs["images"][1]["image_url"]["url"], "data:image/png;base64,IMG2")

class ParseOpenRouterModelTests(TestCase):
    """Tests for _parse_openrouter_model helper."""

    def test_parses_provider_suffix(self):
        clean, providers, allow_fallback = parse_openrouter_model("minimax_27<MYPROVIDER>")
        self.assertEqual(clean, "minimax_27")
        self.assertEqual(providers, ["MYPROVIDER"])
        self.assertFalse(allow_fallback)

    def test_parses_provider_suffix_with_fallback(self):
        clean, providers, allow_fallback = parse_openrouter_model("minimax_27<MYPROVIDER+>")
        self.assertEqual(clean, "minimax_27")
        self.assertEqual(providers, ["MYPROVIDER"])
        self.assertTrue(allow_fallback)

    def test_parses_provider_suffix_multiple(self):
        clean, providers, allow_fallback = parse_openrouter_model("minimax_27<MYPROVIDER1,MYPROVIDER2>")
        self.assertEqual(clean, "minimax_27")
        self.assertEqual(providers, ["MYPROVIDER1","MYPROVIDER2"])
        self.assertFalse(allow_fallback)

    def test_no_suffix_returns_model_unchanged(self):
        clean, providers, allow_fallback = parse_openrouter_model("openai/gpt-4o")
        self.assertEqual(clean, "openai/gpt-4o")
        self.assertIsNone(providers)
        self.assertFalse(allow_fallback)

    def test_no_suffix_with_angle_brackets_not_at_end(self):
        clean, providers, allow_fallback = parse_openrouter_model("minimax_27<PROVIDER>extra")
        self.assertEqual(clean, "minimax_27<PROVIDER>extra")
        self.assertIsNone(providers)
        self.assertFalse(allow_fallback)

    def test_malformed_angle_bracket_no_close(self):
        clean, providers, allow_fallback = parse_openrouter_model("minimax_27<PROVIDER")
        self.assertEqual(clean, "minimax_27<PROVIDER")
        self.assertIsNone(providers)
        self.assertFalse(allow_fallback)


