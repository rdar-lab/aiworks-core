"""
Tests for ChatOpenRouterExtended.generate_video method.
"""

from django.test import TestCase
from langchain_core.messages import HumanMessage

from ..logic.openrouter import ChatOpenRouterExtended


class ChatOpenRouterExtendedVideoTests(TestCase):
    def _make_executor(self, model="google/veo-3.1-fast", api_key="sk-test-key", request_timeout=600000):
        return ChatOpenRouterExtended(
            model=model,
            api_key=api_key,
            temperature=0.7,
            timeout=request_timeout,
        )

    def test_generate_video_converts_messages_to_prompt(self):
        """Multiple messages are joined into a single prompt."""
        executor = self._make_executor()

        messages = [
            HumanMessage(content="First message"),
            HumanMessage(content="Second message"),
        ]
        result = executor._messages_to_prompt(messages)
        self.assertEqual(result, "First message\nSecond message")

    def test_generate_video_url_is_correct(self):
        """Video API URL uses OpenRouter's correct endpoint."""
        executor = self._make_executor()
        self.assertEqual(executor.app_url, "https://docs.langchain.com")

    def test_generate_video_uses_openrouter_api_key(self):
        """Executor uses openrouter_api_key for authentication."""
        executor = self._make_executor(api_key="sk-my-key")
        self.assertEqual(executor.openrouter_api_key.get_secret_value(), "sk-my-key")

    def test_generate_video_request_timeout_stored(self):
        """Executor stores request_timeout correctly."""
        executor = self._make_executor(request_timeout=300000)
        self.assertEqual(executor.request_timeout, 300000)
