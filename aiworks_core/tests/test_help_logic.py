"""
Help chat logic tests for the Ai-Works Core API.
"""

from unittest.mock import patch

from django.test import TestCase

from ..logic.help import help_chat_reply


class HelpChatReplyTests(TestCase):
    MANUAL = "# App Manual\nUse the dashboard."

    @patch("aiworks_core.logic.help.invoke_llm")
    def test_returns_llm_reply(self, mock_invoke_llm):
        mock_invoke_llm.return_value = "Go to the dashboard."
        result = help_chat_reply(
            [{"role": "user", "content": "How do I start?"}], self.MANUAL
        )
        self.assertEqual(result, "Go to the dashboard.")

    @patch("aiworks_core.logic.help.invoke_llm")
    def test_system_message_contains_manual_content(self, mock_invoke_llm):
        mock_invoke_llm.return_value = "Sure!"
        help_chat_reply([{"role": "user", "content": "Help?"}], self.MANUAL)
        args, kwargs = mock_invoke_llm.call_args
        lc_messages = kwargs.get("messages", args[0] if args else [])
        system_msg = lc_messages[0]
        self.assertIn(self.MANUAL, system_msg.content)

    @patch("aiworks_core.logic.help.invoke_llm")
    def test_uses_fast_llm_type(self, mock_invoke_llm):
        mock_invoke_llm.return_value = "reply"
        help_chat_reply([{"role": "user", "content": "Hi"}], self.MANUAL)
        args, kwargs = mock_invoke_llm.call_args
        self.assertEqual(args[0], "help_chat")

    @patch("aiworks_core.logic.help.invoke_llm")
    def test_user_and_assistant_messages_are_mapped(self, mock_invoke_llm):
        mock_invoke_llm.return_value = "reply"
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        help_chat_reply(messages, self.MANUAL)
        args, kwargs = mock_invoke_llm.call_args
        lc_messages = kwargs.get("messages", args[0] if args else [])
        self.assertEqual(len(lc_messages), 4)

    @patch("aiworks_core.logic.help.invoke_llm")
    def test_unknown_role_messages_are_skipped(self, mock_invoke_llm):
        mock_invoke_llm.return_value = "reply"
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "system", "content": "ignored"},
        ]
        help_chat_reply(messages, self.MANUAL)
        args, kwargs = mock_invoke_llm.call_args
        lc_messages = kwargs.get("messages", args[0] if args else [])
        self.assertEqual(len(lc_messages), 2)
