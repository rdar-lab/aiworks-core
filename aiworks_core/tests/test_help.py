"""
Help chat views tests for the Ai-Works Core API.
"""

from unittest.mock import patch

from rest_framework import status
from rest_framework.test import APITestCase

from . import (
    _make_user,
    _auth_header,
)


class HelpChatViewTests(APITestCase):
    def setUp(self):
        self.user = _make_user(username="helpchat_user")
        self.auth = _auth_header(self.user)
        self.url = "/api/help-chat/"

    def test_unauthenticated_returns_401(self):
        response = self.client.post(
            self.url, {"messages": [{"role": "user", "content": "Hi"}]}, format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_missing_messages_returns_400(self):
        response = self.client.post(self.url, {}, format="json", **self.auth)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_empty_messages_list_returns_400(self):
        response = self.client.post(
            self.url, {"messages": []}, format="json", **self.auth
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_non_list_messages_returns_400(self):
        response = self.client.post(
            self.url, {"messages": "not a list"}, format="json", **self.auth
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch("aiworks_core.logic.help.invoke_llm")
    def test_successful_request_returns_reply(self, mock_invoke_llm):
        mock_invoke_llm.return_value = "You can create a session from the dashboard."
        payload = {
            "messages": [{"role": "user", "content": "How do I start?"}],
            "manual_content": "# User Manual\nUse the dashboard",
        }
        response = self.client.post(self.url, payload, format="json", **self.auth)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data["reply"], "You can create a session from the dashboard."
        )

    def test_missing_manual_content_returns_400(self):
        payload = {"messages": [{"role": "user", "content": "Help?"}]}
        response = self.client.post(self.url, payload, format="json", **self.auth)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_empty_manual_content_returns_400(self):
        payload = {
            "messages": [{"role": "user", "content": "Help?"}],
            "manual_content": "",
        }
        response = self.client.post(self.url, payload, format="json", **self.auth)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    @patch("aiworks_core.logic.help.invoke_llm")
    def test_conversation_history_is_forwarded(self, mock_invoke_llm):
        mock_invoke_llm.return_value = "Follow-up reply"
        payload = {
            "messages": [
                {"role": "user", "content": "What is this app?"},
                {
                    "role": "assistant",
                    "content": "It is a virtual tool.",
                },
                {"role": "user", "content": "How do I invite members?"},
            ],
            "manual_content": "# Manual",
        }
        response = self.client.post(self.url, payload, format="json", **self.auth)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        args, kwargs = mock_invoke_llm.call_args
        lc_messages = kwargs.get("messages", args[0] if args else [])
        self.assertEqual(len(lc_messages), 4)

    @patch("aiworks_core.logic.help.invoke_llm", side_effect=RuntimeError("LLM unavailable"))
    def test_llm_error_returns_500(self, _mock):
        payload = {
            "messages": [{"role": "user", "content": "Help?"}],
            "manual_content": "# Manual",
        }
        response = self.client.post(self.url, payload, format="json", **self.auth)
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("error", response.data)
