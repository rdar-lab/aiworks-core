"""
Tests for the spawn_subagent deep agent tool in llm.py.

Verifies:
- spawn_subagent is a LangChain tool with correct name and signature
- spawn_subagent returns error string when backend not set
- spawn_subagent calls invoke_llm with is_sub_agent=True when backend is set
- spawn_subagent passes correct messages and parameters to invoke_llm
"""

from unittest.mock import MagicMock, patch
from django.test import TestCase

from ..logic.llm import spawn_subagent, init_agent_backend, init_active_tools


class SpawnSubagentToolTests(TestCase):
    """Tests for the spawn_subagent tool."""

    def setUp(self):
        init_agent_backend(None)
        init_active_tools(None)

    def tearDown(self):
        init_agent_backend(None)
        init_active_tools(None)

    def test_tool_is_langchain_structured_tool(self):
        """spawn_subagent is a LangChain tool with name and run method."""
        self.assertTrue(hasattr(spawn_subagent, "name"))
        self.assertEqual(spawn_subagent.name, "spawn_subagent")
        self.assertTrue(callable(spawn_subagent.run))

    def test_spawn_subagent_error_when_backend_not_set(self):
        """spawn_subagent returns error string when agent backend is None."""
        result = spawn_subagent.run({"custom_prompt": "Do work"})
        self.assertIn("Error", result)
        self.assertIn("Subagent context not available", result)

    @patch("aiworks_core.logic.llm.invoke_llm")
    def test_spawn_subagent_calls_invoke_llm_with_is_sub_agent(self, mock_invoke):
        """spawn_subagent calls invoke_llm with is_sub_agent=True."""
        mock_invoke.return_value = "subagent response"
        parent_backend = MagicMock()
        init_agent_backend(parent_backend)

        result = spawn_subagent.run({"custom_prompt": "You are a researcher"})

        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args.kwargs
        self.assertEqual(call_kwargs["is_agent"], True)
        self.assertEqual(call_kwargs["is_deep_agent"], True)
        self.assertEqual(call_kwargs["is_sub_agent"], True)
        self.assertEqual(call_kwargs["parse_json"], False)
        from langchain_core.messages import SystemMessage
        self.assertIsInstance(call_kwargs["messages"][0], SystemMessage)
        self.assertEqual(call_kwargs["messages"][0].content, "You are a researcher")
        self.assertEqual(result, "subagent response")