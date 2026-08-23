"""
Unit tests for deepagent_utils — prepare_tools_for_session and LLMConfiguration flag behaviour.
"""

from unittest.mock import MagicMock, patch

from django.test import TestCase

from . import _make_session, _make_user
from ..logic.deepagent_utils import prepare_tools_for_session


class MockLLMConfig:
    def __init__(self, enable_subagents=True, enable_playwright=True):
        self.enable_subagents = enable_subagents
        self.enable_playwright = enable_playwright


class PrepareToolsForSessionLLMConfigFlagsTests(TestCase):
    """Tests for enable_subagents and enable_playwright LLMConfiguration flags."""

    def setUp(self):
        self.user = _make_user(tier="pro")

    def _call_prepare_tools(self, session, cfg_flags=None):
        cfg = MockLLMConfig(**cfg_flags) if cfg_flags else MockLLMConfig()
        with patch("aiworks_core.logic.deepagent_utils.LLMConfiguration.get_solo", return_value=cfg):
            with patch("aiworks_core.logic.deepagent_utils.preprocess_kb_files", return_value=([], None)):
                with patch("aiworks_core.logic.deepagent_utils.build_mcp_tools_from_servers", return_value=[]):
                    with patch("aiworks_core.logic.deepagent_utils.build_tunnel_tools", return_value=[]):
                        with patch("aiworks_core.logic.deepagent_utils.get_deepagent_validation_tools",
                                   return_value=[]):
                            return prepare_tools_for_session(
                                session, update_task_callback=lambda **kw: None
                            )

    def test_spawn_subagent_included_when_enable_subagents_true(self):
        session = _make_session(self.user, session_type="test")
        tools, err = self._call_prepare_tools(session, {"enable_subagents": True})
        self.assertIsNone(err)
        tool_names = [t.name for t in tools]
        self.assertIn("spawn_subagent", tool_names)

    def test_spawn_subagent_excluded_when_enable_subagents_false(self):
        session = _make_session(self.user, session_type="test")
        tools, err = self._call_prepare_tools(session, {"enable_subagents": False})
        self.assertIsNone(err)
        tool_names = [t.name for t in tools]
        self.assertNotIn("spawn_subagent", tool_names)

    def test_playwright_tools_included_when_enable_playwright_true_and_pro_user(self):
        session = _make_session(
            self.user,
            session_type="test",
            is_research_needed=True,
            is_research_online=True,
        )
        mock_playwright = MagicMock(name="playwright_tool")
        with patch("aiworks_core.logic.deepagent_utils.get_web_tools", return_value=[]):
            with patch("aiworks_core.logic.deepagent_utils.get_playwright_tools", return_value=[mock_playwright]):
                tools, err = self._call_prepare_tools(
                    session, {"enable_playwright": True}
                )
        self.assertIsNone(err)
        self.assertIn(mock_playwright, tools)

    def test_playwright_tools_excluded_when_enable_playwright_false(self):
        session = _make_session(
            self.user,
            session_type="test",
            is_research_needed=True,
            is_research_online=True,
        )
        with patch("aiworks_core.logic.deepagent_utils.get_web_tools", return_value=[]):
            with patch("aiworks_core.logic.deepagent_utils.get_playwright_tools",
                       return_value=[MagicMock(name="playwright_tool")]):
                tools, err = self._call_prepare_tools(
                    session, {"enable_playwright": False}
                )
        self.assertIsNone(err)
        tool_names = [t.name for t in tools]
        self.assertNotIn("playwright_tool", tool_names)

    def test_playwright_tools_excluded_when_user_not_pro(self):
        non_pro_user = _make_user(username="freeuser", email="free@example.com", tier="free")
        session = _make_session(
            non_pro_user,
            session_type="test",
            is_research_needed=True,
            is_research_online=True,
        )
        mock_playwright = MagicMock(name="playwright_tool")
        with patch("aiworks_core.logic.deepagent_utils.get_web_tools", return_value=[]):
            with patch("aiworks_core.logic.deepagent_utils.get_playwright_tools", return_value=[mock_playwright]):
                tools, err = self._call_prepare_tools(
                    session, {"enable_playwright": True}
                )
        self.assertIsNone(err)
        self.assertNotIn(mock_playwright, tools)

    def test_web_tools_still_added_when_playwright_disabled(self):
        session = _make_session(
            self.user,
            session_type="test",
            is_research_needed=True,
            is_research_online=True,
        )
        mock_web = MagicMock(name="web_tool")
        with patch("aiworks_core.logic.deepagent_utils.get_web_tools", return_value=[mock_web]):
            with patch("aiworks_core.logic.deepagent_utils.get_playwright_tools",
                       return_value=[MagicMock(name="playwright_tool")]):
                tools, err = self._call_prepare_tools(
                    session, {"enable_playwright": False}
                )
        self.assertIsNone(err)
        self.assertIn(mock_web, tools)

    def test_default_flags_both_true(self):
        session = _make_session(
            self.user,
            session_type="test",
            is_research_needed=True,
            is_research_online=True,
        )
        mock_web = MagicMock(name="web_tool")
        mock_playwright = MagicMock(name="playwright_tool")
        with patch("aiworks_core.logic.deepagent_utils.get_web_tools", return_value=[mock_web]):
            with patch("aiworks_core.logic.deepagent_utils.get_playwright_tools", return_value=[mock_playwright]):
                tools, err = self._call_prepare_tools(session)
        self.assertIsNone(err)
        tool_names = [t.name for t in tools]
        self.assertIn("spawn_subagent", tool_names)
        self.assertIn(mock_playwright, tools)
