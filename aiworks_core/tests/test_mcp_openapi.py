"""
Unit tests for OpenAPI MCP server functionality:
- MCPServer and PredefinedMCPServer model fields (server_type, openapi_spec_url, openapi_spec)
- MCPServerSerializer validation for openapi server_type
- build_mcp_tools_from_servers() for both http and openapi server types
- deepagent_utils.prepare_tools_for_session() single-call integration
- Django admin enumerate/execute for predefined OpenAPI servers
"""

import json
from unittest.mock import MagicMock, patch

from django.test import TestCase
from . import _make_user
from ..logic.mcp_tools import (
    build_mcp_tools_from_servers,
    validate_server_connection,
    validate_server_connection_openapi,
)
from ..models import MCPServer, PredefinedMCPServer
from ..serializers import MCPServerSerializer


# ---------------------------------------------------------------------------
# Model field tests
# ---------------------------------------------------------------------------

class MCPServerModelOpenAPIFieldsTests(TestCase):
    def setUp(self):
        self.user = _make_user()

    def test_server_type_defaults_to_http(self):
        server = MCPServer.objects.create(
            user=self.user,
            name='Test Server',
            url='http://example.com/mcp',
        )
        self.assertEqual(server.server_type, 'http')

    def test_server_type_can_be_set_to_openapi(self):
        server = MCPServer.objects.create(
            user=self.user,
            name='Test OpenAPI Server',
            url='http://example.com/api',
            server_type='openapi',
            openapi_spec_url='https://example.com/openapi.json',
        )
        self.assertEqual(server.server_type, 'openapi')
        self.assertEqual(server.openapi_spec_url, 'https://example.com/openapi.json')

    def test_openapi_spec_can_be_set(self):
        spec_content = json.dumps({'openapi': '3.0.0', 'info': {'title': 'Test', 'version': '1.0'}, 'paths': {}})
        server = MCPServer.objects.create(
            user=self.user,
            name='Test OpenAPI Server',
            url='http://example.com/api',
            server_type='openapi',
            openapi_spec=spec_content,
        )
        self.assertEqual(server.server_type, 'openapi')
        self.assertEqual(server.openapi_spec, spec_content)

    def test_openapi_spec_url_blank_when_not_set(self):
        server = MCPServer.objects.create(
            user=self.user,
            name='Test Server',
            url='http://example.com/mcp',
            server_type='http',
        )
        self.assertEqual(server.openapi_spec_url, '')
        self.assertEqual(server.openapi_spec, '')


class PredefinedMCPServerOpenAPIFieldsTests(TestCase):
    def test_server_type_defaults_to_http(self):
        predefined = PredefinedMCPServer.objects.create(
            name='Test Predefined',
            url='http://example.com/mcp',
        )
        self.assertEqual(predefined.server_type, 'http')

    def test_server_type_can_be_set_to_openapi(self):
        predefined = PredefinedMCPServer.objects.create(
            name='Test OpenAPI Predefined',
            url='http://example.com/api',
            server_type='openapi',
            openapi_spec_url='https://example.com/openapi.json',
        )
        self.assertEqual(predefined.server_type, 'openapi')
        self.assertEqual(predefined.openapi_spec_url, 'https://example.com/openapi.json')


# ---------------------------------------------------------------------------
# Serializer validation tests
# ---------------------------------------------------------------------------

class MCPServerSerializerOpenAPIValidationTests(TestCase):
    def setUp(self):
        self.user = _make_user()

    def test_openapi_requires_spec_url_or_spec(self):
        data = {
            'name': 'OpenAPI Server',
            'url': 'http://example.com/api',
            'server_type': 'openapi',
        }
        serializer = MCPServerSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('openapi_spec_url', serializer.errors)

    def test_openapi_accepts_spec_url(self):
        data = {
            'name': 'OpenAPI Server',
            'url': 'http://example.com/api',
            'server_type': 'openapi',
            'openapi_spec_url': 'https://example.com/openapi.json',
        }
        serializer = MCPServerSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_openapi_accepts_spec_content(self):
        data = {
            'name': 'OpenAPI Server',
            'url': 'http://example.com/api',
            'server_type': 'openapi',
            'openapi_spec': json.dumps({'openapi': '3.0.0', 'info': {'title': 'Test', 'version': '1.0'}, 'paths': {}}),
        }
        serializer = MCPServerSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_http_does_not_require_spec(self):
        data = {
            'name': 'HTTP Server',
            'url': 'http://example.com/mcp',
            'server_type': 'http',
        }
        serializer = MCPServerSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_server_type_in_serialized_output(self):
        server = MCPServer.objects.create(
            user=self.user,
            name='Test Server',
            url='http://example.com/mcp',
            server_type='http',
        )
        serializer = MCPServerSerializer(server)
        self.assertEqual(serializer.data['server_type'], 'http')

    def test_openapi_spec_fields_in_serialized_output(self):
        server = MCPServer.objects.create(
            user=self.user,
            name='Test OpenAPI Server',
            url='http://example.com/api',
            server_type='openapi',
            openapi_spec_url='https://example.com/openapi.json',
            openapi_spec='',
        )
        serializer = MCPServerSerializer(server)
        self.assertEqual(serializer.data['server_type'], 'openapi')
        self.assertEqual(serializer.data['openapi_spec_url'], 'https://example.com/openapi.json')


# ---------------------------------------------------------------------------
# build_mcp_tools_from_servers tests
# ---------------------------------------------------------------------------

class BuildMCPToolsFromServersHTTPTests(TestCase):
    def setUp(self):
        self.user = _make_user()

    @patch('aiworks_core.logic.mcp_tools._load_mcp_tools_async')
    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools._ensure_connection')
    def test_http_server_uses_multiserver_client(self, mock_ensure, mock_ssrf, mock_load_tools):
        mock_ensure.return_value = ('http://example.com/mcp', {})
        mock_tool = MagicMock()
        mock_tool.name = 'test_tool'
        mock_load_tools.return_value = [mock_tool]

        server = MCPServer.objects.create(
            user=self.user,
            name='Test MCP',
            url='http://example.com/mcp',
            auth_type='none',
            server_type='http',
        )

        tools = build_mcp_tools_from_servers([server])

        mock_ensure.assert_called_once_with(server, validate_server_connection)
        mock_load_tools.assert_called_once_with('http://example.com/mcp', {})
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, 'test_tool')

    @patch('aiworks_core.logic.mcp_tools._load_mcp_tools_async')
    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools._ensure_connection')
    def test_http_server_with_auth_headers(self, mock_ensure, mock_ssrf, mock_load_tools):
        mock_ensure.return_value = ('http://example.com/mcp', {'Authorization': 'Bearer my-token'})
        mock_load_tools.return_value = []

        server = MCPServer.objects.create(
            user=self.user,
            name='Test MCP',
            url='http://example.com/mcp',
            auth_type='bearer',
            token='my-token',
            server_type='http',
        )

        build_mcp_tools_from_servers([server])

        mock_load_tools.assert_called_once_with('http://example.com/mcp', {'Authorization': 'Bearer my-token'})


class BuildMCPToolsFromServersOpenAPITests(TestCase):
    def setUp(self):
        self.user = _make_user()

    @patch('aiworks_core.logic.mcp_tools.validate_rest_api_connection')
    @patch('aiworks_core.logic.mcp_tools._load_openapi_tools_async')
    @patch('aiworks_core.logic.mcp_tools._load_openapi_spec')
    def test_openapi_server_uses_fastmcp(self, mock_load_spec, mock_load_openapi, mock_validate):
        mock_load_spec.return_value = {'openapi': '3.0.0', 'info': {'title': 'Test'}, 'paths': {}}
        mock_tool = MagicMock()
        mock_tool.name = 'get_users'
        mock_load_openapi.return_value = [mock_tool]

        server = MCPServer.objects.create(
            user=self.user,
            name='Test OpenAPI',
            url='http://example.com/api',
            server_type='openapi',
            openapi_spec_url='https://example.com/openapi.json',
            auth_type='none',
        )

        tools = build_mcp_tools_from_servers([server])

        mock_load_openapi.assert_called_once()
        spec_arg = mock_load_openapi.call_args[0][0]
        self.assertEqual(spec_arg['info']['title'], 'Test')
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, 'get_users')

    @patch('aiworks_core.logic.mcp_tools.validate_rest_api_connection')
    @patch('aiworks_core.logic.mcp_tools._load_openapi_tools_async')
    def test_openapi_server_with_inline_spec(self, mock_load_openapi, mock_validate):
        mock_load_openapi.return_value = []

        spec_content = json.dumps({
            'openapi': '3.0.0',
            'info': {'title': 'Test API', 'version': '1.0'},
            'paths': {'/users': {'get': {'operationId': 'get_users'}}}
        })
        server = MCPServer.objects.create(
            user=self.user,
            name='Test OpenAPI',
            url='http://example.com/api',
            server_type='openapi',
            openapi_spec=spec_content,
            auth_type='none',
        )

        build_mcp_tools_from_servers([server])

        spec_arg = mock_load_openapi.call_args[0][0]
        self.assertEqual(spec_arg['info']['title'], 'Test API')


class BuildMCPToolsFromServersErrorIsolationTests(TestCase):
    def setUp(self):
        self.user = _make_user()

    @patch('aiworks_core.logic.mcp_tools._load_mcp_tools_async')
    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools._ensure_connection')
    def test_one_failing_server_does_not_break_others(self, mock_ensure, mock_ssrf, mock_load_tools):
        mock_ensure.side_effect = [
            Exception("Connection refused"),
            ('http://ok.com/mcp', {}),
        ]
        working_tool = MagicMock()
        working_tool.name = 'working_tool'
        mock_load_tools.return_value = [working_tool]

        server1 = MCPServer.objects.create(
            user=self.user, name='Failing', url='http://fail.com/mcp',
            auth_type='none', server_type='http',
        )
        server2 = MCPServer.objects.create(
            user=self.user, name='Working', url='http://ok.com/mcp',
            auth_type='none', server_type='http',
        )

        tools = build_mcp_tools_from_servers([server1, server2])

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, 'working_tool')


class BuildMCPToolsFromServersMixedTypesTests(TestCase):
    def setUp(self):
        self.user = _make_user()

    @patch('aiworks_core.logic.mcp_tools._load_openapi_tools_async')
    @patch('aiworks_core.logic.mcp_tools._load_openapi_spec')
    @patch('aiworks_core.logic.mcp_tools._load_mcp_tools_async')
    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools._ensure_connection')
    def test_mixed_http_and_openapi_servers(self, mock_ensure, mock_ssrf, mock_load_tools, mock_load_spec, mock_load_openapi):
        mock_ensure.side_effect = [
            ('http://http.example.com/mcp', {}),
            ('http://api.example.com', {}),
        ]
        http_tool = MagicMock()
        http_tool.name = 'http_tool'
        mock_load_tools.return_value = [http_tool]
        mock_load_spec.return_value = {'openapi': '3.0.0', 'info': {'title': 'Test'}, 'paths': {}}
        openapi_tool = MagicMock()
        openapi_tool.name = 'openapi_tool'
        mock_load_openapi.return_value = [openapi_tool]

        http_server = MCPServer.objects.create(
            user=self.user, name='HTTP Server', url='http://http.example.com/mcp',
            auth_type='none', server_type='http',
        )
        openapi_server = MCPServer.objects.create(
            user=self.user, name='OpenAPI Server', url='http://api.example.com',
            auth_type='none', server_type='openapi',
            openapi_spec_url='https://api.example.com/openapi.json',
        )

        tools = build_mcp_tools_from_servers([http_server, openapi_server])

        self.assertEqual(len(tools), 2)
        tool_names = {t.name for t in tools}
        self.assertIn('http_tool', tool_names)
        self.assertIn('openapi_tool', tool_names)
        mock_ensure.assert_any_call(http_server, validate_server_connection)
        mock_ensure.assert_any_call(openapi_server, validate_server_connection_openapi)


# ---------------------------------------------------------------------------
# deepagent_utils single-call test
# ---------------------------------------------------------------------------

class DeepAgentUtilsMCPSingleCallTests(TestCase):
    def setUp(self):
        self.user = _make_user()

    @patch('aiworks_core.logic.deepagent_utils.build_mcp_tools_from_servers')
    def test_prepare_tools_calls_build_mcp_tools_from_servers(self, mock_build):
        mock_build.return_value = []

        from ..logic.deepagent_utils import prepare_tools_for_session
        from ..models import Session

        session = Session.objects.create(
            id='test-mcp-single-call-session',
            user=self.user,
            session_type='test',
            session_title='Test Session',
        )
        server = MCPServer.objects.create(
            user=self.user,
            name='Test MCP',
            url='http://example.com/mcp',
            server_type='http',
        )
        session.mcp_servers.add(server)

        prepare_tools_for_session(session)

        mock_build.assert_called_once()
        self.assertEqual(len(mock_build.call_args[0][0]), 1)


# ---------------------------------------------------------------------------
# Django admin tests for PredefinedMCPServer
# ---------------------------------------------------------------------------

class PredefinedMCPServerAdminTests(TestCase):
    def setUp(self):
        from django.contrib.admin.sites import AdminSite
        from ..admin import PredefinedMCPServerAdmin
        from ..models import User
        self.site = AdminSite()
        self.admin = PredefinedMCPServerAdmin(PredefinedMCPServer, self.site)
        self.superuser = User.objects.create_superuser('admin', 'admin@test.com', 'password')

    def test_list_display_includes_server_type(self):
        self.assertIn('server_type', self.admin.list_display)

    def test_list_filter_includes_server_type(self):
        self.assertIn('server_type', self.admin.list_filter)


# ---------------------------------------------------------------------------
# build_mcp_tools_from_servers — OpenAPI OAuth tests
# ---------------------------------------------------------------------------

class BuildMCPToolsFromServersOpenAPIOAuthTests(TestCase):
    def setUp(self):
        self.user = _make_user()

    @patch('aiworks_core.logic.mcp_tools._load_openapi_tools_async')
    @patch('aiworks_core.logic.mcp_tools._load_openapi_spec')
    @patch('aiworks_core.logic.mcp_tools._ensure_connection')
    def test_openapi_calls_ensure_connection_with_openapi_validator(
        self, mock_ensure, mock_load_spec, mock_load_openapi
    ):
        mock_ensure.return_value = ('https://api.example.com', {'Authorization': 'Bearer fresh'})
        mock_load_spec.return_value = {'openapi': '3.0.0', 'info': {'title': 'Test'}, 'paths': {}}
        mock_tool = MagicMock()
        mock_tool.name = 'get_users'
        mock_load_openapi.return_value = [mock_tool]

        server = MCPServer.objects.create(
            user=self.user,
            name='OAuth OpenAPI Server',
            url='https://api.example.com',
            auth_type='oauth',
            token='stale',
            refresh_token='refresh123',
            server_type='openapi',
            openapi_spec='{"openapi":"3.0","paths":{}}',
        )

        tools = build_mcp_tools_from_servers([server])

        mock_ensure.assert_called_once_with(server, validate_server_connection_openapi)
        mock_load_openapi.assert_called_once()
        self.assertEqual(tools[0].name, 'get_users')