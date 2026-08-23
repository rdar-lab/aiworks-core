"""
Unit tests for MCP server functionality:
- mcp_tools.py: validate_mcp_connection, build_mcp_tools
- MCPServerSerializer
- MCPServerViewSet (CRUD, ownership, validation)
- OAuth discovery/exchange endpoints (mcp_views)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from ..utils import async_to_sync
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APITestCase
from . import _make_user, _auth_header
from ..logic.logic_utils import safe_get_with_ssrf_check
from ..logic.mcp_tools import (
    build_mcp_tools,
    validate_mcp_connection,
    validate_ssrf_safe_url,
    RefreshableTokenAuth,
    resolve_auth_headers,
    _load_mcp_tools_async,
)
from ..logic.oauth import (
    discover_service_oauth,
    exchange_oauth_token,
    resolve_oauth_tokens,
    refresh_access_token,
)
from ..models import MCPServer, PredefinedMCPServer
from ..serializers import MCPServerSerializer, PredefinedMCPServerSerializer
from ..views.mcp_views import MCPServerViewSet


# ---------------------------------------------------------------------------
# validate_ssrf_safe_url tests
# ---------------------------------------------------------------------------

class ValidateSSRFSafeURLTests(TestCase):
    """Extensive tests for validate_ssrf_safe_url SSRF protection."""

    # ------------------------------------------------------------------
    # Scheme validation
    # ------------------------------------------------------------------

    @patch('socket.gethostbyname_ex', return_value=('example.com', [], ['93.184.216.34']))
    def test_http_scheme_is_allowed(self, _):
        validate_ssrf_safe_url('http://example.com/mcp')  # must not raise

    @patch('socket.gethostbyname_ex', return_value=('example.com', [], ['93.184.216.34']))
    def test_https_scheme_is_allowed(self, _):
        validate_ssrf_safe_url('https://example.com/mcp')  # must not raise

    def test_ftp_scheme_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('ftp://example.com/mcp')
        self.assertIn('ftp', str(ctx.exception))

    def test_file_scheme_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('file:///etc/passwd')

    def test_gopher_scheme_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('gopher://example.com/')

    def test_missing_scheme_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('example.com/mcp')

    # ------------------------------------------------------------------
    # Hostname validation
    # ------------------------------------------------------------------

    def test_url_without_hostname_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://')
        self.assertIn('hostname', str(ctx.exception).lower())

    @patch('socket.gethostbyname_ex', side_effect=__import__('socket').gaierror('NXDOMAIN'))
    def test_unresolvable_hostname_is_rejected(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://nonexistent.invalid/mcp')
        self.assertIn('resolve', str(ctx.exception).lower())

    # ------------------------------------------------------------------
    # Loopback addresses (127.0.0.0/8)
    # ------------------------------------------------------------------

    @patch('socket.gethostbyname_ex', return_value=('localhost', [], ['127.0.0.1']))
    def test_loopback_127_0_0_1_is_blocked(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://localhost/mcp')
        self.assertIn('127.0.0.1', str(ctx.exception))

    @patch('socket.gethostbyname_ex', return_value=('host', [], ['127.0.0.2']))
    def test_loopback_127_0_0_2_is_blocked(self, _):
        """Any 127.x address, not just .1, must be blocked (the entire /8 is loopback)."""
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('http://loopback2/mcp')

    @patch('socket.gethostbyname_ex', return_value=('host', [], ['127.255.255.255']))
    def test_loopback_127_255_255_255_is_blocked(self, _):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('http://loopback-host/mcp')

    # ------------------------------------------------------------------
    # RFC1918 private ranges
    # ------------------------------------------------------------------

    @patch('socket.gethostbyname_ex', return_value=('internal', [], ['10.0.0.1']))
    def test_rfc1918_class_a_10_x_is_blocked(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://internal.corp/mcp')
        self.assertIn('10.0.0.1', str(ctx.exception))

    @patch('socket.gethostbyname_ex', return_value=('internal', [], ['10.255.255.255']))
    def test_rfc1918_class_a_boundary_is_blocked(self, _):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('http://internal/mcp')

    @patch('socket.gethostbyname_ex', return_value=('internal', [], ['172.16.0.1']))
    def test_rfc1918_class_b_172_16_is_blocked(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://corp/mcp')
        self.assertIn('172.16.0.1', str(ctx.exception))

    @patch('socket.gethostbyname_ex', return_value=('internal', [], ['172.31.255.255']))
    def test_rfc1918_class_b_boundary_is_blocked(self, _):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('http://corp/mcp')

    @patch('socket.gethostbyname_ex', return_value=('internal', [], ['192.168.1.100']))
    def test_rfc1918_class_c_192_168_is_blocked(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://router/mcp')
        self.assertIn('192.168.1.100', str(ctx.exception))

    # ------------------------------------------------------------------
    # Link-local / cloud IMDS (169.254.0.0/16)
    # ------------------------------------------------------------------

    @patch('socket.gethostbyname_ex', return_value=('imds', [], ['169.254.169.254']))
    def test_aws_gcp_azure_imds_is_blocked(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://metadata.internal/mcp')
        self.assertIn('169.254.169.254', str(ctx.exception))

    @patch('socket.gethostbyname_ex', return_value=('link', [], ['169.254.0.1']))
    def test_link_local_169_254_0_1_is_blocked(self, _):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('http://link-local/mcp')

    # ------------------------------------------------------------------
    # Carrier-grade NAT (100.64.0.0/10, RFC6598)
    # ------------------------------------------------------------------

    @patch('socket.gethostbyname_ex', return_value=('cgnat', [], ['100.64.0.1']))
    def test_cgnat_100_64_is_blocked(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://cgnat/mcp')
        self.assertIn('100.64.0.1', str(ctx.exception))

    @patch('socket.gethostbyname_ex', return_value=('cgnat', [], ['100.127.255.255']))
    def test_cgnat_boundary_is_blocked(self, _):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('http://cgnat/mcp')

    # ------------------------------------------------------------------
    # Class E / reserved (240.0.0.0/4)
    # ------------------------------------------------------------------

    @patch('socket.gethostbyname_ex', return_value=('reserved', [], ['240.0.0.1']))
    def test_class_e_reserved_is_blocked(self, _):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('http://reserved/mcp')

    # ------------------------------------------------------------------
    # Documentation ranges (RFC5737)
    # ------------------------------------------------------------------

    @patch('socket.gethostbyname_ex', return_value=('doc', [], ['192.0.2.1']))
    def test_test_net_1_is_blocked(self, _):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('http://doc/mcp')

    @patch('socket.gethostbyname_ex', return_value=('doc', [], ['198.51.100.1']))
    def test_test_net_2_is_blocked(self, _):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('http://doc/mcp')

    @patch('socket.gethostbyname_ex', return_value=('doc', [], ['203.0.113.1']))
    def test_test_net_3_is_blocked(self, _):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('http://doc/mcp')

    # ------------------------------------------------------------------
    # IPv6 — loopback, ULA, link-local
    # ------------------------------------------------------------------

    @patch('socket.gethostbyname_ex', return_value=('ip6-localhost', [], ['::1']))
    def test_ipv6_loopback_is_blocked(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://ip6-localhost/mcp')
        self.assertIn('::1', str(ctx.exception))

    @patch('socket.gethostbyname_ex', return_value=('ula', [], ['fc00::1']))
    def test_ipv6_ula_fc00_is_blocked(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://ula/mcp')
        self.assertIn('fc00::1', str(ctx.exception))

    @patch('socket.gethostbyname_ex', return_value=('aws-ipv6-imds', [], ['fd00:ec2::254']))
    def test_aws_ipv6_imds_fd00_is_blocked(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://aws-imds6/mcp')
        self.assertIn('fd00:ec2::254', str(ctx.exception))

    @patch('socket.gethostbyname_ex', return_value=('link6', [], ['fe80::1']))
    def test_ipv6_link_local_is_blocked(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://link6/mcp')
        self.assertIn('fe80::1', str(ctx.exception))

    # ------------------------------------------------------------------
    # Fail-safe: unparseable addresses must be rejected
    # ------------------------------------------------------------------

    @patch('socket.gethostbyname_ex', return_value=('host', [], ['not-an-ip']))
    def test_unparseable_address_is_rejected(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://host/mcp')
        self.assertIn('unparseable', str(ctx.exception))

    # ------------------------------------------------------------------
    # Empty IP list
    # ------------------------------------------------------------------

    @patch('socket.gethostbyname_ex', return_value=('host', [], []))
    def test_empty_ip_list_is_rejected(self, _):
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://host/mcp')
        self.assertIn('No IP', str(ctx.exception))

    # ------------------------------------------------------------------
    # Valid public IP — must NOT raise
    # ------------------------------------------------------------------

    @patch('socket.gethostbyname_ex', return_value=('example.com', [], ['93.184.216.34']))
    def test_public_ipv4_is_allowed(self, _):
        validate_ssrf_safe_url('http://example.com/mcp')  # must not raise

    @patch('socket.gethostbyname_ex', return_value=('example.com', [], ['2606:2800:21f:cb07:6820:80da:af4b:8b2c']))
    def test_public_ipv6_is_allowed(self, _):
        validate_ssrf_safe_url('https://example.com/mcp')  # must not raise

    # ------------------------------------------------------------------
    # Multi-address: any blocked address causes rejection
    # ------------------------------------------------------------------

    @patch('socket.gethostbyname_ex', return_value=('mixed', [], ['93.184.216.34', '10.0.0.1']))
    def test_one_blocked_address_among_many_causes_rejection(self, _):
        with self.assertRaises(ValueError):
            validate_ssrf_safe_url('http://mixed/mcp')

    @patch('socket.gethostbyname_ex', return_value=('totally-legitimate-site.com', [], ['192.168.1.100']))
    def test_dns_rebinding_public_hostname_resolving_to_private_ip_is_blocked(self, _):
        """DNS rebinding: a public-looking hostname that resolves to a private IP must be blocked.

        This guards against a class of SSRF where an attacker registers a domain
        that resolves to an internal RFC1918 address (192.168.0.0/16 Class C in this case).
        """
        with self.assertRaises(ValueError) as ctx:
            validate_ssrf_safe_url('http://totally-legitimate-site.com/mcp')
        self.assertIn('192.168.1.100', str(ctx.exception))


# ---------------------------------------------------------------------------
# mcp_tools tests
# ---------------------------------------------------------------------------

# noinspection HttpUrlsUsage
class ValidateMCPConnectionTests(TestCase):
    """Tests for validate_mcp_connection (sync wrapper around async validation)."""

    @patch('aiworks_core.logic.mcp_tools.MultiServerMCPClient')
    def test_valid_connection_does_not_raise(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client_cls.return_value = mock_client

        # Should not raise
        validate_mcp_connection('http://example.com/mcp', {})

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools.MultiServerMCPClient')
    def test_connection_failure_raises_value_error(self, mock_client_cls, mock_ssrf):
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=ConnectionError('refused'))
        mock_client_cls.return_value = mock_client

        with self.assertRaises(ValueError) as ctx:
            validate_mcp_connection('http://bad-server.example.com/mcp', {})

        self.assertIn('http://bad-server.example.com/mcp', str(ctx.exception))

    @patch('aiworks_core.logic.mcp_tools.MultiServerMCPClient')
    def test_passes_headers_to_client(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client_cls.return_value = mock_client

        headers = {'Authorization': 'Bearer token123'}
        validate_mcp_connection('http://example.com/mcp', headers)

        call_kwargs = mock_client_cls.call_args[0][0]
        self.assertEqual(call_kwargs['server']['headers'], headers)

    @patch('aiworks_core.logic.mcp_tools.MultiServerMCPClient')
    def test_none_headers_passed_as_none_to_client(self, mock_client_cls):
        """validate_mcp_connection passes None headers directly (not converted to {})."""
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client_cls.return_value = mock_client

        validate_mcp_connection('http://example.com/mcp')

        call_kwargs = mock_client_cls.call_args[0][0]
        # headers dict is passed as-is (may be None when not provided)
        self.assertIsNone(call_kwargs['server']['headers'])


# noinspection HttpUrlsUsage
class BuildMCPToolsTests(TestCase):
    """Tests for build_mcp_tools."""

    @patch('aiworks_core.logic.mcp_tools.MultiServerMCPClient')
    def test_returns_tools_from_server(self, mock_client_cls):
        fake_tool = MagicMock()
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[fake_tool])
        mock_client_cls.return_value = mock_client

        configs = [{'name': 'my-server', 'url': 'http://example.com/mcp', 'headers': {}}]
        tools = build_mcp_tools(configs)

        self.assertEqual(tools, [fake_tool])

    @patch('aiworks_core.logic.mcp_tools.MultiServerMCPClient')
    def test_skips_failing_server_and_logs_warning(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=ConnectionError('refused'))
        mock_client_cls.return_value = mock_client

        configs = [{'name': 'bad-server', 'url': 'http://bad.example.com/mcp', 'headers': {}}]
        # Should not raise — failed server is skipped
        tools = build_mcp_tools(configs)
        self.assertEqual(tools, [])

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools.MultiServerMCPClient')
    def test_aggregates_tools_from_multiple_servers(self, mock_client_cls, mock_ssrf):
        tool_a = MagicMock()
        tool_b = MagicMock()

        clients = []
        for tools_returned in ([tool_a], [tool_b]):
            c = MagicMock()
            c.get_tools = AsyncMock(return_value=tools_returned)
            clients.append(c)

        mock_client_cls.side_effect = clients

        configs = [
            {'name': 'server-a', 'url': 'http://a.example.com/mcp', 'headers': {}},
            {'name': 'server-b', 'url': 'http://b.example.com/mcp', 'headers': {}},
        ]
        tools = build_mcp_tools(configs)
        self.assertEqual(tools, [tool_a, tool_b])

    def test_empty_configs_returns_empty_list(self):
        tools = build_mcp_tools([])
        self.assertEqual(tools, [])


# ---------------------------------------------------------------------------
# RefreshableTokenAuth tests
# ---------------------------------------------------------------------------


class RefreshableTokenAuthTests(TestCase):
    """Tests for RefreshableTokenAuth OAuth token refresh on 401."""

    def setUp(self):
        self.user = _make_user()

    def test_read_token_sync_reads_from_db(self):
        """_read_token_sync reads the current token from the DB."""
        server = MCPServer.objects.create(
            user=self.user,
            name='OAuth Server',
            url='https://api.example.com/',
            auth_type='oauth',
            token='current-token',
            refresh_token='refresh-token',
        )
        auth = RefreshableTokenAuth(server.pk)
        self.assertEqual(auth._read_token_sync(), 'current-token')

    def test_read_token_sync_returns_empty_string_when_no_token(self):
        """_read_token_sync returns empty string when server has no token (empty string is falsy)."""
        server = MCPServer.objects.create(
            user=self.user,
            name='OAuth Server',
            url='https://api.example.com/',
            auth_type='oauth',
            token='',
            refresh_token='refresh-token',
        )
        auth = RefreshableTokenAuth(server.pk)
        # Empty string is falsy, so auth flow won't set Authorization header
        self.assertEqual(auth._read_token_sync(), '')

    @patch('aiworks_core.logic.mcp_tools.resolve_oauth_tokens')
    def test_refresh_token_sync_returns_none_when_no_refresh_token(self, mock_resolve):
        """_refresh_token_sync returns None if server has no refresh token."""
        server = MCPServer.objects.create(
            user=self.user,
            name='OAuth Server',
            url='https://api.example.com/',
            auth_type='oauth',
            token='stale-token',
            refresh_token='',
        )
        auth = RefreshableTokenAuth(server.pk)
        result = auth._refresh_token_sync()
        self.assertIsNone(result)
        mock_resolve.assert_not_called()

    @patch('aiworks_core.logic.mcp_tools.resolve_oauth_tokens')
    def test_refresh_token_sync_updates_db_and_returns_new_token(self, mock_resolve):
        """_refresh_token_sync calls resolve_oauth_tokens, saves to DB, returns new token."""
        mock_resolve.return_value = ('new-access-token', 'new-refresh-token')
        server = MCPServer.objects.create(
            user=self.user,
            name='OAuth Server',
            url='https://api.example.com/',
            auth_type='oauth',
            token='stale-token',
            refresh_token='old-refresh-token',
        )
        auth = RefreshableTokenAuth(server.pk)
        result = auth._refresh_token_sync()

        self.assertEqual(result, 'new-access-token')
        server.refresh_from_db()
        self.assertEqual(server.token, 'new-access-token')
        self.assertEqual(server.refresh_token, 'new-refresh-token')

    @patch('aiworks_core.logic.mcp_tools.resolve_oauth_tokens')
    def test_refresh_token_sync_returns_none_on_oauth_error(self, mock_resolve):
        """_refresh_token_sync returns None when resolve_oauth_tokens raises ValueError."""
        mock_resolve.side_effect = ValueError("Token rejected")
        server = MCPServer.objects.create(
            user=self.user,
            name='OAuth Server',
            url='https://api.example.com/',
            auth_type='oauth',
            token='stale-token',
            refresh_token='old-refresh-token',
        )
        auth = RefreshableTokenAuth(server.pk)
        result = auth._refresh_token_sync()

        self.assertIsNone(result)
        server.refresh_from_db()
        self.assertEqual(server.token, 'stale-token')  # unchanged

    def test_refresh_token_sync_uses_select_for_update(self):
        """_refresh_token_sync uses select_for_update inside transaction.atomic."""
        server = MCPServer.objects.create(
            user=self.user,
            name='OAuth Server',
            url='https://api.example.com/',
            auth_type='oauth',
            token='stale-token',
            refresh_token='old-refresh-token',
        )
        auth = RefreshableTokenAuth(server.pk)

        with patch('aiworks_core.logic.mcp_tools.resolve_oauth_tokens', return_value=('new-token', 'new-rt')):
            with self.assertNumQueries(4):  # SAVEPOINT + SELECT FOR UPDATE + UPDATE + RELEASE SAVEPOINT
                auth._refresh_token_sync()


class RefreshableTokenAuthAuthFlowTests(TestCase):
    """Tests for RefreshableTokenAuth sync and async auth flows."""

    def setUp(self):
        self.user = _make_user()

    @patch('aiworks_core.logic.mcp_tools.resolve_oauth_tokens')
    def test_sync_auth_flow_injects_token_and_retries_on_401(self, mock_resolve):
        """sync_auth_flow injects token, yields request, and on 401 retries with fresh token."""
        mock_resolve.return_value = ('fresh-token', 'fresh-rt')
        server = MCPServer.objects.create(
            user=self.user,
            name='OAuth Server',
            url='https://api.example.com/',
            auth_type='oauth',
            token='stale-token',
            refresh_token='old-rt',
        )
        auth = RefreshableTokenAuth(server.pk)

        # Build a fake request
        request = httpx.Request('GET', 'https://api.example.com/tools')
        request.headers['Authorization'] = 'Bearer stale-token'

        # Run the sync auth flow
        flow = auth.sync_auth_flow(request)

        # First yield - should have fresh token
        yielded_request = next(flow)
        self.assertEqual(yielded_request.headers['Authorization'], 'Bearer stale-token')

        # Simulate 401 response
        response_401 = httpx.Response(401, request=yielded_request)

        # Second yield - should have new token from refresh
        retry_request = flow.send(response_401)
        self.assertEqual(retry_request.headers['Authorization'], 'Bearer fresh-token')

        # Exhaust the generator
        with self.assertRaises(StopIteration):
            next(flow)

    @patch('aiworks_core.logic.mcp_tools.resolve_oauth_tokens')
    def test_sync_auth_flow_skips_retry_when_refresh_fails(self, mock_resolve):
        """sync_auth_flow does not yield retry when refresh returns None."""
        mock_resolve.side_effect = ValueError("Refresh failed")
        server = MCPServer.objects.create(
            user=self.user,
            name='OAuth Server',
            url='https://api.example.com/',
            auth_type='oauth',
            token='stale-token',
            refresh_token='old-rt',
        )
        auth = RefreshableTokenAuth(server.pk)

        request = httpx.Request('GET', 'https://api.example.com/tools')
        flow = auth.sync_auth_flow(request)

        yielded_request = next(flow)
        self.assertEqual(yielded_request.headers['Authorization'], 'Bearer stale-token')

        # Simulate 401 - refresh fails, generator ends (return None)
        response_401 = httpx.Response(401, request=yielded_request)
        with self.assertRaises(StopIteration):
            flow.send(response_401)

    @patch('aiworks_core.logic.mcp_tools.resolve_oauth_tokens')
    def test_sync_auth_flow_does_not_retry_on_200(self, mock_resolve):
        """sync_auth_flow does not retry when response is 200."""
        server = MCPServer.objects.create(
            user=self.user,
            name='OAuth Server',
            url='https://api.example.com/',
            auth_type='oauth',
            token='valid-token',
            refresh_token='old-rt',
        )
        auth = RefreshableTokenAuth(server.pk)

        request = httpx.Request('GET', 'https://api.example.com/tools')
        flow = auth.sync_auth_flow(request)

        yielded_request = next(flow)
        self.assertEqual(yielded_request.headers['Authorization'], 'Bearer valid-token')

        # Simulate 200 response - generator ends, no retry
        response_200 = httpx.Response(200, request=yielded_request)
        with self.assertRaises(StopIteration):
            flow.send(response_200)
        with self.assertRaises(StopIteration):
            next(flow)

        # resolve_oauth_tokens should never have been called
        mock_resolve.assert_not_called()


# ---------------------------------------------------------------------------
# resolve_auth_headers OAuth server_id injection tests
# ---------------------------------------------------------------------------


class ResolveAuthHeadersOAuthTests(TestCase):
    """Tests for resolve_auth_headers injecting Oauth-Server-ID header."""

    def test_oauth_injects_oauth_server_id_header(self):
        """When auth_type='oauth' and server_id is provided, Oauth-Server-ID is injected."""
        headers = resolve_auth_headers(
            headers={},
            auth_type='oauth',
            username='',
            password='',
            token='some-token',
            server_id=42,
        )
        self.assertEqual(headers.get('Oauth-Server-ID'), str(42))
        self.assertEqual(headers.get('Authorization'), 'Bearer some-token')

    def test_oauth_does_not_inject_header_when_no_server_id(self):
        """When auth_type='oauth' but server_id is None, Oauth-Server-ID is not injected."""
        headers = resolve_auth_headers(
            headers={},
            auth_type='oauth',
            username='',
            password='',
            token='some-token',
            server_id=None,
        )
        self.assertNotIn('Oauth-Server-ID', headers)
        self.assertEqual(headers.get('Authorization'), 'Bearer some-token')

    def test_non_oauth_does_not_inject_server_id_header(self):
        """When auth_type is not 'oauth', Oauth-Server-ID is not injected."""
        headers = resolve_auth_headers(
            headers={},
            auth_type='bearer',
            username='',
            password='',
            token='some-token',
            server_id=99,
        )
        self.assertNotIn('Oauth-Server-ID', headers)
        self.assertEqual(headers.get('Authorization'), 'Bearer some-token')


# ---------------------------------------------------------------------------
# _load_mcp_tools_async OAuth tests
# ---------------------------------------------------------------------------


class LoadMCPToolsAsyncOAuthTests(TestCase):
    """Tests for _load_mcp_tools_async with OAuth server ID in headers."""

    def setUp(self):
        self.user = _make_user()

    @patch('aiworks_core.logic.mcp_tools.MultiServerMCPClient')
    def test_load_mcp_tools_extracts_oauth_server_id_from_headers(self, mock_client_cls):
        """_load_mcp_tools_async extracts Oauth-Server-ID and passes RefreshableTokenAuth to MultiServerMCPClient."""
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client_cls.return_value = mock_client

        headers = {'Authorization': 'Bearer token', 'Oauth-Server-ID': "42"}
        async_to_sync(_load_mcp_tools_async)('http://example.com/mcp', headers)

        call_kwargs = mock_client_cls.call_args[0][0]
        self.assertIn('auth', call_kwargs['server'])
        self.assertIsInstance(call_kwargs['server']['auth'], RefreshableTokenAuth)
        self.assertEqual(call_kwargs['server']['auth'].server_id, 42)

    @patch('aiworks_core.logic.mcp_tools.MultiServerMCPClient')
    def test_load_mcp_tools_strips_oauth_server_id_from_headers(self, mock_client_cls):
        """_load_mcp_tools_async strips Oauth-Server-ID from headers passed to MultiServerMCPClient."""
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client_cls.return_value = mock_client

        headers = {'Authorization': 'Bearer token', 'Oauth-Server-ID': "42"}
        async_to_sync(_load_mcp_tools_async)('http://example.com/mcp', headers)

        call_headers = mock_client_cls.call_args[0][0]['server']['headers']
        self.assertNotIn('Oauth-Server-ID', call_headers)
        self.assertEqual(call_headers.get('Authorization'), 'Bearer token')

    @patch('aiworks_core.logic.mcp_tools.MultiServerMCPClient')
    def test_load_mcp_tools_no_auth_when_no_oauth_server_id(self, mock_client_cls):
        """_load_mcp_tools_async passes auth=None when no Oauth-Server-ID in headers."""
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client_cls.return_value = mock_client

        headers = {'Authorization': 'Bearer token'}
        async_to_sync(_load_mcp_tools_async)('http://example.com/mcp', headers)

        call_kwargs = mock_client_cls.call_args[0][0]
        self.assertIsNone(call_kwargs.get('auth'))

    @patch('aiworks_core.logic.mcp_tools.MultiServerMCPClient')
    def test_load_mcp_tools_no_auth_when_none_headers(self, mock_client_cls):
        """_load_mcp_tools_async passes auth=None when headers is None."""
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client_cls.return_value = mock_client

        async_to_sync(_load_mcp_tools_async)('http://example.com/mcp', None)

        call_kwargs = mock_client_cls.call_args[0][0]
        self.assertIsNone(call_kwargs.get('auth'))


# ---------------------------------------------------------------------------
# MCPServerSerializer tests
# ---------------------------------------------------------------------------

# noinspection HttpUrlsUsage
class MCPServerSerializerTests(TestCase):
    def setUp(self):
        self.user = _make_user()

    def test_valid_data_serializes_correctly(self):
        data = {'name': 'My Server', 'url': 'http://example.com/mcp', 'headers': {'X-Token': 'abc'}}
        serializer = MCPServerSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_missing_name_is_invalid(self):
        data = {'url': 'http://example.com/mcp'}
        serializer = MCPServerSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('name', serializer.errors)

    def test_missing_url_is_invalid(self):
        data = {'name': 'My Server'}
        serializer = MCPServerSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('url', serializer.errors)

    def test_headers_defaults_to_empty_dict(self):
        data = {'name': 'My Server', 'url': 'http://example.com/mcp'}
        serializer = MCPServerSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        # headers field is optional — serializer should accept omitting it
        instance = serializer.save(user=self.user)
        self.assertEqual(instance.headers, {})

    def test_read_only_fields_are_not_writable(self):
        data = {
            'id': 99999,
            'name': 'My Server',
            'url': 'http://example.com/mcp',
            'created_at': '2000-01-01T00:00:00Z',
            'updated_at': '2000-01-01T00:00:00Z',
        }
        serializer = MCPServerSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        # id and timestamps should be ignored (read-only)
        instance = serializer.save(user=self.user)
        self.assertNotEqual(instance.pk, 99999)

    def test_credential_fields_are_write_only(self):
        """username, password, token, refresh_token must NOT be returned in responses."""
        instance = MCPServer.objects.create(
            user=self.user,
            name='Secret Server',
            url='http://example.com/mcp',
            auth_type='basic',
            username='alice',
            password='s3cret',
        )
        serializer = MCPServerSerializer(instance)
        data = serializer.data
        self.assertNotIn('username', data)
        self.assertNotIn('password', data)
        self.assertNotIn('token', data)
        self.assertNotIn('refresh_token', data)

    def test_has_credentials_false_when_no_credentials(self):
        instance = MCPServer.objects.create(
            user=self.user, name='Open Server', url='http://example.com/mcp',
        )
        data = MCPServerSerializer(instance).data
        self.assertFalse(data['has_credentials'])

    def test_has_credentials_true_when_password_set(self):
        instance = MCPServer.objects.create(
            user=self.user, name='Password Server', url='http://example.com/mcp',
            auth_type='basic', username='user', password='pass',
        )
        data = MCPServerSerializer(instance).data
        self.assertTrue(data['has_credentials'])

    def test_auth_type_is_readable(self):
        instance = MCPServer.objects.create(
            user=self.user, name='Bearer Server', url='http://example.com/mcp',
            auth_type='bearer', token='mytoken',
        )
        data = MCPServerSerializer(instance).data
        self.assertEqual(data['auth_type'], 'bearer')

    def test_update_empty_credential_keeps_existing(self):
        """Sending an empty string for a credential field on update should keep the existing value."""
        instance = MCPServer.objects.create(
            user=self.user, name='Bearer Server', url='http://example.com/mcp',
            auth_type='bearer', token='old-token',
        )
        serializer = MCPServerSerializer(instance,
                                         data={'name': 'Bearer Server', 'url': 'http://example.com/mcp', 'token': ''},
                                         partial=True)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        updated = serializer.save()
        self.assertEqual(updated.token, 'old-token')


# ---------------------------------------------------------------------------
# Credential validation per auth_type
# ---------------------------------------------------------------------------

class MCPCredentialValidationTests(TestCase):
    """MCPServerSerializer must enforce credential requirements per auth_type."""

    def setUp(self):
        self.user = _make_user(username='creduser', email='cred@example.com')

    def _base_data(self, **kwargs):
        return {'name': 'Test', 'url': 'http://example.com/mcp', **kwargs}

    # --- Create: bearer requires token ---

    def test_create_bearer_without_token_fails(self):
        s = MCPServerSerializer(data=self._base_data(auth_type='bearer'))
        self.assertFalse(s.is_valid())
        self.assertIn('auth_type', s.errors)

    def test_create_bearer_with_token_ok(self):
        s = MCPServerSerializer(data=self._base_data(auth_type='bearer', token='mytoken'))
        self.assertTrue(s.is_valid(), s.errors)

    # --- Create: basic requires username + password ---

    def test_create_basic_without_password_fails(self):
        s = MCPServerSerializer(data=self._base_data(auth_type='basic', username='user'))
        self.assertFalse(s.is_valid())
        self.assertIn('auth_type', s.errors)

    def test_create_basic_without_username_fails(self):
        s = MCPServerSerializer(data=self._base_data(auth_type='basic', password='pass'))
        self.assertFalse(s.is_valid())
        self.assertIn('auth_type', s.errors)

    def test_create_basic_with_both_ok(self):
        s = MCPServerSerializer(
            data=self._base_data(auth_type='basic', username='user', password='pass')
        )
        self.assertTrue(s.is_valid(), s.errors)

    # --- Create: oauth requires refresh_token ---

    def test_create_oauth_without_refresh_token_fails(self):
        s = MCPServerSerializer(data=self._base_data(auth_type='oauth'))
        self.assertFalse(s.is_valid())
        self.assertIn('auth_type', s.errors)

    def test_create_oauth_with_refresh_token_ok(self):
        s = MCPServerSerializer(
            data=self._base_data(auth_type='oauth', refresh_token='rt-abc')
        )
        self.assertTrue(s.is_valid(), s.errors)

    # --- Create: none needs no credentials ---

    def test_create_none_without_credentials_ok(self):
        s = MCPServerSerializer(data=self._base_data(auth_type='none'))
        self.assertTrue(s.is_valid(), s.errors)

    # --- Update: switching auth_type validates new requirements ---

    def test_update_switch_to_bearer_without_token_fails(self):
        instance = MCPServer.objects.create(
            user=self.user, name='S', url='http://example.com/mcp', auth_type='none',
        )
        s = MCPServerSerializer(
            instance, data={'auth_type': 'bearer'}, partial=True
        )
        self.assertFalse(s.is_valid())
        self.assertIn('auth_type', s.errors)

    def test_update_switch_to_bearer_with_existing_token_ok(self):
        instance = MCPServer.objects.create(
            user=self.user, name='S', url='http://example.com/mcp',
            auth_type='none', token='pre-existing-token',
        )
        s = MCPServerSerializer(
            instance, data={'auth_type': 'bearer'}, partial=True
        )
        self.assertTrue(s.is_valid(), s.errors)


# ---------------------------------------------------------------------------
# MCPServerViewSet tests
# ---------------------------------------------------------------------------

# noinspection HttpUrlsUsage
class MCPServerViewSetTests(APITestCase):
    def setUp(self):
        self.user = _make_user(username='mcpuser', email='mcp@example.com')
        self.other_user = _make_user(username='other', email='other@example.com')
        self.auth = _auth_header(self.user)
        self.other_auth = _auth_header(self.other_user)
        self.list_url = '/api/mcp-servers/'

    @staticmethod
    def _detail_url(pk):
        return f'/api/mcp-servers/{pk}/'

    def _make_server(self, user=None, name='Test Server', url='http://example.com/mcp'):
        return MCPServer.objects.create(
            user=user or self.user,
            name=name,
            url=url,
            headers={},
        )

    # --- Authentication ---

    def test_list_requires_authentication(self):
        resp = self.client.get(self.list_url)
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_create_requires_authentication(self):
        resp = self.client.post(self.list_url, {'name': 'X', 'url': 'http://x.com'}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    # --- List ---

    def test_list_returns_only_own_servers(self):
        self._make_server(self.user, name='Mine')
        self._make_server(self.other_user, name='Theirs')

        resp = self.client.get(self.list_url, **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        names = [s['name'] for s in resp.data]
        self.assertIn('Mine', names)
        self.assertNotIn('Theirs', names)

    def test_list_empty_when_no_servers(self):
        resp = self.client.get(self.list_url, **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data, [])

    # --- Create ---

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    def test_create_success(self, mock_validate, mock_ssrf):
        mock_validate.return_value = None
        data = {'name': 'New Server', 'url': 'http://new.example.com/mcp', 'headers': {}}
        resp = self.client.post(self.list_url, data, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertEqual(resp.data['name'], 'New Server')
        self.assertTrue(MCPServer.objects.filter(user=self.user, name='New Server').exists())

    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    @patch('aiworks_core.logic.mcp_tools.resolve_auth_headers', return_value={'X-Key': 'val'})
    def test_create_calls_validate_mcp_connection(self, mock_resolve, mock_validate):
        mock_validate.return_value = None
        data = {'name': 'Validated', 'url': 'http://example.com/mcp', 'headers': {'X-Key': 'val'}}
        self.client.post(self.list_url, data, format='json', **self.auth)
        mock_validate.assert_called_once_with('http://example.com/mcp', {'X-Key': 'val'})

    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    def test_create_returns_400_when_connection_fails(self, mock_validate):
        mock_validate.side_effect = ValueError("Could not connect to MCP server at 'http://bad.com/mcp': timeout")
        data = {'name': 'Bad Server', 'url': 'http://bad.com/mcp', 'headers': {}}
        resp = self.client.post(self.list_url, data, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', resp.data)

    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    def test_create_duplicate_name_returns_400(self, mock_validate):
        mock_validate.return_value = None
        self._make_server(self.user, name='Duplicate')
        data = {'name': 'Duplicate', 'url': 'http://example.com/mcp', 'headers': {}}
        resp = self.client.post(self.list_url, data, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', resp.data)

    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    def test_create_assigns_server_to_authenticated_user(self, mock_validate):
        mock_validate.return_value = None
        data = {'name': 'User Server', 'url': 'http://example.com/mcp', 'headers': {}}
        resp = self.client.post(self.list_url, data, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        server = MCPServer.objects.get(pk=resp.data['id'])
        self.assertEqual(server.user, self.user)

    # --- Retrieve ---

    def test_retrieve_own_server(self):
        server = self._make_server()
        resp = self.client.get(self._detail_url(server.pk), **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data['name'], server.name)

    def test_retrieve_other_users_server_returns_404(self):
        server = self._make_server(self.other_user)
        resp = self.client.get(self._detail_url(server.pk), **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    # --- Update ---

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    def test_update_own_server(self, mock_validate, mock_ssrf):
        mock_validate.return_value = None
        server = self._make_server()
        data = {'name': 'Updated', 'url': 'http://updated.example.com/mcp', 'headers': {}}
        resp = self.client.put(self._detail_url(server.pk), data, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        server.refresh_from_db()
        self.assertEqual(server.name, 'Updated')

    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    def test_partial_update_own_server(self, mock_validate):
        mock_validate.return_value = None
        server = self._make_server()
        resp = self.client.patch(
            self._detail_url(server.pk),
            {'name': 'Patched'},
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        server.refresh_from_db()
        self.assertEqual(server.name, 'Patched')

    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    def test_update_validates_connection(self, mock_validate):
        mock_validate.side_effect = ValueError("Cannot connect")
        server = self._make_server()
        data = {'name': server.name, 'url': 'http://bad.com/mcp', 'headers': {}}
        resp = self.client.put(self._detail_url(server.pk), data, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', resp.data)

    def test_update_other_users_server_returns_404(self):
        server = self._make_server(self.other_user)
        data = {'name': 'Hacked', 'url': 'http://example.com/mcp', 'headers': {}}
        resp = self.client.put(self._detail_url(server.pk), data, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    # --- Delete ---

    def test_delete_own_server(self):
        server = self._make_server()
        resp = self.client.delete(self._detail_url(server.pk), **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data, {'status': 'OK'})
        self.assertFalse(MCPServer.objects.filter(pk=server.pk).exists())

    def test_delete_other_users_server_returns_404(self):
        server = self._make_server(self.other_user)
        resp = self.client.delete(self._detail_url(server.pk), **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)
        # Server must still exist
        self.assertTrue(MCPServer.objects.filter(pk=server.pk).exists())


# ---------------------------------------------------------------------------
# OAuth discovery + token exchange tests
# ---------------------------------------------------------------------------

class MCPOAuthDiscoverTests(APITestCase):
    """Tests for POST /api/mcp-servers/discover-oauth/."""

    DISCOVER_URL = '/api/mcp-servers/discover-oauth/'

    def setUp(self):
        self.user = _make_user(username='oauthuser', email='oauth@example.com')
        self.auth = _auth_header(self.user)

    # --- Authentication ---

    def test_requires_authentication(self):
        resp = self.client.post(
            self.DISCOVER_URL,
            {'url': 'http://example.com/mcp', 'redirect_uri': 'http://app.example.com/callback'},
            format='json',
        )
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    # --- Validation ---

    def test_missing_url_returns_400(self):
        resp = self.client.post(
            self.DISCOVER_URL,
            {'redirect_uri': 'http://app.example.com/callback'},
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_missing_redirect_uri_returns_400(self):
        resp = self.client.post(
            self.DISCOVER_URL,
            {'url': 'http://example.com/mcp'},
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    @patch('socket.gethostbyname_ex', return_value=('localhost', [], ['127.0.0.1']))
    def test_ssrf_url_is_blocked(self, _):
        resp = self.client.post(
            self.DISCOVER_URL,
            {'url': 'http://localhost/mcp', 'redirect_uri': 'http://app.example.com/callback'},
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('forbidden', resp.data.get('error', '').lower())

    # --- No OAuth needed ---

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth.discover_service_oauth', return_value={'requires_oauth': False})
    def test_no_oauth_returns_requires_oauth_false(self, mock_discover, _mock_ssrf):
        resp = self.client.post(
            self.DISCOVER_URL,
            {'url': 'http://example.com/mcp', 'redirect_uri': 'http://app.example.com/callback'},
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertFalse(resp.data['requires_oauth'])
        mock_discover.assert_called_once_with(
            'http://example.com/mcp', 'http://app.example.com/callback', client_id=None
        )

    # --- OAuth required ---

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth.discover_service_oauth', return_value={
        'requires_oauth': True,
        'authorization_endpoint': 'https://auth.example.com/authorize',
        'token_endpoint': 'https://auth.example.com/token',
        'client_id': 'test-client-123',
        'scope': 'openid',
    })
    def test_oauth_metadata_returned(self, _mock, _mock_ssrf):
        resp = self.client.post(
            self.DISCOVER_URL,
            {'url': 'http://example.com/mcp', 'redirect_uri': 'http://app.example.com/callback'},
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertTrue(resp.data['requires_oauth'])
        self.assertEqual(resp.data['authorization_endpoint'], 'https://auth.example.com/authorize')
        self.assertEqual(resp.data['token_endpoint'], 'https://auth.example.com/token')
        self.assertEqual(resp.data['client_id'], 'test-client-123')

    # --- Discovery failure propagates as 400 ---

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch(
        'aiworks_core.logic.oauth.discover_service_oauth',
        side_effect=ValueError("Cannot reach MCP server — check the URL and try again."),
    )
    def test_discovery_error_returns_400(self, _mock, _mock_ssrf):
        resp = self.client.post(
            self.DISCOVER_URL,
            {'url': 'http://example.com/mcp', 'redirect_uri': 'http://app.example.com/callback'},
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        # Verify a sanitized error message is returned (not the raw internal exception)
        self.assertIn('error', resp.data)
        self.assertIn('discovery', str(resp.data['error']).lower())

    # --- Non-pro user is rejected ---

    def test_free_user_is_rejected(self):
        free_user = _make_user(username='freeuser2', email='free2@example.com', tier='free')
        auth = _auth_header(free_user)
        resp = self.client.post(
            self.DISCOVER_URL,
            {'url': 'http://example.com/mcp', 'redirect_uri': 'http://app.example.com/callback'},
            format='json',
            **auth,
        )
        self.assertIn(resp.status_code, (status.HTTP_403_FORBIDDEN, status.HTTP_400_BAD_REQUEST))


class MCPOAuthExchangeTests(APITestCase):
    """Tests for POST /api/mcp-servers/oauth-exchange/."""

    EXCHANGE_URL = '/api/mcp-servers/oauth-exchange/'
    VALID_PAYLOAD = {
        'token_endpoint': 'https://auth.example.com/token',
        'code': 'auth-code-abc',
        'code_verifier': 'verifier-xyz',
        'redirect_uri': 'http://app.example.com/callback',
        'client_id': 'test-client-123',
    }

    def setUp(self):
        self.user = _make_user(username='exchangeuser', email='exchange@example.com')
        self.auth = _auth_header(self.user)

    # --- Authentication ---

    def test_requires_authentication(self):
        resp = self.client.post(self.EXCHANGE_URL, self.VALID_PAYLOAD, format='json')
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    # --- Missing fields ---

    def test_missing_token_endpoint_returns_400(self):
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != 'token_endpoint'}
        resp = self.client.post(self.EXCHANGE_URL, payload, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_missing_code_returns_400(self):
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != 'code'}
        resp = self.client.post(self.EXCHANGE_URL, payload, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_missing_code_verifier_returns_400(self):
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != 'code_verifier'}
        resp = self.client.post(self.EXCHANGE_URL, payload, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    # --- SSRF protection on token_endpoint ---

    @patch('socket.gethostbyname_ex', return_value=('localhost', [], ['127.0.0.1']))
    def test_ssrf_token_endpoint_is_blocked(self, _):
        payload = {**self.VALID_PAYLOAD, 'token_endpoint': 'http://localhost/token'}
        resp = self.client.post(self.EXCHANGE_URL, payload, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('not allowed', resp.data.get('error', '').lower())

    # --- Successful exchange ---

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth.exchange_oauth_token', return_value={
        'access_token': 'my-token-123',
        'token_type': 'bearer',
        'expires_in': 3600,
    })
    def test_successful_exchange_returns_token(self, mock_exchange, _mock_ssrf):
        with patch('aiworks_core.views.mcp_views.SiteConfiguration') as mock_config:
            mock_config.get_solo.return_value.site_url = 'http://app.example.com'
            resp = self.client.post(self.EXCHANGE_URL, self.VALID_PAYLOAD, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data['access_token'], 'my-token-123')
        self.assertEqual(resp.data['token_type'], 'bearer')
        mock_exchange.assert_called_once_with(
            token_endpoint='https://auth.example.com/token',
            code='auth-code-abc',
            code_verifier='verifier-xyz',
            redirect_uri='http://app.example.com/callback',
            client_id='test-client-123',
            client_secret='',
        )

    # --- Exchange failure propagates as 400 ---

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch(
        'aiworks_core.logic.oauth.exchange_oauth_token',
        side_effect=ValueError("Token exchange was rejected by the authorization server (HTTP 400)."),
    )
    def test_exchange_failure_returns_400(self, _mock_exchange, _mock_ssrf):
        with patch('aiworks_core.views.mcp_views.SiteConfiguration') as mock_config:
            mock_config.get_solo.return_value.site_url = 'http://app.example.com'
            resp = self.client.post(self.EXCHANGE_URL, self.VALID_PAYLOAD, format='json', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('rejected', resp.data['error'])

    # --- Non-pro user is rejected ---

    def test_free_user_is_rejected(self):
        free_user = _make_user(username='freeuser3', email='free3@example.com', tier='free')
        auth = _auth_header(free_user)
        resp = self.client.post(self.EXCHANGE_URL, self.VALID_PAYLOAD, format='json', **auth)
        self.assertIn(resp.status_code, (status.HTTP_403_FORBIDDEN, status.HTTP_400_BAD_REQUEST))


# ---------------------------------------------------------------------------
# OAuth logic unit tests (discover_mcp_oauth, exchange_oauth_token)
# ---------------------------------------------------------------------------

class DiscoverMCPOAuthLogicTests(TestCase):
    """Unit tests for discover_service_oauth() (the generic OAuth discovery wrapper)."""

    @patch('aiworks_core.logic.oauth.async_to_sync')
    def test_discover_calls_async_wrapper(self, mock_a2s):
        """discover_service_oauth delegates to _discover_service_oauth_async via async_to_sync."""
        mock_inner = MagicMock(return_value={'requires_oauth': False})
        mock_a2s.return_value = mock_inner

        result = discover_service_oauth('http://example.com/mcp', 'http://app/callback')
        self.assertFalse(result['requires_oauth'])
        mock_inner.assert_called_once_with('http://example.com/mcp', 'http://app/callback', None)

    @patch('aiworks_core.logic.oauth.async_to_sync')
    def test_exchange_calls_async_wrapper(self, mock_a2s):
        """exchange_oauth_token delegates to _exchange_oauth_token_async via async_to_sync."""
        mock_inner = MagicMock(return_value={'access_token': 'tok'})
        mock_a2s.return_value = mock_inner

        result = exchange_oauth_token('http://auth/token', 'code', 'verifier', 'http://cb', 'cid')
        self.assertEqual(result['access_token'], 'tok')


# ---------------------------------------------------------------------------
# Token refresh logic unit tests
# ---------------------------------------------------------------------------

class RefreshOAuthTokenLogicTests(TestCase):
    """Unit tests for refresh_access_token() (the generic token refresh wrapper)."""

    @patch('aiworks_core.logic.oauth.async_to_sync')
    def test_refresh_calls_async_wrapper(self, mock_a2s):
        """refresh_oauth_token delegates to _refresh_access_token_async."""
        mock_inner = MagicMock(return_value={'access_token': 'new-token'})
        mock_a2s.return_value = mock_inner

        result = refresh_access_token(
            'https://auth.example.com/token',
            'my-refresh-token',
            'test-client',
        )
        self.assertEqual(result['access_token'], 'new-token')
        mock_inner.assert_called_once_with(
            'https://auth.example.com/token', 'my-refresh-token', 'test-client', ''
        )


# ---------------------------------------------------------------------------
# OAuth resolve_oauth_tokens_async — error propagation tests
# ---------------------------------------------------------------------------

class ResolveOAuthTokensErrorTests(TestCase):
    """resolve_oauth_tokens must propagate OAuth errors instead of silently proceeding."""

    @patch('aiworks_core.logic.oauth._discover_oauth_metadata_async', new_callable=AsyncMock)
    def test_oauth_discovery_failure_propagates(self, mock_discover):
        """If oauth_metadata is missing and metadata discovery fails, the error must propagate."""
        mock_discover.side_effect = ValueError("Cannot reach MCP server")
        with self.assertRaises(ValueError) as ctx:
            resolve_oauth_tokens('http://mcp.example.com', 'my-refresh-token')
        self.assertIn('Cannot reach', str(ctx.exception))

    @patch('aiworks_core.logic.oauth._refresh_access_token_async', new_callable=AsyncMock)
    @patch('aiworks_core.logic.oauth._discover_oauth_metadata_async', new_callable=AsyncMock)
    def test_missing_token_endpoint_in_metadata_raises(self, mock_discover, mock_refresh):
        """If the metadata document has no token_endpoint, an error must be raised."""
        mock_discover.return_value = {'authorization_endpoint': 'https://auth.example.com/authorize'}
        with self.assertRaises(ValueError) as ctx:
            resolve_oauth_tokens('http://mcp.example.com', 'my-refresh-token',
                                 oauth_metadata={'client_id': 'test-client'})
        self.assertIn('token_endpoint', str(ctx.exception))

    @patch('aiworks_core.logic.oauth._refresh_access_token_async', new_callable=AsyncMock)
    @patch('aiworks_core.logic.oauth._discover_oauth_metadata_async', new_callable=AsyncMock)
    def test_no_access_token_in_response_raises(self, mock_discover, mock_refresh):
        """If the token endpoint returns no access_token, an error must be raised."""
        mock_discover.return_value = {'token_endpoint': 'https://auth.example.com/token', 'client_id': 'test-client'}
        mock_refresh.return_value = {}  # no access_token
        with patch('aiworks_core.logic.oauth.validate_ssrf_safe_url'):
            with self.assertRaises(ValueError) as ctx:
                resolve_oauth_tokens('http://mcp.example.com', 'my-refresh-token',
                                     oauth_metadata={'token_endpoint': 'https://auth.example.com/token',
                                                     'client_id': 'test-client'})
        self.assertIn('access token', str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# SSRF-safe redirect following in OAuth discovery
# ---------------------------------------------------------------------------

class OAuthDiscoverySSRFRedirectTests(TestCase):
    """safe_get_with_ssrf_check must validate every redirect target for SSRF."""

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    def test_redirect_to_internal_ip_is_blocked(self, mock_ssrf):
        """A 302 redirect from the MCP server to an internal address must be blocked."""

        redirect_resp = MagicMock(spec=httpx.Response)
        redirect_resp.is_redirect = True
        redirect_resp.headers = {'location': 'http://192.168.1.1/evil'}
        redirect_resp.status_code = 302

        mock_ssrf.side_effect = ValueError("SSRF blocked: private IP")

        async def _run():
            async_client = AsyncMock()
            async_client.get.return_value = redirect_resp
            with self.assertRaises(ValueError) as ctx:
                await safe_get_with_ssrf_check(async_client, 'http://mcp.example.com')
            self.assertIn('SSRF', str(ctx.exception))

        asyncio.run(_run())
        mock_ssrf.assert_called_with('http://192.168.1.1/evil')


# ---------------------------------------------------------------------------
# SSRF validation on redirect_uri in oauth_exchange
# ---------------------------------------------------------------------------

class OAuthExchangeRedirectURISSRFTests(APITestCase):
    """redirect_uri in /oauth-exchange/ must be SSRF-validated."""

    EXCHANGE_URL = '/api/mcp-servers/oauth-exchange/'

    def setUp(self):
        self.user = _make_user(username='ssrf_exchange', email='ssrf_exchange@example.com')
        self.auth = _auth_header(self.user)

    @patch('socket.gethostbyname_ex', return_value=('localhost', [], ['127.0.0.1']))
    def test_ssrf_redirect_uri_is_blocked(self, _):
        """redirect_uri pointing to an internal address must be rejected."""
        resp = self.client.post(
            self.EXCHANGE_URL,
            {
                'token_endpoint': 'https://auth.example.com/token',
                'code': 'abc',
                'code_verifier': 'xyz',
                'redirect_uri': 'http://localhost/callback',
                'client_id': 'test-client',
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('not allowed', resp.data.get('error', '').lower())

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth.exchange_oauth_token', return_value={
        'access_token': 'tok',
        'token_type': 'bearer',
    })
    def test_invalid_localhost_redirect_uri_is_rejected(self, _mock_exchange, _mock_ssrf):
        """A localhost redirect_uri must be blocked."""
        with patch('aiworks_core.views.mcp_views.SiteConfiguration') as mock_config:
            mock_config.get_solo.return_value.site_url = 'http://localhost:3000'
            resp = self.client.post(
                self.EXCHANGE_URL,
                {
                    'token_endpoint': 'https://auth.example.com/token',
                    'code': 'abc',
                    'code_verifier': 'xyz',
                    'redirect_uri': 'http://localhost/callback',
                    'client_id': 'test-client',
                },
                format='json',
                **self.auth,
            )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('configured site', resp.data.get('error', '').lower())

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth.exchange_oauth_token', return_value={
        'access_token': 'tok',
        'token_type': 'bearer',
    })
    def test_valid_redirect_uri_is_accepted(self, _mock_exchange, _mock_ssrf):
        """A valid public redirect_uri must not be blocked."""
        with patch('aiworks_core.views.mcp_views.SiteConfiguration') as mock_config:
            mock_config.get_solo.return_value.site_url = 'https://app.example.com'
            resp = self.client.post(
                self.EXCHANGE_URL,
                {
                    'token_endpoint': 'https://auth.example.com/token',
                    'code': 'abc',
                    'code_verifier': 'xyz',
                    'redirect_uri': 'https://app.example.com/mcp-oauth-callback',
                    'client_id': 'test-client',
                },
                format='json',
                **self.auth,
            )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# OAuth error surfacing during server registration
# ---------------------------------------------------------------------------

class OAuthRegistrationErrorTests(APITestCase):
    """perform_create/perform_update must surface OAuth token refresh errors clearly."""

    LIST_URL = '/api/mcp-servers/'

    def setUp(self):
        self.user = _make_user(username='reg_oauth', email='reg_oauth@example.com')
        self.auth = _auth_header(self.user)

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools.resolve_auth_headers', side_effect=ValueError("refresh token expired"))
    def test_create_oauth_auth_failure_returns_clear_error(self, _mock_resolve, _mock_ssrf):
        """If OAuth refresh fails during registration, a clear message must be returned."""
        resp = self.client.post(
            self.LIST_URL,
            {
                'name': 'OAuth Server',
                'url': 'https://mcp.example.com/',
                'auth_type': 'oauth',
                'refresh_token': 'some-token',
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', resp.data)
        error_msg = str(resp.data['error']).lower()
        self.assertIn('oauth', error_msg)


# ---------------------------------------------------------------------------
# PredefinedMCPServer model and serializer tests
# ---------------------------------------------------------------------------

class PredefinedMCPServerSerializerTests(TestCase):
    """Tests for PredefinedMCPServerSerializer."""

    def _make_predefined(self, name='GitHub Copilot', url='https://mcp.github.com/', auth_type='bearer'):
        return PredefinedMCPServer.objects.create(name=name, url=url, auth_type=auth_type)

    def test_serializes_all_expected_fields(self):
        predefined = self._make_predefined()
        data = PredefinedMCPServerSerializer(predefined).data
        for field in ('id', 'name', 'url', 'auth_type', 'headers', 'created_at', 'updated_at'):
            self.assertIn(field, data)

    def test_client_id_and_secret_not_exposed(self):
        """client_id and client_secret are admin-only fields — must NOT appear in the API response."""
        predefined = PredefinedMCPServer.objects.create(
            name='Secret Server', url='https://mcp.example.com/', auth_type='oauth',
            client_id='cid123', client_secret='csecret456',
        )
        data = PredefinedMCPServerSerializer(predefined).data
        self.assertNotIn('client_id', data)
        self.assertNotIn('client_secret', data)

    def test_headers_defaults_to_empty_dict(self):
        predefined = self._make_predefined()
        data = PredefinedMCPServerSerializer(predefined).data
        self.assertEqual(data['headers'], {})


# noinspection HttpUrlsUsage
class PredefinedMCPServerViewSetTests(APITestCase):
    """Tests for GET /api/predefined-mcp-servers/."""

    LIST_URL = '/api/predefined-mcp-servers/'

    def setUp(self):
        self.user = _make_user(username='preduser', email='pred@example.com')
        self.auth = _auth_header(self.user)

    def _make_predefined(self, name='Test Predefined', url='http://mcp.example.com/', auth_type='none'):
        return PredefinedMCPServer.objects.create(name=name, url=url, auth_type=auth_type)

    def test_list_requires_authentication(self):
        resp = self.client.get(self.LIST_URL)
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_list_returns_all_predefined_servers(self):
        self._make_predefined(name='Server A')
        self._make_predefined(name='Server B', url='http://mcp2.example.com/')
        resp = self.client.get(self.LIST_URL, **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        names = [s['name'] for s in resp.data]
        self.assertIn('Server A', names)
        self.assertIn('Server B', names)

    def test_list_empty_when_no_predefined_servers(self):
        resp = self.client.get(self.LIST_URL, **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data, [])

    def test_create_is_not_allowed(self):
        resp = self.client.post(
            self.LIST_URL,
            {'name': 'New', 'url': 'http://mcp.example.com/'},
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

    def test_delete_is_not_allowed(self):
        predefined = self._make_predefined()
        resp = self.client.delete(f'{self.LIST_URL}{predefined.pk}/', **self.auth)
        self.assertEqual(resp.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

    def test_free_user_can_list(self):
        """Predefined server list is visible regardless of tier."""
        free_user = _make_user(username='freeuser_pred', email='freepred@example.com', tier='free')
        auth = _auth_header(free_user)
        self._make_predefined()
        resp = self.client.get(self.LIST_URL, **auth)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)


# noinspection HttpUrlsUsage
class MCPServerFromPredefinedTests(APITestCase):
    """Tests for creating and editing MCPServer instances derived from a PredefinedMCPServer."""

    LIST_URL = '/api/mcp-servers/'

    def setUp(self):
        self.user = _make_user(username='predmcpuser', email='predmcp@example.com')
        self.auth = _auth_header(self.user)
        self.predefined = PredefinedMCPServer.objects.create(
            name='Example Predefined',
            url='http://mcp.example.com/',
            auth_type='bearer',
            headers={'X-Source': 'predefined'},
        )

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    def test_create_from_predefined_inherits_url_and_headers(self, mock_validate, mock_ssrf):
        mock_validate.return_value = None
        resp = self.client.post(
            self.LIST_URL,
            {
                'predefined_server_id': self.predefined.pk,
                'token': 'mytoken',
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, resp.data)
        server = MCPServer.objects.get(pk=resp.data['id'])
        self.assertEqual(server.url, self.predefined.url)
        self.assertEqual(server.auth_type, self.predefined.auth_type)
        self.assertEqual(server.headers, self.predefined.headers)
        self.assertEqual(server.predefined_server, self.predefined)

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    def test_create_from_predefined_uses_predefined_name_when_name_not_provided(self, mock_validate, mock_ssrf):
        mock_validate.return_value = None
        resp = self.client.post(
            self.LIST_URL,
            {
                'predefined_server_id': self.predefined.pk,
                'token': 'mytoken',
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, resp.data)
        server = MCPServer.objects.get(pk=resp.data['id'])
        self.assertEqual(server.name, self.predefined.name)

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    def test_create_from_predefined_stores_predefined_server_link(self, mock_validate, mock_ssrf):
        mock_validate.return_value = None
        resp = self.client.post(
            self.LIST_URL,
            {
                'predefined_server_id': self.predefined.pk,
                'token': 'mytoken',
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, resp.data)
        self.assertIn('predefined_server_id', resp.data)
        self.assertEqual(resp.data['predefined_server_id'], self.predefined.pk)

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    def test_update_predefined_derived_server_locks_url_and_headers(self, mock_validate, mock_ssrf):
        """Updating an MCPServer derived from a predefined server must not change url/headers."""
        mock_validate.return_value = None
        server = MCPServer.objects.create(
            user=self.user,
            name=self.predefined.name,
            url=self.predefined.url,
            auth_type=self.predefined.auth_type,
            headers=self.predefined.headers,
            predefined_server=self.predefined,
            token='old-token',
        )
        resp = self.client.patch(
            f'{self.LIST_URL}{server.pk}/',
            {
                'url': 'http://attacker.example.com/',
                'headers': {'X-Evil': 'hacked'},
                'token': 'new-token',
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK, resp.data)
        server.refresh_from_db()
        # URL and headers must remain those of the predefined server
        self.assertEqual(server.url, self.predefined.url)
        self.assertEqual(server.headers, self.predefined.headers)
        # Token should be updated
        self.assertEqual(server.token, 'new-token')

    @patch('aiworks_core.logic.mcp_tools.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.mcp_tools.validate_mcp_connection')
    def test_predefined_with_client_id_injects_into_oauth_metadata(self, mock_validate, mock_ssrf):
        """When a predefined server has a client_id, it must be injected into oauth_metadata."""
        mock_validate.return_value = None
        predefined_oauth = PredefinedMCPServer.objects.create(
            name='OAuth Predefined',
            url='http://oauth-mcp.example.com/',
            auth_type='none',
            client_id='predefined-client-id',
            client_secret='predefined-secret',
        )
        resp = self.client.post(
            self.LIST_URL,
            {'predefined_server_id': predefined_oauth.pk},
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, resp.data)
        server = MCPServer.objects.get(pk=resp.data['id'])
        self.assertEqual(server.oauth_metadata.get('client_id'), 'predefined-client-id')
        # client_secret must never be persisted to oauth_metadata
        self.assertIsNone(server.oauth_metadata.get('client_secret'))


# ---------------------------------------------------------------------------
# Custom OAuth endpoints on PredefinedMCPServer
# ---------------------------------------------------------------------------

class PredefinedMCPServerCustomOAuthEndpoints(APITestCase):
    """Tests for custom OAuth endpoints (authorization_endpoint, token_endpoint, scope)
    on PredefinedMCPServer — skip discovery when custom endpoints exist."""

    DISCOVER_URL = '/api/mcp-servers/discover-oauth/'

    def setUp(self):
        self.user = _make_user(username='custom_oauth_user', email='custom_oauth@example.com')
        self.auth = _auth_header(self.user)

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._discover_service_oauth_async')
    def test_discover_oauth_skips_discovery_when_authorization_endpoint(self, mock_discover_async, _mock_ssrf):
        """When predefined has authorization_endpoint, discovery is skipped."""
        predefined = PredefinedMCPServer.objects.create(
            name='Custom Auth Server',
            url='https://custom-oauth.example.com/',
            auth_type='oauth',
            authorization_endpoint='https://auth.example.com/authorize',
            token_endpoint='https://auth.example.com/token',
            scope='custom_scope',
            client_id='custom-client',
        )
        resp = self.client.post(
            self.DISCOVER_URL,
            {
                'url': 'https://custom-oauth.example.com/',
                'redirect_uri': 'http://app.example.com/callback',
                'predefined_server_id': predefined.pk,
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data['authorization_endpoint'], 'https://auth.example.com/authorize')
        self.assertEqual(resp.data['token_endpoint'], 'https://auth.example.com/token')
        self.assertEqual(resp.data['scope'], 'custom_scope')
        mock_discover_async.assert_not_called()

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._discover_service_oauth_async')
    def test_discover_oauth_skips_discovery_when_token_endpoint(self, mock_discover_async, _mock_ssrf):
        """When predefined has both endpoints, discovery is skipped (AND condition)."""
        predefined = PredefinedMCPServer.objects.create(
            name='Custom Token Server',
            url='https://custom-oauth.example.com/',
            auth_type='oauth',
            authorization_endpoint='https://auth.example.com/authorize',
            token_endpoint='https://auth.example.com/token',
            scope='custom_scope',
            client_id='custom-client',
        )
        resp = self.client.post(
            self.DISCOVER_URL,
            {
                'url': 'https://custom-oauth.example.com/',
                'redirect_uri': 'http://app.example.com/callback',
                'predefined_server_id': predefined.pk,
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data['token_endpoint'], 'https://auth.example.com/token')
        self.assertEqual(resp.data['scope'], 'custom_scope')
        mock_discover_async.assert_not_called()

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._discover_service_oauth_async')
    def test_discover_oauth_returns_custom_endpoints_and_scope(self, mock_discover_async, _mock_ssrf):
        """Response includes authorization_endpoint, token_endpoint, and scope from predefined."""
        predefined = PredefinedMCPServer.objects.create(
            name='Full Custom Endpoint Server',
            url='https://custom-oauth.example.com/',
            auth_type='oauth',
            authorization_endpoint='https://custom.auth.example.com/authorize',
            token_endpoint='https://custom.auth.example.com/token',
            scope='profile email read write',
            client_id='my-client-id',
        )
        resp = self.client.post(
            self.DISCOVER_URL,
            {
                'url': 'https://custom-oauth.example.com/',
                'redirect_uri': 'http://app.example.com/callback',
                'predefined_server_id': predefined.pk,
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data['authorization_endpoint'], 'https://custom.auth.example.com/authorize')
        self.assertEqual(resp.data['token_endpoint'], 'https://custom.auth.example.com/token')
        self.assertEqual(resp.data['scope'], 'profile email read write')
        self.assertEqual(resp.data['client_id'], 'my-client-id')
        mock_discover_async.assert_not_called()

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._discover_service_oauth_async', return_value={
        'requires_oauth': True,
        'authorization_endpoint': 'https://discovered.example.com/authorize',
        'token_endpoint': 'https://discovered.example.com/token',
        'client_id': 'discovered-client',
        'scope': 'discovered_scope',
    })
    def test_discover_oauth_scope_only_does_not_skip_discovery(self, mock_discover_async, _mock_ssrf):
        """Predefined with only scope (no custom endpoints) must not skip discovery."""
        predefined = PredefinedMCPServer.objects.create(
            name='Scope Only Server',
            url='https://scope-only.example.com/',
            auth_type='oauth',
            authorization_endpoint='',
            token_endpoint='',
            scope='custom_scope_override',
            client_id='scope-client',
        )
        resp = self.client.post(
            self.DISCOVER_URL,
            {
                'url': 'https://scope-only.example.com/',
                'redirect_uri': 'http://app.example.com/callback',
                'predefined_server_id': predefined.pk,
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        mock_discover_async.assert_called_once()
        self.assertEqual(resp.data['scope'], 'custom_scope_override')

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._discover_service_oauth_async', return_value={
        'requires_oauth': True,
        'authorization_endpoint': 'https://discovered.example.com/authorize',
        'token_endpoint': 'https://discovered.example.com/token',
        'client_id': 'discovered-client',
        'scope': 'discovered_scope',
    })
    def test_discover_oauth_scope_override_after_discovery(self, mock_discover_async, _mock_ssrf):
        """Predefined scope must override discovered scope even when discovery ran."""
        predefined = PredefinedMCPServer.objects.create(
            name='Scope Override Server',
            url='https://scope-override.example.com/',
            auth_type='oauth',
            authorization_endpoint='',
            token_endpoint='',
            scope='custom_scope',
        )
        resp = self.client.post(
            self.DISCOVER_URL,
            {
                'url': 'https://scope-override.example.com/',
                'redirect_uri': 'http://app.example.com/callback',
                'predefined_server_id': predefined.pk,
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        mock_discover_async.assert_called_once()
        self.assertEqual(resp.data['scope'], 'custom_scope')
        self.assertEqual(resp.data['token_endpoint'], 'https://discovered.example.com/token')

    def test_inject_predefined_oauth_fields_injects_authorization_endpoint(self):
        """_inject_predefined_oauth_fields must inject authorization_endpoint into oauth_metadata."""
        predefined = PredefinedMCPServer.objects.create(
            name='Inject Auth Endpoint',
            url='https://inject.example.com/',
            auth_type='oauth',
            authorization_endpoint='https://auth.example.com/authorize',
            client_id='inject-client',
        )
        view = MCPServerViewSet()
        result = view._inject_predefined_oauth_fields(predefined, {})
        self.assertEqual(result.get('authorization_endpoint'), 'https://auth.example.com/authorize')

    def test_inject_predefined_oauth_fields_injects_token_endpoint(self):
        """_inject_predefined_oauth_fields must inject token_endpoint into oauth_metadata."""
        predefined = PredefinedMCPServer.objects.create(
            name='Inject Token Endpoint',
            url='https://inject.example.com/',
            auth_type='oauth',
            token_endpoint='https://auth.example.com/token',
            client_id='inject-client',
        )
        view = MCPServerViewSet()
        result = view._inject_predefined_oauth_fields(predefined, {})
        self.assertEqual(result.get('token_endpoint'), 'https://auth.example.com/token')

    def test_inject_predefined_oauth_fields_does_not_override_existing(self):
        """_inject_predefined_oauth_fields must not overwrite existing metadata keys."""
        predefined = PredefinedMCPServer.objects.create(
            name='No Override Server',
            url='https://no-override.example.com/',
            auth_type='oauth',
            authorization_endpoint='https://predefined.example.com/auth',
            token_endpoint='https://predefined.example.com/token',
            client_id='predefined-client',
        )
        view = MCPServerViewSet()
        existing_metadata = {
            'authorization_endpoint': 'https://existing.example.com/auth',
            'token_endpoint': 'https://existing.example.com/token',
            'client_id': 'existing-client',
        }
        result = view._inject_predefined_oauth_fields(predefined, existing_metadata)
        self.assertEqual(result['authorization_endpoint'], 'https://existing.example.com/auth')
        self.assertEqual(result['token_endpoint'], 'https://existing.example.com/token')
        self.assertEqual(result['client_id'], 'existing-client')


    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    def test_discover_oauth_returns_custom_auth_params_when_skipping(self, _mock_ssrf):
        """When skipping discovery, custom_auth_params must be returned from predefined."""
        predefined = PredefinedMCPServer.objects.create(
            name='Custom Auth Params Server',
            url='https://custom-auth.example.com/',
            auth_type='oauth',
            authorization_endpoint='https://auth.example.com/authorize',
            token_endpoint='https://auth.example.com/token',
            custom_auth_params={"connection_hint": "enterprise", "tenant": "acme"},
        )
        resp = self.client.post(
            self.DISCOVER_URL,
            {
                'url': 'https://custom-auth.example.com/',
                'redirect_uri': 'http://app.example.com/callback',
                'predefined_server_id': predefined.pk,
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data['custom_auth_params'], {"connection_hint": "enterprise", "tenant": "acme"})

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._discover_service_oauth_async', return_value={
        'requires_oauth': True,
        'authorization_endpoint': 'https://discovered.example.com/authorize',
        'token_endpoint': 'https://discovered.example.com/token',
        'client_id': 'discovered-client',
        'scope': '',
    })
    def test_discover_oauth_includes_custom_auth_params_after_discovery(self, mock_discover_async, _mock_ssrf):
        """When discovery ran, custom_auth_params from predefined must still be included."""
        predefined = PredefinedMCPServer.objects.create(
            name='Discovery With Custom Params',
            url='https://discovery-params.example.com/',
            auth_type='oauth',
            custom_auth_params={"custom_field": "custom_value"},
        )
        resp = self.client.post(
            self.DISCOVER_URL,
            {
                'url': 'https://discovery-params.example.com/',
                'redirect_uri': 'http://app.example.com/callback',
                'predefined_server_id': predefined.pk,
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        mock_discover_async.assert_called_once()
        self.assertEqual(resp.data['custom_auth_params'], {"custom_field": "custom_value"})

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._discover_service_oauth_async', return_value={
        'requires_oauth': True,
        'authorization_endpoint': 'https://discovered.example.com/authorize',
        'token_endpoint': 'https://discovered.example.com/token',
        'client_id': 'discovered-client',
        'scope': '',
    })
    def test_custom_auth_params_empty_dict_when_not_set(self, mock_discover_async, _mock_ssrf):
        """When predefined has no custom_auth_params, {} must be returned, not absent."""
        predefined = PredefinedMCPServer.objects.create(
            name='No Custom Params Server',
            url='https://no-params.example.com/',
            auth_type='oauth',
        )
        resp = self.client.post(
            self.DISCOVER_URL,
            {
                'url': 'https://no-params.example.com/',
                'redirect_uri': 'http://app.example.com/callback',
                'predefined_server_id': predefined.pk,
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data.get('custom_auth_params'), {})


# ---------------------------------------------------------------------------
# OAuth improvements: skip registration, client_secret in calls
# ---------------------------------------------------------------------------
# These tests were removed — they duplicate test_oauth.py coverage and
# reference _register_oauth_client_async / _discover_service_oauth_async
# which are internal implementation details of the generic oauth module.


class DiscoverOAuthWithPredefinedServerTests(APITestCase):
    """The discover-oauth endpoint should pass predefined server client_id to skip registration."""

    DISCOVER_URL = '/api/mcp-servers/discover-oauth/'

    def setUp(self):
        self.user = _make_user(username='predoauthuser', email='predoauth@example.com')
        self.auth = _auth_header(self.user)
        self.predefined = PredefinedMCPServer.objects.create(
            name='Predefined OAuth Server',
            url='https://predefined-mcp.example.com/',
            auth_type='oauth',
            client_id='predefined-client-id',
            client_secret='predefined-secret',
        )

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth.discover_service_oauth', return_value={
        'requires_oauth': True,
        'authorization_endpoint': 'https://auth.example.com/authorize',
        'token_endpoint': 'https://auth.example.com/token',
        'client_id': 'predefined-client-id',
        'scope': '',
    })
    def test_predefined_server_id_passes_client_id_to_discover(self, mock_discover, _mock_ssrf):
        """When predefined_server_id is provided, the view looks up client_id and passes it."""
        resp = self.client.post(
            self.DISCOVER_URL,
            {
                'url': 'https://predefined-mcp.example.com/',
                'redirect_uri': 'http://app.example.com/callback',
                'predefined_server_id': self.predefined.pk,
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        mock_discover.assert_called_once_with(
            'https://predefined-mcp.example.com/',
            'http://app.example.com/callback',
            client_id='predefined-client-id',
        )

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth.discover_service_oauth', return_value={'requires_oauth': False})
    def test_unknown_predefined_server_id_ignored(self, mock_discover, _mock_ssrf):
        """An unknown predefined_server_id is silently ignored and discover is called with client_id=None."""
        resp = self.client.post(
            self.DISCOVER_URL,
            {
                'url': 'https://other-mcp.example.com/',
                'redirect_uri': 'http://app.example.com/callback',
                'predefined_server_id': 99999,
            },
            format='json',
            **self.auth,
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        mock_discover.assert_called_once_with(
            'https://other-mcp.example.com/',
            'http://app.example.com/callback',
            client_id=None,
        )


class OAuthExchangeWithPredefinedServerTests(APITestCase):
    """The oauth-exchange endpoint should pass client_secret for predefined servers."""

    EXCHANGE_URL = '/api/mcp-servers/oauth-exchange/'

    def setUp(self):
        self.user = _make_user(username='exchpreduser', email='exchpred@example.com')
        self.auth = _auth_header(self.user)
        self.predefined = PredefinedMCPServer.objects.create(
            name='Predefined OAuth Exchange Server',
            url='https://predefined-mcp.example.com/',
            auth_type='oauth',
            client_id='predefined-client-id',
            client_secret='predefined-secret',
        )

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth.exchange_oauth_token', return_value={
        'access_token': 'tok',
        'token_type': 'bearer',
        'refresh_token': 'rt',
        'oauth_metadata': {'token_endpoint': 'https://auth.example.com/token', 'client_id': 'client'},
    })
    def test_predefined_server_id_passes_client_secret_to_exchange(self, mock_exchange, _mock_ssrf):
        """When predefined_server_id is provided, client_secret is fetched and forwarded."""
        with patch('aiworks_core.views.mcp_views.SiteConfiguration') as mock_config:
            mock_config.get_solo.return_value.site_url = 'http://app.example.com'
            resp = self.client.post(
                self.EXCHANGE_URL,
                {
                    'token_endpoint': 'https://auth.example.com/token',
                    'code': 'code123',
                    'code_verifier': 'verifier',
                    'redirect_uri': 'http://app.example.com/callback',
                    'client_id': 'predefined-client-id',
                    'predefined_server_id': self.predefined.pk,
                },
                format='json',
                **self.auth,
            )
        self.assertEqual(resp.status_code, status.HTTP_200_OK, resp.data)
        mock_exchange.assert_called_once_with(
            token_endpoint='https://auth.example.com/token',
            code='code123',
            code_verifier='verifier',
            redirect_uri='http://app.example.com/callback',
            client_id='predefined-client-id',
            client_secret='predefined-secret',
        )

    @patch('aiworks_core.logic.logic_utils.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth.exchange_oauth_token', return_value={
        'access_token': 'tok', 'token_type': 'bearer', 'refresh_token': 'rt',
    })
    def test_exchange_without_predefined_sends_empty_client_secret(self, mock_exchange, _mock_ssrf):
        """When no predefined_server_id, exchange is called with empty client_secret."""
        with patch('aiworks_core.views.mcp_views.SiteConfiguration') as mock_config:
            mock_config.get_solo.return_value.site_url = 'http://app.example.com'
            resp = self.client.post(
                self.EXCHANGE_URL,
                {
                    'token_endpoint': 'https://auth.example.com/token',
                    'code': 'code123',
                    'code_verifier': 'verifier',
                    'redirect_uri': 'http://app.example.com/callback',
                    'client_id': 'some-client',
                },
                format='json',
                **self.auth,
            )
        self.assertEqual(resp.status_code, status.HTTP_200_OK, resp.data)
        mock_exchange.assert_called_once_with(
            token_endpoint='https://auth.example.com/token',
            code='code123',
            code_verifier='verifier',
            redirect_uri='http://app.example.com/callback',
            client_id='some-client',
            client_secret='',
        )
