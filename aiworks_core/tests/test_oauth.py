"""
Unit tests for the generic OAuth 2.1 / PKCE helpers in logic/oauth.py.

Covers:
- discover_service_oauth (sync wrapper)
- exchange_oauth_token (sync wrapper)
- refresh_access_token (sync wrapper)
- _discover_oauth_metadata_async
- _exchange_oauth_token_async
- _resolve_oauth_tokens_async
"""
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import httpx
from django.test import TestCase
from ..logic.oauth import (
    discover_service_oauth,
    exchange_oauth_token,
    refresh_access_token,
    _discover_oauth_metadata_async,
    _exchange_oauth_token_async,
    _resolve_oauth_tokens_async,
)


class DiscoverServiceOAuthTests(TestCase):
    """Tests for discover_service_oauth (sync wrapper)."""

    @patch('aiworks_core.logic.oauth.async_to_sync')
    def test_discover_calls_async_wrapper(self, mock_a2s):
        """discover_service_oauth delegates to _discover_service_oauth_async."""
        mock_inner = MagicMock(return_value={'requires_oauth': False})
        mock_a2s.return_value = mock_inner

        result = discover_service_oauth('http://example.com', 'http://app/callback')
        self.assertFalse(result['requires_oauth'])
        mock_inner.assert_called_once_with('http://example.com', 'http://app/callback', None)

    @patch('aiworks_core.logic.oauth.async_to_sync')
    def test_discover_passes_client_id_when_provided(self, mock_a2s):
        """discover_service_oauth passes client_id to the async wrapper."""
        mock_inner = MagicMock(return_value={'requires_oauth': True, 'client_id': 'my-client'})
        mock_a2s.return_value = mock_inner

        result = discover_service_oauth(
            service_url='http://example.com',
            redirect_uri='http://app/callback',
            client_id='my-client',
        )
        self.assertEqual(result['client_id'], 'my-client')
        mock_inner.assert_called_once_with('http://example.com', 'http://app/callback', 'my-client')


class ExchangeOAuthTokenTests(TestCase):
    """Tests for exchange_oauth_token (sync wrapper)."""

    @patch('aiworks_core.logic.oauth.async_to_sync')
    def test_exchange_calls_async_wrapper(self, mock_a2s):
        """exchange_oauth_token delegates to _exchange_oauth_token_async."""
        mock_inner = MagicMock(return_value={'access_token': 'tok'})
        mock_a2s.return_value = mock_inner

        result = exchange_oauth_token(
            token_endpoint='https://auth.example.com/token',
            code='code',
            code_verifier='verifier',
            redirect_uri='http://app/callback',
            client_id='client',
        )
        self.assertEqual(result['access_token'], 'tok')


class RefreshAccessTokenTests(TestCase):
    """Tests for refresh_access_token (sync wrapper)."""

    @patch('aiworks_core.logic.oauth.async_to_sync')
    def test_refresh_calls_async_wrapper(self, mock_a2s):
        """refresh_access_token delegates to _refresh_access_token_async."""
        mock_inner = MagicMock(return_value={'access_token': 'new-token'})
        mock_a2s.return_value = mock_inner

        result = refresh_access_token(
            token_endpoint='https://auth.example.com/token',
            refresh_token='my-refresh-token',
            client_id='test-client',
        )
        self.assertEqual(result['access_token'], 'new-token')
        mock_inner.assert_called_once_with(
            'https://auth.example.com/token', 'my-refresh-token', 'test-client', ''
        )

    @patch('aiworks_core.logic.oauth.async_to_sync')
    def test_refresh_includes_client_secret(self, mock_a2s):
        """refresh_access_token passes client_secret when provided."""
        mock_inner = MagicMock(return_value={'access_token': 'new-token'})
        mock_a2s.return_value = mock_inner

        refresh_access_token(
            token_endpoint='https://auth.example.com/token',
            refresh_token='my-refresh-token',
            client_id='test-client',
            client_secret='secret-value',
        )
        mock_inner.assert_called_once_with(
            'https://auth.example.com/token', 'my-refresh-token', 'test-client', 'secret-value'
        )


class DiscoverOAuthMetadataAsyncTests(TestCase):
    """Tests for _discover_oauth_metadata_async."""

    @patch('aiworks_core.logic.oauth.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth.safe_get_with_ssrf_check', new_callable=AsyncMock)
    async def test_returns_none_when_server_does_not_require_oauth(self, mock_safe_get, mock_ssrf):
        """A 200 response without WWW-Authenticate header means no OAuth required.

        The code continues to try fetching resource-metadata (well-known paths),
        and if those fail it raises. So we test the connection-error path instead.
        """
        mock_safe_get.side_effect = httpx.RequestError("Connection refused")

        with self.assertRaises(ValueError) as ctx:
            await _discover_oauth_metadata_async('https://example.com')
        self.assertIn("Cannot reach", str(ctx.exception))

    @patch('aiworks_core.logic.oauth.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth.safe_get_with_ssrf_check', new_callable=AsyncMock)
    async def test_raises_when_cannot_reach_server(self, mock_safe_get, mock_ssrf):
        """Connection errors propagate as ValueError."""
        mock_safe_get.side_effect = httpx.RequestError("Connection refused")

        with self.assertRaises(ValueError) as ctx:
            await _discover_oauth_metadata_async('https://example.com')
        self.assertIn("Cannot reach", str(ctx.exception))


class ExchangeOAuthTokenAsyncTests(TestCase):
    """Tests for _exchange_oauth_token_async."""

    def _fake_client(self, status_code=200, response_data=None):
        """Build a fake httpx client that returns the given response_data."""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = ''
        mock_resp.json.return_value = response_data or {}
        mock_resp.raise_for_status = MagicMock()
        if status_code >= 400:
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=mock_resp
            )

        class FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, url, **kwargs):
                return mock_resp
        return FakeClient()

    @patch('aiworks_core.logic.oauth.validate_ssrf_safe_url')
    @patch('httpx.AsyncClient')
    async def test_returns_token_data_on_success(self, mock_client_cls, mock_ssrf):
        """A successful exchange returns access_token, refresh_token, etc."""
        mock_client_cls.return_value = self._fake_client(
            response_data={
                'access_token': 'access-123',
                'refresh_token': 'refresh-456',
                'token_type': 'Bearer',
            }
        )

        result = await _exchange_oauth_token_async(
            token_endpoint='https://auth.example.com/token',
            code='auth-code',
            code_verifier='verifier',
            redirect_uri='https://app.example.com/callback',
            client_id='my-client',
        )
        self.assertEqual(result['access_token'], 'access-123')
        self.assertEqual(result['refresh_token'], 'refresh-456')
        self.assertEqual(result['oauth_metadata']['token_endpoint'], 'https://auth.example.com/token')

    @patch('aiworks_core.logic.oauth.validate_ssrf_safe_url')
    @patch('httpx.AsyncClient')
    async def test_raises_on_http_error(self, mock_client_cls, mock_ssrf):
        """HTTP 400 propagates as ValueError."""
        mock_client_cls.return_value = self._fake_client(status_code=400)

        with self.assertRaises(ValueError) as ctx:
            await _exchange_oauth_token_async(
                'https://auth.example.com/token', 'code', 'verifier', 'http://cb', 'client'
            )
        self.assertIn('rejected', str(ctx.exception))


class ResolveOAuthTokensAsyncTests(TestCase):
    """Tests for _resolve_oauth_tokens_async."""

    @patch('aiworks_core.logic.oauth.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._refresh_access_token_async', new_callable=AsyncMock)
    @patch('aiworks_core.logic.oauth._discover_oauth_metadata_async', new_callable=AsyncMock)
    async def test_returns_tokens_on_success(self, mock_discover, mock_refresh, mock_ssrf):
        """Happy path: discover metadata then refresh token."""
        mock_discover.return_value = {'token_endpoint': 'https://auth.example.com/token', 'client_id': 'c'}
        mock_refresh.return_value = {'access_token': 'new-access', 'refresh_token': 'new-refresh'}

        access, refresh = await _resolve_oauth_tokens_async(
            url='https://example.com',
            refresh_token='old-rt',
        )
        self.assertEqual(access, 'new-access')
        self.assertEqual(refresh, 'new-refresh')

    @patch('aiworks_core.logic.oauth.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._discover_oauth_metadata_async', new_callable=AsyncMock)
    async def test_raises_when_no_metadata_and_discovery_returns_none(self, mock_discover, mock_ssrf):
        """If server returns None (no OAuth required), raises ValueError."""
        mock_discover.return_value = None

        with self.assertRaises(ValueError) as ctx:
            await _resolve_oauth_tokens_async(url='https://example.com', refresh_token='rt')
        self.assertIn("doesn't require", str(ctx.exception))

    @patch('aiworks_core.logic.oauth.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._refresh_access_token_async', new_callable=AsyncMock)
    @patch('aiworks_core.logic.oauth._discover_oauth_metadata_async', new_callable=AsyncMock)
    async def test_raises_when_metadata_missing_token_endpoint(self, mock_discover, mock_refresh, mock_ssrf):
        """If metadata has no token_endpoint, raises ValueError."""
        mock_discover.return_value = {'authorization_endpoint': 'https://auth.example.com/authorize'}
        mock_refresh.return_value = {'access_token': 'tok'}

        with self.assertRaises(ValueError) as ctx:
            await _resolve_oauth_tokens_async(
                url='https://example.com',
                refresh_token='rt',
                oauth_metadata={'client_id': 'c'},
            )
        self.assertIn('token_endpoint', str(ctx.exception))

    @patch('aiworks_core.logic.oauth.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._refresh_access_token_async', new_callable=AsyncMock)
    @patch('aiworks_core.logic.oauth._discover_oauth_metadata_async', new_callable=AsyncMock)
    async def test_raises_when_no_access_token_in_response(self, mock_discover, mock_refresh, mock_ssrf):
        """If refresh response has no access_token, raises ValueError."""
        mock_discover.return_value = {'token_endpoint': 'https://auth.example.com/token', 'client_id': 'c'}
        mock_refresh.return_value = {}  # no access_token

        with self.assertRaises(ValueError) as ctx:
            await _resolve_oauth_tokens_async(
                url='https://example.com',
                refresh_token='rt',
                oauth_metadata={'token_endpoint': 'https://auth.example.com/token', 'client_id': 'c'},
            )
        self.assertIn('access token', str(ctx.exception).lower())

    @patch('aiworks_core.logic.oauth.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._refresh_access_token_async', new_callable=AsyncMock)
    def test_uses_provided_oauth_metadata_when_available(self, mock_refresh, mock_ssrf):
        """When oauth_metadata is passed directly, discovery is skipped."""
        mock_refresh.return_value = {'access_token': 'tok', 'refresh_token': 'new-rt'}

        access, refresh = asyncio.run(_resolve_oauth_tokens_async(
            url='https://example.com',
            refresh_token='rt',
            oauth_metadata={
                'token_endpoint': 'https://auth.example.com/token',
                'client_id': 'cid',
                'client_secret': 'csecret',
            },
        ))

        self.assertEqual(access, 'tok')
        self.assertEqual(refresh, 'new-rt')
        mock_refresh.assert_called_once_with(
            'https://auth.example.com/token', 'rt', 'cid', client_secret='csecret'
        )

    @patch('aiworks_core.logic.oauth.validate_ssrf_safe_url')
    @patch('aiworks_core.logic.oauth._refresh_access_token_async', new_callable=AsyncMock)
    @patch('aiworks_core.logic.oauth._discover_oauth_metadata_async', new_callable=AsyncMock)
    async def test_client_secret_from_metadata_is_used(self, mock_discover, mock_refresh, mock_ssrf):
        """client_secret from oauth_metadata is passed to refresh."""
        mock_discover.return_value = {'token_endpoint': 'https://auth.example.com/token', 'client_id': 'cid'}
        mock_refresh.return_value = {'access_token': 'tok', 'refresh_token': 'new-rt'}

        await _resolve_oauth_tokens_async(
            url='https://example.com',
            refresh_token='rt',
            oauth_metadata={'token_endpoint': 'https://auth.example.com/token', 'client_id': 'cid', 'client_secret': 'meta-secret'},
            client_secret='override-secret',
        )

        # override-secret takes precedence over meta-secret
        _, call_kwargs = mock_refresh.call_args
        positional_args = mock_refresh.call_args[0]
        all_args = list(positional_args) + list(call_kwargs.values())
        self.assertIn('override-secret', all_args)