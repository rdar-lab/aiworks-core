"""
Middleware tests for the Ai-Works Core API.
"""

from unittest.mock import patch

from django.test import RequestFactory
from django.test import TestCase

from ..middleware import DynamicCsrfMiddleware, _get_csrf_trusted_origins
from ..models import (
    SiteConfiguration,
)


class DynamicCsrfMiddlewareTests(TestCase):
    """Tests for DynamicCsrfMiddleware._origin_verified."""

    def setUp(self):
        # noinspection PyTypeChecker
        self.middleware = DynamicCsrfMiddleware(get_response=lambda r: None)
        # Ensure a clean SiteConfiguration record for each test.
        self.config = SiteConfiguration.get_solo()
        self.config.csrf_trusted_origins = ''
        self.config.save()
        # lru_cache persists across tests — clear it so each test reads from DB.
        _get_csrf_trusted_origins.cache_clear()

    def _set_origins(self, value: str):
        """Save a new csrf_trusted_origins value and clear the lru_cache."""
        self.config.csrf_trusted_origins = value
        self.config.save()
        _get_csrf_trusted_origins.cache_clear()

    @staticmethod
    def _make_request(origin, host='example.com', secure=False):
        factory = RequestFactory()
        request = factory.get('/', HTTP_ORIGIN=origin, SERVER_NAME=host)
        request.META['SERVER_PORT'] = '443' if secure else '80'
        if secure:
            request.META['wsgi.url_scheme'] = 'https'
        return request

    # -- blank config: falls back to Django defaults ---------------------

    def test_blank_config_rejects_cross_origin_request(self):
        """With no trusted origins configured, cross-origin requests are rejected (Django default)."""
        self._set_origins('')
        request = self._make_request('https://anything.ngrok-free.app')
        self.assertFalse(self.middleware._origin_verified(request))

    def test_whitespace_only_config_rejects_cross_origin_request(self):
        """Whitespace-only value is treated the same as blank — cross-origin rejected."""
        self._set_origins('   ')
        request = self._make_request('https://random.example.com')
        self.assertFalse(self.middleware._origin_verified(request))

    # -- same-origin is always trusted ------------------------------------

    def test_same_origin_always_trusted(self):
        """The server's own origin passes even when not on the trusted list."""
        self._set_origins('https://other.example.com')
        # Use 'localhost' which is in ALLOWED_HOSTS; the request comes from the
        # same scheme+host Django is serving, so it must be trusted.
        request = self._make_request('http://localhost', host='localhost')
        self.assertTrue(self.middleware._origin_verified(request))

    # -- exact-match list -------------------------------------------------

    def test_exact_match_in_trusted_list(self):
        """An origin that is in the comma-separated list is allowed."""
        self._set_origins('https://trusted.example.com,https://other.example.com')
        request = self._make_request('https://trusted.example.com')
        self.assertTrue(self.middleware._origin_verified(request))

    def test_untrusted_origin_is_rejected(self):
        """An origin NOT in the trusted list (and not same-origin) is rejected."""
        self._set_origins('https://trusted.example.com')
        request = self._make_request('https://evil.example.com')
        self.assertFalse(self.middleware._origin_verified(request))

    def test_list_with_extra_whitespace_is_parsed_correctly(self):
        """Spaces around entries in the comma-separated list are stripped."""
        self._set_origins('  https://trusted.example.com  ,  https://other.example.com  ')
        request = self._make_request('https://other.example.com')
        self.assertTrue(self.middleware._origin_verified(request))

    def test_empty_entries_in_list_are_ignored(self):
        """Trailing/doubled commas do not cause errors."""
        self._set_origins('https://trusted.example.com,,')
        request = self._make_request('https://trusted.example.com')
        self.assertTrue(self.middleware._origin_verified(request))

    # -- wildcard subdomain matching --------------------------------------

    def test_wildcard_subdomain_match(self):
        """``https://*.example.com`` matches any subdomain of example.com."""
        self._set_origins('https://*.example.com')
        request = self._make_request('https://sub.example.com')
        self.assertTrue(self.middleware._origin_verified(request))

    def test_wildcard_matches_apex_domain(self):
        """``https://*.example.com`` also matches the bare apex domain."""
        self._set_origins('https://*.example.com')
        request = self._make_request('https://example.com')
        self.assertTrue(self.middleware._origin_verified(request))

    def test_wildcard_does_not_match_different_scheme(self):
        """Wildcard entry for https does not match an http origin."""
        self._set_origins('https://*.example.com')
        # noinspection HttpUrlsUsage
        request = self._make_request('http://sub.example.com')
        self.assertFalse(self.middleware._origin_verified(request))

    def test_wildcard_does_not_match_different_domain(self):
        """Wildcard entry does not match a completely different domain."""
        self._set_origins('https://*.example.com')
        request = self._make_request('https://evil.notexample.com')
        self.assertFalse(self.middleware._origin_verified(request))

    # -- DB unavailable fallback -----------------------------------------

    def test_db_error_falls_back_to_django_default(self):
        """If the DB raises an exception, falls back to Django's default check."""
        with patch('aiworks_core.models.SiteConfiguration.get_solo', side_effect=Exception('DB down')):
            # Same-origin should still pass via the Django default path
            request = self._make_request('http://localhost', host='localhost')
            self.assertTrue(self.middleware._origin_verified(request))
