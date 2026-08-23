import logging
from functools import lru_cache
from typing import Awaitable
from urllib.parse import urlparse

from django.core.exceptions import DisallowedHost
from django.http import HttpRequest, HttpResponseBase
from django.middleware.csrf import CsrfViewMiddleware

from .models import SiteConfiguration

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_csrf_trusted_origins():
    """Return cached CSRF trusted origins from SiteConfiguration.

    Uses @lru_cache (no TTL) — cached for the entire process lifetime.
    This is safe because CSRF origins change extremely rarely (only when
    an admin updates SiteConfiguration in Django admin). On any config change,
    the Django process must be restarted anyway to pick up the new value.
    """
    trusted_origin = SiteConfiguration.get_solo().csrf_trusted_origins.strip()
    logger.debug(f"Retrieved CSRF trusted origins from SiteConfiguration: {trusted_origin}")
    return trusted_origin


# noinspection PyUnresolvedReferences,PyProtectedMember,PyBroadException
class DynamicCsrfMiddleware(CsrfViewMiddleware):
    """
    Drop-in replacement for CsrfViewMiddleware that reads trusted origins from
    SiteConfiguration rather than from settings.CSRF_TRUSTED_ORIGINS.

    - When ``csrf_trusted_origins`` is blank (the default), the standard Django
      CSRF origin check is applied (same-origin only).
    - When it contains a comma-separated list of origins, only those origins
      (plus the request's own host) pass the origin check.  Wildcard subdomain
      patterns such as ``https://*.example.com`` are supported.
    - Setting ``csrf_trusted_origins`` to ``*`` trusts **all** origins and
      effectively disables CSRF origin enforcement.  Use this only in controlled
      environments (e.g. behind a reverse proxy that enforces origin checks) and
      be aware of the security implications.
    """

    def _origin_verified(self, request):

        try:
            raw = _get_csrf_trusted_origins()
        except Exception as exp:
            logger.warning(f"Error retrieving SiteConfiguration: {exp}")
            # DB not yet available (e.g. during initial migrations)
            return super()._origin_verified(request)

        if not raw:
            logger.debug("No CSRF trusted origins configured; using default Django behavior.")
            # No restriction configured
            return super()._origin_verified(request)

        if raw == '*':
            logger.debug("CSRF trusted origins set to '*'; all origins are trusted.")
            # All origins are trusted
            return True

        # Same-origin is always trusted regardless of the configured list.
        request_origin = request.META.get('HTTP_ORIGIN', '')
        logger.debug(f"Request origin: {request_origin}")
        try:
            good_host = request.get_host()
        except DisallowedHost:
            logger.debug("DisallowedHost error when getting request host; treating as untrusted origin.")
            pass
        else:
            logger.debug(f"Request host: {good_host}")
            good_origin = '{}://{}'.format(
                'https' if request.is_secure() else 'http',
                good_host,
            )
            logger.debug(f"Request origin: {request_origin}, Good origin: {good_origin}")
            if request_origin == good_origin:
                logger.debug("Request origin matches request host; same-origin trusted.")
                return True

        trusted_list = [o.strip() for o in raw.split(',') if o.strip()]
        logger.debug(f"CSRF trusted origins list: {trusted_list}")

        # Exact match.
        if request_origin in trusted_list:
            logger.debug("Request origin matches trusted origins list; origin trusted.")
            return True

        # Wildcard subdomain match (e.g. ``https://*.example.com``).
        try:
            parsed_origin = urlparse(request_origin)
        except ValueError:
            logger.debug("Request origin is not a valid URL; treating as untrusted origin.")
            return False

        for trusted in trusted_list:
            try:
                parsed_trusted = urlparse(trusted)
            except ValueError:
                continue
            netloc = parsed_trusted.netloc
            if netloc.startswith('*.') and parsed_origin.scheme == parsed_trusted.scheme:
                domain = netloc[2:]
                if parsed_origin.netloc == domain or parsed_origin.netloc.endswith('.' + domain):
                    logger.debug(
                        f"Request origin {request_origin} matches wildcard trusted origin {trusted}; origin trusted.")
                    return True

        logger.debug("Request origin does not match any trusted origins; origin not trusted.")
        return False

    def __call__(self, request: HttpRequest) -> HttpResponseBase | Awaitable[HttpResponseBase]:
        logger.debug("DynamicCsrfMiddleware: processing request %s", request.path)
        return super().__call__(request)
