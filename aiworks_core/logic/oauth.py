"""Generic OAuth 2.1 / PKCE helpers for external integrations (e.g., LinkedIn).

"""
import logging
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx
from ..utils import async_to_sync, raise_for_status
from .logic_utils import base_url, validate_ssrf_safe_url, safe_get_with_ssrf_check

logger = logging.getLogger(__name__)

_OAUTH_PROBE_TIMEOUT = 10
_OAUTH_REGISTER_TIMEOUT = 15


async def _resolve_oauth_tokens_async(url: str,
                                      refresh_token: str = "",
                                      oauth_metadata: Optional[dict] = None,
                                      client_secret: str = ""):
    as_metadata = oauth_metadata if oauth_metadata else {}

    # If we don't have the token_endpoint in this stage, we have to discover it
    if not as_metadata.get('token_endpoint'):
        as_metadata = await _discover_oauth_metadata_async(url)

    if as_metadata:
        client_id: str = as_metadata.get("client_id") or "aiworks-client"
        effective_client_secret = client_secret or as_metadata.get("client_secret") or ""
        token_endpoint: str | None = as_metadata.get("token_endpoint")
        if not token_endpoint:
            raise ValueError(
                "OAuth authorization-server metadata does not contain a token_endpoint."
            )
        validate_ssrf_safe_url(token_endpoint)
        token_data = await _refresh_access_token_async(
            token_endpoint, refresh_token, client_id, client_secret=effective_client_secret
        )
        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError(
                "OAuth token refresh succeeded but the server returned no access token."
            )
        new_refresh_token = token_data.get("refresh_token") or refresh_token
        return access_token, new_refresh_token
    raise ValueError("Server doesn't require Oauth authentication")


def resolve_oauth_tokens(url: str,
                         refresh_token: str = "",
                         oauth_metadata: Optional[dict] = None,
                         client_secret: str = ""):
    """Synchronous wrapper around resolve_oauth_tokens_async."""
    return async_to_sync(_resolve_oauth_tokens_async)(url, refresh_token, oauth_metadata, client_secret)



async def _refresh_access_token_async(
        token_endpoint: str,
        refresh_token: str,
        client_id: str,
        client_secret: str = "",
) -> Dict[str, Any]:
    """
    Use a refresh token to obtain a new access token.
    Returns the parsed token response dict.
    """
    validate_ssrf_safe_url(token_endpoint)

    if not refresh_token:
        raise ValueError("Refresh token is missing")

    payload: Dict[str, Any] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    if client_secret:
        payload["client_secret"] = client_secret

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(token_endpoint, data=payload)
            logger.info("[OAuth] Token refresh POST to %s — HTTP %d body: %s", token_endpoint, resp.status_code,
                        resp.text[:500])
            raise_for_status(resp)
        except httpx.RequestError as exc:
            logger.warning("Token refresh request failed: %s", exc)
            raise ValueError("Token refresh request failed — check network connectivity.") from exc
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Token refresh returned HTTP %d: %s",
                exc.response.status_code,
                exc.response.text[:400],
            )
            raise ValueError(
                f"Token refresh was rejected by the authorization server "
                f"(HTTP {exc.response.status_code})."
            ) from exc
        return resp.json()


def refresh_access_token(
        token_endpoint: str,
        refresh_token: str,
        client_id: str,
        client_secret: str = "",
) -> Dict[str, Any]:
    """Synchronous wrapper around _refresh_access_token_async."""
    return async_to_sync(_refresh_access_token_async)(token_endpoint, refresh_token, client_id, client_secret)


async def _discover_oauth_metadata_async(service_url: str) -> Optional[Dict[str, Any]]:
    """
    Probe *service_url* without credentials to detect whether it is protected by
    OAuth 2.1.  If it is, fetch and return the authorization-server metadata
    document.  Returns ``None`` if the server does not require OAuth.

    Discovery order (follows the MCP 2025-03-26 spec):
    1. GET the MCP URL — if the response is not 401, no OAuth is required.
    2. Parse the ``WWW-Authenticate`` header for a ``resource_metadata`` URI.
    3. Fetch that URI (or fall back to ``{base_url}/.well-known/oauth-protected-resource``).
    4. Follow the ``authorization_servers`` list from the resource-metadata doc
       (or fall back to ``{base_url}/.well-known/oauth-authorization-server``).
    5. Return the parsed authorization-server metadata dict.

    ``follow_redirects=False`` is intentional; each redirect target is
    re-validated by ``safe_get_with_ssrf_check`` before following to
    prevent SSRF via server-controlled 302 responses.
    """
    async with httpx.AsyncClient(follow_redirects=False, timeout=_OAUTH_PROBE_TIMEOUT) as client:
        try:
            resp = await safe_get_with_ssrf_check(client, service_url)
        except httpx.RequestError as exc:
            logger.warning("Cannot probe service server '%s' for OAuth: %s", service_url, exc)
            raise ValueError("Cannot reach service server — check the URL and try again.") from exc

        # --- Step 2: read WWW-Authenticate for resource_metadata ---
        www_auth = resp.headers.get("WWW-Authenticate", "")
        resource_metadata_url: Optional[str] = None
        # Parse simple `key="value"` pairs; handles the common Bearer challenge format.
        for token in www_auth.split(","):
            key_val = token.strip()
            if "=" in key_val:
                key, _, val = key_val.partition("=")
                if key.strip().lower() == "resource_metadata":
                    resource_metadata_url = val.strip().strip('"')
                    break

        base = base_url(service_url)

        # --- Step 3: fetch resource-metadata document ---
        if resource_metadata_url:
            try:
                validate_ssrf_safe_url(resource_metadata_url)
                rm_resp = await safe_get_with_ssrf_check(client, resource_metadata_url)
                raise_for_status(rm_resp)
                resource_metadata = rm_resp.json()
            except Exception:
                resource_metadata = None
        else:
            protected_resource_metadata_url = urljoin(
                base.rstrip("/") + "/",
                ".well-known/oauth-protected-resource",
            )
            try:
                validate_ssrf_safe_url(protected_resource_metadata_url)
                rm_resp = await safe_get_with_ssrf_check(client, protected_resource_metadata_url)
                raise_for_status(rm_resp)
                resource_metadata = rm_resp.json()
            except Exception:
                resource_metadata = None

        # --- Step 4: determine authorization-server metadata URL ---
        auth_server_url: Optional[str] = None
        if resource_metadata:
            auth_servers = resource_metadata.get("authorization_servers") or []
            if auth_servers:
                auth_server_url = auth_servers[0]

        if not auth_server_url:
            auth_server_url = base

        # Try both RFC 8414 paths: /.well-known/oauth-authorization-server and
        # the older /.well-known/openid-configuration (some implementations use it).
        # Use only the scheme+host from auth_server_url so the well-known path is
        # always relative to the root, not to any path component in auth_server_url
        # (e.g. https://mcp.example.com/auth  →  https://mcp.example.com/.well-known/...)
        auth_origin = base_url(auth_server_url)
        for well_known_path in (
                ".well-known/oauth-authorization-server",
                ".well-known/openid-configuration",
        ):
            candidate = urljoin(auth_origin.rstrip("/") + "/", well_known_path)
            try:
                validate_ssrf_safe_url(candidate)
                as_resp = await safe_get_with_ssrf_check(client, candidate)
                if as_resp.status_code == 200:
                    return as_resp.json()
            except Exception:
                continue

        raise ValueError(
            "Service server requires OAuth but authorization-server metadata could not be discovered."
        )


async def _register_oauth_client_async(
        registration_endpoint: str,
        redirect_uri: str,
        client_name: str = "aiworks-client",
) -> str:
    """
    Perform RFC 7591 Dynamic Client Registration.
    Returns the newly issued ``client_id``.
    """
    validate_ssrf_safe_url(registration_endpoint)
    payload = {
        "client_name": client_name,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",  # public client (PKCE)
    }
    async with httpx.AsyncClient(timeout=_OAUTH_REGISTER_TIMEOUT) as client:
        try:
            resp = await client.post(registration_endpoint, json=payload)
            raise_for_status(resp)
        except httpx.RequestError as exc:
            logger.warning("Dynamic client registration request failed: %s", exc)
            raise ValueError(
                "Dynamic client registration request failed — check network connectivity."
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Dynamic client registration returned HTTP %d: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            raise ValueError(
                f"Dynamic client registration was rejected (HTTP {exc.response.status_code})."
            ) from exc
        data = resp.json()
        client_id = data.get("client_id")
        if not client_id:
            raise ValueError(
                "Dynamic client registration did not return a client_id."
            )
        return client_id


async def _discover_service_oauth_async(
        service_url: str,
        redirect_uri: str,
        client_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full OAuth discovery: probe *service_url*, fetch authorization-server metadata,
    optionally perform dynamic client registration, and return everything the
    frontend needs to start a PKCE authorization flow.

    When *client_id* is provided dynamic client registration is skipped entirely.

    Returns a dict with keys:
    - ``requires_oauth`` (bool)
    - ``authorization_endpoint`` (str, present when ``requires_oauth`` is True)
    - ``token_endpoint`` (str, present when ``requires_oauth`` is True)
    - ``client_id`` (str, present when ``requires_oauth`` is True)
    - ``scope`` (str, may be empty)
    """
    validate_ssrf_safe_url(service_url)
    as_metadata = await _discover_oauth_metadata_async(service_url)
    if as_metadata is None:
        return {"requires_oauth": False}

    authorization_endpoint = as_metadata.get("authorization_endpoint")
    token_endpoint = as_metadata.get("token_endpoint")
    if not authorization_endpoint or not token_endpoint:
        raise ValueError(
            "Authorization-server metadata is missing required endpoints "
            "(both authorization_endpoint and token_endpoint are required)."
        )

    if client_id:
        # Pre-registered client credentials supplied  —
        # skip dynamic client registration to avoid creating unnecessary client entries.
        resolved_client_id = client_id
    else:
        registration_endpoint: str | None = as_metadata.get("registration_endpoint")
        if registration_endpoint:
            resolved_client_id = await _register_oauth_client_async(
                registration_endpoint, redirect_uri
            )
        else:
            # Use a stable default client ID for servers without dynamic registration.
            resolved_client_id = "aiworks-client"

    scope = as_metadata.get("scopes_supported", [])
    scope_str = " ".join(scope) if isinstance(scope, list) else (scope or "")

    return {
        "requires_oauth": True,
        "authorization_endpoint": authorization_endpoint,
        "token_endpoint": token_endpoint,
        "client_id": resolved_client_id,
        "scope": scope_str,
    }


def discover_service_oauth(service_url: str, redirect_uri: str, client_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronous wrapper around ``_discover_service_oauth_async``.
    Raises ``ValueError`` on any failure.
    """
    return async_to_sync(_discover_service_oauth_async)(service_url, redirect_uri, client_id)


async def _exchange_oauth_token_async(
        token_endpoint: str,
        code: str,
        code_verifier: Optional[str],
        redirect_uri: str,
        client_id: str,
        client_secret: str = "",
) -> Dict[str, Any]:
    """
    Exchange an authorization *code* for tokens using the PKCE ``code_verifier``.
    Returns the full token response dict (access_token, token_type, etc.).
    """
    validate_ssrf_safe_url(token_endpoint)
    payload: Dict[str, Any] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
    }

    if code_verifier:
        payload["code_verifier"] = code_verifier

    if client_secret:
        payload["client_secret"] = client_secret

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(token_endpoint, data=payload)
            raise_for_status(resp)
        except httpx.RequestError as exc:
            logger.warning("Token exchange request failed: %s", exc)
            raise ValueError("Token exchange request failed — check network connectivity.") from exc
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Token exchange returned HTTP %d: %s",
                exc.response.status_code,
                exc.response.text[:400],
            )
            raise ValueError(
                f"Token exchange was rejected by the authorization server "
                f"(HTTP {exc.response.status_code})."
            ) from exc
        token_data = resp.json()

        # Build the oauth_metadata that.
        # record and fetched server-side at token-exchange time.
        token_data["oauth_metadata"] = {
            "token_endpoint": token_endpoint,
            "client_id": client_id,
            "client_secret": client_secret
        }
        return token_data


def exchange_oauth_token(
        token_endpoint: str,
        code: str,
        code_verifier: Optional[str],
        redirect_uri: str,
        client_id: str,
        client_secret: str = "",
) -> Dict[str, Any]:
    """
    Synchronous wrapper around ``_exchange_oauth_token_async``.
    Raises ``ValueError`` on any failure.
    """
    return async_to_sync(_exchange_oauth_token_async)(
        token_endpoint, code, code_verifier, redirect_uri, client_id, client_secret
    )