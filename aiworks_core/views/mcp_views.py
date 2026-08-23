import logging
from typing import cast

from django.db import IntegrityError, transaction
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.exceptions import (
    PermissionDenied,
    ValidationError as DRFValidationError,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .views_utils import require_pro
from ..models import (
    User,
    MCPServer,
    PredefinedMCPServer,
    SiteConfiguration,
)
from ..serializers import (
    MCPServerSerializer,
    PredefinedMCPServerSerializer,
)

logger = logging.getLogger(__name__)


class PredefinedMCPServerViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only ViewSet exposing the admin-managed catalogue of predefined MCP servers."""

    serializer_class = PredefinedMCPServerSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return PredefinedMCPServer.objects.all()


class MCPServerViewSet(viewsets.ModelViewSet):
    """ViewSet for managing user MCP server registrations"""

    serializer_class = MCPServerSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return MCPServer.objects.filter(user=self.request.user)

    @staticmethod
    def _inject_predefined_oauth_fields(
        predefined_server: PredefinedMCPServer | None,
        oauth_metadata: dict,
    ) -> dict:
        """Inject predefined server OAuth fields into oauth_metadata.

        Injects client_id, authorization_endpoint, and token_endpoint.
        Existing metadata values are not overwritten.
        scope is NOT injected — it is not used in token refresh requests.
        """
        if predefined_server is None:
            return oauth_metadata
        metadata = dict(oauth_metadata or {})
        if predefined_server.client_id:
            metadata.setdefault("client_id", predefined_server.client_id)
        if predefined_server.authorization_endpoint:
            metadata.setdefault("authorization_endpoint", predefined_server.authorization_endpoint)
        if predefined_server.token_endpoint:
            metadata.setdefault("token_endpoint", predefined_server.token_endpoint)
        return metadata

    def perform_create(self, serializer):
        from ..logic.mcp_tools import validate_mcp_connection, validate_ssrf_safe_url, resolve_auth_headers, \
            resolve_oauth_tokens, validate_rest_api_connection

        require_pro(cast(User, self.request.user), "mcp_server")

        predefined_server: PredefinedMCPServer | None = serializer.validated_data.get("predefined_server")

        if predefined_server is not None:
            # When creating from a predefined server, inherit URL, auth_type, headers and name.
            url = predefined_server.url
            headers = predefined_server.headers or {}
            auth_type = predefined_server.auth_type
            name = predefined_server.name
            server_type = predefined_server.server_type
            openapi_spec_url = predefined_server.openapi_spec_url or ""
            openapi_spec = predefined_server.openapi_spec or ""
        else:
            url = serializer.validated_data.get("url", "")
            headers = serializer.validated_data.get("headers") or {}
            auth_type = serializer.validated_data.get("auth_type", "none")
            name = serializer.validated_data.get("name", "")
            server_type = serializer.validated_data.get("server_type", "http")
            openapi_spec_url = serializer.validated_data.get("openapi_spec_url", "")
            openapi_spec = serializer.validated_data.get("openapi_spec", "")

        username = serializer.validated_data.get("username", "")
        password = serializer.validated_data.get("password", "")
        token = serializer.validated_data.get("token", "")
        refresh_token = serializer.validated_data.get("refresh_token", "")
        oauth_metadata = self._inject_predefined_oauth_fields(
            predefined_server,
            serializer.validated_data.get("oauth_metadata") or {},
        )

        try:
            validate_ssrf_safe_url(url)
        except ValueError as exc:
            logger.info(
                "SSRF validation blocked MCP server create for url=%r: %s", url, exc
            )
            raise DRFValidationError(
                {"error": "Hostname or IP is forbidden to be used for MCP server"}
            )
        if auth_type == "oauth":
            try:
                token, refresh_token = resolve_oauth_tokens(url, refresh_token, oauth_metadata=oauth_metadata, client_secret=predefined_server.client_secret if predefined_server else "")
            except Exception as exc:
                logger.info(
                    "OAuth token refresh failed during server registration (url=%r): %s", url, exc
                )
                raise DRFValidationError(
                    {
                        "error": (
                            "OAuth token refresh failed — the refresh token may have expired. "
                            "Please re-authenticate."
                        )
                    }
                ) from exc

        effective_headers = resolve_auth_headers(
            headers, auth_type, username, password, token
        )
        try:
            if server_type == "openapi":
                validate_rest_api_connection(url, effective_headers, openapi_spec_url, openapi_spec)
            else:
                validate_mcp_connection(url, effective_headers)
        except ValueError:
            raise DRFValidationError({"error": "Could not connect to MCP server"})
        try:
            with transaction.atomic():
                instance = serializer.save(
                    user=self.request.user,
                    url=url,
                    headers=headers,
                    auth_type=auth_type,
                    name=name,
                    server_type=server_type,
                    openapi_spec_url=openapi_spec_url,
                    openapi_spec=openapi_spec,
                )
                instance.oauth_metadata = oauth_metadata
                instance.token = token
                instance.refresh_token = refresh_token
                instance.save(update_fields=["oauth_metadata", "token", "refresh_token"])
        except IntegrityError:
            if (
                    name
                    and MCPServer.objects.filter(user=self.request.user, name=name).exists()
            ):
                raise DRFValidationError(
                    {"error": "An MCP server with this name already exists."}
                )
            raise

    def perform_update(self, serializer: MCPServerSerializer):
        from ..logic.mcp_tools import validate_mcp_connection, validate_ssrf_safe_url, resolve_auth_headers, \
            resolve_oauth_tokens, validate_rest_api_connection

        if serializer.instance.user != self.request.user:
            raise PermissionDenied("Cannot update another user's MCP server")

        predefined_server = serializer.instance.predefined_server

        if predefined_server is not None:
            # For predefined-derived servers, URL, headers, auth_type and name are locked.
            url = predefined_server.url
            headers = predefined_server.headers or {}
            auth_type = predefined_server.auth_type
            server_type = predefined_server.server_type
            openapi_spec_url = predefined_server.openapi_spec_url or ""
            openapi_spec = predefined_server.openapi_spec or ""
        else:
            url = serializer.validated_data.get("url", serializer.instance.url)
            headers = (
                    serializer.validated_data.get("headers")
                    or serializer.instance.headers
                    or {}
            )
            # Use updated auth fields or fall back to the stored values.
            # On PATCH, validate() removes empty strings from validated_data so that
            # .get(field, instance.field) correctly returns the existing stored credential
            # for any field the client did not change.
            auth_type = serializer.validated_data.get("auth_type", serializer.instance.auth_type)
            server_type = serializer.validated_data.get("server_type", serializer.instance.server_type)
            openapi_spec_url = serializer.validated_data.get("openapi_spec_url", serializer.instance.openapi_spec_url or "")
            openapi_spec = serializer.validated_data.get("openapi_spec", serializer.instance.openapi_spec or "")

        username = serializer.validated_data.get("username", serializer.instance.username)
        password = serializer.validated_data.get("password", serializer.instance.password)
        token = serializer.validated_data.get("token", serializer.instance.token)
        refresh_token = serializer.validated_data.get("refresh_token", serializer.instance.refresh_token)
        oauth_metadata = self._inject_predefined_oauth_fields(
            predefined_server,
            serializer.validated_data.get("oauth_metadata") or serializer.instance.oauth_metadata or {},
        )

        try:
            validate_ssrf_safe_url(url)
        except ValueError as exc:
            logger.info(
                "SSRF validation blocked MCP server update for url=%r: %s", url, exc
            )
            raise DRFValidationError(
                {"error": "Hostname or IP is forbidden to be used for MCP server"}
            )
        if auth_type == "oauth":
            try:
                token, refresh_token = resolve_oauth_tokens(url, refresh_token, oauth_metadata=oauth_metadata, client_secret=predefined_server.client_secret if predefined_server else "")
            except Exception as exc:
                logger.info(
                    "OAuth token refresh failed during server registration (url=%r): %s", url, exc
                )
                raise DRFValidationError(
                    {
                        "error": (
                            "OAuth token refresh failed — the refresh token may have expired. "
                            "Please re-authenticate."
                        )
                    }
                ) from exc

        effective_headers = resolve_auth_headers(
            headers, auth_type, username, password, token
        )
        try:
            if server_type == "openapi":
                validate_rest_api_connection(url, effective_headers, openapi_spec_url, openapi_spec)
            else:
                validate_mcp_connection(url, effective_headers)
        except ValueError:
            raise DRFValidationError({"error": "Could not connect to MCP server"})
        try:
            with transaction.atomic():
                save_kwargs: dict = dict(
                    user=self.request.user,
                    url=url,
                    headers=headers,
                    auth_type=auth_type,
                    server_type=server_type,
                    openapi_spec_url=openapi_spec_url,
                    openapi_spec=openapi_spec,
                )
                if predefined_server is not None:
                    save_kwargs["name"] = predefined_server.name
                instance = serializer.save(**save_kwargs)
                instance.oauth_metadata = oauth_metadata
                instance.token = token
                instance.refresh_token = refresh_token
                instance.save(update_fields=["oauth_metadata", "token", "refresh_token"])
        except IntegrityError:
            name = serializer.validated_data.get("name")
            if (
                    name
                    and MCPServer.objects.filter(user=self.request.user, name=name)
                    .exclude(pk=serializer.instance.pk)
                    .exists()
            ):
                raise DRFValidationError(
                    {"error": "An MCP server with this name already exists."}
                )
            raise

    def perform_destroy(self, instance: MCPServer):
        if instance.user != self.request.user:
            raise PermissionDenied("Cannot delete another user's MCP server")
        instance.delete()

    def destroy(self, request, *args, **kwargs):
        super().destroy(request, *args, **kwargs)
        return Response({"status": "OK"}, status=status.HTTP_200_OK)

    # ------------------------------------------------------------------
    # OAuth helpers
    # ------------------------------------------------------------------

    @action(detail=False, methods=["post"], url_path="discover-oauth")
    def discover_oauth(self, request):
        """
        Probe an MCP server URL to discover whether it requires OAuth 2.1
        authentication and, if so, return the information needed to start a
        PKCE authorization-code flow (authorization endpoint, token endpoint,
        client_id).

        Request body:
            {
                "url":          "<MCP server URL>",
                "redirect_uri": "<OAuth callback URL in the frontend>"
            }

        Response:
            {
                "requires_oauth": false
            }
        or:
            {
                "requires_oauth":        true,
                "authorization_endpoint": "...",
                "token_endpoint":         "...",
                "client_id":              "...",
                "scope":                  "..."   // may be empty
            }
        """
        from ..logic.logic_utils import validate_ssrf_safe_url
        from ..logic.oauth import discover_service_oauth

        require_pro(cast(User, request.user), "mcp_server")
        url = (request.data.get("url") or "").strip()
        redirect_uri = (request.data.get("redirect_uri") or "").strip()
        if not url:
            raise DRFValidationError({"error": "url is required"})
        if not redirect_uri:
            raise DRFValidationError({"error": "redirect_uri is required"})

        try:
            validate_ssrf_safe_url(url)
        except ValueError as exc:
            raise DRFValidationError(
                {"error": "Hostname or IP is forbidden to be used for MCP server"}
            ) from exc

        # If the caller specifies a predefined server, use its pre-registered client_id
        # so that dynamic client registration is skipped.
        predefined_client_id: str | None = None
        predefined_server_id = request.data.get("predefined_server_id")
        predefined: PredefinedMCPServer | None = None
        if predefined_server_id:
            try:
                predefined = PredefinedMCPServer.objects.get(pk=predefined_server_id)
                predefined_client_id = predefined.client_id or None
            except PredefinedMCPServer.DoesNotExist:
                pass

        # Skip discovery if predefined has authorization_endpoint OR token_endpoint
        if predefined and predefined.authorization_endpoint and predefined.token_endpoint:
            return Response({
                "requires_oauth": True,
                "authorization_endpoint": predefined.authorization_endpoint or "",
                "token_endpoint": predefined.token_endpoint or "",
                "client_id": predefined.client_id or "aiworks-client",
                "scope": predefined.scope or "",
                "custom_auth_params": predefined.custom_auth_params or {},
            })

        try:
            result = discover_service_oauth(url, redirect_uri, client_id=predefined_client_id)
        except ValueError as exc:
            logger.info("OAuth discovery failed for url=%r: %s", url, exc)
            raise DRFValidationError(
                {"error": "OAuth discovery failed for this server. Check the URL and try again."}
            ) from exc

        # Override scope if predefined has it (even when discovery ran)
        if predefined and predefined.scope:
            result["scope"] = predefined.scope

        # Include custom_auth_params if predefined has them (even if {})
        if predefined and predefined.custom_auth_params is not None:
            result["custom_auth_params"] = predefined.custom_auth_params

        return Response(result, status=status.HTTP_200_OK)

    @action(detail=False, methods=["post"], url_path="oauth-exchange")
    def oauth_exchange(self, request):
        """
        Exchange an OAuth authorization code (together with the PKCE
        ``code_verifier``) for an access token.

        The exchange is performed server-side so that the ``code_verifier``
        never has to leave the backend, and to avoid cross-origin issues with
        the token endpoint.

        Request body:
            {
                "token_endpoint": "...",
                "code":           "...",
                "code_verifier":  "...",
                "redirect_uri":   "...",
                "client_id":      "...",
                "client_secret":  "...",   // optional; overrides predefined or metadata
                "oauth_metadata": {...}   // optional; contains client_secret if not in body
            }

        Response (pass-through from the token endpoint):
            {
                "access_token":  "...",
                "token_type":    "bearer",
                "expires_in":    3600,       // optional
                "refresh_token": "..."        // optional
            }
        """
        from ..logic.logic_utils import validate_ssrf_safe_url
        from ..logic.oauth import exchange_oauth_token

        require_pro(cast(User, request.user), "mcp_server")

        token_endpoint = (request.data.get("token_endpoint") or "").strip()
        code = (request.data.get("code") or "").strip()
        code_verifier = (request.data.get("code_verifier") or "").strip()
        redirect_uri = (request.data.get("redirect_uri") or "").strip()
        client_id = (request.data.get("client_id") or "").strip()
        client_secret = (request.data.get("client_secret") or "").strip()
        oauth_metadata = request.data.get("oauth_metadata") or {}

        for field, value in [
            ("token_endpoint", token_endpoint),
            ("code", code),
            ("code_verifier", code_verifier),
            ("redirect_uri", redirect_uri),
            ("client_id", client_id),
        ]:
            if not value:
                raise DRFValidationError({field: "This field is required."})

        # If client_secret not provided in body, check oauth_metadata
        if not client_secret:
            client_secret = (oauth_metadata.get("client_secret") or "").strip()

        try:
            validate_ssrf_safe_url(token_endpoint)
        except ValueError as exc:
            raise DRFValidationError(
                {"error": "token_endpoint hostname or IP is not allowed"}
            ) from exc

        try:
            site_config = SiteConfiguration.get_solo()
            site_url = (site_config.site_url or "").strip().rstrip("/")
            if site_url and not redirect_uri.startswith(site_url):
                raise ValueError("redirect_uri does not belong to this site")
        except ValueError as exc:
            raise DRFValidationError(
                {"error": "redirect_uri must be on the configured site URL"}
            ) from exc

        # If the caller specifies a predefined server, use its client_secret for the exchange
        # (confidential clients must authenticate at the token endpoint).
        predefined_server_id = request.data.get("predefined_server_id")
        if predefined_server_id:
            try:
                predefined = PredefinedMCPServer.objects.get(pk=predefined_server_id)
                client_secret = predefined.client_secret or ""
            except PredefinedMCPServer.DoesNotExist:
                pass

        try:
            token_data = exchange_oauth_token(
                token_endpoint=token_endpoint,
                code=code,
                code_verifier=code_verifier,
                redirect_uri=redirect_uri,
                client_id=client_id,
                client_secret=client_secret,
            )

            # Preserve all oauth metadata fields
            # This is needed for re-authentication attempts
            token_data = {**oauth_metadata, **token_data}

            # Remove client secret if predefined server
            if predefined_server_id and token_data and "client_secret" in token_data:
                del token_data["client_secret"]

        except ValueError as exc:
            logger.info("OAuth token exchange failed for token_endpoint=%r: %s", token_endpoint, exc)
            raise DRFValidationError(
                {"error": "Token exchange was rejected by the authorization server."}
            ) from exc

        return Response(token_data, status=status.HTTP_200_OK)
