from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView

from .views import (
    kb_views,
    mcp_views,
    mcp_tunnel_views,
    auth_views,
    help_views,
    memory_views,
    notification_views,
)

router = DefaultRouter()
router.register(
    r"knowledge-bases", kb_views.KnowledgeBaseViewSet, basename="knowledge-base"
)
router.register(r"mcp-servers", mcp_views.MCPServerViewSet, basename="mcp-server")
router.register(
    r"predefined-mcp-servers",
    mcp_views.PredefinedMCPServerViewSet,
    basename="predefined-mcp-server",
)

# Core API URL patterns — used when aiworks_core is mounted at root
api_urlpatterns = [
    path("", include(router.urls)),
    path("server-settings/", auth_views.server_settings, name="server_settings"),
    path("auth/register/", auth_views.register, name="register"),
    path("auth/verify-email/", auth_views.verify_email, name="verify_email"),
    path("auth/login/", auth_views.login, name="login"),
    path("auth/logout/", auth_views.logout, name="logout"),
    path("auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("auth/me/", auth_views.current_user, name="current_user"),
    path(
        "auth/password-reset/",
        auth_views.request_password_reset,
        name="request_password_reset",
    ),
    path(
        "auth/password-reset/confirm/",
        auth_views.confirm_password_reset,
        name="confirm_password_reset",
    ),
    path("auth/google/", auth_views.google_auth, name="google_auth"),
    path("help-chat/", help_views.help_chat, name="help_chat"),
    path("memory/", memory_views.memory_list, name="memory_list"),
    path("memory/import/", memory_views.memory_import, name="memory_import"),
    path("memory/<int:memory_id>/", memory_views.memory_delete, name="memory_delete"),
    path("mcp-tunnel/connect/", mcp_tunnel_views.MCPTunnelConnectView.as_view(), name="mcp_tunnel_connect"),
    path("mcp-tunnel/", mcp_tunnel_views.MCPTunnelListView.as_view(), name="mcp_tunnel_list"),
    path(
        "mcp-tunnel/<str:tunnel_id>/claim/",
        mcp_tunnel_views.MCPTunnelClaimView.as_view(),
        name="mcp_tunnel_claim",
    ),
    path(
        "mcp-tunnel/<str:tunnel_id>/servers/",
        mcp_tunnel_views.MCPTunnelServersView.as_view(),
        name="mcp_tunnel_servers",
    ),
    path(
        "mcp-tunnel/<str:tunnel_id>/tools/",
        mcp_tunnel_views.MCPTunnelToolsView.as_view(),
        name="mcp_tunnel_tools",
    ),
    path(
        "notifications/",
        notification_views.notifications_list,
        name="notifications_list",
    ),
]

urlpatterns = [
    path("admin/", admin.site.urls),
    # aiworks_core.urls mounted at root — URLs are at /api/*
    path("api/", include(api_urlpatterns)),
]
