from django.urls import path

from .consumers import TunnelConsumer

websocket_urlpatterns = [
    path("ws/mcp-tunnel/connect", TunnelConsumer.as_asgi()),
]
