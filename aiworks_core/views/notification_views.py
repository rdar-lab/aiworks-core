import logging

from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import (
    api_view,
    permission_classes,
)
from rest_framework.exceptions import (
    MethodNotAllowed,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ..models import (
    Notification,
)
from ..serializers import (
    NotificationSerializer,
)

logger = logging.getLogger(__name__)


@api_view(["GET", "PUT"])
@permission_classes([IsAuthenticated])
def notifications_list(request):
    """List or update notifications.

    GET: List notifications for the user.
    PUT: Update notification status (mark as read, dismiss).
    """
    user = request.user

    if request.method == "GET":
        type_filter = request.query_params.get("type")
        limit = min(int(request.query_params.get("limit", 20)), 100)
        offset = int(request.query_params.get("offset", 0))

        queryset = Notification.objects.filter(user=user)

        if type_filter:
            queryset = queryset.filter(type=type_filter)

        notifications = queryset.order_by("-created_at")[offset: offset + limit]

        unread_count = queryset.exclude(
            status__in=["read", "dismissed", "expired"]
        ).count()

        serializer = NotificationSerializer(notifications, many=True)
        return Response(
            {
                "data": serializer.data,
                "meta": {
                    "unread_count": unread_count,
                    "total": queryset.count(),
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < queryset.count(),
                },
            }
        )

    elif request.method == "PUT":
        notification_id = request.data.get("notification_id")
        new_status = request.data.get("status")

        if not notification_id or not new_status:
            return Response(
                {"error": "notification_id and status are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if new_status not in ("read", "dismissed"):
            return Response(
                {"error": "status must be one of: read, dismissed"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            notification = Notification.objects.get(id=notification_id, user=user)
        except Notification.DoesNotExist:
            return Response(
                {"error": "Notification not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        notification.status = new_status
        if new_status == "read":
            notification.read_at = timezone.now()
        elif new_status == "dismissed":
            notification.dismissed_at = timezone.now()

        notification.save(update_fields=["status", "read_at", "dismissed_at"])

        return Response({"data": NotificationSerializer(notification).data})
    else:
        raise MethodNotAllowed(request.method)
