import logging

from django.utils import timezone

from ..models import (
    Notification,
)

logger = logging.getLogger(__name__)


def clear_expired_notifications():
    now = timezone.now()
    expired_count = Notification.objects.filter(
        expires_at__lte=now,
        status__in=["sent", "read"],
    ).update(status="expired", dismissed_at=now)

    if expired_count:
        logger.info(
            "daily_insights_job | expired %d notifications",
            expired_count,
        )
