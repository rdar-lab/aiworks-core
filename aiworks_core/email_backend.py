"""
Custom email backend that reads SMTP configuration from the SiteConfiguration
database record, allowing email settings to be managed via the Django admin
without requiring a server restart or environment-variable changes.
"""

import logging

from django.conf import settings
from django.core.mail.backends.base import BaseEmailBackend

logger = logging.getLogger(__name__)


class DatabaseEmailBackend(BaseEmailBackend):
    """
    Delegate to an underlying Django email backend whose settings are stored in
    ``SiteConfiguration`` (singleton DB record).

    Falls back to the console backend when the database is unavailable (e.g.
    during migrations or test set-up before the schema exists).
    """

    def _build_backend(self):
        try:
            from .models import SiteConfiguration
            config = SiteConfiguration.get_solo()
        except Exception as exc:
            from django.db import OperationalError, ProgrammingError
            if isinstance(exc, (OperationalError, ProgrammingError)):
                # DB not yet ready (e.g. first migration) – fall back gracefully
                logger.debug("DatabaseEmailBackend: DB not ready, falling back to console (%s)", exc)
            else:
                logger.warning(
                    "DatabaseEmailBackend: unexpected error reading SiteConfiguration, "
                    "falling back to console backend: %s",
                    exc,
                )
            from django.core.mail.backends.console import EmailBackend
            return EmailBackend(fail_silently=self.fail_silently), None

        if config.email_backend == 'console':
            from django.core.mail.backends.console import EmailBackend
            return EmailBackend(fail_silently=self.fail_silently), config
        elif config.email_backend == 'dummy':
            from django.core.mail.backends.dummy import EmailBackend
            return EmailBackend(fail_silently=self.fail_silently), config
        else:  # 'smtp' (default)
            from django.core.mail.backends.smtp import EmailBackend
            return EmailBackend(
                host=config.email_host,
                port=config.email_port,
                username=config.email_host_user,
                password=config.email_host_password,
                use_tls=config.email_use_tls,
                use_ssl=config.email_use_ssl,
                fail_silently=self.fail_silently,
            ), config

    def send_messages(self, email_messages):
        backend, config = self._build_backend()

        # Override the "From" address on every outgoing message with the value
        # stored in the DB so that changes take effect without a restart.
        if config and config.default_from_email:
            for msg in email_messages:
                if not msg.from_email or msg.from_email == settings.DEFAULT_FROM_EMAIL:
                    msg.from_email = config.default_from_email

        with backend:
            return backend.send_messages(email_messages)
