"""Central email service for AiWorksCore.

All outbound emails go through this module.  Call sites (views.py, logic.py)
should import and use EmailService static methods — never call send_mail
directly.
"""

import logging

from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from premailer import transform

from ..models import SiteConfiguration

logger = logging.getLogger(__name__)

# Suppress cssutils warnings about vendor-prefixed CSS properties (-webkit-*, -ms-*, mso-*)
# that premailer's CSS parser doesn't recognise but are required for email-client compatibility.
# cssutils uses the logger name 'CSSUTILS' (uppercase) via its own ErrorHandler wrapper.
logging.getLogger("CSSUTILS").setLevel(logging.ERROR)


def _get_site_config():
    """Return the SiteConfiguration singleton."""
    return SiteConfiguration.get_solo()


class EmailService:
    """Thin service layer for sending emails.

    Each method renders both HTML and plain-text templates and sends a
    multipart email.  Failures are logged but never re-raised so callers
    do not crash on email errors.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def send_verification_email(user, token: str) -> None:
        """Send an email-verification token to a newly registered user."""
        config = _get_site_config()
        site_url = (config.site_url or "").strip().rstrip("/")
        from_email = config.default_from_email or None

        app_name = getattr(settings, "AIWORKS_CORE_APP_NAME", "AiWorksCore")

        context = {
            "user": user,
            "token": token,
            "site_url": site_url,
            "app_name": app_name,
        }
        subject = "Verify your email address"
        EmailService.send(
            subject=subject,
            template_name="verify_email",
            context=context,
            recipient_email=user.email,
            from_email=from_email,
        )
        logger.info("email_service | verification email sent to %s", user.email)

    @staticmethod
    def send_password_reset_email(user, token: str) -> None:
        """Send a password-reset token."""
        config = _get_site_config()
        site_url = (config.site_url or "").strip().rstrip("/")
        from_email = config.default_from_email or None

        app_name = getattr(settings, "AIWORKS_CORE_APP_NAME", "AiWorksCore")

        context = {
            "user": user,
            "token": token,
            "site_url": site_url,
            "app_name": app_name,
        }
        subject = "Reset your password"
        EmailService.send(
            subject=subject,
            template_name="password_reset",
            context=context,
            recipient_email=user.email,
            from_email=from_email,
        )
        logger.info("email_service | password reset email sent to %s", user.email)

    @staticmethod
    def send(
        subject: str,
        template_name: str,
        context: dict,
        recipient_email: str,
        from_email: str | None = None,
    ) -> None:
        """Render templates and dispatch a multipart email.

        Never raises — any exception is caught and logged.
        """
        try:
            txt_body = render_to_string(f"emails/{template_name}.txt", context)
            html_body = transform(
                render_to_string(f"emails/{template_name}.html", context)
            )

            msg = EmailMultiAlternatives(
                subject=subject,
                body=txt_body,
                from_email=from_email,
                to=[recipient_email],
            )
            msg.attach_alternative(html_body, "text/html")
            msg.send(fail_silently=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "email_service | failed to send '%s' to %s: %s",
                template_name,
                recipient_email,
                exc,
            )
