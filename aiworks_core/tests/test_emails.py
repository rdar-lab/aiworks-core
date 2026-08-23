"""Unit tests for aiworks_core.logic.emails.EmailService."""

from unittest.mock import patch

from django.core import mail
from django.test import TestCase, override_settings

from . import _make_user
from ..logic.emails import EmailService
from ..models import SiteConfiguration


@override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend')
class EmailServiceVerificationTests(TestCase):
    """Tests for EmailService.send_verification_email."""

    def setUp(self):
        self.user = _make_user(username='ver_user', email='ver@example.com')
        SiteConfiguration.objects.update_or_create(pk=1, defaults={
            'site_url': 'https://app.example.com',
            'default_from_email': 'no-reply@app.com',
        })

    def test_sends_multipart_email(self):
        EmailService.send_verification_email(self.user, 'ABC123')
        self.assertEqual(len(mail.outbox), 1)
        msg = mail.outbox[0]
        self.assertEqual(msg.subject, 'Verify your email address')
        self.assertIn('ver@example.com', msg.to)

    def test_subject_correct(self):
        EmailService.send_verification_email(self.user, 'TOK999')
        self.assertEqual(mail.outbox[0].subject, 'Verify your email address')

    def test_plain_text_contains_token(self):
        EmailService.send_verification_email(self.user, 'MYTOKEN')
        self.assertIn('MYTOKEN', mail.outbox[0].body)

    def test_html_alternative_contains_token(self):
        EmailService.send_verification_email(self.user, 'HTMLTOK')
        msg = mail.outbox[0]
        html_content = msg.alternatives[0][0]
        self.assertIn('HTMLTOK', html_content)

    def test_failure_does_not_raise(self):
        """A render/send error must be silently caught."""
        with patch('aiworks_core.logic.emails.EmailMultiAlternatives.send', side_effect=Exception('SMTP down')):
            # Must not raise
            EmailService.send_verification_email(self.user, 'FAILTOKEN')


@override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend')
class EmailServicePasswordResetTests(TestCase):
    """Tests for EmailService.send_password_reset_email."""

    def setUp(self):
        self.user = _make_user(username='pw_user', email='pw@example.com')
        SiteConfiguration.objects.update_or_create(pk=1, defaults={
            'site_url': 'https://app.example.com',
            'default_from_email': 'no-reply@app.com',
        })

    def test_sends_email_with_correct_subject(self):
        EmailService.send_password_reset_email(self.user, 'RESETTOKEN')
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].subject, 'Reset your password')

    def test_plain_text_contains_token(self):
        EmailService.send_password_reset_email(self.user, 'RTOKEN')
        self.assertIn('RTOKEN', mail.outbox[0].body)

    def test_failure_does_not_raise(self):
        with patch('aiworks_core.logic.emails.EmailMultiAlternatives.send', side_effect=Exception('err')):
            EmailService.send_password_reset_email(self.user, 'FAIL')
