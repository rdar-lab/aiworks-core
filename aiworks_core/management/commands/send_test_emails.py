"""Management command: send one of every email type to a given address.

Usage:
    python manage.py send_test_emails recipient@example.com

Sends all four email types using the live EmailService (and therefore whatever
email backend is currently configured in SiteConfiguration / Django settings),
so this is a real end-to-end test of the email pipeline.
"""

from django.core.management.base import BaseCommand, CommandError

from ...logic.emails import EmailService


class _FakeUser:
    """Minimal user-like object sufficient for all EmailService methods."""

    def __init__(self, email):
        self.email = email
        self.username = email
        self.first_name = 'Test'


class _FakeSession:
    """Minimal session-like object sufficient for all EmailService methods."""

    def __init__(self, session_id, title, dilemma):
        self.id = session_id
        self.session_title = title
        self.dilemma = dilemma


class Command(BaseCommand):
    help = (
        'Send one of every email type to the given address using the configured '
        'email backend — useful for visually testing email templates.'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            'email',
            type=str,
            help='Recipient email address',
        )

    def handle(self, *args, **options):
        recipient = options['email']
        if '@' not in recipient:
            raise CommandError(f"'{recipient}' does not look like a valid email address.")

        user = _FakeUser(email=recipient)

        emails = [
            (
                'Verification email',
                lambda: EmailService.send_verification_email(user, token='ABC-123-XYZ'),
            ),
            (
                'Password reset email',
                lambda: EmailService.send_password_reset_email(user, token='DEF-456-UVW'),
            )
        ]

        self.stdout.write(f"Sending {len(emails)} test emails to {recipient} …\n")

        all_ok = True
        for label, send_fn in emails:
            try:
                send_fn()
                self.stdout.write(self.style.SUCCESS(f"  ✓ {label}"))
            except Exception as exc:  # noqa: BLE001
                self.stdout.write(self.style.ERROR(f"  ✗ {label}: {exc}"))
                all_ok = False

        self.stdout.write('')
        if all_ok:
            self.stdout.write(self.style.SUCCESS('All emails sent successfully.'))
        else:
            raise CommandError('One or more emails failed — check output above.')
