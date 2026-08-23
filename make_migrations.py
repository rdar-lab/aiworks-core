#!/usr/bin/env python
"""Generate migrations for aiworks_core using the test settings."""
import os
import sys

def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aiworks_core.tests.settings")
    import django
    django.setup()
    from django.core.management import call_command
    call_command("makemigrations", "aiworks_core")

if __name__ == "__main__":
    main()
