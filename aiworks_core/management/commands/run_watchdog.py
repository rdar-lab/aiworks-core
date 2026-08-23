"""Django management command to run the aiworks-core watchdog daemon."""

import logging
import signal
import sys
import threading

from django.core.management.base import BaseCommand

from aiworks_core.aiworks_core.apps import AiWorksCoreConfig


class Command(BaseCommand):
    help = "Run the aiworks-core watchdog daemon"

    def add_arguments(self, parser):
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging",
        )

    def handle(self, *args, **options):
        if options["verbose"]:
            logging.getLogger("aiworks_core").setLevel(logging.DEBUG)

        self.stdout.write(self.style.SUCCESS("Starting aiworks-core watchdog..."))

        stop_event = threading.Event()

        def signal_handler(signum, frame):
            self.stdout.write(self.style.WARNING("Shutting down watchdog..."))
            stop_event.set()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        watchdog_thread = threading.Thread(
            target=AiWorksCoreConfig._watchdog,
            daemon=True,
        )
        watchdog_thread.start()

        try:
            while not stop_event.is_set():
                stop_event.wait(timeout=1)
        except KeyboardInterrupt:
            pass

        self.stdout.write(self.style.WARNING("Watchdog stopped"))
