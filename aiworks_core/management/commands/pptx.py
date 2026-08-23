"""Management command: pptx read / read_master / write

Usage:
    python manage.py pptx read [PPTX_FILE] [OUTPUT_JSON_FILE]
    python manage.py pptx read_master [PPTX_FILE] [OUTPUT_JSON_FILE]
    python manage.py pptx write [INPUT_JSON_FILE] [OUTPUT_PPTX_FILE]
"""

import json as _json

from django.core.management.base import BaseCommand, CommandError
from ...logic.pptx_generator import render_presentation_pptx
from ...logic.pptx_reader import parse_pptx


class Command(BaseCommand):
    help = "Read or write PPTX files using the presentation JSON schema."

    def add_arguments(self, parser):
        subparsers = parser.add_subparsers(dest="action", required=True)

        read_parser = subparsers.add_parser("read", help="Parse a PPTX file to presentation.json")
        read_parser.add_argument("pptx_file", type=str, help="Path to the input PPTX file")
        read_parser.add_argument("output_json_file", type=str, help="Path to the output JSON file")

        write_parser = subparsers.add_parser("write", help="Generate a PPTX file from presentation.json")
        write_parser.add_argument("input_json_file", type=str, help="Path to the input JSON file")
        write_parser.add_argument("output_pptx_file", type=str, help="Path to the output PPTX file")

    def handle(self, *args, **options):
        action = options["action"]

        if action == "read":
            self._handle_read(options["pptx_file"], options["output_json_file"])
        elif action == "write":
            self._handle_write(options["input_json_file"], options["output_pptx_file"])
        else:
            raise CommandError(f"Unknown action: {action}")

    def _handle_read(self, pptx_file: str, output_json_file: str):
        try:
            with open(pptx_file, "rb") as f:
                pptx_bytes = f.read()
        except OSError as exc:
            raise CommandError(f"Cannot read PPTX file '{pptx_file}': {exc}")

        result = parse_pptx(pptx_bytes)
        try:
            parsed = _json.loads(result)
        except _json.JSONDecodeError as exc:
            raise CommandError(f"parse_pptx returned invalid JSON: {exc}")

        with open(output_json_file, "w", encoding="utf-8") as f:
            _json.dump(parsed, f, indent=2)

        self.stdout.write(
            self.style.SUCCESS(
                f"Wrote {len(parsed.get('slides', []))} slides to '{output_json_file}'."
            )
        )

    def _handle_write(self, input_json_file: str, output_pptx_file: str):
        try:
            with open(input_json_file, "r", encoding="utf-8") as f:
                presentation_json = _json.load(f)
        except OSError as exc:
            raise CommandError(f"Cannot read JSON file '{input_json_file}': {exc}")
        except _json.JSONDecodeError as exc:
            raise CommandError(f"Invalid JSON in '{input_json_file}': {exc}")

        image_files = {}
        pptx_bytes = render_presentation_pptx(presentation_json, image_files)

        try:
            with open(output_pptx_file, "wb") as f:
                f.write(pptx_bytes)
        except OSError as exc:
            raise CommandError(f"Cannot write PPTX file '{output_pptx_file}': {exc}")

        self.stdout.write(
            self.style.SUCCESS(
                f"Wrote {len(pptx_bytes):,} bytes to '{output_pptx_file}'."
            )
        )