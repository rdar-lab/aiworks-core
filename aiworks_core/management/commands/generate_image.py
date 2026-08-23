"""Management command: generate an image via invoke_llm.

Usage:
    python manage.py generate_image "A futuristic city at sunset"

    python manage.py generate_image --save /tmp/image.txt

Required LLMConfiguration settings (in Django admin):
    - image_provider = "openrouter"
    - openrouter_image_model = "google/gemini-2.5-flash-image"
    - openrouter_api_key = "sk-..."
"""

import base64
import re
from pathlib import Path
from typing import Optional

from django.core.management.base import BaseCommand, CommandError
from langchain_core.messages import HumanMessage


class Command(BaseCommand):
    help = "Generate an image using the configured image LLM."

    def add_arguments(self, parser):
        parser.add_argument(
            "prompt",
            type=str,
            nargs="?",
            default="A professional LinkedIn post image about innovation and technology",
            help="Image prompt (default: a generic professional prompt)",
        )
        parser.add_argument(
            "--save",
            type=str,
            metavar="PATH",
            help="Save the base64 image data URL to a file at PATH",
        )

    def handle(self, *args, **options):
        from ...logic.llm import invoke_llm
        from ...utils import async_to_sync

        prompt = options["prompt"]
        save_path_str: Optional[str] = options.get("save")

        self.stdout.write(f"Generating image with prompt: {prompt}\n")

        try:
            image_url = async_to_sync(invoke_llm)(
                "generate_image",
                messages=[HumanMessage(content=prompt)],
                render_output_image=True,
            )
            self.stdout.write(self.style.SUCCESS("Image generated!"))
            self.stdout.write(f"Data URL length: {len(image_url)} chars")

            if save_path_str:
                # Parse data URL: data:image/png;base64,iVBORw0KGgo...
                match = re.match(r"data:([^;]+);base64,(.+)", image_url)
                if not match:
                    raise CommandError("Invalid image data URL format")
                mime_type = match.group(1)
                b64_data = match.group(2)

                # Determine extension from mime type
                ext = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/jpg": ".jpg",
                    "image/gif": ".gif",
                    "image/webp": ".webp",
                }.get(mime_type, "")

                save_path = Path(save_path_str)
                if save_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
                    save_path = save_path.with_suffix(ext)

                image_bytes = base64.b64decode(b64_data)
                save_path.write_bytes(image_bytes)
                self.stdout.write(f"Saved to: {save_path}")

            else:
                self.stdout.write(f"\nData URL (first 200 chars):\n{image_url[:200]}")

        except CommandError:
            raise
        except Exception as exc:
            raise CommandError(f"Image generation failed: {exc}")
