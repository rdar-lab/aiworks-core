"""Management command: generate a video via invoke_llm.

Usage:
    python manage.py generate_video "A person walking through a futuristic city at sunset"

    python manage.py generate_video --save /tmp/video

Required LLMConfiguration settings (in Django admin):
    - video_provider = "openrouter"
    - openrouter_video_model = "<video-capable-model>"
    - openrouter_api_key = "sk-..."
"""

import shutil
from pathlib import Path
from typing import Optional

from django.core.management.base import BaseCommand, CommandError
from langchain_core.messages import HumanMessage


class Command(BaseCommand):
    help = "Generate a video using the configured video LLM."

    def add_arguments(self, parser):
        parser.add_argument(
            "prompt",
            type=str,
            nargs="?",
            default="A person walking through a futuristic city at sunset, cinematic style",
            help="Video prompt (default: a generic cinematic prompt)",
        )
        parser.add_argument(
            "--save",
            type=str,
            metavar="PATH",
            help="Copy the generated video to PATH",
        )

    def handle(self, *args, **options):
        from ...logic.llm import invoke_llm
        from ...utils import async_to_sync

        prompt = options["prompt"]
        save_path_str: Optional[str] = options.get("save")

        self.stdout.write(f"Generating video with prompt: {prompt}\n")

        try:
            video_path = async_to_sync(invoke_llm)(
                "generate_video",
                messages=[HumanMessage(content=prompt)],
                render_output_video=True,
            )
            self.stdout.write(self.style.SUCCESS("Video generation completed!"))
            self.stdout.write(f"Video file: {video_path}")

            if save_path_str:
                save_path = Path(save_path_str)
                shutil.copy2(video_path, save_path)
                self.stdout.write(f"Copied video to: {save_path}")
            else:
                self.stdout.write(f"\nVideo saved at: {video_path}")

        except CommandError:
            raise
        except Exception as exc:
            raise CommandError(f"Video generation failed: {exc}")
