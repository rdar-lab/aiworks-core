"""Management command: generate TTS audio via invoke_llm.

Usage:
    python manage.py generate_audio "Hello, world!" --voice nova

Required LLMConfiguration settings (in Django admin):
    - audio_provider = "openrouter"
    - openrouter_audio_model = "<audio-capable-model>"
    - openrouter_api_key = "sk-..."
    - audio_voices = ["alloy", "echo", "fable", "nova", ...]
"""

import shutil
from pathlib import Path
from typing import Optional

from django.core.management.base import BaseCommand, CommandError
from langchain_core.messages import HumanMessage


class Command(BaseCommand):
    help = "Generate TTS audio using the configured audio LLM."

    def add_arguments(self, parser):
        parser.add_argument(
            "text",
            type=str,
            help="Text to convert to speech",
        )
        parser.add_argument(
            "--voice",
            type=str,
            help="Voice ID to use",
        )
        parser.add_argument(
            "--save",
            type=str,
            metavar="PATH",
            help="Copy the generated audio to PATH",
        )

    def handle(self, *args, **options):
        from ...logic.llm import invoke_llm
        from ...utils import async_to_sync

        text = options["text"]
        voice = options["voice"]
        save_path_str: Optional[str] = options.get("save")

        self.stdout.write(f"Generating audio with voice={voice}: {text[:50]}{'...' if len(text) > 50 else ''}\n")

        try:
            audio_path = async_to_sync(invoke_llm)(
                "generate_audio",
                render_output_audio=True,
                messages=[HumanMessage(content=text)],
                audio_config={"voice": voice},
            )
            self.stdout.write(self.style.SUCCESS("Audio generation completed!"))
            self.stdout.write(f"Audio file: {audio_path}")

            if save_path_str:
                save_path = Path(save_path_str)
                shutil.copy2(audio_path, save_path)
                self.stdout.write(f"Copied audio to: {save_path}")
            else:
                self.stdout.write(f"\nAudio saved at: {audio_path}")

        except CommandError:
            raise
        except Exception as exc:
            raise CommandError(f"Audio generation failed: {exc}")
