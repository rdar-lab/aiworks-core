"""Conversion utilities for file format transformations."""

import logging
import os
import subprocess
import tempfile
from typing import Tuple

logger = logging.getLogger(__name__)


def pptx_to_pdf(pptx_bytes: bytes) -> Tuple[bytes | None, bool]:
    """Convert PPTX bytes to PDF bytes using LibreOffice headless.

    Returns (pdf_bytes, was_successful).
    If LibreOffice is not available or conversion fails, returns (None, False).
    Does not raise — failures are non-fatal.
    """
    try:
        result = subprocess.run(
            ["soffice", "--headless", "--version"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.info("pptx_to_pdf | LibreOffice not available (--version failed)")
            return None, False
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        logger.info("pptx_to_pdf | LibreOffice not available: %s", e)
        return None, False

    with tempfile.TemporaryDirectory() as tmpdir:
        pptx_path = os.path.join(tmpdir, "input.pptx")
        with open(pptx_path, "wb") as f:
            f.write(pptx_bytes)

        try:
            result = subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    tmpdir,
                    pptx_path,
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.warning("pptx_to_pdf | conversion failed: %s, %s", result.stdout.decode(), result.stderr.decode())
                return None, False

            pdf_path = os.path.join(tmpdir, "input.pdf")
            if not os.path.exists(pdf_path):
                logger.warning("pptx_to_pdf | PDF file not created")
                return None, False

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            return pdf_bytes, True
        except subprocess.TimeoutExpired:
            logger.warning("pptx_to_pdf | conversion timed out")
            return None, False
        except Exception as e:
            logger.warning("pptx_to_pdf | conversion error: %s", e)
            return None, False