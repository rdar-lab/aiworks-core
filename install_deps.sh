#!/bin/bash
set -euo pipefail
python -m pip install --upgrade pip uv
uv pip sync --torch-backend=cpu requirements-dev.txt
