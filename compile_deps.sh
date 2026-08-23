#!/bin/bash
set -euo pipefail
python -m pip install --upgrade pip uv

if [[ "${1:-}" == "upgrade" ]]; then
    UV_FLAGS="-U"
else
    UV_FLAGS=""
fi

uv pip compile $UV_FLAGS --torch-backend=cpu requirements.in -o requirements.txt
uv pip compile $UV_FLAGS --torch-backend=cpu requirements-dev.in -o requirements-dev.txt
