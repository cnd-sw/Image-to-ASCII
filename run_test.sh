#!/bin/bash
# run_test.sh - Run the full smoke test
set -e
PROJ="/Users/chandan/Documents/Code/Image to ASCII"
PYTHON="$PROJ/.venv/bin/python3"
export PYTHONPATH="$PROJ"
cd "$PROJ"
echo "Running smoke test..."
"$PYTHON" test_smoke.py
