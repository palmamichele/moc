#!/bin/bash

DEST="/Users/palmamichele/Desktop/toWorkMachineOnly/Research Projects/Beyond Lipschitz/6a8ff1cf4469cbc36d0201bb/experiments"


set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/.venv/bin/python"

osascript <<EOF
tell application "Terminal"
    activate
    do script "cd '$SCRIPT_DIR' && '$PYTHON' mnist.py --batch-size 64"
    do script "cd '$SCRIPT_DIR' && '$PYTHON' mnist.py --batch-size 4096"
end tell
EOF

