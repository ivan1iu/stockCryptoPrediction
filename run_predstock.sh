#!/usr/bin/env bash
# Helper to run predstock.py with the project's virtualenv
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Project venv python not found at $VENV_PY"
  echo "You can activate your venv manually: source $SCRIPT_DIR/.venv/bin/activate"
  exit 1
fi
"$VENV_PY" "$SCRIPT_DIR/predstock.py" "$@"
