#!/bin/bash
set -euo pipefail
: "${LOCVAR_DATA:?set LOCVAR_DATA}"
module load miniforge
mkdir -p "$LOCVAR_DATA"
python -m venv "$LOCVAR_DATA/venv"
"$LOCVAR_DATA/venv/bin/pip" install --upgrade pip
"$LOCVAR_DATA/venv/bin/pip" install -r "$(cd "$(dirname "$0")" && pwd)/requirements.txt"
