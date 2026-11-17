#!/bin/bash
#SBATCH -J marimo
#SBATCH -p pi_manoli
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH -t 04:00:00
#SBATCH -o %x_%j.log
set -euo pipefail
: "${LOCVAR_DATA:?set LOCVAR_DATA}"
: "${NOTEBOOK:?set NOTEBOOK}"
PORT="${PORT:-2718}"
echo "node: $(hostname)  port: $PORT"
cd "$LOCVAR_DATA"
"$LOCVAR_DATA/venv/bin/python" -m marimo edit "$SLURM_SUBMIT_DIR/$NOTEBOOK" --headless --host 0.0.0.0 --port "$PORT"
