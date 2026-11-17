#!/bin/bash
#SBATCH -J localization
#SBATCH -p pi_manoli
#SBATCH -c 12
#SBATCH --mem=120GB
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:1
#SBATCH -o %x_%j.log
set -euo pipefail
: "${LOCVAR_DATA:?set LOCVAR_DATA}"
cd "$LOCVAR_DATA"
PY="$LOCVAR_DATA/venv/bin/python"
if [ "${STEP:-all}" != "score" ]; then "$PY" -u "$SLURM_SUBMIT_DIR/loc_pipeline.py" prepare; fi
if [ "${STEP:-all}" != "prepare" ]; then "$PY" -u "$SLURM_SUBMIT_DIR/loc_pipeline.py" score; fi
