#!/bin/bash
#SBATCH -J gtex
#SBATCH -p pi_manoli
#SBATCH -c 8
#SBATCH --mem=48GB
#SBATCH -t 04:00:00
#SBATCH -o %x_%j.log
set -euo pipefail
: "${LOCVAR_DATA:?set LOCVAR_DATA}"
cd "$LOCVAR_DATA"
"$LOCVAR_DATA/venv/bin/python" -u "$SLURM_SUBMIT_DIR/gtex_curate.py" --steps gtf,annotate,gnomad
