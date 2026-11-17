#!/bin/bash
#SBATCH -J gnomad_extract
#SBATCH -p pi_manoli
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH -t 24:00:00
#SBATCH -o %x_%j.log
set -euo pipefail
: "${LOCVAR_DATA:?set LOCVAR_DATA}"
: "${GNOMAD_VCF_DIR:?set GNOMAD_VCF_DIR}"
cd "$LOCVAR_DATA"
"$LOCVAR_DATA/venv/bin/python" -u "$SLURM_SUBMIT_DIR/gnomad_extract.py" "$GNOMAD_VCF_DIR" --out gnomad_slim --workers "${WORKERS:-12}"
