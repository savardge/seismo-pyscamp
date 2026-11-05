#!/bin/bash
#SBATCH --job-name=scamp-stage1
#SBATCH --partition=glados-gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%A_%a.out
#
# Stage 1 (association -> windows -> families) for one day of a SCAMP run.
# CPU only: no GPU is requested. Memory is set by the day cache in
# s2_cut_windows.py (42 channels at 250 Hz = 3.6 GB) plus the correlation
# blocks in s3_cluster.py.
#
# Usage (from the repo on Glados):
#   cd ~/seismo-pyscamp/scripts_frs/stage1 && mkdir -p logs
#   sbatch stage1_day.slurm RUN_DIR DAY [OUT_DIR]
#   sbatch --array=0-N stage1_day.slurm RUN_DIR DAYS.txt [OUT_DIR]   # one day per task
#
# DAY may be a YYYYMMDD string or a file with one day per line, indexed by
# SLURM_ARRAY_TASK_ID. OUT_DIR defaults to RUN_DIR/stage1.
set -uo pipefail

RUN="${1:?usage: stage1_day.slurm RUN_DIR DAY|DAYS.txt [OUT_DIR]}"
DAYARG="${2:?usage: stage1_day.slurm RUN_DIR DAY|DAYS.txt [OUT_DIR]}"
OUT="${3:-$RUN/stage1}"
REPO="${SEISCAMP_REPO:-$HOME/seismo-pyscamp}"
S1="$REPO/scripts_frs/stage1"

if [ -f "$DAYARG" ]; then
    DAY=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$DAYARG")
else
    DAY="$DAYARG"
fi
echo "=== $(date) on $(hostname) | stage 1 | run $RUN | day $DAY | out $OUT ==="

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
set +u   # conda's activation hooks read unset variables; re-enabled right after
conda activate seismo-pyscamp
set -u

SECONDS=0
python "$S1/s1_associate.py" --run-dir "$RUN" --day "$DAY" --out "$OUT" || { echo "s1 FAILED"; exit 1; }
echo "[s1] ${SECONDS}s"; SECONDS=0
python "$S1/s2_cut_windows.py" --day "$DAY" --out "$OUT" || { echo "s2 FAILED"; exit 1; }
echo "[s2] ${SECONDS}s"; SECONDS=0
python "$S1/s3_cluster.py" --day "$DAY" --out "$OUT" || { echo "s3 FAILED"; exit 1; }
echo "[s3] ${SECONDS}s"
echo "=== finished $(date) ==="
