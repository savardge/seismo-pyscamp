#!/bin/bash
#SBATCH --job-name=scamp-group
#SBATCH --partition=glados-gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#
# Re-run step 4 (event-family grouping) only, one station per array task,
# on the combined detection CSVs of an existing run directory. Step 4 is
# CPU-only (networkx connected components + eqcorrscan alignment, re-reading
# the raw daily SAC files for every detection), so it needs no GPU and can
# be separated from the GPU pass. Used to time step 4 after the benchmark
# (job 3869428), whose step-4 calls crashed on the archive-layout bug.
#
# Usage (from the repo on Glados):
#   cd ~/seismo-pyscamp/scripts_frs/glados && mkdir -p logs
#   N=$(ls ~/scamp_runs/<run>/detections_combined/*_detections.csv | wc -l)
#   sbatch --array=0-$((N-1)) group_only.slurm ~/scamp_runs/<run>
#
# Array task k processes the k-th CSV (sorted) in <run>/detections_combined;
# output goes to <run>/detections_times, the log to <run>/logs/step4_*.log,
# and the wall time of each station is printed as "[step4] <csv> done in Ns".
set -uo pipefail

RUN="${1:?usage: sbatch --array=0-N group_only.slurm RUN_DIR}"
REPO="${SEISCAMP_REPO:-$HOME/seismo-pyscamp}"
PYDIR="$REPO/scripts_frs/python_scripts"
export SEISCAMP_WF_DIR="${SEISCAMP_WF_DIR:-/home/gilbert_lab/cami_frs/all_daily_symlinks}"
export SEISCAMP_DET_TIMES="$RUN/detections_times"
mkdir -p "$SEISCAMP_DET_TIMES" "$RUN/logs"

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
set +u   # conda's activation hooks read unset variables; re-enabled right after
conda activate seismo-pyscamp
set -u

# Pick this task's CSV: sorted list, indexed by SLURM_ARRAY_TASK_ID.
mapfile -t CSVS < <(ls "$RUN"/detections_combined/*_detections.csv | sort)
COMB="${CSVS[${SLURM_ARRAY_TASK_ID:-0}]}"
BASE=$(basename "$COMB")
LOG="$RUN/logs/step4_${BASE%.csv}.log"
NDET=$(( $(wc -l < "$COMB") - 1 ))

echo "=== $(date) on $(hostname) | task ${SLURM_ARRAY_TASK_ID:-0} | $BASE | $NDET detections ==="
SECONDS=0
if python "$PYDIR/step4_group.py" "$COMB" > "$LOG" 2>&1; then
    NFAM=$(grep -c . "$SEISCAMP_DET_TIMES/${BASE/detections.csv/times.csv}")
    echo "[step4] $BASE done in ${SECONDS}s ($NDET detections -> $((NFAM-1)) family members)"
else
    echo "[step4] FAILED for $BASE after ${SECONDS}s, see $LOG"
    tail -5 "$LOG"
fi
echo "=== finished $(date) ==="
