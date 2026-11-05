#!/bin/bash
#SBATCH --job-name=scamp-prod
#SBATCH --partition=glados-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=14G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%A_%a.out
#
# Production SCAMP scan of the CaMI.FRS borehole archive on glados-gpu:
# one array task = one day, all borehole stations present that day, three
# components, steps 1-4 via run_frs_day.sh on a single GTX 1080 Ti.
#
# Resources per task (from the 2022-07-19 benchmark, job 3869428): 42 traces
# x ~293 s = 3.4 h at 100 Hz single precision; MaxRSS 10.4 GB. --mem=14G lets
# four tasks share a 62 GB glados12 node (one per GPU). --cpus-per-task=1:
# step 1 is GPU-bound and step 2 single-threaded, and the QOS caps a user at
# 200 running jobs and 200 CPUs in total, shared with any other arrays.
# Only ~33 MB of output per day survives: matrix profiles are
# deleted after step 2 (SEISCAMP_KEEP_MP=0).
#
# Usage (from the repo on Glados):
#   cd ~/seismo-pyscamp/scripts_frs/glados && mkdir -p logs
#   # day list: every day with at least one borehole station, as YYYYMMDD
#   awk -F, 'NR>1 && $2>0 {gsub(/-/,"",$1); print $1}' \
#       /home/gilbert_lab/cami_frs/all_daily_symlinks/coverage_daily.csv > days.txt
#   N=$(wc -l < days.txt)
#   sbatch --array=0-$((N-1))%64 production_scan.slurm days.txt [FS=100] [SUBLEN=0.5] [PRECISION=single]
#
# Array task k runs line k+1 of the day list. A day that finished without any
# "FAILED" step gets a marker in $RUN/done/, and a requeued task exits at once
# if the marker exists; within a day, run_frs_day.sh skips traces that already
# have a detections CSV, so partial days resume cheaply. Re-submitting the
# same array over the same day list is therefore always safe.
#
# Change the running-task cap later with:
#   scontrol update JobId=<arrayid> ArrayTaskThrottle=<n>
set -uo pipefail

DAYLIST="${1:?usage: sbatch --array=0-N production_scan.slurm DAYS.txt [FS] [SUBLEN] [PRECISION]}"
FS="${2:-100}"
SUBLEN="${3:-0.5}"
PRECISION="${4:-single}"
REPO="${SEISCAMP_REPO:-$HOME/seismo-pyscamp}"
ARCHIVE=/home/gilbert_lab/cami_frs/all_daily_symlinks
RUN="${SEISCAMP_RUN_DIR:-$HOME/scamp_runs/prod_${FS}Hz_${PRECISION}_win${SUBLEN}s}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
DAY=$(sed -n "$((TASK + 1))p" "$DAYLIST")
if [ -z "$DAY" ]; then
    echo "task $TASK: no day at line $((TASK + 1)) of $DAYLIST"; exit 1
fi
mkdir -p "$RUN/done" "$RUN/daylogs"
if [ -e "$RUN/done/$DAY" ]; then
    echo "task $TASK: day $DAY already done ($(cat "$RUN/done/$DAY")), skipping"; exit 0
fi

echo "=== $(date) on $(hostname) | task $TASK | day $DAY | fs $FS Hz | win $SUBLEN s | $PRECISION | run dir $RUN ==="

module purge   # CUDA runtime comes from the conda env, not a module
source "$HOME/miniconda3/etc/profile.d/conda.sh"
set +u   # conda's activation hooks read unset variables; re-enabled right after
conda activate seismo-pyscamp
set -u

export SEISCAMP_WF_DIR="$ARCHIVE"
export SEISCAMP_OUT="$RUN/matrix_profiles"       # transient, deleted after step 2
export SEISCAMP_DET_SINGLE="$RUN/detections_single"
export SEISCAMP_DET_COMBINED="$RUN/detections_combined"
export SEISCAMP_DET_TIMES="$RUN/detections_times"
export SEISCAMP_LOG_DIR="$RUN/logs"
export SEISCAMP_PRECISION="$PRECISION"
export SEISCAMP_KEEP_MP=0
export SEISCAMP_PLOT=0

# Borehole stations with files that day: field 2 of 8O.BHxxx..DPZ.D.<date>...
YEAR=${DAY:0:4}
DOY=$(date -d "${DAY:0:4}-${DAY:4:2}-${DAY:6:2}" +%j)
mapfile -t STATIONS < <(ls "$ARCHIVE/$YEAR/$DOY" 2>/dev/null | awk -F. '$2 ~ /^BH/ {print $2}' | sort -u)
if [ "${#STATIONS[@]}" -eq 0 ]; then
    echo "day $DAY: no borehole files in $ARCHIVE/$YEAR/$DOY"; exit 1
fi
echo "stations (${#STATIONS[@]}): ${STATIONS[*]}"

DAYLOG="$RUN/daylogs/$DAY.out"
SECONDS=0
"$REPO/scripts_frs/run_frs_day.sh" "$DAY" "$SUBLEN" "$FS" DP "${STATIONS[@]}" 2>&1 | tee "$DAYLOG"
WALL=$SECONDS
NFAIL=$(grep -c "FAILED" "$DAYLOG" || true)
NDET=$(ls "$RUN"/detections_single/${DAY}*_detections.csv 2>/dev/null | wc -l)
echo "=== day $DAY: ${#STATIONS[@]} stations, $NDET detection CSVs, $NFAIL failed steps, wall ${WALL}s ==="
if [ "$NFAIL" -eq 0 ] && [ "$NDET" -eq $(( 3 * ${#STATIONS[@]} )) ]; then
    echo "$(date +%FT%T) job ${SLURM_JOB_ID:-local} wall ${WALL}s" > "$RUN/done/$DAY"
    echo "marked done"
else
    echo "NOT marked done; resubmit this task to retry the unfinished traces"
    exit 2
fi
