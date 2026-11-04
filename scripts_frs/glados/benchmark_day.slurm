#!/bin/bash
#SBATCH --job-name=scamp-bench
#SBATCH --partition=glados-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#
# One-day end-to-end benchmark of the scripts_frs pipeline on a single GTX
# 1080 Ti: every borehole station present in the archive that day, all three
# components, step1 (SCAMP self-join) through step4 (families), via
# run_frs_day.sh. Afterwards two extra single-trace step1 runs give the
# scaling anchors the cost model needs: 250 Hz single precision (n^2 scaling)
# and 100 Hz double precision (FP64 penalty).
#
# Usage (from the repo on Glados):
#   cd ~/seismo-pyscamp/scripts_frs/glados && mkdir -p logs
#   sbatch benchmark_day.slurm [DAY=20220719] [FS=100] [SUBLEN_SEC=0.5]
#
# Outputs go to $SEISCAMP_RUN_DIR (default ~/scamp_runs/bench_<DAY>_<FS>Hz_<jobid>):
#   detections_single/ detections_combined/ detections_times/ logs/
#   timing_step1.csv  (one row per trace: station,channel,fs,precision,seconds)
# Matrix profiles are deleted after step2 (run_frs_day.sh default).
set -uo pipefail

DAY="${1:-20220719}"          # 2022-07-19: all 14 borehole stations present, 500 Hz native
FS="${2:-100}"
SUBLEN="${3:-0.5}"
REPO="${SEISCAMP_REPO:-$HOME/seismo-pyscamp}"
ARCHIVE=/home/gilbert_lab/cami_frs/all_daily_symlinks
RUN="${SEISCAMP_RUN_DIR:-$HOME/scamp_runs/bench_${DAY}_${FS}Hz_${SLURM_JOB_ID}}"

echo "=== $(date) on $(hostname) | day $DAY | fs $FS Hz | win $SUBLEN s | run dir $RUN ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

module purge   # CUDA runtime comes from the conda env, not a module
source "$HOME/miniconda3/etc/profile.d/conda.sh"
set +u   # conda's activation hooks read unset variables; re-enabled right after
conda activate seismo-pyscamp
set -u
python -c "import pyscamp; assert pyscamp.gpu_supported(), 'no CUDA in pyscamp'; print('pyscamp GPU OK')" || exit 1

# Pipeline I/O: archive is the Glados year/julian-day tree (data.py handles the
# layout), everything written goes under $RUN so the run is self-contained.
export SEISCAMP_WF_DIR="$ARCHIVE"
export SEISCAMP_OUT="$RUN/matrix_profiles"
export SEISCAMP_DET_SINGLE="$RUN/detections_single"
export SEISCAMP_DET_COMBINED="$RUN/detections_combined"
export SEISCAMP_DET_TIMES="$RUN/detections_times"
export SEISCAMP_LOG_DIR="$RUN/logs"
export SEISCAMP_PRECISION=single
mkdir -p "$RUN"

# Borehole stations that have data this day (BH*, DP? channels).
YEAR=${DAY:0:4}; DOY=$(date -u -d "${DAY:0:4}-${DAY:4:2}-${DAY:6:2}" +%j)
mapfile -t STATIONS < <(ls "$ARCHIVE/$YEAR/$DOY" | awk -F. '$2 ~ /^BH/ {print $2}' | sort -u)
echo "stations (${#STATIONS[@]}): ${STATIONS[*]}"

# --- full pipeline, one day --------------------------------------------------
SECONDS=0
bash "$REPO/scripts_frs/run_frs_day.sh" "$DAY" "$SUBLEN" "$FS" DP "${STATIONS[@]}"
echo "=== run_frs_day.sh wall time: ${SECONDS}s ==="

# --- scaling anchors: one trace each -----------------------------------------
STEP1="$REPO/scripts_frs/python_scripts/step1_pyscamp_1chan.py"
ANCHOR_OUT="$RUN/anchors"; mkdir -p "$ANCHOR_OUT"
for spec in "250 single" "100 double"; do
    set -- $spec; afs=$1; aprec=$2
    echo "[anchor] ${STATIONS[0]} DPZ @ ${afs} Hz, ${aprec}"
    SEISCAMP_OUT="$ANCHOR_OUT" SEISCAMP_PRECISION="$aprec" \
        python "$STEP1" "${STATIONS[0]}" "$DAY" 1 "$SUBLEN" DPZ "$afs" \
        > "$RUN/logs/anchor_${afs}Hz_${aprec}.log" 2>&1 \
        || echo "[anchor] FAILED, see $RUN/logs/anchor_${afs}Hz_${aprec}.log"
    # ${VAR:?} aborts instead of expanding to "/" if the variable were empty
    rm -f "${ANCHOR_OUT:?}"/*.npy
done

# --- timing table from the step1 logs ----------------------------------------
# step1 logs one line "Finished processing component in X seconds (precision=P, GPU)".
TIMING="$RUN/timing_step1.csv"
echo "station,channel,fs_hz,precision,seconds" > "$TIMING"
for LOG in "$RUN"/logs/step1_*.log "$RUN"/logs/anchor_*.log; do
    [ -e "$LOG" ] || continue
    secs=$(grep -o "Finished processing component in [0-9.]*" "$LOG" | awk '{print $NF}')
    prec=$(grep -o "precision=[a-z]*" "$LOG" | head -1 | cut -d= -f2)
    sta=$(grep -o "station = [A-Z0-9]*" "$LOG" | head -1 | awk '{print $3}')
    chan=$(grep -o "channel = [A-Z0-9]*" "$LOG" | head -1 | awk '{print $3}')
    fs=$(grep -o "Sampling rate: [0-9]*" "$LOG" | head -1 | awk '{print $3}')
    [ -n "$secs" ] && echo "$sta,$chan,$fs,$prec,$secs" >> "$TIMING"
done
echo "=== step1 timings ($TIMING) ==="; column -s, -t "$TIMING"
echo "=== detections ==="; ls "$RUN/detections_single" | wc -l; ls "$RUN/detections_times" 2>/dev/null | head
echo "=== disk use of run dir ==="; du -sh "$RUN"
echo "=== finished $(date) ==="
