#!/bin/bash
# Run the full scripts_frs pipeline (step1-4) for one day of /mnt/cmc3 data,
# without SLURM. Usage:
#   ./run_frs_day.sh DAY SUBLEN_SEC FS CHANPREFIX STATION [STATION ...]
# e.g.
#   ./run_frs_day.sh 20250204 0.5 250 DP BH001 BH012 BH013
set -u

DAY=$1
SUBLEN=$2
FS=$3
CHANPRE=$4
shift 4
STATIONS=("$@")

NUMDAYS=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYDIR="$SCRIPT_DIR/python_scripts"

MP_DIR="${SEISCAMP_OUT:-$HOME/seismo-pyscamp/scripts_frs/matrix_profiles}"
DET_SINGLE="${SEISCAMP_DET_SINGLE:-$HOME/seismo-pyscamp/scripts_frs/detections_single}"
DET_COMBINED="${SEISCAMP_DET_COMBINED:-$HOME/seismo-pyscamp/scripts_frs/detections_combined}"
DET_TIMES="${SEISCAMP_DET_TIMES:-$HOME/seismo-pyscamp/scripts_frs/detections_times}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$MP_DIR" "$DET_SINGLE" "$DET_COMBINED" "$DET_TIMES" "$LOG_DIR"

echo "=== Day $DAY | sublen ${SUBLEN}s | fs ${FS}Hz | stations: ${STATIONS[*]} ==="

# ---- Step 1: SCAMP self-join per trace (GPU-bound, sequential) ----
for STA in "${STATIONS[@]}"; do
    for COMP in N E Z; do
        CHAN="${CHANPRE}${COMP}"
        LOG="$LOG_DIR/step1_${DAY}_${STA}_${CHAN}.log"
        echo "[step1] $STA $CHAN (log: $LOG)"
        SECONDS=0
        python "$PYDIR/step1_pyscamp_1chan.py" "$STA" "$DAY" "$NUMDAYS" "$SUBLEN" "$CHAN" "$FS" > "$LOG" 2>&1 \
            || { echo "[step1] FAILED for $STA $CHAN, see $LOG"; continue; }
        echo "[step1] $STA $CHAN done in ${SECONDS}s"
    done
done

# ---- Step 2: extract detections per trace ----
for MPFILE in "$MP_DIR"/${DAY}*_mp.npy; do
    [ -e "$MPFILE" ] || continue
    BASE=$(basename "$MPFILE")
    LOG="$LOG_DIR/step2_${BASE%.npy}.log"
    echo "[step2] $BASE (log: $LOG)"
    python "$PYDIR/step2_extract_detections.py" "$BASE" > "$LOG" 2>&1 \
        || { echo "[step2] FAILED for $BASE, see $LOG"; continue; }
done

# ---- Step 3: combine N/E/Z per station ----
for STA in "${STATIONS[@]}"; do
    ZCSV=$(ls "$DET_SINGLE"/${DAY}*_${STA}_${CHANPRE}Z_*_detections.csv 2>/dev/null | head -1)
    if [ -z "$ZCSV" ]; then
        echo "[step3] no ${CHANPRE}Z detections CSV for $STA, skipping"
        continue
    fi
    BASE=$(basename "$ZCSV")
    PREFIX="${BASE%%${CHANPRE}Z*}"
    SUFFIX="${BASE##*${CHANPRE}Z}"
    LOG="$LOG_DIR/step3_${DAY}_${STA}.log"
    echo "[step3] $STA prefix=$PREFIX suffix=$SUFFIX (log: $LOG)"
    python "$PYDIR/step3_combine.py" "$PREFIX" "$SUFFIX" "$CHANPRE" > "$LOG" 2>&1 \
        || { echo "[step3] FAILED for $STA, see $LOG"; continue; }
done

# ---- Step 4: cluster into families per station ----
for COMB in "$DET_COMBINED"/${DAY}*COMBINED*_detections.csv; do
    [ -e "$COMB" ] || continue
    BASE=$(basename "$COMB")
    LOG="$LOG_DIR/step4_${BASE%.csv}.log"
    echo "[step4] $BASE (log: $LOG)"
    python "$PYDIR/step4_group.py" "$COMB" > "$LOG" 2>&1 \
        || { echo "[step4] FAILED for $BASE, see $LOG"; continue; }
done

echo "=== Pipeline finished for day $DAY ==="
echo "Outputs:"
echo "  matrix profiles:      $MP_DIR"
echo "  per-trace detections: $DET_SINGLE"
echo "  combined detections:  $DET_COMBINED"
echo "  families/times:       $DET_TIMES"
