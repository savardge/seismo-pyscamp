#!/bin/bash
# Progress summary of a production scan: array state, days done/failed,
# throughput and a rough ETA. Light enough for the login node (reads a few
# small files and squeue/sacct; no data processing).
#
# Usage:  status.sh [ARRAY_JOBID] [RUN_DIR]
#   defaults: the newest scamp-prod array in the queue, ~/scamp_runs/prod_100Hz_single_win0.5s
set -u
JOB="${1:-$(squeue -h -u "$USER" -n scamp-prod -o %F | sort -u | tail -1)}"
RUN="${2:-$HOME/scamp_runs/prod_100Hz_single_win0.5s}"
NDAYS=$(wc -l < "$RUN/days.txt")
NDONE=$(ls "$RUN/done" 2>/dev/null | wc -l)

echo "array $JOB | run dir $RUN"
echo "queue:   $(squeue -h -u "$USER" -n scamp-prod -o %T | sort | uniq -c | awk '{printf "%s %s  ", $1, $2}')"
echo "days:    $NDONE / $NDAYS done"
if [ -n "$JOB" ]; then
    # Finished-task outcomes from the accounting database (one line per array task).
    sacct -j "$JOB" -X -n -P --format=State | sort | uniq -c | awk '{printf "tasks:   %s %s  ", $1, $2} END {print ""}'
fi
# Throughput from the done markers (each holds the wall time of its day).
if [ "$NDONE" -gt 0 ]; then
    awk '{s+=$(NF)+0; n++} END {printf "mean wall per finished day: %.0f s (%.2f h)\n", s/n, s/n/3600}' \
        <(sed 's/s$//' "$RUN"/done/*)
    FIRST=$(ls -tr "$RUN/done" | head -1); T0=$(stat -c %Y "$RUN/done/$FIRST"); NOW=$(date +%s)
    RATE=$(awk -v n="$NDONE" -v dt=$((NOW - T0)) 'BEGIN {if (dt>0) printf "%.2f", n/dt*3600; else print 0}')
    awk -v r="$RATE" -v left=$((NDAYS - NDONE)) 'BEGIN {if (r>0) printf "rate since first finished day: %s days/h -> ~%.1f days of wall-clock left\n", r, left/r/24}'
fi
echo "failed steps so far: $(grep -l FAILED "$RUN"/daylogs/*.out 2>/dev/null | wc -l) day(s)"
du -sh "$RUN/detections_single" "$RUN/matrix_profiles" 2>/dev/null | awk '{printf "%s %s  ", $1, $2} END {print ""}'
