"""
Stage 1, step 1: associate the SCAMP detections of one day into events.

Reads every ``*_COMBINED_*_detections.csv`` of the day from a production
or benchmark run directory, collapses the window pairs into per-station
occurrences, associates them across stations and writes::

    <out>/<DAY>/occurrences.csv     one row per collapsed occurrence (+ event_id)
    <out>/<DAY>/event_stations.csv  one row per event member
    <out>/<DAY>/events.csv          one row per event

Usage::

    python s1_associate.py --run-dir ~/scamp_runs/prod_100Hz_single_win0.5s \\
        --day 20220719 --out ~/scamp_runs/stage1 [--tol 1.0] [--collapse 5] \\
        [--min-stations 3]

See :mod:`seiscamp.families.associate` for the algorithm and the meaning
of the tolerances.
"""
import argparse
import os
from glob import glob

import pandas as pd

from seiscamp.families.associate import associate, collapse_occurrences, load_occurrences


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="pipeline run directory (holds detections_combined/)")
    p.add_argument("--day", required=True, help="day as YYYYMMDD")
    p.add_argument("--out", required=True, help="stage-1 output root; <out>/<DAY>/ is created")
    p.add_argument("--tol", type=float, default=1.0, help="association window in s from the anchor (default 1.0)")
    p.add_argument("--collapse", type=float, default=5.0, help="per-station collapse tolerance in s (default 5)")
    p.add_argument("--min-stations", type=int, default=3, help="minimum distinct stations per event (default 3)")
    a = p.parse_args()

    files = sorted(glob(os.path.join(a.run_dir, "detections_combined", "%s*COMBINED*_detections.csv" % a.day)))
    if not files:
        raise SystemExit("no combined CSVs for %s under %s" % (a.day, a.run_dir))
    out_dir = os.path.join(a.out, a.day)
    os.makedirs(out_dir, exist_ok=True)

    raw = pd.concat([load_occurrences(f) for f in files], ignore_index=True)
    occ = collapse_occurrences(raw, tol_s=a.collapse)
    events, event_stations, occ = associate(occ, tol_s=a.tol, min_stations=a.min_stations)

    occ.to_csv(os.path.join(out_dir, "occurrences.csv"), index=False)
    event_stations.to_csv(os.path.join(out_dir, "event_stations.csv"), index=False)
    events.to_csv(os.path.join(out_dir, "events.csv"), index=False)

    print("day %s: %d stations, %d window times -> %d occurrences -> %d events (>= %d stations)"
          % (a.day, len(files), len(raw), len(occ), len(events), a.min_stations))
    per_sta = raw.groupby("station").size().rename("windows").to_frame()
    per_sta["occurrences"] = occ.groupby("station").size()
    per_sta["in_events"] = occ[occ.event_id >= 0].groupby("station").size()
    print(per_sta.fillna(0).astype(int).to_string())
    if len(events):
        print("events by number of stations:")
        print(events["n_stations"].value_counts().sort_index().to_string())
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
