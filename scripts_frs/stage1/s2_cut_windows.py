"""
Stage 1, step 2: cut filtered three-component windows around every event.

Reads ``<out>/<DAY>/events.csv`` (from ``s1_associate.py``), reads each
station-day of the raw archive once (decimated to the working rate), cuts
one window per station and component around each event's ``t_ref`` and
writes ``<out>/<DAY>/windows.npz`` (see
:func:`seiscamp.families.windows.save_windows`).

Usage::

    python s2_cut_windows.py --day 20220719 --out ~/scamp_runs/stage1 \\
        [--archive /home/gilbert_lab/cami_frs/all_daily_symlinks] \\
        [--fs 250] [--pre 1] [--post 3] [--fmin 5] [--fmax 60]

Memory: the cache holds all channels of the day at ``--fs``; 42 channels
at 250 Hz are 3.6 GB. Run it on a compute node, never on the login node.
"""
import argparse
import os
import time

import pandas as pd

from seiscamp.families.windows import DayArchive, cut_windows, save_windows

DEFAULT_ARCHIVE = os.environ.get("SEISCAMP_WF_DIR", "/home/gilbert_lab/cami_frs/all_daily_symlinks")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day", required=True, help="day as YYYYMMDD")
    p.add_argument("--out", required=True, help="stage-1 output root (holds <DAY>/events.csv)")
    p.add_argument("--archive", default=DEFAULT_ARCHIVE, help="waveform archive root")
    p.add_argument("--fs", type=float, default=250.0, help="working sampling rate in Hz (default 250)")
    p.add_argument("--pre", type=float, default=1.0, help="seconds before t_ref (default 1)")
    p.add_argument("--post", type=float, default=3.0, help="seconds after t_ref (default 3)")
    p.add_argument("--fmin", type=float, default=5.0, help="band-pass low corner in Hz (default 5)")
    p.add_argument("--fmax", type=float, default=60.0, help="band-pass high corner in Hz (default 60)")
    a = p.parse_args()

    day_dir = os.path.join(a.out, a.day)
    events = pd.read_csv(os.path.join(day_dir, "events.csv"), parse_dates=["t_ref", "t_first", "t_last"])
    if events.empty:
        print("day %s: no events, nothing to cut" % a.day)
        return
    # Station order of the array = every station that joined any event.
    stations = sorted({s for row in events["stations"] for s in row.split(";")})
    print("day %s: %d events, %d stations: %s" % (a.day, len(events), len(stations), " ".join(stations)))

    archive = DayArchive(a.archive, fs_work=a.fs)
    t0 = time.time()
    data, live = cut_windows(events, archive, stations, pre_s=a.pre, post_s=a.post,
                             fmin=a.fmin, fmax=a.fmax)
    out = os.path.join(day_dir, "windows.npz")
    save_windows(out, data, live, events, stations, a.fs, a.pre, a.post, a.fmin, a.fmax)
    print("cut %s in %.0f s; live channels per event: mean %.1f of %d; wrote %s (%.1f MB)"
          % (data.shape, time.time() - t0, live.sum(axis=(1, 2)).mean(), live.shape[1] * live.shape[2],
             out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
