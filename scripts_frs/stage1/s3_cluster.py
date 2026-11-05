"""
Stage 1, step 3: correlate, cluster into families, summarise and plot.

Reads ``<out>/<DAY>/windows.npz`` and ``events.csv``, computes the network
correlation of every event pair, cuts near (0.7) and tight (0.9) families
from an average-linkage dendrogram and writes::

    <out>/<DAY>/cc.npy, lag.npy          correlation and lag matrices
    <out>/<DAY>/families.csv             events + family label per tier
    <out>/<DAY>/family_summary_<tier>.csv  one row per family
    <out>/<DAY>/figures/cc_matrix_<tier>.png
    <out>/<DAY>/figures/family_<tier>_<id>.png   (families with >= --plot-min members)

Usage::

    python s3_cluster.py --day 20220719 --out ~/scamp_runs/stage1 \\
        [--max-lag 1.0] [--near 0.7] [--tight 0.9] [--plot-min 3]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from seiscamp.families.cluster import cluster_families, family_summary, reference_member
from seiscamp.families.plotting import plot_cc_matrix, plot_family
from seiscamp.families.similarity import align_to_reference, network_cc
from seiscamp.families.windows import load_windows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day", required=True, help="day as YYYYMMDD")
    p.add_argument("--out", required=True, help="stage-1 output root")
    p.add_argument("--max-lag", type=float, default=1.0, help="largest lag searched, in s each way (default 1)")
    p.add_argument("--near", type=float, default=0.7, help="near-repeater threshold (default 0.7)")
    p.add_argument("--tight", type=float, default=0.9, help="tight-repeater threshold (default 0.9)")
    p.add_argument("--min-common", type=int, default=6, help="min shared live channels per pair (default 6)")
    p.add_argument("--plot-min", type=int, default=3, help="plot families with at least this many members")
    a = p.parse_args()

    day_dir = os.path.join(a.out, a.day)
    fig_dir = os.path.join(day_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    w = load_windows(os.path.join(day_dir, "windows.npz"))
    events = pd.read_csv(os.path.join(day_dir, "events.csv"), parse_dates=["t_ref", "t_first", "t_last"])
    assert (events["event_id"].values == w["event_id"]).all(), "events.csv and windows.npz disagree"
    data, live, fs = w["data"], w["live"], w["fs"]
    n = len(events)
    print("day %s: %d events, windows %s at %g Hz" % (a.day, n, data.shape, fs))

    t0 = time.time()
    cc, lag, n_common = network_cc(data, live, max_lag=int(round(a.max_lag * fs)), min_common=a.min_common)
    print("correlation matrix in %.0f s" % (time.time() - t0))
    np.save(os.path.join(day_dir, "cc.npy"), cc)
    np.save(os.path.join(day_dir, "lag.npy"), lag)

    tiers = {"near": a.near, "tight": a.tight}
    labels, _ = cluster_families(cc, tiers=tiers)
    fam = events.copy()
    for tier in tiers:
        fam["family_%s" % tier] = labels[tier]
    fam.to_csv(os.path.join(day_dir, "families.csv"), index=False)

    for tier, thr in tiers.items():
        lab = labels[tier]
        summ = family_summary(events, lab, cc)
        summ.to_csv(os.path.join(day_dir, "family_summary_%s.csv" % tier), index=False)
        n_fam = len(summ)
        n_in = int((lab >= 0).sum())
        print("%s tier (cc >= %.2f): %d families holding %d of %d events; %d periodic"
              % (tier, thr, n_fam, n_in, n, int(summ["periodic"].sum()) if n_fam else 0))
        if n_fam:
            print(summ.head(15).to_string(index=False, float_format=lambda x: "%.2f" % x))
        plot_cc_matrix(cc, lab, os.path.join(fig_dir, "cc_matrix_%s.png" % tier),
                       title="%s, %s tier (cc >= %.2f)" % (a.day, tier, thr))
        for _, row in summ.iterrows():
            if row["n"] < a.plot_min:
                continue
            members = np.where(lab == row["family"])[0]
            ref = reference_member(cc, members)
            aligned, members = align_to_reference(data, lag, ref, members)
            title = ("%s  %s family %d: %d members, mean cc %.2f, median IET %.0f s%s"
                     % (a.day, tier, row["family"], row["n"], row["cc_mean"],
                        row["iet_median_s"], ", periodic" if row["periodic"] else ""))
            plot_family(aligned, live[members], w["stations"], w["components"], fs, w["pre_s"],
                        os.path.join(fig_dir, "family_%s_%03d.png" % (tier, row["family"])), title=title)
    print("wrote", day_dir)


if __name__ == "__main__":
    main()
