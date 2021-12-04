from obspy import Stream, read
import os
import matplotlib.dates as mdates
from eqcorrscan.utils.stacking import align_traces
from eqcorrscan.utils.clustering import corr_cluster
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from glob import glob
import os
import networkx as nx
from networkx.algorithms.components import *
import matplotlib.pyplot as plt
import sys

input_dir = os.path.split(sys.argv[1])[0] #"/home/genevieve.savard/seismo-pyscamp/scripts_frs/detections_combined/"
input_csv = os.path.split(sys.argv[1])[1]
wf_dir = "/home/gilbert_lab/cami_frs/all_daily_symlinks"
doplot = False
dowrite = True
output_dir = "/home/genevieve.savard/seismo-pyscamp/scripts_frs/detections_times/"
output_file = input_csv.replace("detections.csv", "times.csv")
sta = input_csv.split("_")[2]
print(f"Station {sta}\ninput CSV: {input_csv}\nInput dir: {input_dir}\nOutput files: {output_file}\nOutput dir: {output_dir} ")

def merge_intervals(intervals):
    """
    A simple algorithm can be used:
    1. Sort the intervals in increasing order
    2. Push the first interval on the stack
    3. Iterate through intervals and for each one compare current interval
       with the top of the stack and:
       A. If current interval does not overlap, push on to stack
       B. If current interval does overlap, merge both intervals in to one
          and push on to stack
    4. At the end return stack
    https://codereview.stackexchange.com/questions/69242/merging-overlapping-intervals
    """
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = np.array([lower[0], upper_bound])  # replace by merged interval
            else:
                merged.append(higher)
    return np.array(merged)

def idx_to_windows(idx1, idx2, tol=10):
    """
    Combine indices from peak detection into unique, merged windows.
    :param tol: Tolerance in samples before and after peak to merge
    """
    allidx = np.hstack((idx1, idx2))

    # Find unique intervals for all window pairs
    x = allidx - tol
    y = allidx + tol
    intervals = np.array((x, y)).T
    return merge_intervals(intervals)
        
def extract(input_csv):
    df = pd.read_csv(input_csv)

    # Add timestamps
    ts1 = np.array([UTCDateTime(t).timestamp for t in df.time1.values]).astype(int)
    ts2 = np.array([UTCDateTime(t).timestamp for t in df.time2.values]).astype(int)
    df["timestamp1"] = ts1
    df["timestamp2"] = ts2

    # Get unique windows
    tol = 1.0
    merged = idx_to_windows(ts1, ts2, tol=tol)
    lb = merged[:,0].astype(int)
    ub = merged[:,1].astype(int)
    dfref = pd.DataFrame({"lb": lb, "ub": ub, "window_id": np.array(range(merged.shape[0]))})
    dfref.drop_duplicates()

    # Find window id for first indices of pairs
    i, j = np.where((ts1[:, None] >= lb) & (ts1[:, None] <= ub))

    dfw = pd.DataFrame(
        np.column_stack([df.iloc[i], dfref.iloc[j]]),
        columns=df.columns.append(pd.Index(data=["lb1", "ub1", "win_id_1"]))
    )

    # Find window id for second indices of pairs
    i, j = np.where((ts2[:, None] >= lb) & (ts2[:, None] <= ub))

    df = pd.DataFrame(
        np.column_stack([dfw.iloc[i], dfref.iloc[j]]),
        columns=dfw.columns.append(pd.Index(data=["lb2", "ub2", "win_id_2"]))
    )

    # Remove bounds from df
    df = df.drop(columns=["lb1", "ub1", "lb2", "ub2"])

    # Fix dtypes
    df["mpval"] = df["mpval"].astype(float)
    df["prominence"] = df["prominence"].astype(float)
    df["peak_std_sec"] = df["peak_std_sec"].astype(float)
    df["win_id_1"] = df["win_id_1"].astype(int)
    df["win_id_2"] = df["win_id_2"].astype(int)
    df["timestamp1"] = df["timestamp1"].astype(int)
    df["timestamp2"] = df["timestamp2"].astype(int)

    # Add time to dfref
    tub = [UTCDateTime(i) for i in ub]
    tlb = [UTCDateTime(i) for i in lb]
    dfref["tub"] = tub
    dfref["tlb"] = tlb

    # from collections import Counter
    # ids = np.hstack((df.win_id_1.values, df.win_id_2.values))
    # values, counts = np.unique(ids, return_counts=True)
    # print(sorted(zip(values, counts), key=lambda x: x[1], reverse=True))

    # GROUPS USING NETWORKX
    G = nx.Graph()
    subset = df[["win_id_1", "win_id_2", "mpval"]]
    edgelist = [tuple(x) for x in subset.to_numpy()]
    G.add_weighted_edges_from(edgelist)
    #print(is_connected(G))
    print(f"Number of connected components: {number_connected_components(G)}")

    groups = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    return df, dfref, groups


def plot_3comps(stream, savefile=None):
    nrows = np.min([len(stream.select(channel="*Z")), 100])
    stn = stream.select(channel="*X").sort(keys=['starttime'])
    ste = stream.select(channel="*Y").sort(keys=["starttime"])
    stz = stream.select(channel="*Z").sort(keys=["starttime"])
    fig, axs = plt.subplots(nrows, 3, figsize=(16, 5*nrows))
    for k in range(nrows):
        axs[k][0].plot_date(stn[k].times("matplotlib"), stn[k].data, "k")
        axs[k][1].plot_date(ste[k].times("matplotlib"), ste[k].data, "k")
        axs[k][2].plot_date(stz[k].times("matplotlib"), stz[k].data, "k")
        axs[k][1].set_title(stn[k].stats.starttime)
    fig.autofmt_xdate()
    if savefile:
        plt.savefig(savefile, format="PNG")
    plt.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# First read csv and get groups
df, dfref, groups = extract(os.path.join(input_dir, input_csv))

# Now plot/write groups

if dowrite:
    print("Opening output file: %s" % os.path.join(output_dir, output_file))
    fout = open(os.path.join(output_dir, output_file), "w")
    fout.write("timeUTC,max_amplitude,stations_available,group_id\n")

for i, group in enumerate(sorted(groups, key=len, reverse=True)):
    print(f"Number of nodes: {len(group.nodes)}")

    # Align traces
    trace_list = []
    node_list = []
    for node in group.nodes:
        t1 = dfref.loc[dfref.window_id == node].tlb.values[0] - 4
        t2 = dfref.loc[dfref.window_id == node].tub.values[0] + 4
        #print(f"{t1}, {t2}, {t2-t1}")
        tmp = read(os.path.join(wf_dir, t1.strftime("%Y/%j"), "*%s*Z*" % sta), starttime=t1, endtime=t2)[0]
        tmp.detrend()
        tmp.filter("highpass", freq=2)
        tmp.filter("lowpass", freq=10)
        #tmp.plot()
        trace_list.append(tmp)
        node_list.append(node)
    shifts, ccs = align_traces(trace_list, shift_len=int(4.*100), master=False, positive=False, plot=False)

    stream = Stream()
    for node, shift in zip(node_list, shifts):
        t1 = dfref.loc[dfref.window_id == node].tlb.values[0] - shift
        t2 = t1 + 6
        tmp = read(os.path.join(wf_dir, t1.strftime("%Y/%j"), "*%s*" % sta), starttime=t1, endtime=t2)
        tmp.detrend()
        tmp.filter("highpass", freq=2)
#        tmp.filter("lowpass", freq=10)
        #tmp.plot()
        stream += tmp

        if dowrite:
            nsta = len(glob(os.path.join(wf_dir, t1.strftime("%Y/%j"), "*Z*")))
            maxamp = np.max(np.abs(tmp.max()))
            print("Adding a line to file")
            fout.write("%s,%.0f,%d,%d\n" % (t1.strftime("%Y-%m-%dT%H:%M:%S.%f"),maxamp,nsta,i))

    if doplot:
        num_nodes = len(node_list)
        plotfile = os.path.join(output_dir, "plot_group%03d_%dnodes.png" % (i, num_nodes))
        plot_3comps(stream, savefile=plotfile)

if dowrite:
    fout.close()
    print("Closed file")
