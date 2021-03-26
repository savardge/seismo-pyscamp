from statsmodels import robust
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import os
import pickle
from seiscamp.GraphLib import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import logging
Logger = logging.getLogger(__name__)


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


def find_group(row, colname, cc):
    """
    Find the group index from the window ID for a the connected components cc
    :param row: dataframe row containing column "colname" with window IDs used to build undirected graph
    :param colname: column name to use that contains window ID
    :param cc: connected components, list of lists (see GraphLib)
    :return:
    """
    id = row[colname]
    match = [i for i, group in enumerate(cc) if id in group]
    return match[0] if match else None


class MatrixProfile:
    """
    Class for Matrix profile data
    """

    def __init__(self, mp=None, ind=None, trace=None, station=None, channel=None, fs=None, sublen=None, starttime=None, endtime=None):
        """

        :param mp:  Matrix profile (MP) values (Pearson CC)
        :param ind: Indices of matching windows
        :param station:
        :param channel:
        :param fs: sampling rate in Hz
        :param sublen: window length in samples used for building MP
        :param starttime: start time of array UTCDateTime (does not take into account sublen)
        :param endtime: End time of array UTCDateTime (does not take into account sublen)
        """
        self.mp = mp
        if mp.any():
            self.length = len(mp)
        self.ind = ind
        self.sublen = sublen

        if trace:
            self.add_trace(trace, use_stats=True)
        else:
            self.station = station
            self.channel = channel
            self.fs = fs
            self.starttime = starttime
            self.mp_starttime = starttime + sublen/fs
            self.endtime = endtime
            self.trace = None
            self._datac = None

        # Properties to be determined by class methods
        self.num_peaks = None
        self._mad_thresh_mult = None
        self._peak_idx1 = None
        self._peak_idx2 = None
        self._peak_properties = None
        self._merge_tol = None
        self._merged_windows = None
        self.groups = None
        self.pairs = None
        self.ref_windows = None

    def load_from_files(self, mp_file, ind_file, trace=None):

        self.mp = np.load(mp_file)
        self.ind = np.load(ind_file)
        if trace:
            self.add_trace(trace, use_stats=True)

    def add_trace(self, trace, use_stats=False):
        """
        Add Obspy trace
        :param trace:
        :param use_stats: use stats from trace object to fill object fields (station, fs, etc.)
        :return:
        """

        if use_stats:
            self.station = trace.stats.station
            self.channel = trace.stats.channel
            self.fs = trace.stats.sampling_rate
            self.starttime = trace.stats.starttime
            self.mp_starttime = self.starttime + self.sublen/self.fs
            self.endtime = trace.stats.endtime

        else:
            # Perform some checks for consistency
            if trace.stats.station != self.station or trace.stats.channel != self.channel:
                Logger.error("Trace provided does not match station and channel of MP!")

            else:
                if trace.stats.sampling_rate != self.fs:
                    Logger.warning("Trace sampling rate didn't match MP sampling rate. Resampling for consistency.")
                    trace.resample(self.fs)

                if trace.stats.starttime != self.starttime or trace.stats.endtime != self.endtime:
                    Logger.warning("Trace start/end times don't match. Trimming trace to fit MP data.")
                    trace.trim(starttime=self.starttime, endtime=self.endtime)

        # Now add trace data and check for length consistency
        datac = trace.data[self.sublen-1:]
        if len(datac) != self.length:
            Logger.error("Mismatch in trace data length in samples and MP length. Check input!")
        else:
            self.trace = trace
            self._datac = datac  # trace data cut to fit length of MP

    def get_peaks(self, mad_thresh_mult=6, width=None):
        """
        Get peaks in matrix profile.
        :param mad_thresh_mult: muliple of MP mad value to use as height theshold
        :param width: minimum width of peaks in samples
        :return: peaks (indices), properties
        """

        self._mad_thresh_mult = mad_thresh_mult

        if not width:
            width = int(0.5 * self.sublen)
        lowthresh = np.median(self.mp) + mad_thresh_mult * robust.mad(self.mp)
        peaks, properties = find_peaks(self.mp, height=[lowthresh, 1.0], width=width)  # , prominence=0.3)

        # Remove detections where CC > 1
        nbefore = len(peaks)
        mask = properties["peak_heights"] < 1.0
        peaks = peaks[mask]
        for k, v in properties.items():
            #print(k)
            properties[k] = properties[k][mask]
        nafter = len(peaks)
        Logger.info("Remove %d peak detections where CC > 1" % (nafter - nbefore))

        self._peak_properties = properties
        self._peak_idx1 = peaks.astype(int)
        self._peak_idx2 = np.take(self.ind, self._peak_idx1).astype(int)
        self.num_peaks = len(peaks)
        
        Logger.info("# peaks found: %d" % self.num_peaks)

    def _idx_to_windows(self, tol=None):
        """
        Combine indices from peak detection into unique, merged windows.
        :param tol: Tolerance in samples before and after peak to merge
        """
        if not tol:
            tol = int(self.sublen / 2)

        self._merge_tol = tol

        allidx = np.hstack((self._peak_idx1, self._peak_idx2))

        # Find unique intervals for all window pairs
        x = allidx - tol
        y = allidx + tol
        intervals = np.array((x, y)).T
        self._merged_windows = merge_intervals(intervals)

    def _get_connected_components(self, df, ref, colid1, colid2):
        """
        Find connected components for window ID pairs
        :param df: Pandas.Daframe with index pairs
        :param ref: Reference Pandas.DataFrame with window ID definition
        :param colid1: column name for window ID, first element of pair
        :param colid2: column name for window ID, second element of pair
        """

        # Initialize graph
        Logger.info("Building the graph...")
        nrow = len(ref)
        g = Graph(nrow)

        for i, row in df.iterrows():
            id1 = int(row[colid1])
            id2 = int(row[colid2])
            g.add_edge(id1, id2)

        # Find islands in the graph
        Logger.info("Getting connected components...")
        cc = g.connected_components()
        cc = sorted(cc, key=lambda x: len(x), reverse=True)  # Sort by decreasing # of links
        Logger.info("Found %d groups." % len(cc))

        self.groups = cc

    def group_ids(self, tol=None, connect=False):
        """
        Associate windows to each index for each pair
        return: Pandas DataFrame
        https://stackoverflow.com/questions/44367672/best-way-to-join-merge-by-range-in-pandas
        :param tol: Tolerance in samples before and after peak to merge
        :return:
            groups: connected pairs arranged in groups (list of lists)
            pairs: Pandas.DataFrame with every pairs from peak detection of Matrix profile
            ref: Pandas.DataFrame with merged windows and their ID and group index
        """
        # Get unique windows
        self._idx_to_windows(tol=tol)

        # Setup for pandas
        lb = self._merged_windows[:, 0]  # Lower bound of windows
        ub = self._merged_windows[:, 1]  # upper bound of windows
        dfref = pd.DataFrame({"lower_bound": lb,
                              "upper_bound": ub,
                              "window_id": np.array(range(len(lb)))})
        self.ref_windows = dfref

        pairs = pd.DataFrame({"Index1": self._peak_idx1,
                              "Index2": self._peak_idx2,
                              "t1": np.array([self.mp_starttime + _t for _t in self._peak_idx1.astype(float)/self.fs]),
                              "t2": np.array([self.mp_starttime + _t for _t in self._peak_idx2.astype(float)/self.fs]),
                              "peak_cc": self._peak_properties["peak_heights"],
                              "peak_mad_mult": self._peak_properties["peak_heights"]/robust.mad(self.mp),
                              "peak_width": self._peak_properties["widths"],
                              "peak_prominence": self._peak_properties["prominences"]})

        # Find window id for first indices of pairs
        i, j = np.where((self._peak_idx1[:, None] >= lb) & (self._peak_idx1[:, None] <= ub))

        dfw = pd.DataFrame(
            np.column_stack([pairs.iloc[i], dfref.iloc[j]]),
            columns=pairs.columns.append(pd.Index(data=["lb1", "ub1", "win_id_1"]))
        )

        # Find window id for second indices of pairs
        i, j = np.where((self._peak_idx2[:, None] >= lb) & (self._peak_idx2[:, None] <= ub))

        df = pd.DataFrame(
            np.column_stack([dfw.iloc[i], dfref.iloc[j]]),
            columns=dfw.columns.append(pd.Index(data=["lb2", "ub2", "win_id_2"]))
        )

        # Remove bounds from df
        df = df.drop(columns=["lb1", "ub1", "lb2", "ub2"])

        # Fix dtypes
        df["Index1"] = df["Index1"].astype(int)
        df["Index2"] = df["Index2"].astype(int)
        df["win_id_1"] = df["win_id_1"].astype(int)
        df["win_id_2"] = df["win_id_2"].astype(int)

        if connect:
            # Get connected components
            self._get_connected_components(df, ref=dfref, colid1="win_id_1", colid2="win_id_2")

            # Add column with group index to dataframe
            df["group_idx"] = df.apply(find_group, axis=1, args=("win_id_1", self.groups))
            dfref["group_idx"] = dfref.apply(find_group, axis=1, args=("window_id", self.groups))
        else:
            self.groups = None
            
        # Add time to ref windows
        dfref["lb_time"] = np.array([self.mp_starttime + _t for _t in lb.astype(float)/self.fs])
        dfref["ub_time"] = np.array([self.mp_starttime + _t for _t in ub.astype(float)/self.fs])
        dfref["window_duration"] = dfref["ub_time"] - dfref["lb_time"]        
        
        # Finally, Attach to MP object
        self.pairs = df
        self.ref_windows = dfref

        return self.groups, self.pairs, self.ref_windows

    def save(self, filename):
        """
        Save MatrixProfile object to pickle file
        :param filename: Full file name with path to save to
        """
        filepath = os.path.split(filename)[0]
        if not os.path.exists(filepath):
            Logger.warning("File path %s does not exist. Creating it now.")
            os.makedirs(filepath)
        with open(filename, "wb") as output:
            pickle.dump(self, output, -1)

        Logger.info("MP object saved to %s" % filename)

    def plot_all_pairs(self, outdir, prepick=1.0, postpick=1.0):
        """

        :param outdir:
        :return:
        """

        for i, row in self.pairs.iterrows():
            i1 = row["Index1"]
            i2 = row["Index2"]
            t1 = self.mp_starttime + i1/self.fs
            t2 = self.mp_starttime + i2/self.fs
            group_idx = row["group_idx"]

            outsubdir = os.path.join(os.path.join(outdir, "group%d" % group_idx))
            if not os.path.exists(outsubdir):
                os.makedirs(outsubdir)

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            data1 = self.trace.slice(starttime=t1 - prepick, endtime=t1 + postpick).data
            axs[0].plot(data1, "k")
            axs[0].axvline(prepick * self.fs, np.min(data1), np.max(data1), color="r", linestyle="--")
            axs[0].set_title("%s" % t1)

            data2 = self.trace.slice(starttime=t2 - prepick, endtime=t2 + postpick).data
            axs[1].plot(data2, "k")
            axs[1].axvline(prepick * self.fs, np.min(data2), np.max(data2), color="r", linestyle="--")
            axs[1].set_title("%s" % t2)

            plt.suptitle("Group #%d, CC = %3.2f, width = %3.2f s, prominence = %3.2f" % (group_idx, row["peak_cc"], row["peak_width"]/self.fs, row["peak_prominence"]))
            fname = os.path.join(outsubdir, "pairs_%s_%s.png" % (t1.strftime("%Y%m%d%H%M%S"), t2.strftime("%Y%m%d%H%M%S")))
            plt.savefig(fname)
            plt.close()

    def plot_group(self, outdir, Nmax=40, prepick=1.0, postpick=1.0):
        """

        :param outdir:
        :param Nmax:
        :param prepick:
        :param postpick:
        :return:
        """

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for igroup, group in enumerate(self.groups):

            dfg = self.pairs.loc[self.pairs["group_idx"] == igroup]
            indices = np.unique(np.hstack((dfg["Index1"], dfg["Index2"])))

            if len(indices) > Nmax:
                indices = indices[:Nmax]
                Logger.info("more than %d detections in group %d. Only keep the first %d" % (Nmax, igroup, Nmax))

            ndets = len(indices)
            ncols = 3
            nrows = ndets // ncols + 1
            fig, axs = plt.subplots(nrows, ncols, figsize=(5*nrows, 5*ncols))
            for i, ix in enumerate(indices):
                tix = self.mp_starttime + ix/self.fs
                data = self.trace.slice(starttime=tix - prepick, endtime=tix + postpick).data
                tutc = np.array([self.mp_starttime + t for t in np.array(range(len(data))).astype(float) / self.fs])
                tplt = [mdates.date2num(t._get_datetime()) for t in tutc]
                axs[i].plot_date(tplt, data, "k")
                axs[i].set_title("%s" % tix)

            plt.suptitle("Group #%d, %d detections" % (igroup, ndets))
            fname = os.path.join(outdir, "group%d_%ddets.png" % (igroup, ndets))
            plt.savefig(fname)
            plt.close()
