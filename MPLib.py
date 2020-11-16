from statsmodels import robust
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import logging
import os
import pickle


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
        if mp:
            self.length = len(mp)
        self.ind = ind
        self.sublen = sublen

        self.num_peaks = None
        self.peak_idx1 = None
        self.peak_idx2 = None
        self.peak_properties = None
        self.merged_windows = None

        if trace:
            self.add_trace(trace, use_stats=True)
        else:
            self.station = station
            self.channel = channel
            self.fs = fs
            self.starttime = starttime
            self.endtime = endtime
            self.trace = None
            self.datac = None

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
            self.datac = datac  # trace data cut to fit length of MP

    def get_peaks(self, mad_mul_thresh, width=None):
        """
        Get peaks in matrix profile.

        :param mad_mul_thresh: muliple of MP mad value to use as height theshold
        :param width: minimum width of peaks in samples
        :return: peaks (indices), properties
        """
        if not width:
            width = int(0.5 * self.sublen)
        lowthresh = np.median(self.mp) + mad_mul_thresh * robust.mad(self.mp)
        peaks, properties = find_peaks(self.mp, height=lowthresh, width=width)  # , prominence=0.3)

        self.num_peaks = len(peaks)
        self.peak_properties = properties
        self.peak_idx1 = peaks.astype(int)
        self.peak_idx2 = np.take(self.ind, self.peak_idx1).astype(int)

        Logger.info("# peaks found: %d" % self.num_peaks)

    def idx_to_windows(self, tol=None):
        """
        Combine peak indices into unique, merged windows.
        :param tol: Tolerance in samples before and after peak to merge
        :return:
        """
        if not tol:
            tol = int(self.sublen / 2)

        allidx = np.hstack((self.peak_idx1, self.peak_idx2))

        # Find unique intervals for all window pairs
        x = allidx - tol
        y = allidx + tol
        intervals = np.array((x, y)).T
        self.merged_windows = merge_intervals(intervals)

    def associate_ids(self):
        """
        Associate windows to each index for each pair
        return: Pandas DataFrame
        https://stackoverflow.com/questions/44367672/best-way-to-join-merge-by-range-in-pandas

        :return: pandas dataframe with one row per peak and associated window ids and groups

        """
        # Setup for pandas
        lb = self.merged_windows[:, 0]  # Lower bound of windows
        ub = self.merged_windows[:, 1]  # upper bound of windows
        dfref = pd.DataFrame({"lower_bound": lb,
                              "upper_bound": ub,
                              "window_id": range(len(lb))})
        pairs = pd.DataFrame({"Index1": self.peak_idx1, "Index2": self.peak_idx2})

        # Find window id for first indices of pairs
        i, j = np.where((self.peak_idx1[:, None] >= lb) & (self.peak_idx1[:, None] <= ub))

        df = pd.DataFrame(
            np.column_stack([pairs.iloc[i], dfref.iloc[j]]),
            columns=pd.Index(data=["Index1", "Index2", "lb1", "ub1", "win_id_1"])  # df1.columns.append(dfref.columns)
        )

        # Find window id for second indices of pairs
        i, j = np.where((self.peak_idx2[:, None] >= lb) & (self.peak_idx2[:, None] <= ub))

        dfw = pd.DataFrame(
            np.column_stack([df.iloc[i], dfref.iloc[j]]),
            columns=df.columns.append(pd.Index(data=["lb2", "ub2", "win_id_2"]))  # df1.columns.append(dfref.columns)
        )

        # TODO: add group id to dfw

        return dfw, dfref

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