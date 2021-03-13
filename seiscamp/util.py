import numpy as np
from statsmodels import robust
from obspy import read, Stream
from glob import glob
import os

import logging

Logger = logging.getLogger(__name__)

#WF_DIR = "/home/gilbert_lab/cami_frs/borehole_data/sac_daily_nez_500Hz/"
WF_DIR = "/home/gilbert_lab/cami_frs/all_daily_symlinks/"


def detect_time_gaps(trace, min_samples=10, epsilon=1e-20, thresh_disc=100):
    """
    Detect time gaps where data is filled with zeros for a given Obspy trace.
    :param trace: obspy.core.trace
    :param min_samples: Minimum number of samples for time gap
    :param epsilon: Machine precision (use to define zero)
    :param thresh_disc: Threshold for discontinuity for multiple gaps
    :return: num_gaps, gap_start_ind, gap_end_ind
    """

    tdata = trace.data

    indz = np.where(abs(tdata) < epsilon)[0]  # indices where we have 0
    diff_indz = indz[min_samples:] - indz[
                                     0:-min_samples]  # Need min_samples consecutive samples with 0's to identify as time gap
    ind_des = np.where(diff_indz == min_samples)[0]  # desired indices: value is equal to min_samples in the time gap
    ind_gap = indz[ind_des]  # indices of the time gaps

    gap_start_ind = []
    gap_end_ind = []

    if 0 == len(ind_gap):  # No time gaps found
        num_gaps = 0
    else:
        # May have more than 1 time gap
        ind_diff = np.diff(ind_gap)  # discontinuities in indices of the time gaps, if there is more than 1 time gap
        ind_disc = np.where(ind_diff > thresh_disc)[0]

        # N-1 time gaps
        # Logger.info(ind_gap)
        curr_ind_start = ind_gap[0]
        # Logger.info(curr_ind_start)
        for igap in range(len(ind_disc)):  # do not enter this loop if ind_disc is empty
            gap_start_ind.append(curr_ind_start)
            last_index = ind_gap[ind_disc[igap]] + min_samples
            gap_end_ind.append(last_index)
            curr_ind_start = ind_gap[ind_disc[igap] + 1]  # update for next iteration

        # Last time gap
        gap_start_ind.append(curr_ind_start)
        gap_end_ind.append(ind_gap[-1] + min_samples)
        num_gaps = len(gap_start_ind)

    return [num_gaps, gap_start_ind, gap_end_ind]


def fill_time_gaps_noise(stream, min_samples=10, epsilon=1e-20, thresh_disc=100, ntest=2000, verbose=False):
    """
    Fill time gaps where data is filled with zeros for a given Obspy stream.
    :param stream: obspy.core.stream
    :param min_samples: Minimum number of samples for time gap
    :param epsilon: Machine precision (use to define zero)
    :param thresh_disc: Threshold for discontinuity for multiple gaps
    :param ntest: Number of test samples in data - assume they are noise
    :param verbose: Show details
    :return: stream filled with noise for time gaps.
    """

    for trace in stream:

        dt = trace.stats.delta

        # 1) Find time gaps
        [num_gaps, gap_start_ind, gap_end_ind] = detect_time_gaps(trace, min_samples, epsilon, thresh_disc)

        if verbose and num_gaps > 0:
            Logger.info('Number of time gaps for %s: %d' % (trace.id, num_gaps))
            for igap in range(num_gaps):
                tstart = trace.stats.starttime + gap_start_ind[igap] * dt
                duration = (gap_end_ind[igap] - gap_start_ind[igap]) * dt
                Logger.info("Time: %s,  Duration: %d s." % (tstart, duration))

        # 2) Fill with uncorrelated noise

        for igap in range(num_gaps):
            ngap = gap_end_ind[igap] - gap_start_ind[igap] + 1

            # Bounds check
            if gap_start_ind[igap] - ntest < 0:  # not enough data on left side
                # median of ntest noise values on right side of gap
                median_gap_right = np.median(trace.data[gap_end_ind[igap] + 1:gap_end_ind[igap] + ntest])
                median_gap = median_gap_right
                # MAD of ntest noise values on right side of gap
                mad_gap_right = robust.mad(trace.data[gap_end_ind[igap] + 1:gap_end_ind[igap] + ntest])
                mad_gap = mad_gap_right
            elif gap_end_ind[igap] + ntest >= len(trace.data):  # not enough data on left side
                # median of ntest noise values on left side of gap
                median_gap_left = np.median(trace.data[gap_start_ind[igap] - ntest:gap_start_ind[igap] - 1])
                median_gap = median_gap_left
                # MAD of ntest noise values on left side of gap
                mad_gap_left = robust.mad(trace.data[gap_start_ind[igap] - ntest:gap_start_ind[igap] - 1])
                mad_gap = mad_gap_left
            else:
                # median of ntest noise values on left side of gap
                median_gap_left = np.median(trace.data[gap_start_ind[igap] - ntest:gap_start_ind[igap] - 1])
                # median of ntest noise values on right side of gap
                median_gap_right = np.median(trace.data[gap_end_ind[igap] + 1:gap_end_ind[igap] + ntest])

                median_gap = 0.5 * (median_gap_left + median_gap_right)
                # MAD of ntest noise values on left side of gap
                mad_gap_left = robust.mad(trace.data[gap_start_ind[igap] - ntest:gap_start_ind[igap] - 1])
                # MAD of ntest noise values on right side of gap
                mad_gap_right = robust.mad(trace.data[gap_end_ind[igap] + 1:gap_end_ind[igap] + ntest])
                mad_gap = 0.5 * (mad_gap_left + mad_gap_right)

            # Fill in gap with uncorrelated white Gaussian noise
            gap_x = mad_gap * np.random.randn(ngap) + median_gap
            trace.data[gap_start_ind[igap]:gap_end_ind[igap] + 1] = gap_x

    return stream


def get_stream_1day(station, channel, starttime, endtime, fs, gain=1e18):
    Logger.info("Getting data stream for station %s, channel %s..." % (station, channel))

    #day = starttime.strftime("%Y%m%d")
    #path_search = os.path.join(WF_DIR, day, "*.%s*%s*" % (station, channel))
    day = starttime.strftime("%j")
    year = starttime.strftime("%Y")
    path_search = os.path.join(WF_DIR, year, day, "*.%s..%s*" % (station, channel))
    
    file_list = glob(path_search)
    st = Stream()
    if len(file_list) > 0:
        for file in file_list:
            Logger.info("Reading file %s" % file)
            tmp = read(file, starttime=starttime, endtime=endtime)
            if len(tmp) > 1:
                raise ValueError("More than one trace read from file, that's weird...")
            if tmp[0].stats.sampling_rate != fs:
                tmp.resample(fs)
            st.append(tmp[0])
    else:
        Logger.info("No data found for day %s" % day)
        Logger.info("\t\tSearch string was: %s" % path_search)

    # Fill gaps with noise
    st = fill_time_gaps_noise(st)

    # Convert to nm/s
    trace = st[0]
    trace.data *= gain

    Logger.info("\tFinal Stream:")
    Logger.info("\tSampling rate: %f" % fs)
    Logger.info("\tStart time: %s" % trace.stats.starttime.strftime("%Y-%m-%d %H:%M:%S"))
    Logger.info("\tEnd time: %s" % trace.stats.endtime.strftime("%Y-%m-%d %H:%M:%S"))

    return trace


def get_stream_days(station, channel, first_day, num_days, fs, gain=1e18):
    Logger.info("Getting data stream for station %s, channel %s..." % (station, channel))

    days = [first_day + n*24*3600 for n in range(num_days)]
    
    st = Stream()
    for day in days:
        #daystr = day.strftime("%Y%m%d")
        #path_search = os.path.join(WF_DIR, daystr, "*.%s*%s*" % (station, channel))
        daystr = day.strftime("%j")
        year = day.strftime("%Y")
        path_search = os.path.join(WF_DIR, year, daystr, "*.%s..%s*" % (station, channel))

        Logger.info("Looking for data for day %s" % daystr)
        file_list = glob(path_search)
        
        if len(file_list) > 0:
            for file in file_list:
                Logger.info("Reading file %s" % file)
                tmp = read(file)
                tmp.merge(method=1)
                if len(tmp) > 1:
                    raise ValueError("More than one trace read from file, that's weird...")
                if tmp[0].stats.sampling_rate != fs:
                    tmp.resample(fs)
                st.append(tmp[0])
        else:
            Logger.info("No data found for day %s" % day)
            Logger.info("\t\tSearch string was: %s" % path_search)
    
    # Merge
    Logger.info("Merging stream")
    st.merge(method=1, fill_value=0)
    Logger.info(st)
    
    # Fill gaps with noise
    st = fill_time_gaps_noise(st)

    # Convert to nm/s
    trace = st[0]
    trace.data *= gain

    # High-pass at 2 Hz
    trace.filter("highpass", freq=2)
    
    Logger.info("Final trace: ")
    Logger.info(trace)
    
    Logger.info("\tFinal Stream:")
    Logger.info("\tSampling rate: %f" % fs)
    Logger.info("\tStart time: %s" % trace.stats.starttime.strftime("%Y-%m-%d %H:%M:%S"))
    Logger.info("\tEnd time: %s" % trace.stats.endtime.strftime("%Y-%m-%d %H:%M:%S"))

    return trace


def get_filename_root(stats, sublen_samp, out_dir):
    # times
    trace_start = stats.starttime
    trace_end = stats.endtime
    trace_start.precision = 3
    trace_end.precision = 3

    station = stats.station
    channel = stats.channel
    fs = stats.sampling_rate

    filename_root = "%s_%s_%s_%s_%dHz_win%dsamp" % (
        trace_start.strftime("%Y%m%d%H%M%S.%f"),
        trace_end.strftime("%Y%m%d%H%M%S.%f"),
        station,
        channel,
        int(round(fs, 0)),
        sublen_samp)
    filepath = os.path.join(out_dir, filename_root)

    return filepath
