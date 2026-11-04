from obspy import read, Stream
from glob import glob
import os
import logging
from seiscamp.util import *

Logger = logging.getLogger(__name__)

DEFAULT_WF_DIR = os.environ.get("SEISCAMP_WF_DIR", "/mnt/cmc3/")


def day_directory(wf_dir, day):
    """
    Return the directory holding one day of daily waveform files.

    Two archive layouts are supported, because the same pipeline runs on two
    machines whose archives are organised differently:

    * ``{wf_dir}/YYYYMMDD/`` -- flat day folders (the ``/mnt/cmc3`` mirror on
      work-lumidas-1);
    * ``{wf_dir}/YYYY/DDD/`` -- year / julian-day tree (the
      ``/home/gilbert_lab/cami_frs/all_daily_symlinks`` archive on Glados).

    The flat layout is tried first; if that folder does not exist the
    year/julian-day path is returned (whether or not it exists, so the
    caller's "no data found" message still names a sensible path).

    :type wf_dir: str
    :param wf_dir: Root of the waveform archive.
    :type day: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param day: Any time within the requested day (UTC).
    :rtype: str
    :return: Path of the day's directory.

    >>> from obspy import UTCDateTime
    >>> day_directory("/archive", UTCDateTime("2022-07-19"))  # no flat folder
    '/archive/2022/200'
    """
    flat = os.path.join(wf_dir, day.strftime("%Y%m%d"))
    if os.path.isdir(flat):
        return flat
    return os.path.join(wf_dir, day.strftime("%Y"), day.strftime("%j"))


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


def get_stream_1day(station, channel, starttime, endtime, fs, gain=1e18, wf_dir=None):
    Logger.info("Getting data stream for station %s, channel %s..." % (station, channel))

    if not wf_dir:
        wf_dir = DEFAULT_WF_DIR
    Logger.info("Getting data stream from directory %s..." % (wf_dir))
    
    day = starttime.strftime("%Y%m%d")
    path_search = os.path.join(day_directory(wf_dir, starttime),
                               "*.%s..%s*" % (station, channel))
    
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


def get_stream_days(station, channel, first_day, num_days, fs, gain=1e18, wf_dir=None):
    Logger.info("Getting data stream for station %s, channel %s..." % (station, channel))

    if not wf_dir:
        wf_dir = DEFAULT_WF_DIR
    Logger.info("Getting data stream from directory %s..." % (wf_dir))
    
    days = [first_day + n*24*3600 for n in range(num_days)]
    
    st = Stream()
    for day in days:
        daystr = day.strftime("%Y%m%d")
        path_search = os.path.join(day_directory(wf_dir, day),
                                   "*.%s..%s*" % (station, channel))

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