import pyscamp
from obspy import read, UTCDateTime, Stream
import os
import numpy as np
import sys
import time
from glob import glob
from functions import *


def get_stream(station, channel, starttime, endtime, fs):
    print("Getting data stream for station %s, channel %s..." % (station, channel))
    
    day = starttime.strftime("%Y%m%d")
    path_search = os.path.join(wf_dir, day, "BH.%s..%s.%s*" % (station, channel, day))
    file_list = glob(path_search)
    st = Stream()
    if len(file_list) > 0:
        for file in file_list:
            print("Reading file %s" % file)
            tmp = read(file, starttime=starttime, endtime=endtime)    
            if len(tmp) > 1:
                raise ValueError("More than one trace read from file, that's weird...")
            if tmp[0].stats.sampling_rate != fs:
                tmp.resample(fs)
            st.append(tmp[0])
    else:
        print("No data found for day %s" % day)
        print("\t\tSearch string was: %s" % path_search)
    
    # Fill gaps with noise
    st = fill_time_gaps_noise(st)
    
    # Convert to nm/s
    trace = st[0]    
    trace.data *= 1e18 #1e9        
    
    print("\tFinal Stream:")
    print("\tSampling rate: %f" % fs)
    print("\tStart time: %s" % trace.stats.starttime.strftime("%Y-%m-%d %H:%M:%S"))
    print("\tEnd time: %s" % trace.stats.endtime.strftime("%Y-%m-%d %H:%M:%S"))
    
    return trace

def get_mp(data, sublen_samp):
    proc_start = time.time()
    mp, mpind = pyscamp.selfjoin(data, sublen_samp, gpus=[0, 1, 2, 3], pearson=True, precision="double")
    proc_end = time.time()
    print("Finished processing component in %f seconds" % (proc_end - proc_start))
    return mp, mpind

def get_filename_root(stats, sublen_samp):
    
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

# MAIN -----------------------------------------------
station = sys.argv[1]
day_str = sys.argv[2]
starttime = UTCDateTime(day_str)
endtime = starttime + 86400

out_dir = "/home/gilbert_lab/cami_frs/scamp/borehole/matrix_profiles"
wf_dir = "/home/gilbert_lab/cami_frs/borehole_data/sac_daily_nez_500Hz/"
sublen_sec = 0.5
fs = 250.0
sublen_samp = int(sublen_sec * fs)

# Process DPN --------------------------------------
trace1 = get_stream(station=station, channel="DPN", starttime=starttime, endtime=endtime, fs=fs)
filename_root = get_filename_root(stats=trace1.stats, sublen_samp=sublen_samp)

if not os.path.exists(filename_root + ".mp"):
    
    mp1, mpind1 = get_mp(data=trace1.data, sublen_samp=sublen_samp)

    # Write to csv
    filename_root = get_filename_root(stats=trace1.stats, sublen_samp=sublen_samp)
    np.save(filename_root + ".ind", mpind1)
    del mpind1
    np.save(filename_root + ".mp", mp1)
    del mp1
    
del trace1

# Process DPE ---------------------------------------
trace2 = get_stream(station=station, channel="DPE", starttime=starttime, endtime=endtime, fs=fs)
filename_root = get_filename_root(stats=trace2.stats, sublen_samp=sublen_samp)

if not os.path.exists(filename_root + "_mp.npy"):
    
    mp2, mpind2 = get_mp(data=trace2.data, sublen_samp=sublen_samp)

    # Write to csv
    np.save(filename_root + "_ind.npy", mpind2)
    del mpind2
    np.save(filename_root + "_mp.npy", mp2)
    del mp2
    
del trace2

# Process DPZ -----------------------------------------
trace3 = get_stream(station=station, channel="DPZ", starttime=starttime, endtime=endtime, fs=fs)
filename_root = get_filename_root(stats=trace3.stats, sublen_samp=sublen_samp)

if not os.path.exists(filename_root + "_mp.npy"):
    
    mp3, mpind3 = get_mp(data=trace3.data, sublen_samp=sublen_samp)

    # Write indices to csv
#    np.savetxt(filename_root + ".ind", mpind3)
    np.save(filename_root + "_ind.npy", mpind3)
    del mpind3
    np.save(filename_root + "_mp.npy", mp3)
#    np.savetxt(filename_root + ".mp", mp3)
    del mp3
    
del trace3

# Now stack -----------------------------------------
# mpstack = (mp1 + mp2 + mp3) / 3
#mpstack = mpstack / 3

# Write to csv
#filename_root = "matrix_profiles/%s_%s_%s_stack_win%3.1fs" % (
#traceZ.stats.starttime.strftime("%Y%m%d%H%M%S"), traceZ.stats.endtime.strftime("%Y%m%d%H%M%S"), station, sublen_sec)
#np.savetxt(filename_root + ".mp", mpstack)

