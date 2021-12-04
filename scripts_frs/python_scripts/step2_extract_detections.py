import os
import numpy as np
import sys
import time
from glob import glob
from obspy import UTCDateTime
import scipy
import pandas as pd
from seiscamp.MPLib import *
from seiscamp.util import *
from seiscamp.data import *
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)

procs = time.time()

# INPUT
doplot=False
fname_mp = sys.argv[1]
input_dir = "/home/genevieve.savard/seismo-pyscamp/scripts_frs/matrix_profiles/"
fname_out = os.path.join(input_dir, fname_mp.replace("_mp.npy", "_detections.csv"))
Logger.info(f"Input filename {fname_mp}\nInput directory: {input_dir}")
Logger.info(f"Output filename {fname_out}")

if os.path.exists(fname_out):
	Logger.info("Already processed. Exiting.")
	sys.exit()

fname_ind = fname_mp.replace("_mp", "_ind")
mp = np.load(os.path.join(input_dir, fname_mp))
ind = np.load(os.path.join(input_dir, fname_ind))

station = os.path.split(fname_mp)[1].split("_")[2]
channel = os.path.split(fname_mp)[1].split("_")[3]
starttime = UTCDateTime(os.path.split(fname_mp)[1].split("_")[0])
endtime = UTCDateTime(os.path.split(fname_mp)[1].split("_")[1])
num_days = int(np.ceil((endtime - starttime)/(3600*24.)))
fs = float(os.path.split(fname_mp)[1].split("_")[4].split("Hz")[0])
sublen_samp = int(os.path.split(fname_mp)[1].split("_")[5].split("win")[1].split("samp")[0])

Logger.info("Station %s, channel %s" % (station, channel))
Logger.info("Start time %s" % starttime)
Logger.info("End time %s" % endtime)
Logger.info("# days %s" % num_days)
Logger.info("Sampling rate: %s" % fs)
Logger.info("sublen_samp = %d" % sublen_samp)
Logger.info("Length of MP: %d samples" % mp.shape[0])

# GET DATA
wf_dir = "/home/gilbert_lab/cami_frs/all_daily_symlinks"
trace1 = get_stream_days(station=station, channel=channel, first_day=starttime, num_days=num_days, fs=fs, gain=1, wf_dir=wf_dir)
trace1.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
Logger.info(trace1)
tvec = trace1.times("matplotlib")

mpObj = MatrixProfile(mp=mp, ind=ind, trace=trace1, sublen=sublen_samp)

del mp, ind

# DETECTION

# Params
N = int(sublen_samp * 0.5) # Rolling window length for median and std stats
width = int(0.75 * sublen_samp) # Minimum width of peaks
prepad = 5 # time buffer before for stack
postpad = 10
nsplit = 12. # Split days into chunks

# Initialize lists
time1 = []
time2 = []
mpval = []
prominence = []
index_std = []

# DEFINE WINDOW
num_days = round((mpObj.endtime - mpObj.starttime)/(3600*24.))
Logger.info(f"Number of days to scan: {num_days}, # chunks / day: {nsplit}")

for nday in range(int(num_days * nsplit)):
    print(f"WINDOW {nday}")
    print(f"Starting # peaks at window {nday}: {len(time1)}")
    
    # Define day window
    tutc1 = mpObj.starttime + nday * (3600.*24 / nsplit)
    tutc2 = tutc1 + (3600.*24 / nsplit)
    print(f"start time =  {tutc1}, end time = {tutc2}")
    t1 = mdates.date2num(tutc1._get_datetime())
    t2 = mdates.date2num(tutc2._get_datetime())
    i1 = np.argmax(tvec>=t1)
    i2 = np.argmax(tvec>=t2)

    if i2 == 0:
        print("Data does not extend to window's end. Stopping.")
        continue
    
    # CUT
    mp = mpObj.mp[i1:i2].copy()
    ind = mpObj.ind[i1:i2].copy()    
    tcut = tvec[i1:i2]
    
    # STATS
    mpmed = RunningMedian(mp, N=N)
    indstd = RunningStd(ind, N=N)

    # Find peaks w/ Scipy
    peaks, properties = find_peaks(mpmed,  width=width, prominence=0.1)
    print(f"# of peaks found by scipy: {len(peaks)}")
    
    # Enumerate over peaks
    for ip, p in enumerate(peaks):
        
        # Get start-end of peak
        lips = int(properties["left_ips"][ip])
        rips = int(properties["right_ips"][ip])
        
        # Get index std inside peak width and decide if keeping
        instd_imin = np.argmin(indstd[lips:rips])
        indstd_min = indstd[lips:rips][instd_imin]
        if indstd_min >= sublen_samp:
            #Logger.info("REJECTED DETECTION")            
            continue
            
        # Get index of minimum std 
        ind_min = int(ind[lips:rips][instd_imin])
        mp_min = mp[lips:rips][instd_imin]
        tpk = UTCDateTime(mdates.num2date(tcut[lips:rips][instd_imin]))
#        print(f"tpk = {tpk}")
        print(f"ind_min = {ind_min}")
        if ind_min > len(tvec):
            Logger.error("Index is higher than vector length! Skipping")
            continue
        tpk2 = UTCDateTime(mdates.num2date(tvec[ind_min]))
#        print(f"tpk2 = {tpk2}")

        # Get stack for more precise times of peak       
        data1 = trace1.slice(starttime=tpk - prepad, endtime=tpk + postpad).copy().normalize().trim(starttime=tpk - prepad, endtime=tpk + postpad, pad=True, fill_value=0)
        data2 = trace1.slice(starttime=tpk2 - prepad, endtime=tpk2 + postpad).copy().normalize().trim(starttime=tpk2 - prepad, endtime=tpk2 + postpad, pad=True, fill_value=0)
        stack = np.abs(scipy.signal.hilbert(data1.data)) + np.abs(scipy.signal.hilbert(data2.data))
        imax = np.argmax(stack)
        tstack1 = data1.times("matplotlib")[imax]
        tstack2 = data2.times("matplotlib")[imax]
        
        # Now add values to lists
        time1.append(tstack1)
        time2.append(tstack2)
        mpval.append(mp_min)
        prominence.append(properties["prominences"][ip])
        index_std.append(indstd_min / fs)
        
    Logger.info(f"# of peaks after scanning window {nday}: {len(time1)}")
    
# Now build dataframe
Logger.info("Now saving detections to dataframe")
tutc1 = [UTCDateTime(mdates.num2date(t)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4] for t in time1]
tutc2 = [UTCDateTime(mdates.num2date(t)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4] for t in time2]
dfpeaks = pd.DataFrame({"time1": tutc1, 
                       "time2": tutc2, 
                       "mpval": mpval, 
                       "prominence": prominence,
                       "peak_std_sec": index_std})
dfpeaks = dfpeaks.round({'mpval': 4, 'prominence': 4, 'index_std': 0})
dfpeaks.to_csv(fname_out, index=False)
Logger.info(f"Saved results to {fname_out}")

proce = time.time()
Logger.info(f"Total processing time: {proce-procs} s.")
