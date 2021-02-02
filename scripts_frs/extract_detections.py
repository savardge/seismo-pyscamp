import matplotlib.pyplot as plt
from obspy import UTCDateTime
import time
from util import *
from MPLib import *
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)

# Input
doplot = False
mad_thresh_mult = 6
fname_mp = sys.argv[1]
#fname_mp = "data_borehole/matrix_profiles/20200310000000.000000_20200310235959.996000_G12_DPZ_250Hz_win125samp_mp.npy"
detect_dir = "/home/genevieve.savard/seismo-pyscamp/data_borehole/detections"

fname_ind = fname_mp.replace("_mp.npy", "_ind.npy")
Logger.info("MP file: %s" % fname_mp)
Logger.info("Indices file: %s" % fname_ind)

# Get metadata
Logger.info("Getting metadata...")
station = os.path.split(fname_mp)[1].split("_")[2]
channel = os.path.split(fname_mp)[1].split("_")[3]
print("Station %s, channel %s" % (station, channel))
starttime = UTCDateTime(os.path.split(fname_mp)[1].split("_")[0])
endtime = UTCDateTime(os.path.split(fname_mp)[1].split("_")[1])
fs = float(os.path.split(fname_mp)[1].split("_")[4].split("Hz")[0])
Logger.info("Start time %s" % starttime)
Logger.info("End time %s" % endtime)
Logger.info("Sampling rate: %s" % fs)
sublen_samp = int(os.path.split(fname_mp)[1].split("_")[5].split("win")[1].split("samp")[0])
Logger.info("sublen_samp = %d" % sublen_samp)

# Load MP and indices
Logger.info("Loading data...")
mp = np.load(fname_mp)
ind = np.load(fname_ind)
Logger.info("Length of MP: %d samples" % mp.shape[0])

# Build MP object
mpObj = MatrixProfile(mp=mp, ind=ind, station=station, channel=channel, fs=fs, sublen=sublen_samp,
                      starttime=starttime, endtime=endtime)

# Get peaks in MP
procs = time.time()
mpObj.get_peaks(mad_thresh_mult=mad_thresh_mult)
proce = time.time()
Logger.info("Time for find_peaks: %f s." % (proce - procs))

# Find connected groups
cc, pairs, ref_windows = mpObj.group_ids()

# Save detections
fname_root = os.path.split(fname_mp)[1].split("_mp.npy")[0]
fname_pairs = os.path.join(detect_dir, fname_root + "_pairs.csv")
fname_ref = os.path.join(detect_dir, fname_root + "_refwin.csv")
pairs.to_csv(fname_pairs, index=False)
ref_windows.to_csv(fname_ref, index=False)

if doplot:
    # Get seismogram
    #trace = get_stream_1day(station=station, channel=channel, starttime=starttime, endtime=endtime, fs=fs)
    trace = read("data_borehole/waveforms/*.sac")[0]
    mpObj.add_trace(trace)

    # Plot
    # mpObj.plot_all_pairs(outdir="/Users/genevieve/seismo-pyscamp/data_borehole/figs/all_pairs_bygroup")
    mpObj.plot_group(outdir="/Users/genevieve/seismo-pyscamp/data_borehole/figs/groups")
