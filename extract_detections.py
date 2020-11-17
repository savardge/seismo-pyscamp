import matplotlib.pyplot as plt
from obspy import UTCDateTime
import time
from util import *
from MPLib import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)

# Input
fname_mp = "data_borehole/matrix_profiles/20200310000000.000000_20200310235959.996000_G12_DPZ_250Hz_win125samp_mp.npy"
fname_ind = "data_borehole/matrix_profiles/20200310000000.000000_20200310235959.996000_G12_DPZ_250Hz_win125samp_ind.npy"
mp = np.load(fname_mp)
ind = np.load(fname_ind)

station = os.path.split(fname_mp)[1].split("_")[2]
channel = os.path.split(fname_mp)[1].split("_")[3]
print("Station %s, channel %s" % (station, channel))
starttime = UTCDateTime(os.path.split(fname_mp)[1].split("_")[0])
endtime = UTCDateTime(os.path.split(fname_mp)[1].split("_")[1])
fs = float(os.path.split(fname_mp)[1].split("_")[4].split("Hz")[0])
print("Start time %s" % starttime)
print("End time %s" % endtime)
print("Sampling rate: %s" % fs)

sublen_samp = int(os.path.split(fname_mp)[1].split("_")[5].split("win")[1].split("samp")[0])
print("sublen_samp = %d" % sublen_samp)
print("Length of MP: %d samples" % mp.shape[0])

mpObj = MatrixProfile(mp=mp, ind=ind, station=station, channel=channel, fs=fs, sublen=sublen_samp,
                      starttime=starttime, endtime=endtime)


# Get seismogram
#trace = get_stream_1day(station=station, channel=channel, starttime=starttime, endtime=endtime, fs=fs)
trace = read("data_borehole/waveforms/*.sac")[0]
mpObj.add_trace(trace)

# Get peaks in MP
procs = time.time()
mpObj.get_peaks(mad_thresh_mult=6)
proce = time.time()
print("Time for find_peaks: %f s." % (proce - procs))

# Find connected groups
cc, df, dfref = mpObj.group_ids()

# Plot
# mpObj.plot_all_pairs(outdir="/Users/genevieve/seismo-pyscamp/data_borehole/figs/all_pairs_bygroup")
mpObj.plot_group(outdir="/Users/genevieve/seismo-pyscamp/data_borehole/figs/groups")
