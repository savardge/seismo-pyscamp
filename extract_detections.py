import matplotlib.pyplot as plt
from obspy import UTCDateTime, read
import time
from statsmodels.robust import mad
from util import *
from GraphLib import *
from MPLib import *


# Input
fname_mp = "matrix_profiles/20200310000000.000000_20200310235959.996000_G12_DPZ_250Hz_win125samp_mp.npy"
fname_ind = "matrix_profiles/20200310000000.000000_20200310235959.996000_G12_DPZ_250Hz_win125samp_mp.npy"
mp = np.load("matrix_profiles/20200310000000.000000_20200310235959.996000_G12_DPZ_250Hz_win125samp_mp.npy")
ind = np.load("matrix_profiles/20200310000000.000000_20200310235959.996000_G12_DPZ_250Hz_win125samp_ind.npy")

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
trace = get_stream(station=station, channel=channel, starttime=starttime, endtime=endtime, fs=fs)
mpObj.add_trace(trace)

# Get peaks in MP
lowthresh = np.median(mp) + 5 * mad(mp)
procs = time.time()
mpObj.get_peaks(mad_mul_thresh=6)
proce = time.time()
print("Time for find_peaks: %f s." % (proce - procs))

# Get index pairs corresponding to peaks and associate to unique windows
indices1 = peaks.astype(int)
indices2 = np.take(ind, indices1).astype(int)
merged_windows = idx_to_windows(indices1, indices2, sublen_samp)
df, dfref = associate_ids(merged_windows, indices1, indices2)

# Initialize graph
print("Building the graph...")
nrow = len(dfref)
print(nrow)
g = Graph(nrow)
for i, row in df.iterrows():
    id1 = row["win_id_1"]
    id2 = row["win_id_2"]
    g.add_edge(id1, id2)

print("Getting connected components...")
procs = time.time()
cc = g.connected_components()
proce = time.time()
print("Time to get connected components: %f s." % (proce - procs))
cc = sorted(cc, key=lambda x: len(x), reverse=True)  # Sort by decreasing # of links
tstart = starttime + sublen_samp / fs
print(tstart)

plt.close("all")
for group in cc:
    ndet = len(group)
    print(ndet)

    if ndet > 40:
        continue

    if ndet > 1:

        N = ndet
        fig, axs = plt.subplots(N, 1, figsize=(12, 3 * N))
        for i, wid in enumerate(group[:N]):
            lb = dfref.loc[dfref["window_id"] == wid]["lower_bound"].values[0]
            ub = dfref.loc[dfref["window_id"] == wid]["upper_bound"].values[0]
            wstart = tstart + lb / fs
            wend = tstart + ub / fs
            detdata = trace.slice(starttime=wstart - 1.0, endtime=wend + 1.0).data
            axs[i].plot(detdata, "k")
            axs[i].set_title("%s" % wstart)
        plt.show()
        plt.close()

    if ndet == 1:

        wid = group[0]
        print("ID: %d" % wid)
        row = dfref.loc[dfref["window_id"] == wid]
        if len(row) > 0:
            lb = dfref.loc[dfref["window_id"] == wid]["lower_bound"].values[0]
            ub = dfref.loc[dfref["window_id"] == wid]["upper_bound"].values[0]
            wstart = tstart + lb / fs
            wend = tstart + ub / fs
            detdata = trace.slice(starttime=wstart - 1.0, endtime=wend + 1.0).data
            plt.plot(detdata, "k")
            plt.set_title("%s" % wstart)
        else:
            print("Window ID not found in dfref?")

    print("%s" % "*" * 100)