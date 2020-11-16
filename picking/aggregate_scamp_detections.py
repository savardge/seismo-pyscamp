from obspy import read, read_events, UTCDateTime
import os
from phasepapy.phasepicker import ktpicker, fbpicker, aicdpicker
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from obspy.core.event import Pick, WaveformStreamID
from obspy.core.event.base import QuantityError, Comment
import time
import sys
from glob import glob

def get_picks(tr, picker="aic", show_plot=False):
    if picker == "kt":  # PhasePApy Kurtosis Picker

        picker = ktpicker.KTPicker(t_win=0.04, t_ma=0.12, nsigma=6, t_up=0.02, nr_len=0.06,
                                   nr_coeff=2, pol_len=10, pol_coeff=10, uncert_coeff=3)
        scnl, picks, polarity, snr, uncert = picker.picks(tr)
        if show_plot:
            print('scnl:', scnl)
            print('picks:', picks)
            print('polarity:', polarity)
            print('signal to noise ratio:', snr)
            print('uncertainty:', uncert)
            summary = ktpicker.KTSummary(picker, tr)
            summary.plot_summary()
            summary.plot_picks()

    elif picker == "fb":  # PhasePApy FBPicker

        picker = fbpicker.FBPicker(t_long=0.05, freqmin=10, mode='rms', t_ma=0.12, nsigma=6,
                                   t_up=0.02, nr_len=2, nr_coeff=2, pol_len=10, pol_coeff=10, uncert_coeff=3)
        scnl, picks, polarity, snr, uncert = picker.picks(tr)
        if show_plot:
            print('scnl:', scnl)
            print('picks:', picks)
            print('polarity:', polarity)
            print('signal to noise ratio:', snr)
            print('uncertainty:', uncert)
            summary = fbpicker.FBSummary(picker, tr)
            summary.plot_bandfilter()
            summary.plot_statistics()
            summary.plot_summary()

    elif picker == "aic":  # PhasePApy AIC differentiation picker

        picker = aicdpicker.AICDPicker(t_ma=0.12, nsigma=6, t_up=0.02, nr_len=0.06, nr_coeff=2, pol_len=10,
                                       pol_coeff=10, uncert_coeff=3)
        scnl, picks, polarity, snr, uncert = picker.picks(tr)
        if show_plot:
            print('scnl:', scnl)
            print('picks:', picks)
            print('polarity:', polarity)
            print('signal to noise ratio:', snr)
            print('uncertainty:', uncert)
            summary = aicdpicker.AICDSummary(picker, tr)
            summary.plot_summary()
            summary.plot_picks()
    else:
        raise ValueError("Unrecognized picker option")

#    print("Found %d picks." % len(picks))
    return scnl, picks, polarity, snr, uncert


def plot_picks(ax, tr, method="aic", color="red"):
    transform = ax.get_xaxis_transform()
    scnl, picks, polarity, snr, uncert = get_picks(tr, picker=method, show_plot=False)
    for ind, pick in enumerate(picks):
        if snr[ind] > 0:
            ax.axvline(mdates.date2num(pick._get_datetime()), color=color, linestyle="-")
            plt.text(mdates.date2num(pick._get_datetime()), .1+ind*0.2, "SNR=%f" % snr[ind], transform=transform)
            plt.text(mdates.date2num(pick._get_datetime()), .2+ind*0.2, "uncert=%f" % uncert[ind], transform=transform)


def add_picks(tr, method, prev_picks, pick_tol=0.025):
    wav_id = WaveformStreamID(station_code=tr.stats.station,
                              channel_code=tr.stats.channel,
                              network_code=tr.stats.network)
    scnl, tpicks, polarity, snr, uncert = get_picks(tr, picker=method, show_plot=False)
    for ind, tpick in enumerate(tpicks):
        p = Pick(time=tpick,
                 waveform_id=wav_id,
                 time_errors=QuantityError(uncertainty=uncert[ind]),
                 method_id=method,
                 comments=[Comment(text="SNR = %f" % snr[ind])])
        # Check if there is a pick within pick tolerance threshold
        if prev_picks:
            prev_tpick = [pick.time for pick in prev_picks]
            if np.abs(np.array(prev_tpick) - p.time).min() < pick_tol:
                ix = np.abs(np.array(prev_tpick) - p.time).argmin()
                if prev_picks[ix].time < p.time:
                    #print("This pick is within pick_tol from previous pick. Keeping previous pick.")
                    continue  # Don't add pick
                else:
                    #print("This pick is within pick_tol from previous pick. Keeping this new pick.")
                    prev_picks.remove(prev_picks[ix])
                    prev_picks.append(p)
        else:
            #print("No previous pick. Appending this one.")
            prev_picks = [p]

    return prev_picks


#########################
proc_start = time.time()

#wf_dir = "/home/gilbert_lab/cami_frs/hawk_data/sac_data_corrected/X7_Jan-Feb2020_sac_daily_250Hz/"
#wf_dir = "/home/gilbert_lab/cami_frs/hawk_data/sac_data_raw/X6_Nov-Dec2019_sac_daily_250Hz/"
wf_dir = "/home/gilbert_lab/cami_frs/hawk_data/sac_data_raw/X8_March-April2020_sac_daily_250Hz/"
wf_dir2 = "/home/gilbert_lab/cami_frs/hawk_data/sac_data_raw/X9_May2020_sac_daily_250Hz/"

file = sys.argv[1]
fname = os.path.split(file)[1].split(".xml")[0]
date_range = "%s_%s" % (fname.split("_")[2], fname.split("_")[3])
station = fname.split("_")[4]


cat = read_events(file)
buf = 15
pick_tol = 0.025
for event in cat.events:
    tO_utc = event.origins[0].time
    if len( glob( os.path.join(wf_dir, station, "*%s*" % tO_utc.strftime("%Y%m%d")) ) ) > 0:
        st = read(os.path.join(wf_dir, station, "*%s*" % tO_utc.strftime("%Y%m%d")),
                  starttime=tO_utc-buf, endtime=tO_utc+buf)
    else:
        st = read(os.path.join(wf_dir2, station, "*%s*" % tO_utc.strftime("%Y%m%d")),
                  starttime=tO_utc-buf, endtime=tO_utc+buf)
    st.decimate(factor=2)
    st.detrend()
    st.filter("bandpass", freqmin=10, freqmax=60)

    sta_picks = []
    for tr in st.traces:
        # Pick using AIC Picker
        sta_picks = add_picks(tr, method="aic", prev_picks=sta_picks, pick_tol=pick_tol)

        # Pick using FB Picker
        sta_picks = add_picks(tr, method="fb", prev_picks=sta_picks, pick_tol=pick_tol)

        # Pick using kurtosis
        sta_picks = add_picks(tr, method="kt", prev_picks=sta_picks, pick_tol=pick_tol)
    event.picks = sta_picks

cat.write(file.split(".xml")[0] + "_wpicks.xml", format="QUAKEML")

proc_end = time.time()

print("File processed in %f seconds." % (proc_end - proc_start))
