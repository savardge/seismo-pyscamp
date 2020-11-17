import pyscamp
from obspy import UTCDateTime
import sys
import time
from util import *
from MPLib import *
from GraphLib import *

import logging
Logger = logging.getLogger(__name__)


def get_mp(data, sublen_samp):
    proc_start = time.time()
    mp, mpind = pyscamp.selfjoin(data, sublen_samp, gpus=[0, 1, 2, 3], pearson=True, precision="double")
    proc_end = time.time()
    Logger.info("Finished processing component in %f seconds" % (proc_end - proc_start))
    return mp, mpind


if __name__ == "__main__":

    # User inputs
    station = sys.argv[1]
    day_str = sys.argv[2]
    starttime = UTCDateTime(day_str)
    num_days = int(day_str[3])
    out_dir = "/home/genevieve.savard/seismo-pyscamp/borehole_data/matrix_profiles"
    wf_dir = "/home/gilbert_lab/cami_frs/borehole_data/sac_daily_nez_500Hz/"

    # Define SCAMP sliding window length
    sublen_sec = 0.5
    fs = 250.0
    sublen_samp = int(sublen_sec * fs)

    for channel in ["DPN", "DPE", "DPZ"]:
        Logger.info("Processing channel %s" % channel)
        Logger.info("Fetching data...")
        if num_days == 1:
            endtime = starttime + 86400
            trace1 = get_stream_1day(station=station, channel=channel, starttime=starttime, endtime=endtime, fs=fs)
        else:
            trace1 = get_stream_1day(station=station, channel=channel, starttime=starttime, num_days=num_days, fs=fs)

        filename_root = get_filename_root(stats=trace1.stats, sublen_samp=sublen_samp)

        if not os.path.exists(filename_root + ".pickle"):  # Check file doesn't exist yet

            # Run SCAMP
            Logger.info("Running SCAMP...")
            mp1, mpind1 = get_mp(data=trace1.data, sublen_samp=sublen_samp)

            # Make Object
            Logger.info("Making MatrixProfile object")
            mpObj = MatrixProfile(mp=mp1, ind=mpind1, trace=trace1, sublen=sublen_samp)

            # Write
            filename_root = get_filename_root(stats=trace1.stats, sublen_samp=sublen_samp, out_dir=out_dir)
            mpObj.save(filename=filename_root + ".pickle")

            # To Save Matrix Profile as numpy array binaries:
            # np.save(filename_root + ".ind", mpind1)
            # del mpind1
            # np.save(filename_root + ".mp", mp1)
            # del mp1

