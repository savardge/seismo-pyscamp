import pyscamp
from obspy import UTCDateTime
import sys
import time
from seiscamp.util import *
from seiscamp.MPLib import *
#from seiscamp.GraphLib import *

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)


def get_mp(data, sublen_samp):
    proc_start = time.time()
    mp, mpind = pyscamp.selfjoin(data, sublen_samp, gpus=[0, 1, 2, 3], pearson=True, precision="double")
#    mp, mpind = pyscamp.selfjoin(data, sublen_samp, gpus=[0, 1], pearson=True, precision="double")
    proc_end = time.time()
    Logger.info("Finished processing component in %f seconds" % (proc_end - proc_start))
    return mp, mpind


if __name__ == "__main__":

    # User inputs
    station = sys.argv[1]
    day_str = sys.argv[2]
    starttime = UTCDateTime(day_str)
    num_days = int(sys.argv[3])
    sublen_sec = float(sys.argv[4])
    chan = sys.argv[5]
    fs = float(sys.argv[6])
    Logger.info("Input parameters are:\n\tstation = %s\n\tchannel = %s\n\tday_str = %s\n\tnum_days = %d\n\tsublen_sec = %4.2f\n\tSampling rate: %.0f" % (station, chan, day_str, num_days, sublen_sec, fs))
    out_dir = "/home/genevieve.savard/seismo-pyscamp/scripts_frs/matrix_profiles"

    # Define SCAMP sliding window length
    sublen_samp = int(sublen_sec * fs)

    for channel in [chan]:
        Logger.info("Processing channel %s" % channel)
        Logger.info("Fetching data...")
        if num_days == 1:
            endtime = starttime + 86400
            trace1 = get_stream_1day(station=station, channel=channel, starttime=starttime, endtime=endtime, fs=fs)
        else:
            trace1 = get_stream_days(station=station, channel=channel, first_day=starttime, num_days=num_days, fs=fs)

        filename_root = get_filename_root(stats=trace1.stats, sublen_samp=sublen_samp, out_dir=out_dir)

        if not os.path.exists(filename_root + "_mp.npy"):  # Check file doesn't exist yet

            # Run SCAMP
            Logger.info("Running SCAMP...")
            mp1, mpind1 = get_mp(data=trace1.data, sublen_samp=sublen_samp)

            nodd = len(mp1[mp1 > 1.0])
            if nodd > 0:
                Logger.warning("WARNING! Number of MP values above 1 is %d" % nodd)

            # Make Object
#            Logger.info("Making MatrixProfile object")
#            mpObj = MatrixProfile(mp=mp1, ind=mpind1, trace=trace1, sublen=sublen_samp)

            # Write
            filename_root = get_filename_root(stats=trace1.stats, sublen_samp=sublen_samp, out_dir=out_dir)
#            mpObj.save(filename=filename_root + ".pickle")

            # To Save Matrix Profile as numpy array binaries:
            np.save(filename_root + "_ind.npy", mpind1)
            Logger.info("Wrote indices to: %s" % filename_root + "_ind.npy")
            del mpind1
            np.save(filename_root + "_mp.npy", mp1)
            Logger.info("Wrote MP to: %s" % filename_root + "_mp.npy")
            del mp1

