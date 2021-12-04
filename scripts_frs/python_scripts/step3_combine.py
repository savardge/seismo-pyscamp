from glob import glob
import os
import pandas as pd
import sys

input_dir = "/home/genevieve.savard/seismo-pyscamp/scripts_frs/detections_single"
output_dir = "/home/genevieve.savard/seismo-pyscamp/scripts_frs/detections_combined"

prefix = sys.argv[1] #"20201005000000.000000_20201005235959.996000_BH012_"
chan2 = sys.argv[3] #"HH"
suffix = sys.argv[2] #"_500Hz_win500samp_detections.csv"

filex, filey, filez = [os.path.join(input_dir, prefix + chan + suffix) for chan in [chan2+"N", chan2+"E", chan2+"Z"]]
if not all([os.path.isfile(f) for f in [filex, filey, filez]]):
    print("One of the required file doesn't exist!")
    for f in [filex, filey, filez]:
        print(f"{f}: {os.path.exists(f)}")
    sys.exit()

dfx = pd.read_csv(filex, header=0, parse_dates=["time1", "time2"])
dfx[chan2+"N"] = True
dfy = pd.read_csv(filey, header=0, parse_dates=["time1", "time2"])
dfy[chan2+"E"] = True
dfz = pd.read_csv(filez, header=0, parse_dates=["time1", "time2"])
dfz[chan2+"Z"] = True

if any([d.shape[0]==0 for d in [dfx,dfy,dfz]]):
    print("one of the dataframe is empty")
    print(f"{chan2}N: {dfx.shape[0]}, {chan2}E: {dfy.shape[0]}, {chan2}Z: {dfz.shape[0]}")
    sys.exit()

df = pd.merge_asof(dfx.sort_values(by=["time1"]), dfy.sort_values(by=["time1"]), 
                   on="time1", 
                   direction="nearest",
                   tolerance=pd.Timedelta("2s")).dropna()
df = pd.merge_asof(dfz.sort_values(by=["time1"]), df.sort_values(by=["time1"]), 
                   on="time1", 
                   direction="nearest",
                   tolerance=pd.Timedelta("2s")).dropna()
df.head()

fname = prefix + "COMBINED" + suffix
df.to_csv(os.path.join(output_dir, fname), index=False)
