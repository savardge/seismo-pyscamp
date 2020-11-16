import numpy as np
from glob import glob
import os
import sys

f = sys.argv[1]

#flist = glob("/home/gilbert_lab/cami_frs/scamp/borehole/matrix_profiles/*.ind")
#for f in flist:
#    print(f)
a = np.loadtxt(f)
if ".ind" in f:
    fname = f.replace(".ind", "_ind.npy")
if ".mp" in f:
    fname = f.replace(".mp", "_mp.npy")

if fname:
    np.save(fname, a)
    if os.path.exists(fname):
        os.remove(f)