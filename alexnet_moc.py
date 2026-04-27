import math 
import sys
import numpy as np 
from pathlib import Path 
from utils import lipschitz_from_fmoc, pad_moc_with_last, save_moc
sys.path.append(str(Path("fmca") / "build" / "py"))
import FMCA

np.random.seed(0)
save_path = Path("experiments")/"imagenet"
filename = Path("data")/"imagenet"
norms = ["EUCLIDEAN", "TAXICAB"]

N = 1000000
C = 10000 #batch size that fits in memory
B = math.floor(N/C) #(each batch shall have same size)

#compute u_b, l_b for TX, qX 
qX = 0.0001
TX = 1000
nbins =10000

dataList = [] #containing data paths? 


for norm in norms:

    #load the reference dataset, in chunks if it is too large.

    Pdata = dataList[C:C+C]

    dmoc = FMCA.DiscreteModulusOfContinuity()
    dmoc.init(P,F, TX,qX, nbins, norm, norm)
    final_moc = dmoc.omegat()
    t_values = dmoc.tgrid()
    NT = len(t_values)


    save_moc(t_values,folder_path, f"batchdeltas_dmoc_{C}_{norm[0]}")

    #save on disk the very first grid, it will be the same for every other batch

    for b in range(1,B):
        Pdata = dataList[b*C:b*C+C]
        #build actual P matrix

        #also build corresponding f matrix

        #dmoc 
        dmoc = FMCA.DiscreteModulusOfContinuity()
        dmoc.init(P,F, TX,qX, nbins, norm, norm)
        batch_m = dmoc.omegat()
        
        for i in range(NT):
            final_moc[i] = max(final_moc[i], batch_m[i])

        #data_t = time.time() - start_time 

    save_moc(final_moc,folder_path, f"batchdata_dmoc_{C}_{norm[0]}")



