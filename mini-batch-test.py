import math 
import sys
import time 
import csv
import numpy as np 
from pathlib import Path 
from utils import lipschitz_from_fmoc, save_moc
sys.path.append(str(Path("fmca") / "build" / "py"))
import FMCA

np.random.seed(0)
save_path = Path("experiments")
norms = ["EUCLIDEAN", "TAXICAB"]
folder_path = save_path / "MNIST" /"minibatch"
folder_path.mkdir(parents=True, exist_ok=True)
filename = Path("data")/ "MNIST"


#compute u_b, l_b for TX, qX 
qX = 1.64
TX = 52
nbins =10000
#norms = ["EUCLIDEAN", "TAXICAB"]

i=0
X_train = np.loadtxt(filename/("X_train.csv"), delimiter=",",ndmin=2)
X_test = np.loadtxt(filename/("X_test.csv"), delimiter=",",ndmin=2)
Y_train = np.loadtxt(filename/("Y_train.csv"), delimiter="," , ndmin=2)
Y_test = np.loadtxt(filename/("Y_test.csv"), delimiter="," , ndmin=2)
F_train = np.loadtxt(filename/(f"F_train_{i}.csv"), delimiter="," , ndmin=2)
F_test = np.loadtxt(filename/(f"F_test_{i}.csv"), delimiter="," , ndmin=2)
# F_untrain = np.loadtxt(filename/(f"F_un_train_{i}.csv"), delimiter="," , ndmin=2)
# F_untest = np.loadtxt(filename/(f"F_un_test_{i}.csv"), delimiter="," , ndmin=2)

X_union = np.vstack([X_train, X_test])
#Y_union = np.vstack([Y_train, Y_test])
F_union = np.vstack([F_train, F_test])
# F_un_union = np.vstack([F_untrain, F_untest])

Pfull = X_union.transpose()
Ffull = F_union.transpose()
N = len(X_union)

for C in [10, 100, 1000, 10000, N]:

    B = math.floor(N/C) #(each batch shall have same size)
    print(f"n. batches {B}")

    for norm in ["EUCLIDEAN", "TAXICAB"]:
       
        P = Pfull[:, :C]
        F = Ffull[:, :C]

        start_time = time.time()
        dmoc = FMCA.DiscreteModulusOfContinuity()
        dmoc.init(P,F, TX,qX, nbins, norm, norm)
        final_moc = dmoc.omegat()
        t_values = dmoc.tgrid()
        NT = len(t_values)
        b_time = time.time()- start_time  

        #save on disk the very first grid, it will be the same for every other batch

        for b in range(1,B):
            P = Pfull[:,b*C:b*C+C]
            F = Ffull[:, b*C:b*C+C]

            #dmoc 
            dmoc = FMCA.DiscreteModulusOfContinuity()
            dmoc.init(P,F, TX,qX, nbins, norm, norm)
            batch_m = dmoc.omegat()
            
            for i in range(NT):
                final_moc[i] = max(final_moc[i], batch_m[i])

            b_time += time.time() - start_time 
    
        #save on disk together with b_time
        save_moc(final_moc,folder_path, f"union_trained_dmoc_{C}_{norm[0]}") 
        save_moc(t_values,folder_path, f"deltas_dmoc_{C}_{norm[0]}")
   
        csv_path = folder_path / f"minibatch_{C}.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"dmoc_trained_{norm[0]}", "just moc time", b_time ])
            #writer.writerow(["lshmoc", lip_eclipse, lip_eclipse_t])
        print(f"done batch size {C}")

print(f"end")





