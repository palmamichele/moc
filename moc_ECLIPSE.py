import os
import numpy as np
import csv 
import time 
import csv
import sys
import matplotlib.pyplot as plt
sys.path.append("fmca/build/py")
import FMCA



print(os.environ.get("OMP_NUM_THREADS"))

parent= "/Users/palmamichele/Documents/Research Projects/moc/code/moc/data"



#usage
# edmoc = FMCA.ExactDiscreteModulusOfContinuity()
# edmoc.init(x,f1, 0, "EUCLIDEAN", "EUCLIDEAN", "NO")
# edmoc.computeMocPlot(x, f1, delta_step)
# m1 = edmoc.getOmegaT()


#usage
# delta_step = 1
# r = 0.0005
# R = 2
# TX = len(m1)
# min_csize=1

# edmoc = FMCA.EpsilonDiscreteModulusOfContinuity()
# edmoc.init(x,f1,r,R,TX,min_csize,"EUCLIDEAN", "EUCLIDEAN")

# e1=[]
# t_values = np.zeros(TX,dtype=np.float64)
# for i in range(1, len(t_values)):
#     t_values[i]=t_values[i-1]+delta_step


# for t in t_values:
#     e1.append(edmoc.omega(t,x,f1))




def plot_mocs(exactMoc, epsilonMoc, lshMoc, lshEpsilon, deltas, savename):
    """
    mocs contains moc values (from 0 to >= max_dist) for k different functions, it is an array of size k x T
    """

    if exactMoc!=[]:
        plt.plot(deltas, exactMoc, label = "exact", linestyle='None', marker='o', markersize=2, alpha=0.7)

    if epsilonMoc!=[]:
        plt.plot(deltas, epsilonMoc, label = "epsilon", linestyle='None', marker='o', markersize=2, alpha=0.7)

    if lshMoc!=[]:
        plt.plot(deltas, lshMoc, label = "lsh", linestyle='None', marker='o', markersize=2, alpha=0.7)  
    
    if lshEpsilon!=[]:
        plt.plot(deltas, lshEpsilon, label = "epsilon lsh", linestyle='None', marker='o', markersize=2, alpha=0.7)

    plt.legend()
    plt.xlabel("t")
    plt.ylabel("omega(t)")
    plt.savefig(f"{savename}", dpi=300)  # high-resolution PNG
    plt.clf() 
    plt.close('all')

def lipschitz_from_fmoc(fmocs, deltas):
    fmocs = np.asarray(fmocs)
    deltas = np.asarray(deltas)
    mask = deltas > 0          # safety
    return np.max(fmocs[mask] / deltas[mask])


lyrs = [2, 20, 100]#[2, 5, 10, 20, 30, 50, 75, 100]
neurons = [20,100] #[20, 40, 60, 80, 100]

total_files=0
count_smaller_than_eclipse=0
count_smaller_than_eclipse_fast=0
count_greater_than_eclipse=0
count_greater_than_eclipse_fast=0
greater_eclipse_files = []
greater_eclipse_fast_files = []

worst_ratio_trivial=[-1]
worst_ratio_eclipse=[-1]
worst_ratio_eclipse_fast=[-1]


how_many=10000
delta_step=1



for l in lyrs:
    for n in neurons:
        savename = os.path.join(parent, f"simplenet-{n}-{l}")

        if not(os.path.isdir(savename)):
            continue
        total_files += 1

        print("processing ", savename)

        filename = os.path.join(savename, "X.csv")


        for how_many in range(10000,80000,10000):

            X = np.loadtxt(filename, delimiter=",")

            print(X.shape)
            if X.ndim == 1:
                break

            how_many = min(how_many, 70000)

            X = X[:how_many,:]

            X = X.transpose()

            F = [[],[]]
            filename = os.path.join(savename, "F_0.csv")
            F[0]= np.loadtxt(filename, delimiter=",")
            F[0] = F[0][:how_many]
            F[0] = F[0].reshape(-1, 1)
            F[0] = F[0].transpose()

            print("X shape:", X.shape)
            print("F0 shape:", F[0].shape)


            # filename = os.path.join(savename, "F_1.csv")
            # F[1]= np.loadtxt(filename, delimiter=",")
            # F[1] = F[1][:how_many]

            # F[1] = F[1].reshape(-1, 1)
            # F[1] = F[1].transpose()

            dmoc = FMCA.DiscreteModulusOfContinuity()
            edmoc = FMCA.EpsilonDiscreteModulusOfContinuity()
            lshmoc = FMCA.LSHDiscreteModulusOfContinuity()
           
            max_distance = 20

            print("start exact moc")
            start_time = time.time()  # start timer for fair comparison
            dmoc.init(X,F[0], max_distance, delta_step, "EUCLIDEAN", "EUCLIDEAN")
            #dmoc.computeMocPlot(X, F[0], delta_step)
            m1 = dmoc.omegat()
            print(m1)
            elapsed = time.time() - start_time  # compute elapsed time
            print("took (sec) ", elapsed)
           
            #reconstruct values of t_values, having the size of mocplot
            t_values = dmoc.tgrid()
            TX = max_distance
        

            print("start epsilon lsh moc")
            start_time = time.time()  # start timer for fair 
            lshmoc = FMCA.LSHDiscreteModulusOfContinuity()
            lshmoc.init(X,F[0],TX, delta_step)
            e1_lsh =[]
            for t in t_values:
                e1_lsh.append(lshmoc.omega(t,X,F[0]))

            elapsedEPSLSH = time.time() - start_time  # compute 
            print("took (sec) ", elapsedEPSLSH)

    
            print("start epsilon moc")
            start_time = time.time()  # start timer for fair 
            edmoc = FMCA.EpsilonDiscreteModulusOfContinuity()
            edmoc.init(X,F[0],TX, delta_step)
            e1=[]
            e1 = [edmoc.omega(t,X,F[0]) for t in t_values]

            elapsedEps = time.time() - start_time  # compute 
            print("took (sec) ", elapsedEps)
            
        


            plot_mocs(m1, e1, [], e1_lsh, t_values,  os.path.join(savename, f"{type, how_many}refact_MOC.png"))

            L = lipschitz_from_fmoc(m1, t_values)

            print(L, "from exact")

            L = lipschitz_from_fmoc(e1, t_values)

            print(L, "from epsilon")


            L = lipschitz_from_fmoc(e1_lsh, t_values)

            print(L, "from epsilon lsh")

            summary_csv = os.path.join(savename, str(how_many)+"summary_F0.csv")
            with open(summary_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "L from exact moc",
                    "L from epsilon moc",
                    "L from epsilon lsh moc",
                    "time (secs) for exact moc",
                    "time for epsilon moc",
                    "time for epsilon lsh"
                ])

                writer.writerow([
                lipschitz_from_fmoc(m1, t_values),
                    lipschitz_from_fmoc(e1, t_values),
                    lipschitz_from_fmoc(e1_lsh, t_values),
                    elapsed,
                    elapsedEps,
                    elapsedEPSLSH
                ])





lyrs = [3] 
neurons = [100,200,300,400]#[1,128]#[1, 20, 50] #[100, 200, 300, 400]

total_files = 0
smaller_eclipse = {"":0,"_train":0,"_test":0}
smaller_eclipse_fast = {"":0,"_train":0,"_test":0}
greater_eclipse = {"":0,"_train":0,"_test":0}
greater_eclipse_fast = {"":0,"_train":0,"_test":0}
greater_eclipse_files = []
greater_eclipse_fast_files = []

worst_ratio_trivial=[-1]
worst_ratio_eclipse=[-1]
worst_ratio_eclipse_fast=[-1]


for l in lyrs:
    for n in neurons:
        savename = os.path.join(parent, f"mnist-{n}-{l}")
        
        print("processing "+savename)

        total_files += 1

        
        filename = os.path.join(savename, "X.csv")

          
        X = np.loadtxt(filename, delimiter=",")
        print(X.shape)

        X = X.transpose()

        filename = os.path.join(savename, "X_train.csv")
        X_train = np.loadtxt(filename, delimiter=",")
        X_train = X_train.transpose()

        filename = os.path.join(savename, "X_test.csv")
        X_test = np.loadtxt(filename, delimiter=",")
        X_test = X_test.transpose()


       
        F = [[],[]]
        F_train = [[],[]]
        F_test = [[],[]]
        filename = os.path.join(savename, "F_0.csv")
        F[0]= np.loadtxt(filename, delimiter=",")

        F[0]=F[0].transpose()
       

        filename = os.path.join(savename, "F_1.csv")
        F[1]= np.loadtxt(filename, delimiter=",")

        F[1]=F[1].transpose()
        
        filename = os.path.join(savename, "F_0_train.csv")
        F_train[0]= np.loadtxt(filename, delimiter=",")

        F_train[0] = F_train[0].transpose()

        filename = os.path.join(savename, "F_1_train.csv")
        F_train[1]= np.loadtxt(filename, delimiter=",")

        F_train[1] =F_train[1].transpose()
        
        filename = os.path.join(savename, "F_0_test.csv")
        F_test[0]= np.loadtxt(filename, delimiter=",")

        F_test[0] = F_test[0].transpose()

        filename = os.path.join(savename, "F_1_test.csv")
        F_test[1]= np.loadtxt(filename, delimiter=",")

        F_test[1] = F_test[1].transpose()

        file_path = os.path.join(savename, "newlip_constants_with_time.csv")

        
        
        for type in ["", "_train", "_test"]:
            
            xdataset=None
            ydataset=None
            if type=="":
                xdataset=X
                ydataset=F
            elif type=="_train":
                xdataset=X_train
                ydataset=F_train

            elif type=="_test":
                xdataset=X_test
                ydataset=F_test

            delta_step=1
            dmoc = FMCA.DiscreteModulusOfContinuity()
            lshmoc = FMCA.LSHDiscreteModulusOfContinuity()

            print("start exact moc")
            start_time = time.time()  # start timer for fair comparison
            dmoc.init(xdataset,ydataset[0], 0, delta_step,"EUCLIDEAN", "EUCLIDEAN")
            m1 = dmoc.omegat()
            elapsed = time.time() - start_time  # compute elapsed time
            print("took (sec) ", elapsed)
            
            #reconstruct values of t_values, having the size of mocplot
            t_values = dmoc.tgrid()
            TX = len(t_values)



            print("start epsilon lsh moc")
            start_time = time.time()  # start timer for fair 
            lshmoc = FMCA.LSHDiscreteModulusOfContinuity()
            lshmoc.init(xdataset,ydataset[0], TX, delta_step)
            e1_lsh =[]
            for t in t_values:
                e1_lsh.append(lshmoc.omega(t,xdataset,ydataset[0]))

            elapsedEPSLSH = time.time() - start_time  # compute 
            print("took (sec) ", elapsedEPSLSH)
           

            # print("start epsilon moc")
            # start_time = time.time()  # start timer for fair 
            # edmoc = FMCA.EpsilonDiscreteModulusOfContinuity()
            # edmoc.init(xdataset,ydataset[0],r,R,TX,min_csize,"EUCLIDEAN", "EUCLIDEAN")
            # e1=[]
            # e1 = [edmoc.omega(t,xdataset,ydataset[0]) for t in t_values]

            # elapsedEps = time.time() - start_time  # compute 
            # print("took (sec) ", elapsedEps)
            
           



            plot_mocs(m1, [], [], e1_lsh, t_values,  os.path.join(savename, f"{type, how_many}refact_MOC.png"))

            L = lipschitz_from_fmoc(m1, t_values)

            print(L, "from exact")


            L = lipschitz_from_fmoc(e1_lsh, t_values)

            print(L, "from epsilon lsh")

            summary_csv = os.path.join(savename, str(how_many)+"summary_F0.csv")
            with open(summary_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "L from exact moc",
                    #"L from epsilon moc",
                    "L from epsilon lsh moc",
                    "time (secs) for exact moc",
                    #"time from epsilon moc",
                    "time from exact lsh moc",
                    "time from epsilon lsh moc"
                ])

                writer.writerow([
                lipschitz_from_fmoc(m1, t_values),
                    #lipschitz_from_fmoc(e1, t_values),
                    lipschitz_from_fmoc(e1_lsh, t_values),
                    elapsed,
                    elapsedEPSLSH,
                ])


print("end mnist")


# summary_csv = os.path.join(parent, "simplenet_summary_F0.csv")
# with open(summary_csv, mode="w", newline="") as f:
#     writer = csv.writer(f)

#     writer.writerow([
#         "total_files (including none comparisons)",

#         "L_smaller_than_ECLipsE",
#         "L_smaller_than_ECLipsE_Fast",
#         "L_greater_than_ECLipsE",
#         "L_greater_than_ECLipsE_Fast",

#         "files_where_L_greater_than_ECLipsE",
#         "files_where_L_greater_than_ECLipsE_Fast",
#          "worst ratio L trivial",
#         "worst ratio L eclipse",
#         "worst ratio L eclipse_fast"
#     ])

#     writer.writerow([
#         total_files,
#         count_smaller_than_eclipse,
#         count_smaller_than_eclipse_fast,
#         count_greater_than_eclipse,
#         count_greater_than_eclipse_fast,
    
#         ";".join(greater_eclipse_files),
#         ";".join(greater_eclipse_fast_files),

#         np.max(worst_ratio_trivial),
#         np.max(worst_ratio_eclipse),
#         np.max(worst_ratio_eclipse_fast)
#     ])


