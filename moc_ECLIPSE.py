import os
import numpy as np
import csv 
import time 
import csv
import sys
import matplotlib.pyplot as plt
sys.path.append("fmca/build/py")
import FMCA

dmoc = FMCA.ExactDiscreteModulusOfContinuity()
epsmoc = FMCA.EpsilonDiscreteModulusOfContinuity()

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




def plot_mocs(fmocs, lshmocs, deltas, savename):
    """
    mocs contains moc values (from 0 to >= max_dist) for k different functions, it is an array of size k x T
    """


    num_functions = len(fmocs)
    fig, ax = plt.subplots(num_functions, 2)

    for j in range(num_functions):
        ax[j,0].plot(deltas, fmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
        ax[j,0].set_title(f"exact MOC for f{j}")
        ax[j, 1].set_title(f"epsilon moc for f{j}")
        if lshmocs ==[]:

            ax[j, 1].plot(deltas, deltas*0, linestyle='None', marker='o', markersize=2, alpha=0.7)
            
            

        else:
            ax[j,1].plot(deltas, lshmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
           

    fig.tight_layout()
    #plt.savefig(f"{savename}", dpi=300)  # high-resolution PNG
    plt.show()
    plt.clf() 
    plt.close('all')

def lipschitz_from_fmoc(fmocs, deltas):
    fmocs = np.asarray(fmocs)
    deltas = np.asarray(deltas)
    mask = deltas > 0          # safety
    return np.max(fmocs[mask] / deltas[mask])

lyrs = [3] 
neurons = [100,200,300,400]#[1,128]#[1, 20, 50] #[100, 200, 300, 400]

parent= "/Users/palmamichele/Documents/Research Projects/moc/code/moc/data"

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


r = 0.0005
R = 2
min_csize=1



for l in lyrs:
    for n in neurons:
        savename = os.path.join(parent, f"mnist-{n}-{l}")
        
        print("processing "+savename)

        total_files += 1

        filename = os.path.join(savename, "X.csv")
        X = np.loadtxt(filename, delimiter=",")
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
            dmoc.init(xdataset,ydataset[0], 0, "EUCLIDEAN", "EUCLIDEAN", "NO")
            dmoc.computeMocPlot(xdataset, ydataset[0], delta_step)
            m1 = dmoc.getOmegaT()
            
            TX = len(m1)
            #reconstruct values of t_values, having the size of mocplot
            t_values = np.zeros(TX,dtype=np.float64)
            for i in range(1, TX):
                t_values[i]=t_values[i-1]+delta_step



            edmoc = FMCA.EpsilonDiscreteModulusOfContinuity()
            edmoc.init(xdataset,ydataset[0],r,R,TX,min_csize,"EUCLIDEAN", "EUCLIDEAN")
            e1=[]
            for t in t_values:
                e1.append(edmoc.omega(t,xdataset,ydataset[0]))


            start_time = time.time()  # start timer for fair comparison
            dmoc.init(xdataset,ydataset[1], len(m1), "EUCLIDEAN", "EUCLIDEAN", "NO")
            dmoc.computeMocPlot(xdataset, ydataset[1], delta_step)
            m2 = dmoc.getOmegaT()
            elapsed = time.time() - start_time  # compute elapsed time
            print("exact moc required "+str(elapsed)+" on "+str(type))

            start_time = time.time()  # start timer for fair comparison
            edmoc.init(xdataset,ydataset[1],r,R,TX,min_csize,"EUCLIDEAN", "EUCLIDEAN")
            e2=[]
            for t in t_values:
                e2.append(edmoc.omega(t,xdataset,ydataset[1]))
            elapsed = time.time() - start_time  # compute elapsed time
            print("epsilon moc required "+str(elapsed)+" on "+str(type))


            plot_mocs([m1,m2], [e1, e2], t_values,  os.path.join(savename, f"{type}NEWmoc.png"))

            L = lipschitz_from_fmoc(m1, t_values)

            print(L, "from exact")

            L = lipschitz_from_fmoc(e1, t_values)

            print(L, "from epsilon")

            with open(file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"lipschitz from moc {type} for F0", L, elapsed])
            
            exit()
# exit()   
# summary_csv = os.path.join(parent, "mnist_summary_F0.csv")
# with open(summary_csv, mode="w", newline="") as f:
#     writer = csv.writer(f)

#     writer.writerow([
#         "total_files (including none comparisons)",

#         "L_smaller_than_ECLipsE",
#         "L_smaller_than_ECLipsE_Fast",
#         "L_greater_than_ECLipsE",
#         "L_greater_than_ECLipsE_Fast",

#         "L_smaller_than_ECLipsE on train only ",
#         "L_smaller_than_ECLipsE_Fast on train only",
#         "L_greater_than_ECLipsE on train only",
#         "L_greater_than_ECLipsE_Fast on train only",


#         "L_smaller_than_ECLipsE on test only ",
#         "L_smaller_than_ECLipsE_Fast on test only",
#         "L_greater_than_ECLipsE on test only",
#         "L_greater_than_ECLipsE_Fast on test only",

#         "files_where_L_greater_than_ECLipsE",
#         "files_where_L_greater_than_ECLipsE_Fast",

#         "worst ratio L trivial",
#         "worst ratio L eclipse",
#         "worst ratio L eclipse_fast"
#     ])

#     writer.writerow([
#         total_files,
#         smaller_eclipse[""],
#         smaller_eclipse_fast[""],
#         greater_eclipse[""],
#         greater_eclipse_fast[""],
        
#         smaller_eclipse["_train"],
#         smaller_eclipse_fast["_train"],
#         greater_eclipse["_train"],
#         greater_eclipse_fast["_train"],

#         smaller_eclipse["_test"],
#         smaller_eclipse_fast["_test"],
#         greater_eclipse["_test"],
#         greater_eclipse_fast["_test"],

#         ";".join(greater_eclipse_files),
#         ";".join(greater_eclipse_fast_files),

#         np.max(worst_ratio_trivial),
#         np.max(worst_ratio_eclipse),
#         np.max(worst_ratio_eclipse_fast)
#     ])

# print("end mnist")


# lyrs = [2, 5, 10, 20, 30, 50, 75, 100]
# neurons = [20, 40, 60, 80, 100]

# total_files=0
# count_smaller_than_eclipse=0
# count_smaller_than_eclipse_fast=0
# count_greater_than_eclipse=0
# count_greater_than_eclipse_fast=0
# greater_eclipse_files = []
# greater_eclipse_fast_files = []

# worst_ratio_trivial=[-1]
# worst_ratio_eclipse=[-1]
# worst_ratio_eclipse_fast=[-1]


# how_many=70000
# delta_step=0.05


# for l in lyrs:
#     for n in neurons:
#         savename = os.path.join(parent, f"simplenet-{n}-{l}")

#         if not(os.path.isdir(savename)):
#             continue
#         total_files += 1

#         filename = os.path.join(savename, "X.csv")
#         X = np.loadtxt(filename, delimiter=",")
        
#         print(X.shape)

#         X = X[:how_many,:]

#         X = X.transpose()

#         F = [[],[]]
#         filename = os.path.join(savename, "F_0.csv")
#         F[0]= np.loadtxt(filename, delimiter=",")
#         F[0] = F[0][:how_many]
#         F[0] = F[0].reshape(-1, 1)
#         F[0] = F[0].transpose()

#         print("X shape:", X.shape)
#         print("F0 shape:", F[0].shape)


#         filename = os.path.join(savename, "F_1.csv")
#         F[1]= np.loadtxt(filename, delimiter=",")
#         F[1] = F[1][:how_many]

#         F[1] = F[1].reshape(-1, 1)
#         F[1] = F[1].transpose()
#         start_time = time.time()  # start timer
        

#         edmoc = exactdmoc.ExactDiscreteModulusOfContinuity()
#         edmoc.init(X,F[0])

        
#         print("hey")
#         elapsed = time.time() - start_time  # compute elapsed time
#         m1 = edmoc.computeMocPlot(X, F[0], delta_step)
       
        
#         m2 = edmoc.computeMocPlot(X, F[1], delta_step)
#         print("required "+str(elapsed))

#         #reconstruct values of t_values, having the size of mocplot
#         t_values = np.zeros(len(m1),dtype=np.float64)

#         for i in range(1, len(t_values)):
#             t_values[i]=t_values[i-1]+delta_step

#         plot_mocs([m1,m2], [], t_values,  os.path.join(savename, f"{delta_step, how_many}NEWmoc.png"))

#         #load it back via:  np.load('array.npy')
#         save_name = os.path.join(savename, f"old_moc") 
#         gt_m = np.load(save_name+'.npy')
#         gt_t = np.load(save_name+'t.npy')
#         if (not(np.array_equal(gt_m, m1) and np.array_equal(gt_t, t_values))):
#             print("problem!!!" + save_name)


#             print("exact one", m1)
#             print("\n old one:", gt_m)

#             print("deltas one", t_values)
#             print("\n old deltas:", gt_t)
#         else:
#             print("ok")
        
#         L = lipschitz_from_fmoc(m1, t_values)
#         print(L)

#         exit()
        
#         #plot_mocs(fmocs, lshmocs, deltas, os.path.join(savename, f"moc.png"))
        
#         file_path = os.path.join(savename, "lip_constants_with_time.csv")
#         if os.path.exists(file_path):
#             eclipse, eclipse_fast,trivial = read_eclipse_constants(file_path)
#         else:
#             eclipse = eclipse_fast =trivial = None  # skip counting


#         

#         print(L)

#         if eclipse is not None:

#             worst_ratio_eclipse.append(L/eclipse)

#             if L <= eclipse:
#                 count_smaller_than_eclipse += 1
#             else:
#                 count_greater_than_eclipse +=1
#                 greater_eclipse_files.append(savename)
#         if eclipse_fast is not None:

#             worst_ratio_eclipse_fast.append(L/eclipse_fast)
#             if L<= eclipse_fast:
#                 count_smaller_than_eclipse_fast += 1
#             else:
#                 count_greater_than_eclipse_fast +=1
#                 greater_eclipse_fast_files.append(savename)

#         if trivial is not None:
#             worst_ratio_eclipse.append(L/trivial)


#         with open(file_path, mode="a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([f"lipschitz from moc for F0", L, elapsed])

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



# print(total_files)
# print("end random")