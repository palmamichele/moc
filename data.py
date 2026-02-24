import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append("./build")
import exactdmoc


def plot_mocs(fmocs, lshmocs, deltas, savename):
    """
    mocs contains moc values (from 0 to >= max_dist) for k different functions, it is an array of size k x T
    """


    num_functions = len(fmocs)
    fig, ax = plt.subplots(num_functions, 2)

    for j in range(num_functions):
        ax[j,0].plot(deltas, fmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
        ax[j,0].set_title(f"exact MOC for f{j}")
        if lshmocs ==[]:

            ax[j, 1].plot(deltas, deltas*0, linestyle='None', marker='o', markersize=2, alpha=0.7)
            ax[j, 1].set_title(f"lsh moc for f{j}")
        #ax[j, 2].plot(t, aannfmocs[j], linestyle='None', marker='o', markersize=2, alpha=0.7)
        #ax[j, 2].set_title(f"aANN MOC for f{j}")

    fig.tight_layout()
    plt.savefig(f"{savename}", dpi=300)  # high-resolution PNG
    plt.clf() 
    plt.close('all')






f1 = np.array([[0,1],[3,2]], dtype=np.float64) 
#f2 = np.array([3,2,3,3,4,3], dtype=np.float64).reshape(1, -1)#must have shape (d,n), so use this on 1D array
x = np.array([[0,1] , [2,3]], dtype=np.float64)


delta_step = 1
edmoc = exactdmoc.ExactDiscreteModulusOfContinuity()
edmoc.init(x,f1)

m1 = edmoc.computeMocPlot(x, f1, delta_step)
#m2 = edmoc.computeMocPlot(x, f2, delta_step)

#reconstruct values of t_values, having the size of mocplot
t_values = np.zeros(len(m1),dtype=np.float64)

for i in range(1, len(t_values)):
    t_values[i]=t_values[i-1]+delta_step

plot_mocs([m1,m1], [], t_values, "testmoc.png")
