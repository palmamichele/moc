import math 
import sys
import numpy as np 
from pathlib import Path 
from utils import lipschitz_from_fmoc, pad_moc_with_last, save_moc
sys.path.append(str(Path("fmca") / "build" / "py"))
import FMCA

np.random.seed(0)
save_path = Path("experiments")
norms = ["EUCLIDEAN", "TAXICAB"]
folder_path = save_path / "MNIST"
folder_path.mkdir(parents=True, exist_ok=True)

N = 1000000
#compute u_b, l_b for TX, qX 
qX = 0.0001
TX = 1000
nbins =10000
norms = ["EUCLIDEAN", "TAXICAB"]


for C in [10, 100, 1000, 10000]:
    B = math.floor(N/C) #(each batch shall have same size)

    for norm in norms:
        dataList = [] #containing data paths? 

        #load the reference dataset, in chunks if it is too large.

        Pdata = dataList[C:C+C]

        dmoc = FMCA.DiscreteModulusOfContinuity()
        dmoc.init(P,F, TX,qX, nbins, norm, norm)
        final_moc = dmoc.omegat()
        t_values = dmoc.tgrid()
        NT = len(t_values)

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








for mdl in ["MNIST"]:


    n_experiments = len({
        p.name
        for p in (Path("data")/str(mdl)).iterdir()
        if p.is_file() and re.match(r"^F_train_.+", p.name)
    })

    print(n_experiments)



    for norm in norms:
        max_distance = None  #will compute bounding box trick 
        min_distance = None  
        # delta_values = []
        # header = ["bound Lipschitz","dmoc Lipschitz", "lsh Lipschitz","n_points in the dataset","exact Lipschitz Time", "full dmoc Time", "full lsh moc Time"]
        # rows = []
        # m=[]
        # e=[]
        # l=[]
        filename = Path("data")/str(mdl)
        for i in range(n_experiments):

            #always try to load train, test, to make the union 
            X_train = np.loadtxt(filename/("X_train.csv"), delimiter=",",ndmin=2)
            X_test = np.loadtxt(filename/("X_test.csv"), delimiter=",",ndmin=2)
            Y_train = np.loadtxt(filename/("Y_train.csv"), delimiter="," , ndmin=2)
            Y_test = np.loadtxt(filename/("Y_test.csv"), delimiter="," , ndmin=2)
            F_train = np.loadtxt(filename/(f"F_train_{i}.csv"), delimiter="," , ndmin=2)
            F_test = np.loadtxt(filename/(f"F_test_{i}.csv"), delimiter="," , ndmin=2)
            F_untrain = np.loadtxt(filename/(f"F_un_train_{i}.csv"), delimiter="," , ndmin=2)
            F_untest = np.loadtxt(filename/(f"F_un_test_{i}.csv"), delimiter="," , ndmin=2)
            
            X_union = np.vstack([X_train, X_test])
            Y_union = np.vstack([Y_train, Y_test])
            F_union = np.vstack([F_train, F_test])
            F_un_union = np.vstack([F_untrain, F_untest])

            X_union = X_union.transpose()
            Y_union = Y_union.transpose()
            n = len(F_union)
            print("model:", filename)
            print("-> n points:"+str(n))
            F_union = F_union.transpose()
            F_un_union= F_un_union.transpose()

            #start by computing dmoc on union (this will contain largest tgrid)
            dmoc = FMCA.DiscreteModulusOfContinuity()
            start_time = time.time()  
            dmoc.init(X_union,Y_union, max_distance,min_distance, nbins, norm, norm)
            data_m = dmoc.omegat()
            data_t = time.time() - start_time 
            t_values = dmoc.tgrid()

            #compute dmoc of trained net
            dmoc = FMCA.DiscreteModulusOfContinuity()
            start_time = time.time()  
            dmoc.init(X_union,F_union, max_distance,min_distance, nbins,norm, norm)
            tr_m = dmoc.omegat()
            tr_t = time.time() - start_time 
            

            #compute dmoc of untrained net
            dmoc = FMCA.DiscreteModulusOfContinuity()
            start_time = time.time() 
            dmoc.init(X_union,F_un_union, max_distance,min_distance, nbins,norm, norm)
            un_m = dmoc.omegat()


            save_moc(un_m,folder_path, f"union_untrained_dmoc_{i}_{norm[0]}")
            save_moc(tr_m,folder_path, f"union_trained_dmoc_{i}_{norm[0]}")
            save_moc(data_m,folder_path, f"union_data_dmoc_{norm[0]}")
            save_moc(t_values,folder_path, f"deltas_dmoc_{i}_{norm[0]}")

            start_time = time.time()
            lip_moc = lipschitz_from_fmoc(tr_m, t_values)
            lip_moc_t = time.time()-start_time

            csv_path = folder_path / f"model_{i}.csv"
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"dmoc_union_{norm[0]}", lip_moc, tr_t+lip_moc_t ])
                #writer.writerow(["lshmoc", lip_eclipse, lip_eclipse_t])


            max_distance = t_values[-1]
            min_distance = t_values[0]

            for type in ["train", "test"]:
                X = np.loadtxt(filename/("X_"+type+".csv"), delimiter=",",ndmin=2)
                Y = np.loadtxt(filename/("Y_"+type+".csv"), delimiter="," , ndmin=2)
                F = np.loadtxt(filename/("F_"+type+f"_{i}.csv"), delimiter="," , ndmin=2)
                F_un = np.loadtxt(filename/("F_un_"+type+f"_{i}.csv"), delimiter="," , ndmin=2)

                X = X.transpose()
                Y=Y.transpose()
                F = F.transpose()
                F_un = F_un.transpose()

                #start by computing dmoc on union (this will contain largest tgrid)
                dmoc = FMCA.DiscreteModulusOfContinuity()
                start_time = time.time()  
                dmoc.init(X,Y, max_distance, min_distance, nbins, norm, norm)
                data_m = dmoc.omegat()
                data_t = time.time() - start_time 

                #compute dmoc of trained net
                dmoc = FMCA.DiscreteModulusOfContinuity()
                start_time = time.time()  
                dmoc.init(X,F, max_distance, min_distance, nbins,norm, norm)
                tr_m = dmoc.omegat()
                tr_t = time.time() - start_time 
                

                #compute dmoc of untrained net
                dmoc = FMCA.DiscreteModulusOfContinuity()
                start_time = time.time() 
                dmoc.init(X,F_un, max_distance, min_distance, nbins,norm, norm)
                un_m = dmoc.omegat()

                #un_m= pad_moc_with_last(un_m, len(t_values))

                
                
                #tr_m= pad_moc_with_last(tr_m, len(t_values))
                #data_m= pad_moc_with_last(data_m, len(t_values))

                save_moc(un_m,folder_path, type+f"_untrained_dmoc_{i}_{norm[0]}")
                save_moc(tr_m,folder_path, type+f"_trained_dmoc_{i}_{norm[0]}")
                save_moc(data_m,folder_path, type+f"_data_dmoc_{norm[0]}")
                save_moc(t_values,folder_path, type+f"_deltas_dmoc_{i}_{norm[0]}")

                start_time = time.time()
                lip_moc = lipschitz_from_fmoc(tr_m, t_values)
                lip_moc_t = time.time()-start_time

                csv_path = folder_path / f"model_{i}.csv"
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([f"dmoc_{type}_{norm[0]}", lip_moc, tr_t+lip_moc_t ])
                    #writer.writerow(["lshmoc", lip_eclipse, lip_eclipse_t])


