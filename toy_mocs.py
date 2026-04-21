import numpy as np 
import csv 
import time 
import sys
import re 
import matplotlib.pyplot as plt
from pathlib import Path
from utils import lipschitz_from_fmoc, pad_moc_with_last
sys.path.append(str(Path("fmca") / "build" / "py"))
import FMCA


np.random.seed(0)
delta_step = 0.05
save_path = Path("experiments")

for mdl in ["MNIST"]:
    folder_path = save_path / str(mdl) 
    folder_path.mkdir(parents=True, exist_ok=True)

    n_experiments = len({
        p.name
        for p in (Path("data")/str(mdl)).iterdir()
        if p.is_file() and re.match(r"^F_train_.+", p.name)
    })

    max_distance = None #will compute bounding box trick 
    delta_values = []
    header = ["bound Lipschitz","dmoc Lipschitz", "lsh Lipschitz","n_points in the dataset","exact Lipschitz Time", "full dmoc Time", "full lsh moc Time"]
    rows = []
    m=[]
    e=[]
    l=[]

    for i in range(n_experiments):
        for type in ["union", "train", "test"]:
            filename = Path("data")/str(mdl)
            labelList = ["trained_model"]
            if type=="union":
                X_train = np.loadtxt(filename/("X_train.csv"), delimiter=",",ndmin=2)
                X_test = np.loadtxt(filename/("X_test.csv"), delimiter=",",ndmin=2)
                Y_train = np.loadtxt(filename/("Y_train.csv"), delimiter="," , ndmin=2)
                Y_test = np.loadtxt(filename/("Y_test.csv"), delimiter="," , ndmin=2)
                F_train = np.loadtxt(filename/(f"F_train_{i}.csv"), delimiter="," , ndmin=2)
                F_test = np.loadtxt(filename/(f"F_test_{i}.csv"), delimiter="," , ndmin=2)
                
                X = np.vstack([X_train, X_test])
                Y = np.vstack([Y_train, Y_test])
                F = np.vstack([F_train, F_test])
            else:
                X = np.loadtxt(filename/("X_"+type+".csv"), delimiter=",",ndmin=2)
                Y = np.loadtxt(filename/("Y_"+type+".csv"), delimiter="," , ndmin=2)
                F = np.loadtxt(filename/("F_"+type+f"_{i}.csv"), delimiter="," , ndmin=2)
            
            X = X.transpose()
            Y = Y.transpose()
            n = len(F)
            print("-> n points:"+str(n))
            F = F.transpose()
            d = len(F)
            print("-> dim points:"+str(d))

            #compute the data moc 
            dmoc = FMCA.DiscreteModulusOfContinuity()
            emoc = FMCA.EpsilonDiscreteModulusOfContinuity()
            lshmoc = FMCA.LSHDiscreteModulusOfContinuity()

            start_time = time.time()  
            dmoc.init(X,Y, max_distance, delta_step,"EUCLIDEAN", "EUCLIDEAN")
            data_m = dmoc.omegat()
            dmoc_t = time.time() - start_time 
            print(f"required {dmoc_t}")
            t_values = dmoc.tgrid()
            
           
            # start_time = time.time()  
            # dmoc.init(str(filename/("X_"+type+".csv")),str(filename/("Y_"+type+".csv")), max_distance, delta_step,"EUCLIDEAN", "EUCLIDEAN")
            # data_mb = dmoc.omegat()
            # dmoc_t = time.time() - start_time 
            # print(f"block required {dmoc_t}")


            # delta_values = t_values
            # start_time = time.time()  
            # lshmoc = FMCA.LSHDiscreteModulusOfContinuity()
            # lshmoc.init(X,Y, max_distance, delta_step)
            # data_l = lshmoc.omegat()
            # lmoc_t = time.time() - start_time   
            
           

            dmoc = FMCA.DiscreteModulusOfContinuity()
            dmoc.init(X,F, max_distance, delta_step,"EUCLIDEAN", "EUCLIDEAN")
            model_m = dmoc.omegat()
            lip_moc = lipschitz_from_fmoc(model_m, t_values)
            print(f'Lip from exact moc = {lip_moc}')
            model_m= pad_moc_with_last(model_m, len(delta_values))
            m.append(model_m)

            if type=="train":
                dmoc = FMCA.DiscreteModulusOfContinuity()
                F_un = np.loadtxt(filename/("F_un"+f"_{i}.csv"), delimiter="," , ndmin=2)
                F_un = F_un.transpose()
                dmoc.init(X,F_un, max_distance, delta_step,"EUCLIDEAN", "EUCLIDEAN")
                un_model_m = dmoc.omegat()
                un_model_m= pad_moc_with_last(un_model_m, len(delta_values))

                m.append(un_model_m)
                labelList.append("untrained_model")


            # with open(os.path.join(folder_path, f"lipschitz_constants.csv"), mode="w", newline="", encoding="utf-8") as file:
            #     writer = csv.writer(file)
            #     writer.writerow(header)  
            #     writer.writerows(rows)   
            
            #plot_mocs(labelList,m, None, None, delta_values,  folder_path/f"MOCs-{type}.png")

            data_m = pad_moc_with_last(data_m, len(delta_values))
            save_mocs(["data"],type,[data_m],folder_path)

            save_mocs(labelList,f"dmoc-{i}-"+type,m,folder_path)
            save_mocs(["deltas"],type,m,folder_path)
            #save_mocs(labelList, f"lsh-{i}-", type, l, folder_path)

#         #compute the net moc 
#         dmoc = FMCA.DiscreteModulusOfContinuity()
#         emoc = FMCA.EpsilonDiscreteModulusOfContinuity()
#         lshmoc = FMCA.LSHDiscreteModulusOfContinuity()

      
        

      

#         if n == max_n: #moc is monotone
#             delta_values = t_values
        
#         TX = dmoc.TX()
#         if max_distance == None:
#             max_distance= TX
        

#         e1=[None] * len(delta_values)
#         # if in_s <=5:
#         #     
#         #     start_time = time.time()  
#         #     emoc = FMCA.EpsilonDiscreteModulusOfContinuity()
#         #     emoc.init(X,Y,max_distance)
#         #     e1=[]
#         #     e1 = [emoc.omega(t,X,Y) for t in t_values]

#         #     emoc_t = time.time() - start_time  
#         #     
#         emoc_t = 0

#         l1= [None] * len(delta_values)
#         


#         lip_moc = lipschitz_from_fmoc(m1, t_values)
#         print(f'Lip from exact moc = {lip_moc}')

#         # lip_emoc =lipschitz_from_fmoc(e1, t_values)
#         # print(f'Lip from epsilon moc = {lip_emoc}')

#         lip_lshmoc = lipschitz_from_fmoc(l1, t_values)
#         print(f'Lip from lsh moc = {lip_lshmoc}')

#         #trained net, is moc on trained better? we might also study the moc of the network compared to the moc of the gt function (data)
#         #Must also generate some Y

#         rows.append([lip_trivial, lip_moc, lip_emoc, lip_lshmoc,n, exact_t,dmoc_t, emoc_t, lmoc_t])

#         m1= pad_moc_with_last(m1, len(delta_values))
#         e1 = pad_moc_with_last(e1, len(delta_values))
#         l1 = pad_moc_with_last(l1, len(delta_values))

#         m.append(m1)
#         e.append(e1)
#         l.append(l1)

#         #plot moc from data vs moc from function
    

    






# for in_s in input_sizes:
#     for out_s in output_sizes:

       
#         max_n = max(npoints)
#         data = np.random.randn(max_n, in_s)
        
#         print(folder_path)
        

#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         #randomly generated net (affine map)
#         model = OneLayerReLUNet(in_s, out_s, False).double()
#         model.eval()

#         start_time = time.time()  
#         est = LipConstEstimator(model=model) #same across obs/data points 
#         lip_trivial = est.estimate(method='trivial')
#         exact_t = time.time() - start_time 
#         print(f'exact Lip Const = {lip_trivial}')

       


#we know that without ReLU, the Lipschitz constant coincides with the operator norm, we check how far we are from it.

#might be beneficial to visualize the normalized lipschitz estimates

#the color will denote different models/benchmarks (exact, upper bound, eclipse, our moc,..., etc)

#one plot for each configuration  e.g. (in_s, out_s), the x axis denotes the number of datapoints used in moc, lsh moc....


