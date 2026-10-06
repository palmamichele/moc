from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from eclipse_nn.LipConstEstimator import LipConstEstimator
from sklearn.datasets import load_iris
from utils import NeuralNet, export_split_to_csv, LipConstEstimatorL1, save_moc
from pgdmoc.pgd_moc import pgd_moc
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import time
import csv
import copy


np.random.seed(0)
torch.manual_seed(0)


M=40
nrestart=50
nbins=100
batch_size=150

norm="L2"

def l2_distance(x,y):
    diff = x-y
    return diff.flatten(1).norm(p=2, dim=1)



n_experiments=1
base_path = Path("data")
save_path = base_path / "iris"
save_path.mkdir(parents=True, exist_ok=True)

folder_path =  Path("experiments") / str("iris")
save_path = Path("data") / "iris"
out_dir = Path("experiments") / "iris"
save_path.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

iris_data = load_iris()
X = iris_data.data.astype(np.float32)
y = iris_data.target.astype(np.int64)
num_classes = len(iris_data.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
y_train_onehot = np.eye(num_classes, dtype=np.float32)[y_train]
y_test_onehot = np.eye(num_classes, dtype=np.float32)[y_test]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_class_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_class_tensor = torch.tensor(y_test, dtype=torch.long)

y_train_export_tensor = torch.tensor(y_train_onehot, dtype=torch.float32)
y_test_export_tensor = torch.tensor(y_test_onehot, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_export_tensor),
    batch_size=1,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_export_tensor),
    batch_size=1,
    shuffle=False
)


n_epochs=500
lyrs = [3, 20, 5]  
neurons = [50, 100, 200]
j=0
for l in lyrs:
    for n in neurons:
        for i in range(n_experiments):
            model = NeuralNet(hidden_layers=l, hidden_units=n, input_size=len(X_train[0]), output_size=num_classes) 
            un_model = copy.deepcopy(model)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            
            if l==lyrs[-1]: #e.g. last size, overfitting case
                X_train_tensor=X_train_tensor[:5]
                y_train_class_tensor=y_train_class_tensor[:5]
                n_epochs=100 
        

            model.train()
            for epoch in range(n_epochs):
                optimizer.zero_grad()

                logits = model(X_train_tensor)
                loss = criterion(logits, y_train_class_tensor)

                loss.backward()
                optimizer.step()

                print(f"epoch {epoch}, train loss = {loss.item():.6f}")

           
            model.eval()
            with torch.no_grad():
                    train_logits = model(X_train_tensor)
                    test_logits = model(X_test_tensor)

                    train_probs = torch.softmax(train_logits, dim=1)
                    test_probs = torch.softmax(test_logits, dim=1)

                    train_pred = train_probs.argmax(dim=1)
                    test_pred = test_probs.argmax(dim=1)

                    train_acc = (train_pred == y_train_class_tensor).float().mean().item()
                    test_acc = (test_pred == y_test_class_tensor).float().mean().item()

            print(f"train acc = {train_acc:.4f}")
            print(f"test acc = {test_acc:.4f}")


            # wrap models so it includes softmax head as last layer
            model = nn.Sequential(model, nn.Softmax(dim=1))
            un_model = nn.Sequential(un_model, nn.Softmax(dim=1))
            export_split_to_csv(train_loader, "train", model, un_model, save_path, j)
            export_split_to_csv(test_loader, "test", model, un_model, save_path, j)



            X_train = np.loadtxt(save_path/("X_train.csv"), delimiter=",",ndmin=2)
            X_test = np.loadtxt(save_path/("X_test.csv"), delimiter=",",ndmin=2)
            Y_train = np.loadtxt(save_path/("Y_train.csv"), delimiter="," , ndmin=2)
            Y_test = np.loadtxt(save_path/("Y_test.csv"), delimiter="," , ndmin=2)
            F_train = np.loadtxt(save_path/(f"F_train_{i}.csv"), delimiter="," , ndmin=2)
            F_test = np.loadtxt(save_path/(f"F_test_{i}.csv"), delimiter="," , ndmin=2)
            F_untrain = np.loadtxt(save_path/(f"F_un_train_{i}.csv"), delimiter="," , ndmin=2)
            F_untest = np.loadtxt(save_path/(f"F_un_test_{i}.csv"), delimiter="," , ndmin=2)
            
            X_union = np.vstack([X_train, X_test])
            Y_union = np.vstack([Y_train, Y_test])
            F_union = np.vstack([F_train, F_test])
            F_un_union = np.vstack([F_untrain, F_untest])

 
            X_torch = torch.from_numpy(X_union).float()
            Y_torch = torch.from_numpy(Y_union).float()
            F_union_torch = torch.from_numpy(F_union).float()
            F_un_union_torch = torch.from_numpy(F_un_union).float()

            p_moc_tr, t_values = pgd_moc(
                        model, #f_\theta
                        X_torch,
                        F_union_torch, #either f_\theta(X) or original labels for X
                        l2_distance, #d_Y as loss function (assuming it satisfies metric properties)
                        None,
                        norm, #L2, L1, Linf
                        [10**(-2), 10**(2+0.2)], #t_1,...,t_K
                        None, 
                        M,
                        nrestart,
                        nbins,
                        batch_size
            )
        
        
                        
                        
            p_moc_un, _ = pgd_moc(
                un_model, #f_\theta
                X_torch,
                F_un_union_torch, #either f_\theta(X) or original labels for X
                l2_distance, #d_Y as loss function (assuming it satisfies metric properties)
                None,
                norm, #L2, L1, Linf
                t_values, #t_1,...,t_K
                None, 
                M,
                nrestart,
                nbins,
                batch_size
            )

            save_moc(p_moc_un,folder_path, f"pgd_untrained_dmoc_{j}_{norm}")
            save_moc(p_moc_tr,folder_path, f"pgd_trained_dmoc_{j}_{norm}")
            #save_moc(data_m,folder_path, type+f"_data_dmoc_{norm[0]}")
            save_moc(t_values,folder_path, f"pgd_deltas_dmoc_{j}_{norm}")


            # start_time = time.time()
            # est = LipConstEstimator(model=model)
            # lip_trivial = est.estimate(method="trivial")
            # lip_trivial_t = time.time()-start_time

            # lip_eclipse = 0
            # lip_eclipse_t=0
            # lip_eclipse_fast=0
            # lip_eclipse_fast_t=0


            # start_time = time.time()
            # est = LipConstEstimator(model=model)
            # lip_eclipse = est.estimate(method="ECLipsE")
            # lip_eclipse_t = time.time()-start_time

            # start_time = time.time()
            # est = LipConstEstimator(model=model)
            # lip_eclipse_fast = est.estimate(method="ECLipsE_Fast")
            # lip_eclipse_fast_t = time.time()-start_time

            # start_time = time.time()
            # estimator_l1 = LipConstEstimatorL1(model=model)
            # l1_bound = estimator_l1.estimate_trivial_l1()
            # l1_bound_t = time.time()-start_time

                
            # csv_path = out_dir / f"model_{j}.csv"
            # with open(csv_path, "w", newline="") as f:
            #     writer = csv.writer(f)
            #     writer.writerow(["constant type", "value", "seconds required"]) #trivial must be here adj for softmaxhead
            #     writer.writerow(["trivial_l2", lip_trivial, lip_trivial_t ])
            #     writer.writerow(["trivial_l1", l1_bound, l1_bound_t ]) 
            #     writer.writerow(["ECLipsE", lip_eclipse, lip_eclipse_t])
            #     writer.writerow(["ECLipsE_Fast", lip_eclipse_fast, lip_eclipse_fast_t])
            #     writer.writerow(["accuracy on train", train_acc, 0])
            #     writer.writerow(["accuracy on test", test_acc, 0])

            
            j=j+1
