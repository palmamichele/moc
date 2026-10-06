from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from torch.utils.data import TensorDataset, DataLoader
from eclipse_nn.LipConstEstimator import LipConstEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from utils import NeuralNet, export_split_to_csv, LipConstEstimatorL1, save_moc
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import time
import csv
import copy
from pgdmoc.pgd_moc import pgd_moc

np.random.seed(0)
torch.manual_seed(0)

M=40
nrestart=1
nbins=100
batch_size=1

norm="L2"

def l2_distance(x,y):
    diff = x-y
    return diff.flatten(1).norm(p=2, dim=1)



n_experiments=1
base_path = Path("data")
save_path = base_path / "linear-california"
save_path.mkdir(parents=True, exist_ok=True)

folder_path =  Path("experiments") / str("linear-california")

cal_data = fetch_california_housing()
X = cal_data.data.astype(np.float32)
y = cal_data.target.astype(np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = x_scaler.transform(X_test).astype(np.float32)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train).astype(np.float32)
y_test_scaled = y_scaler.transform(y_test).astype(np.float32)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


export_train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=1,
    shuffle=False
)

export_test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor),
    batch_size=1,
    shuffle=False
)

n_epochs=100
out_dir = Path("experiments") / "linear-california"
out_dir.mkdir(parents=True, exist_ok=True)

for i in range(n_experiments):
    model = NeuralNet(hidden_layers=0, hidden_units=0, input_size=8, output_size=1) 
    un_model = copy.deepcopy(model)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(X_train_tensor)
        loss = criterion(pred, y_train_tensor)
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch}, train loss = {loss.item():.6f}")

    model.eval()
    un_model.eval()
    with torch.no_grad():
        pred_train = model(X_train_tensor)
        pred_test = model(X_test_tensor)

    train_mse = criterion(pred_train, y_train_tensor).item()
    test_mse = criterion(pred_test, y_test_tensor).item()
    print(f"train MSE = {train_mse} of {i}")
    print(f"test MSE = {test_mse} of {i}")

    train_r2 = r2_score(
    y_train_tensor.cpu().numpy().ravel(),
    pred_train.cpu().numpy().ravel()
    )

    test_r2 = r2_score(
        y_test_tensor.cpu().numpy().ravel(),
        pred_test.cpu().numpy().ravel()
    )

    print(f"train R^2 = {train_r2:.6f} of {i}")
    print(f"test R^2 = {test_r2:.6f} of {i}")


    export_split_to_csv(export_train_loader, "train", model, un_model, save_path, i)
    export_split_to_csv(export_test_loader, "test", model, un_model, save_path, i)


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

    p_moc_un, t_values = pgd_moc(
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


    save_moc(p_moc_un,folder_path, f"pgd_untrained_dmoc_0_{norm}")
    save_moc(p_moc_tr,folder_path, f"pgd_trained_dmoc_0_{norm}")
    #save_moc(data_m,folder_path, type+f"_data_dmoc_{norm[0]}")
    save_moc(t_values,folder_path, f"pgd_deltas_dmoc_0_{norm}")


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

        
    # csv_path = out_dir / f"model_{i}.csv"
    # with open(csv_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["constant type", "value", "seconds required"])
    #     writer.writerow(["trivial_l2", lip_trivial, lip_trivial_t ])
    #     writer.writerow(["trivial_l1", l1_bound, l1_bound_t ])
    #     writer.writerow(["train MSE", train_mse, 0])
    #     writer.writerow(["test MSE", test_mse, 0])
        # writer.writerow(["ECLipsE", lip_eclipse, lip_eclipse_t])
        # writer.writerow(["ECLipsE_Fast", lip_eclipse_fast, lip_eclipse_fast_t])
   



