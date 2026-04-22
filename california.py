from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from torch.utils.data import TensorDataset, DataLoader
from eclipse_nn.LipConstEstimator import LipConstEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from utils import NeuralNet, export_split_to_csv
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import time
import csv
import copy
import pandas as pd 

np.random.seed(0)
torch.manual_seed(0)

n_experiments=1
base_path = Path("data")
save_path = base_path / "california"
save_path.mkdir(parents=True, exist_ok=True)

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

n_epochs=500
out_dir = Path("experiments") / "california"
out_dir.mkdir(parents=True, exist_ok=True)

lyrs = [3, 20, 100] #[2, 5, 10, 20, 30, 50, 75, 100]
neurons = [50, 100, 200] #[20, 40, 60, 80, 100]
j=0
for l in lyrs:
    for n in neurons:
        for i in range(n_experiments):
            model = NeuralNet(hidden_layers=l, hidden_units=n, input_size=8, output_size=1) #if hidden_layers=0, hidden_units=0 -> linear-california
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
                    

            train_mse = criterion(pred_train, y_train_tensor).item()
            
            print(f"train MSE = {train_mse} of {j}")

            train_r2 = r2_score(
            y_train_tensor.cpu().numpy().ravel(),
            pred_train.cpu().numpy().ravel()
            )

            model.eval()
            with torch.no_grad():
                    pred_train = model(X_train_tensor)
                    pred_test = model(X_test_tensor)

            test_mse = criterion(pred_test, y_test_tensor).item()
            test_r2 = r2_score(
                y_test_tensor.cpu().numpy().ravel(),
                pred_test.cpu().numpy().ravel()
            )

            print(f"train R^2 = {train_r2:.6f} of {j}")
            print(f"test R^2 = {test_r2:.6f} of {j}")


            export_split_to_csv(export_train_loader, "train", model, un_model, save_path, j)
            export_split_to_csv(export_test_loader, "test", model, un_model, save_path, j)

            start_time = time.time()
            est = LipConstEstimator(model=model)
            lip_trivial = est.estimate(method="trivial")
            lip_trivial_t = time.time()-start_time

            lip_eclipse = 0
            lip_eclipse_t=0
            lip_eclipse_fast=0
            lip_eclipse_fast_t=0


            start_time = time.time()
            est = LipConstEstimator(model=model)
            lip_eclipse = est.estimate(method="ECLipsE")
            lip_eclipse_t = time.time()-start_time

            start_time = time.time()
            est = LipConstEstimator(model=model)
            lip_eclipse_fast = est.estimate(method="ECLipsE_Fast")
            lip_eclipse_fast_t = time.time()-start_time


                
            csv_path = out_dir / f"model_{j}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["constant type", "value", "seconds required"])
                writer.writerow(["trivial", lip_trivial, lip_trivial_t ])
                writer.writerow(["ECLipsE", lip_eclipse, lip_eclipse_t])
                writer.writerow(["ECLipsE_Fast", lip_eclipse_fast, lip_eclipse_fast_t])
                writer.writerow(["R^2 on train", train_r2, 0])
                writer.writerow(["R^2 on test", test_r2, 0])

            
            j=j+1

        print("done")

exit()
