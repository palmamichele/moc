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

n_epochs=100
out_dir = Path("experiments") / "california"
out_dir.mkdir(parents=True, exist_ok=True)


lyrs = [3, 20, 100] #[2, 5, 10, 20, 30, 50, 75, 100]
neurons = [100, 300, 400] #[20, 40, 60, 80, 100]
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
                pred_test = model(X_test_tensor)

            train_mse = criterion(pred_train, y_train_tensor).item()
            test_mse = criterion(pred_test, y_test_tensor).item()
            print(f"train MSE = {train_mse} of {j}")
            print(f"test MSE = {test_mse} of {j}")

            train_r2 = r2_score(
            y_train_tensor.cpu().numpy().ravel(),
            pred_train.cpu().numpy().ravel()
            )

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

#iris,
save_path = base_path / "iris"
out_dir = Path("experiments") / "iris"
save_path.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

iris_data = load_iris()
X = iris_data.data.astype(np.float32)
y = iris_data.target.astype(np.int64)
num_classes = len(iris_data.target_names)

# Split into train/test sets
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

# loaders for export_split_to_csv
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



for i in range(n_experiments):
    # model
    model = ReLUNet(in_s=len(X_train[0]), hiddens=[], out_s=num_classes, relu_enabled=False)
    un_model = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    # train
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()

        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_class_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"epoch {epoch}, loss = {loss.item():.4f}")

    # evaluation
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

    # wrap models so export function writes probabilities instead of raw logits
    model = nn.Sequential(model, nn.Softmax(dim=1))
    un_model = nn.Sequential(un_model, nn.Softmax(dim=1))

    # export using your existing function
    export_split_to_csv(train_loader, "train", model, un_model, save_path, i)
    export_split_to_csv(test_loader, "test", model, un_model, save_path, i)


    start_time = time.time()
    est = LipConstEstimator(model=model)
    lip_trivial = est.estimate(method="trivial")
    lip_trivial_t = time.time()-start_time


    start_time = time.time()
    est = LipConstEstimator(model=model)
    lip_eclipse = est.estimate(method="ECLipsE")
    lip_eclipse_t = time.time()-start_time

    start_time = time.time()
    est = LipConstEstimator(model=model)
    lip_eclipse_fast = est.estimate(method="ECLipsE_Fast")
    lip_eclipse_fast_t = time.time()-start_time

    csv_path = out_dir / f"model_{i}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["constant type", "value", "seconds required"])
        writer.writerow(["trivial", lip_trivial, lip_trivial_t ])
        writer.writerow(["ECLipsE", lip_eclipse, lip_eclipse_t])
        writer.writerow(["ECLipsE_Fast", lip_eclipse_fast, lip_eclipse_fast_t])

exit()





#we know that without ReLU, the Lipschitz constant coincides with the operator norm, we check how far we are from it.

#might be beneficial to visualize the normalized lipschitz estimates

#the color will denote different models/benchmarks (exact, upper bound, eclipse, our moc,..., etc)

#one plot for each configuration  e.g. (in_s, out_s), the x axis denotes the number of datapoints used in moc, lsh moc....






#for each network, export the reference dataset (divided among train and test), in the same format as the network would see it (e.g. after transforms), export the reference output for the net/layers for which moc has to be computed. 

#using l_2 we do flattening for matrices/tensors 


