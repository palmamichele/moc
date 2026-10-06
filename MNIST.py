from pathlib import Path
import torch
import copy 
import csv
import sys 
import matplotlib.pyplot as plt
import pandas as pd 
import time 
import torchvision.transforms as transforms
import numpy as np 
from torch import nn, optim
from torch.utils.data import DataLoader,  TensorDataset, Subset
from torchvision.datasets import MNIST
from utils import export_split_to_csv, NeuralNet, LipConstEstimatorL1, accuracy_under_attack, box_clipping
from eclipse_nn.LipConstEstimator import LipConstEstimator
from pgdmoc.pgd_moc import pgd_moc
from torch.utils.data import Subset


###PARS
norm_dx = "L2" #L2, L1, Linf
def l2_distance(x,y):
    diff = x-y
    return diff.flatten(1).norm(p=2, dim=1)





alpha = 0.01
num_iter = 40
num_restarts=1
batch_size= 70000
###

aua_results = []


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def l2_distance(x,y):
    diff = x-y
    return diff.flatten(1).norm(p=2, dim=1)





np.random.seed(0)
torch.manual_seed(0)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


out_dir = Path("experiments") / "MNIST"
out_dir.mkdir(parents=True, exist_ok=True)

log_file = open(out_dir/"log.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

lyrs = [3, 20, 5] 
neurons = [50, 100, 200]
num_classes = 10
n_experiments=1
num_epochs = 10 #following ECLipsE mnist code

data_path = Path("data")
# Load the training and test sets

# Transform the data to torch tensors and normalize it
transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5,), (0.5,))
])

train_data = MNIST(root=str(data_path), train=True, download=True, transform=transform)
test_data = MNIST(root=str(data_path), train=False, download=True, transform=transform)

save_path = Path("data") / "MNIST"
save_path.mkdir(parents=True, exist_ok=True)

# Data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

#tensors/loaders only for export_split_to_csv
X_train_list, y_train_list = [], []
for x, y in train_data:
    X_train_list.append(x.view(-1))
    y_train_list.append(y)

X_test_list, y_test_list = [], []
for x, y in test_data:
    X_test_list.append(x.view(-1))
    y_test_list.append(y)

X_train_tensor = torch.stack(X_train_list).float()
X_test_tensor = torch.stack(X_test_list).float()


y_train = np.array(y_train_list, dtype=np.int64)
y_test = np.array(y_test_list, dtype=np.int64)


y_train_class_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_class_tensor = torch.tensor(y_test, dtype=torch.long)

y_train_onehot = np.eye(num_classes, dtype=np.float32)[y_train]
y_test_onehot = np.eye(num_classes, dtype=np.float32)[y_test]

Y_union = np.vstack([
    y_train_onehot,
    y_test_onehot
])

Y_union = torch.tensor(
    Y_union,
    dtype=torch.float32
)

y_train_export_tensor = torch.tensor(y_train_onehot, dtype=torch.float32)
y_test_export_tensor = torch.tensor(y_test_onehot, dtype=torch.float32)


y_train_export_tensor = torch.tensor(y_train_onehot, dtype=torch.float32)
y_test_export_tensor = torch.tensor(y_test_onehot, dtype=torch.float32)

# export_train_loader = DataLoader(
#     TensorDataset(X_train_tensor, y_train_export_tensor),
#     batch_size=1,
#     shuffle=False
# )


# export_test_loader = DataLoader(
#     TensorDataset(X_test_tensor, y_test_export_tensor),
#     batch_size=1,
#     shuffle=False
# )

j=0
for l in lyrs:
    for n in neurons:
        for i in range(n_experiments):
            model = NeuralNet(hidden_layers=l, hidden_units=n).to(device)
            un_model = copy.deepcopy(model).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            if l==lyrs[-1]: #overfitting case
                small_train_dataset = Subset(train_loader.dataset, range(5))
                train_loader = DataLoader(
                    small_train_dataset,
                    batch_size=train_loader.batch_size,
                    shuffle=False
                )
                num_epochs=100 

            model.train()
            
            for epoch in range(num_epochs):
                loss = 0.0
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

           
            model.eval()
            with torch.no_grad():
                train_correct = 0
                train_total = 0
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    probs = torch.softmax(model(images), dim=1)
                    preds = probs.argmax(dim=1)
                    train_total += labels.size(0)
                    train_correct += (preds == labels).sum().item()

                test_correct = 0
                test_total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    probs = torch.softmax(model(images), dim=1)
                    preds = probs.argmax(dim=1)
                    test_total += labels.size(0)
                    test_correct += (preds == labels).sum().item()

                train_acc = train_correct / train_total
                test_acc = test_correct / test_total
            print(f"train acc = {train_acc:.4f}")
            print(f"test acc = {test_acc:.4f}")


            #save softmax output from the model
            model = nn.Sequential(
                model,
                nn.Softmax(dim=1)
            )


            un_model = nn.Sequential(
                un_model,
                nn.Softmax(dim=1)
            )

            model.eval()
            un_model.eval()

            with torch.no_grad():
                F_train = model(X_train_tensor)
                F_test = model(X_test_tensor)

                F_un_train = un_model(X_train_tensor)
                F_un_test = un_model(X_test_tensor)

            X_union = torch.cat(
                [X_train_tensor, X_test_tensor],
                dim=0
            ).to(device)

           
            F_union = torch.cat(
                [F_train, F_test],
                dim=0
            ).to(device)

            F_un_union = torch.cat(
                [F_un_train, F_un_test],
                dim=0
            ).to(device)

            trained_moc, t_values = pgd_moc(
                model, #f_\theta
                X_union,
                F_union, #either f_\theta(X) or original labels for X
                l2_distance, #d_Y as loss function (assuming it satisfies metric properties)
                box_clipping,
                norm_dx, #L2, L1, Linf
                t_values=None, #t_1,...,t_K
                step_size=None,
                numiter=1,
                nbins=100,
                batch_size=batch_size
            )




            untrained_moc, _ = pgd_moc(
                un_model, #f_\theta
                X_union,
                F_un_union, #either f_\theta(X) or original labels for X
                l2_distance, #d_Y as loss function (assuming it satisfies metric properties)
                box_clipping,
                norm_dx, #L2, L1, Linf
                t_values, #t_1,...,t_K
                step_size=None,
                numiter=1,
                nbins=100,
                batch_size=batch_size
            )

            plt.figure(figsize=(8, 5))


            plt.plot(
                t_values,
                trained_moc,
                label="trained",
                linewidth=2
            )


            plt.plot(
                t_values,
                untrained_moc,
                "o-",
                label="untrained",
                linewidth=2,
                markersize=4
            )


            plt.xlabel(r"$t$")
            plt.ylabel(r"$\omega(t)$")

            plt.xscale("log")
            plt.yscale("log")

            # plt.xlim(1e-286, 1e2)
            # plt.ylim(1e-12, 1e2)

            plt.title(
                f"MNIST: {l} hidden layers, {n} neurons, experiment {i+1}"
            )

            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(out_dir / f'mnist_{l}_{n}_{i+1}.png')

            for epsilon in t_values:
                aua_test = accuracy_under_attack(
                                model,
                                test_loader,
                                criterion,
                                box_clipping,
                                norm_dx,
                                epsilon,
                                alpha,
                                num_iter
                )
                
                aua_results.append({
                    "hidden_layers": l,
                    "neurons": n,
                    "experiment": i + 1,
                    "epsilon": float(epsilon),
                    "aua_test": float(aua_test),
                })


    




            # export_split_to_csv(export_train_loader, "train", model, un_model, save_path, j)
            # export_split_to_csv(export_test_loader, "test", model, un_model, save_path, j)

            start_time = time.time()
            est = LipConstEstimator(model=model)
            lip_trivial = est.estimate(method="trivial")
            lip_trivial_t = time.time() - start_time


            lip_eclipse = 0
            lip_eclipse_t=0
            lip_eclipse_fast=0
            lip_eclipse_fast_t=0

            
            start_time = time.time()
            est = LipConstEstimator(model=model)
            lip_eclipse = est.estimate(method="ECLipsE")
            lip_eclipse_t = time.time() - start_time

           
            start_time = time.time()
            est = LipConstEstimator(model=model)
            lip_eclipse_fast = est.estimate(method="ECLipsE_Fast")
            lip_eclipse_fast_t = time.time() - start_time

            start_time = time.time()
            estimator_l1 = LipConstEstimatorL1(model=model)
            l1_bound = estimator_l1.estimate_trivial_l1()
            l1_bound_t = time.time()-start_time

            csv_path = out_dir / f"model_{j}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["constant type", "value", "seconds required"])
                writer.writerow(["trivial_l2", lip_trivial, lip_trivial_t ]) #trivial must be adj for softmaxhead
                writer.writerow(["trivial_l1", l1_bound, l1_bound_t ])
                writer.writerow(["ECLipsE", lip_eclipse, lip_eclipse_t])
                writer.writerow(["ECLipsE_Fast", lip_eclipse_fast, lip_eclipse_fast_t])
                writer.writerow(["accuracy on train", train_acc, 0])
                writer.writerow(["accuracy on test", test_acc, 0])
            j=j+1


aua_df = pd.DataFrame(aua_results)
aua_df.to_csv(out_dir / "accuracy_under_attack.csv", index=False)
