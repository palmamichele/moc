#inspired by ECLipsE MNIST code
from pathlib import Path
import torch
import copy 
import csv
import time 
import torchvision
import torchvision.transforms as transforms
import numpy as np 
from torch import nn, optim
from torch.utils.data import DataLoader,  TensorDataset
from torchvision.datasets import MNIST
from utils import export_split_to_csv, NeuralNet
from eclipse_nn.LipConstEstimator import LipConstEstimator

np.random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


out_dir = Path("experiments") / "MNIST"
out_dir.mkdir(parents=True, exist_ok=True)

lyrs = [3, 20, 100] #[2, 5, 10, 20, 30, 50, 75, 100]
neurons = [50, 100, 200]#[20, 40, 60, 80, 100]
num_classes = 10
n_experiments=1

data_path = Path("data")
# Load the training and test sets

# Transform the data to torch tensors and normalize it
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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

y_train_export_tensor = torch.tensor(y_train_onehot, dtype=torch.float32)
y_test_export_tensor = torch.tensor(y_test_onehot, dtype=torch.float32)


y_train_export_tensor = torch.tensor(y_train_onehot, dtype=torch.float32)
y_test_export_tensor = torch.tensor(y_test_onehot, dtype=torch.float32)

export_train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_export_tensor),
    batch_size=1,
    shuffle=False
)

export_test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_export_tensor),
    batch_size=1,
    shuffle=False
)

j=0
for l in lyrs:
    for n in neurons:
        for i in range(n_experiments):
            model = NeuralNet(hidden_layers=l, hidden_units=n)
            un_model = copy.deepcopy(model)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            # Training Loop
            num_epochs = 10
            for epoch in range(num_epochs):
                for images, labels in train_loader:
                    # Forward pass
                    optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            # evaluation on original loaders
            model.eval()
            with torch.no_grad():
                train_correct = 0
                train_total = 0
                for images, labels in train_loader:
                    probs = torch.softmax(model(images), dim=1)
                    preds = probs.argmax(dim=1)
                    train_total += labels.size(0)
                    train_correct += (preds == labels).sum().item()

                test_correct = 0
                test_total = 0
                for images, labels in test_loader:
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

            export_split_to_csv(export_train_loader, "train", model, un_model, save_path, j)
            export_split_to_csv(export_test_loader, "test", model, un_model, save_path, j)

            start_time = time.time()
            est = LipConstEstimator(model=model)
            lip_trivial = est.estimate(method="trivial")
            lip_trivial_t = time.time() - start_time


            lip_eclipse = 0
            lip_eclipse_t=0
            lip_eclipse_fast=0
            lip_eclipse_fast_t=0

            print("ok")
            start_time = time.time()
            est = LipConstEstimator(model=model)
            lip_eclipse = est.estimate(method="ECLipsE")
            lip_eclipse_t = time.time() - start_time

            print("okk")
            start_time = time.time()
            est = LipConstEstimator(model=model)
            lip_eclipse_fast = est.estimate(method="ECLipsE_Fast")
            lip_eclipse_fast_t = time.time() - start_time

            csv_path = out_dir / f"model_{j}.csv"
            print("okkk")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["constant type", "value", "seconds required"])
                writer.writerow(["trivial", lip_trivial, lip_trivial_t])
                writer.writerow(["ECLipsE", lip_eclipse, lip_eclipse_t])
                writer.writerow(["ECLipsE_Fast", lip_eclipse_fast, lip_eclipse_fast_t])
                writer.writerow(["accuracy on train", train_acc, 0])
                writer.writerow(["accuracy on test", test_acc, 0])
            j=j+1



