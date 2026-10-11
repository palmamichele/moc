from pathlib import Path
import torch
import copy 
import random
import torchvision.transforms as transforms
import numpy as np 
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from utils import Tee, NeuralNet, accuracy_under_attack, box_clipping, pgd_m_x, pgd_moc, make_t_grid
from pgdmoc.pgd_moc import moc_regularizer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score


#PARS
num_experiments=1
epsilon = 0.3 #usually be 0.3 for MNIST, 8/255 for CIFAR-10, 0.5 for CIFAR-100
alpha = 0.01 #learning rate for PGD
num_iter = 10 #of PGD
num_classes = 10
num_epochs=500
num_restarts = 1
seed=0
batch_size=512
nbins=10 #for MOC
learning_rate = alpha
norm="L2" #on X space
clipping=None
lambda_k = 0.20
#


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using a GPU

# Enforce deterministic algorithms in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if norm=="L2":
    p=2
elif norm=="L1":
    p=1


def per_example_mse(pred, target):
    # nn.MSELoss(reduction='none') returns shape (batch_size, 1)
    # We squeeze it to get (batch_size,)
    loss_fn = nn.MSELoss(reduction='none')
    return loss_fn(pred, target).squeeze(-1)


def l2_distance(x,y):
    diff = x-y
    return diff.flatten(1).norm(p=2, dim=1)




base_path = Path("data")
save_path = base_path / "california"
save_path.mkdir(parents=True, exist_ok=True)

folder_path =  Path("experiments") / str("california")
cal_data = fetch_california_housing()
X = cal_data.data.astype(np.float32)
y = cal_data.target.astype(np.float32).reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train).astype(np.float32) #live in [-1,1]
X_test_scaled = x_scaler.transform(X_test).astype(np.float32)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train).astype(np.float32)
y_test_scaled = y_scaler.transform(y_test).astype(np.float32)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

t_values =  np.linspace(0.001, epsilon, nbins)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


out_dir = Path("experiments") / "california"
out_dir.mkdir(parents=True, exist_ok=True)


for i in range(num_experiments):
    base_model = NeuralNet(hidden_layers=2, hidden_units=16, input_size=8, output_size=1)
    criterion = nn.MSELoss()
    

    for model_type in ["vanilla", "adv", "mocreg"]:
        model = copy.deepcopy(base_model)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)
        
        model.train()

        #based on type do adv traing, moc regularization, or vanilla training.
        
        if model_type == "vanilla":
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                total_samples = 0

                for xb, yb in train_loader:
                    optimizer.zero_grad()

                    pred = model(xb)
                    loss = criterion(pred, yb)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * xb.size(0)
                    total_samples += xb.size(0)

                epoch_loss = total_loss / total_samples

                if (epoch + 1) % 100 == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs}, "
                        f"train MSE = {epoch_loss:.6f}"
                    )

            

                #print(f"epoch {epoch}, train loss = {loss.item():.6f}")

        elif model_type == "adv":
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                total_samples = 0

                for xb, yb in train_loader:
                    # Generate adversarial perturbations for this batch
                    _, deltas = pgd_m_x(
                        model,
                        xb,
                        yb,
                        epsilon,
                        per_example_mse,
                        norm,
                        clipping,
                        num_iter,
                        alpha
                    )

                    adv_input = xb + deltas

                    # Update model on adversarial examples
                    optimizer.zero_grad()

                    outputs = model(adv_input)
                    loss = criterion(outputs, yb)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * xb.size(0)
                    total_samples += xb.size(0)

                epoch_loss = total_loss / total_samples

                if (epoch + 1) % 100 == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs}, "
                        f"adversarial MSE = {epoch_loss:.6f}"
                    )

            
                #print(f'Epoch [{epoch+1}/{num_epochs}], 'f'Training Loss (adv): {loss.item():.4f}')

        elif model_type == "mocreg":
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                total_clean_loss = 0.0
                total_moc_loss = 0.0
                total_samples = 0

                for xb, yb in train_loader:
                    optimizer.zero_grad()

                    # Clean loss on the current mini-batch
                    outputs = model(xb)
                    clean_loss = criterion(outputs, yb)

                    # MOC regularizer on the current mini-batch
                    moc_loss, omega_values, t_grid = moc_regularizer(
                        model=model,
                        X=xb,
                        dy_f=per_example_mse,
                        norm=norm,
                        t_values=t_values,
                        clipping_f=clipping,
                        step_size=alpha,
                        num_iter=num_iter,
                        num_restarts=num_restarts,
                        batch_size=batch_size,
                    )


                    clean_loss.backward(retain_graph=True)
                    grad_clean = torch.nn.utils.parameters_to_vector(
                        [p.grad for p in model.parameters() if p.grad is not None]
                    )
                    norm_clean = grad_clean.norm(p=p)

                    optimizer.zero_grad()
                    moc_loss.backward(retain_graph=True)
                    grad_moc = torch.nn.utils.parameters_to_vector(
                        [p.grad for p in model.parameters() if p.grad is not None]
                    )
                    norm_moc = grad_moc.norm(p=2)
                    lambda_moc = norm_clean / (norm_moc + 1e-12)

                    optimizer.zero_grad()
                    loss = clean_loss + lambda_k*lambda_moc * moc_loss
                    loss.backward()

                    
                    optimizer.step()

                    # torch.nn.utils.clip_grad_norm_(
                    #     model.parameters(),
                    #     max_norm=1.0
                    # )

                    # n = xb.size(0)
                    # total_loss += loss.item() * n
                    # total_clean_loss += clean_loss.item() * n
                    # total_moc_loss += moc_loss.item() * n
                    # total_samples += n

                if (epoch + 1) % 100 == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs}, Loss {loss.item()}"
                        # f"total={total_loss / total_samples:.6f}, "
                        # f"clean={total_clean_loss / total_samples:.6f}, "
                        # f"MOC={total_moc_loss / total_samples:.6f}"
                    )

        else:
            pass


        #regardless of the type, compute clean test and train accuracy/mse, 
        model.eval()
        with torch.no_grad():
            pred_train = model(X_train_tensor)
            pred_test = model(X_test_tensor)
                
        train_mse = criterion(pred_train, y_train_tensor).item()
        test_mse = criterion(pred_test, y_test_tensor).item()

        print(model_type)
        print(f"test MSE ={test_mse}")
        #print(f"train MSE ={train_mse}")


        train_r2 = r2_score(
                y_train_tensor.cpu().numpy().ravel(),
                pred_train.cpu().numpy().ravel()
        )

        test_r2 = r2_score(
                y_test_tensor.cpu().numpy().ravel(),
                pred_test.cpu().numpy().ravel()
        )

        print(f"test R^2 ={test_r2}")
        #print(f"train R^2 ={train_r2}")

        test_mse_adv = 0
        #regardless of the type, compute (adversarial) test accuracy/mse, 
        _, deltas = pgd_m_x(model,X_test_tensor, y_test_tensor,epsilon,per_example_mse,norm,clipping,num_iter,alpha)
        adv_input = X_test_tensor+deltas

        # Evaluate adversarial examples
        with torch.no_grad():
            pred = model(adv_input)
        
        test_mse_adv = criterion(pred, y_test_tensor).item()

        test_r2_adv = r2_score(
                    y_test_tensor.cpu().numpy().ravel(),
                    pred.cpu().numpy().ravel()
        )

        print(f"test mse (adv) ={test_mse_adv}")
        print(f"test R^2 (adv) ={test_r2_adv}")

        #save the stats 

        #regardless of the type, compute the moc of trained network. 

        # aua_test = accuracy_under_attack(
        #         model,
        #         test_loader,
        #         per_example_mse,
        #         clipping,
        #         norm,
        #         epsilon,
        #         alpha,
        #         num_iter
        #     )
        
        # print(f"test accuracy under attack  = {aua_test:.4f}")