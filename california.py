import argparse
import torch
import copy 
import random
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
from torch import nn, optim
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from utils import Tee, NeuralNet, accuracy_under_attack, box_clipping, pgd_m_x, pgd_moc, make_t_grid, compute_robustness_auc, per_example_mse, l2_distance
from pgdmoc.pgd_moc import moc_regularizer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score


#PARS
num_experiments=5
epsilon = 0.3  #usually be 0.3 for MNIST, 8/255 for CIFAR-10, 0.5 for CIFAR-100
num_attacks = 5
alpha = 0.01 #learning rate for PGD
num_iter = 10 #10 #of PGD
num_classes = 10
num_epochs= 500 #500 #500
num_restarts = 1
seed=0
nbins=5 #for MOC
learning_rate = alpha
norm="L2" #on X space
clipping=None
lambda_k = 1
model_types = ["vanilla", "adv", "mocreg", "spectral"]
lambda_spectral = 0.01  
#


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(
    description="Run"
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="Batch size for training and evaluation (default: 1)"
)

args = parser.parse_args()
batch_size = args.batch_size

if batch_size <= 0:
    parser.error("--batch-size must be a positive integer")


base_path = Path("data")
save_path = base_path / "california"
save_path.mkdir(parents=True, exist_ok=True)

folder_path =  Path("experiments") / str("california")
file_ext = "_"+str(batch_size)+"_"+norm+"_"+str(lambda_k)

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

t_values = np.linspace(0, epsilon, nbins)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
attacks_budget = np.linspace(epsilon/10, 10*epsilon, num_attacks)


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

all_experiment_results = []
all_experiment_mocs = {model_type: [] for model_type in model_types}

all_learning_curves = {model_type: [] for model_type in model_types}
for i in range(num_experiments):
    all_mocs={}

    all_mse = {} 

    all_adv_mse = {}

    all_gen_gap ={}

    base_model = NeuralNet(hidden_layers=2, hidden_units=16, input_size=8, output_size=1)
    criterion = nn.MSELoss()

 
    for model_type in model_types:
        history = {
        "epoch": [],
        "train": [],
        "test": [],
        }
    
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
                    total_samples += xb.size(0) #in case of remainder of batch 

                epoch_loss = total_loss / total_samples

                if (epoch + 1) % 100 == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs}, "
                        f"train MSE = {epoch_loss:.6f}"
                    )


                model.eval()
                with torch.no_grad():
                    pred_test = model(X_test_tensor)
                    test_mse = criterion(pred_test, y_test_tensor).item()

                history["epoch"].append(epoch + 1)
                history["train"].append(epoch_loss)
                history["test"].append(test_mse)

            

                #print(f"epoch {epoch}, train loss = {loss.item():.6f}")

        elif model_type == "adv":
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                total_samples = 0
                train_loss_on_clean = 0 

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
                    outputs_for_clean = model(xb)
                    train_loss_on_clean += criterion(outputs_for_clean, yb).item() * xb.size(0)
                    optimizer.step()

                    total_loss += loss.item() * xb.size(0)
                    total_samples += xb.size(0)

                    

                epoch_loss = total_loss / total_samples

                if (epoch + 1) % 100 == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs}, "
                        f"adversarial MSE = {epoch_loss:.6f}"
                    )


                model.eval()
                with torch.no_grad():
                    pred_test = model(X_test_tensor)
                    test_mse = criterion(pred_test, y_test_tensor).item()


                history["epoch"].append(epoch + 1)
                history["train"].append(train_loss_on_clean/ total_samples)
                history["test"].append(test_mse)

            
                #print(f'Epoch [{epoch+1}/{num_epochs}], 'f'Training Loss (adv): {loss.item():.4f}')

        elif model_type == "mocreg":
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
               
                total_moc_loss = 0.0
                total_samples = 0
                train_loss_on_clean = 0 

                for xb, yb in train_loader:
                    optimizer.zero_grad()

                    # Clean loss on the current mini-batch
                    outputs = model(xb)
                    clean_loss = criterion(outputs, yb)

                    # MOC regularizer on the current mini-batch
                    moc_loss, omega_values, t_grid = moc_regularizer(
                        model=model,
                        X=xb,
                        dy_f=l2_distance,
                        norm=norm,
                        t_values=t_values,
                        clipping_f=clipping,
                        step_size=alpha,
                        num_iter=num_iter,
                        num_restarts=num_restarts
                    )
                    params = tuple(model.parameters())

                    clean_grads = torch.autograd.grad(
                        clean_loss,
                        params,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )

                    moc_grads = torch.autograd.grad(
                        moc_loss,
                        params,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )

                    norm_clean = torch.sqrt(sum(
                        (g.detach() ** 2).sum()
                        for g in clean_grads
                        if g is not None
                    ))

                    norm_moc = torch.sqrt(sum(
                        (g.detach() ** 2).sum()
                        for g in moc_grads
                        if g is not None
                    ))

                    #lambda_moc = norm_clean / (norm_moc + 1e-12)
                    

                    # clean_loss.backward(retain_graph=True)
                    # grad_clean = torch.nn.utils.parameters_to_vector(
                    #     [p.grad for p in model.parameters() if p.grad is not None]
                    # )
                    # norm_clean = grad_clean.norm(p=p)

                    # optimizer.zero_grad()
                    # moc_loss.backward(retain_graph=True)
                    # grad_moc = torch.nn.utils.parameters_to_vector(
                    #     [p.grad for p in model.parameters() if p.grad is not None]
                    # )
                    # norm_moc = grad_moc.norm(p=2)
                    # lambda_moc = norm_clean / (norm_moc + 1e-12)
                    optimizer.zero_grad()
                    #loss = clean_loss + lambda_k*lambda_moc.detach() * moc_loss
                    loss = clean_loss + lambda_k* moc_loss
                    loss.backward()
                    train_loss_on_clean +=clean_loss.item() * xb.size(0)
                    total_samples += xb.size(0)
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

                with torch.no_grad():
                    pred_test = model(X_test_tensor)
                    test_mse = criterion(pred_test, y_test_tensor).item()


                history["epoch"].append(epoch + 1)
                history["train"].append(train_loss_on_clean/ total_samples)
                history["test"].append(test_mse)

        elif model_type == "spectral":
            v_vectors = {}
            for name, param in model.named_parameters():
                if 'weight' in name:
                    v_vectors[name] = torch.randn(param.size(1), 1, device=param.device)
                    v_vectors[name] = v_vectors[name] / v_vectors[name].norm()

            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                total_samples = 0

                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    loss.backward() 

                    # Add Spectral Norm Regularization Gradient
                    for name, param in model.named_parameters():
                        if 'weight' in name and param.grad is not None:
                            W = param.data
                            v = v_vectors[name]

                            #one step of Power iteration
                            u = torch.matmul(W, v)
                            u = u / u.norm()
                            v = torch.matmul(W.t(), u)
                            v = v / v.norm()

                            v_vectors[name] = v.detach()

                            sigma = torch.matmul(u.t(), torch.matmul(W, v))

                            #add penalty gradient: lambda * sigma * u * v^T
                            spectral_grad = lambda_spectral * sigma * torch.matmul(u, v.t())
                            param.grad.data.add_(spectral_grad)


                    total_loss += loss.item() * xb.size(0)
                    total_samples += xb.size(0)
                    optimizer.step()

                    
               


                epoch_loss = total_loss / total_samples
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Spectral MSE = {epoch_loss:.6f}")

                with torch.no_grad():
                    pred_test = model(X_test_tensor)
                    test_mse = criterion(pred_test, y_test_tensor).item()


                history["epoch"].append(epoch + 1)
                history["train"].append(total_loss/ total_samples)
                history["test"].append(test_mse)





        elif model_type == "lcc":
            pass
        else:
            pass


        all_learning_curves[model_type].append(pd.DataFrame(history))

        #regardless of the type, compute clean test and train accuracy/mse, 

        # print(model_type)
        # print(f"test MSE ={test_mse}")
        #print(f"train MSE ={train_mse}")


        # train_r2 = r2_score(
        #         y_train_tensor.cpu().numpy().ravel(),
        #         pred_train.cpu().numpy().ravel()
        # )

        # test_r2 = r2_score(
        #         y_test_tensor.cpu().numpy().ravel(),
        #         pred_test.cpu().numpy().ravel()
        # )

        # print(f"test R^2 ={test_r2}")
        #print(f"train R^2 ={train_r2}")

        all_adv_mse[model_type]={}

        for b in attacks_budget:
            test_mse_adv = 0
            #regardless of the type, compute (adversarial) test accuracy/mse, 
            _, deltas = pgd_m_x(model,X_test_tensor, y_test_tensor,b,per_example_mse,norm,clipping,num_iter,alpha)
            adv_input = X_test_tensor+deltas

            # Evaluate adversarial examples
            with torch.no_grad():
                pred = model(adv_input)
            
            test_mse_adv = criterion(pred, y_test_tensor).item()

            # test_r2_adv = r2_score(
            #             y_test_tensor.cpu().numpy().ravel(),
            #             pred.cpu().numpy().ravel()
            # )
            all_adv_mse[model_type][b] = test_mse_adv #later we also use an array for epsilon value 

        # print(f"test mse (adv) ={test_mse_adv}")
        # print(f"test R^2 (adv) ={test_r2_adv}")



        #compute the network moc, 
        moc, _ = pgd_moc(
        model, #f_\theta
        X_train_tensor,
        model(X_train_tensor), #f_\theta(X) 
        l2_distance, #d_Y as loss function (assuming it satisfies metric properties)
        clipping_f=clipping,
        norm=norm, #L2, L1, Linf
        t_values=t_values, #t_1,...,t_K
        step_size=alpha, 
        numiter=num_iter,
        numrestarts=num_restarts,
        batch_size=batch_size)


        if isinstance(moc, torch.Tensor):
            moc = moc.detach().cpu().numpy()
        else:
            moc = np.asarray(moc)

        moc = moc.reshape(-1)
        all_mocs[model_type] = moc
        all_experiment_mocs[model_type].append(moc.copy())
        all_mse[model_type] = history["test"][-1]
        all_gen_gap[model_type] = np.abs(history["test"][-1]-history["train"][-1])


    results_data = []
    for model_type in model_types:

        auc = compute_robustness_auc(
        all_adv_mse[model_type],
        attacks_budget,
        )

        row = ({
            "Experiment": i,
            "Model": model_type,
            "Clean": all_mse[model_type],
            "Generalization Gap": all_gen_gap[model_type],
            "AUC": auc,
        })
        for b in attacks_budget:
            row[f"{b}"] =  all_adv_mse[model_type][b]
        results_data.append(row)
    df_results = pd.DataFrame(results_data)


    df_results["Cumulative Robustness"] = 0.0
    for b in attacks_budget:
        col_name = f"{b}"
        best_adv_mse_for_eps = df_results[col_name].min()
        
        df_results["Cumulative Robustness"] += df_results[col_name] - best_adv_mse_for_eps

  
    all_experiment_results.append(df_results)



#compute mean and stdev across experiments
df_all = pd.concat(all_experiment_results, ignore_index=True)
o_metrics = [
    "Clean",
    "Generalization Gap",
    "Cumulative Robustness",
    "AUC"
]

metrics = o_metrics.copy()
for b in attacks_budget:
    metrics.append(f"{b}")

df_summary = (
    df_all
    .groupby("Model")[metrics]
    .agg(["mean", "std"])
)



#df_all.to_csv(out_dir / f"experiments_results{file_ext}.csv", index=False)
#df_summary.to_csv(out_dir / f"summary{file_ext}.csv")



data = {}
for metric in o_metrics:
    for model_type in df_summary.index:
        mean = df_summary.loc[model_type, (metric, "mean")]
        std = df_summary.loc[model_type, (metric, "std")]

        data[(metric, model_type)] = rf"{mean:.4f} $\pm$ {std:.4f}"

df_latex = pd.DataFrame([data], index=[f"{folder_path.name}"])


df_latex.columns = pd.MultiIndex.from_tuples(
    df_latex.columns,
    names=["Metric", "Model type"]
)

df_transposed = df_latex.T
df_transposed.columns = [folder_path.name+f"{file_ext}"]
df_transposed.index.names = ["Metric", "Model type"]
df_transposed.to_pickle(folder_path / f"df_transposed{file_ext}.pkl")


adv_summary = {}

for model_type in model_types:
    adv_mean = np.array([
        df_summary.loc[model_type, (f"{b}", "mean")]
        for b in attacks_budget
    ])

    adv_std = np.array([
        df_summary.loc[model_type, (f"{b}", "std")]
        for b in attacks_budget
    ])

    adv_summary[model_type] = {
        "mean": adv_mean,
        "std": adv_std,
    }

    # Create CSV for PGFPlots
    df_adv = pd.DataFrame({
        "t": attacks_budget,
        "mean": adv_mean,
        "std": adv_std,
    })

    df_adv.to_csv(
        out_dir / f"adv_{model_type}{file_ext}.csv",
        index=False,
    )


moc_summary = {}
for model_type in model_types:
    moc_array = np.stack(all_experiment_mocs[model_type], axis=0)

    moc_mean = moc_array.mean(axis=0)

    # Sample standard deviation across initializations
    moc_std = moc_array.std(axis=0, ddof=1)

    moc_summary[model_type] = {
        "mean": moc_mean,
        "std": moc_std,
        "values": moc_array,
    }

    #create dataframe for PGFPlots
    df_moc = pd.DataFrame({
        "t": t_values,
        "mean": moc_mean,
        "std": moc_std,
    })

    df_moc.to_csv(
        out_dir / f"moc_{model_type}{file_ext}.csv",
        index=False
    )


for model_type in model_types:

    train_array = np.stack([
        df["train"].to_numpy()
        for df in all_learning_curves[model_type]
    ])

    test_array = np.stack([
        df["test"].to_numpy()
        for df in all_learning_curves[model_type]
    ])

    df_curve = pd.DataFrame({
        "epoch": all_learning_curves[model_type][0]["epoch"],
        "train_mean": train_array.mean(axis=0),
        "train_std": train_array.std(axis=0, ddof=1),
        "test_mean": test_array.mean(axis=0),
        "test_std": test_array.std(axis=0, ddof=1),
    })

    df_curve.to_csv(
        out_dir / f"learning_curve_{model_type}{file_ext}.csv",
        index=False,
    )