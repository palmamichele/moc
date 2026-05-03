
import numpy as np 
import matplotlib.pyplot as plt 
import csv
import torch
from torch import nn, optim
from pathlib import Path

class LipConstEstimatorL1():
    def __init__(self, model):
        """
        Extract weights directly from PyTorch model (no extract_model_info needed).
        Assumes sequential fully-connected layers: Linear -> optional activation.
        """
        self.weights = []
        self.num_layers = 0
        
        # Traverse model modules to extract Linear weights
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                self.weights.append(module.weight.data)  
                self.num_layers += 1
        
        if self.num_layers == 0:
            raise ValueError("No Linear layers found in model.")
        
        print(f"Extracted {self.num_layers} linear layer weights.")

    def estimate_trivial_l1(self):
        """trivial bound:|W|_1 = max column sum of |W|"""
        l1_norms = []
        for w in self.weights:
            col_sums = torch.sum(torch.abs(w), dim=0)  # Sum over input dim (rows)
            l1_norm = torch.max(col_sums)
            l1_norms.append(l1_norm)
        
        bound = torch.prod(torch.tensor(l1_norms)).item()
        return bound




def save_moc(moc, savepath, lbl):
    """
    moc is a vector ...
    """
            
    with open(Path(savepath) / f"{lbl}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for x in moc:
                writer.writerow([x])


def lipschitz_from_fmoc(fmocs, deltas):
    """Computes the discrete lipschitz constant from discrete modulus of continuity, as sup moc(d)/d for all d>0"""
    fmocs = np.asarray(fmocs)
    deltas = np.asarray(deltas)
    mask = deltas > 0          # safety
    return np.max(fmocs[mask] / deltas[mask])


def pad_moc_with_last(moc, target_len):
    """
    Pad a monotone MOC with its last value until target_len.
    If moc is longer, truncate it.
    If empty, return [None] * target_len.
    """
    moc = list(moc)
    if len(moc) == 0:
        return [None] * target_len

    if len(moc) >= target_len:
        return moc[:target_len]

    last_val = moc[-1]
    return moc + [last_val] * (target_len - len(moc))


def export_split_to_csv(loader, split_name, tr_model, un_model, output_path, model_id):
    x_file = output_path / f"X_{split_name}.csv"
    y_file = output_path / f"Y_{split_name}.csv"
    tr_file = output_path / f"F_{split_name}_{model_id}.csv"
    utr_file = output_path / f"F_un_{split_name}_{model_id}.csv"


    with (
        open(x_file, "w", newline="") as fx,
        open(y_file, "w", newline="") as fy,
        open(tr_file, "w", newline="") as ftr,
        open(utr_file, "w", newline="") as fun 

    ):
        x_writer = csv.writer(fx)
        y_writer = csv.writer(fy)
        tr_writer = csv.writer(ftr)
        un_writer = csv.writer(fun)

        tr_model.eval()
        un_model.eval()

        with torch.no_grad():
            for x, y in loader:
                un_output = un_model(x)
                tr_output = tr_model(x)

                x_row = x[0].flatten().cpu().tolist()
                y_row = y[0].flatten().cpu().tolist()
                tr_row = tr_output[0].flatten().cpu().tolist()

                x_writer.writerow(x_row)
                y_writer.writerow(y_row)
                tr_writer.writerow(tr_row)

                un_row = un_output[0].flatten().cpu().tolist()
                un_writer.writerow(un_row)

    print(f"Exported {split_name} split to {output_path}")


class NeuralNet(nn.Module):
    def __init__(self, hidden_layers=1, hidden_units=512, input_size=28*28, output_size=10):
        super(NeuralNet, self).__init__()
        
        self.flatten = nn.Flatten()
        
        layers = []
        
        # first hidden layer
        if hidden_layers > 0:
            layers.append(nn.Linear(input_size, hidden_units))
            layers.append(nn.ReLU())
            
            # remaining hidden layers
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(nn.ReLU())
            
            # output layer
            layers.append(nn.Linear(hidden_units, output_size))
        else:
            # no hidden layer case
            layers.append(nn.Linear(input_size, output_size))
        
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output