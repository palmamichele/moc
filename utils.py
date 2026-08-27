import numpy as np 
import matplotlib.pyplot as plt 
import csv
import torch
from torch import nn, optim
from pathlib import Path


def pgd_attack(
    model,
    images,
    labels,
    criterion,
    epsilon=0.3,
    alpha=0.01,
    num_iter=40
):
    """
    Untargeted L2-PGD adversarial attack.

    images:
        normalized MNIST images in [-1, 1]

    epsilon:
        L2 perturbation radius in original [0, 1] pixel space

    alpha:
        L2 PGD step size in original [0, 1] pixel space
    """

    #denormalize images to [0, 1] range (specific to MNIST)
    original_images = images * 0.5 + 0.5
    original_images = original_images.detach()
    delta = torch.randn_like(original_images)
    delta_flat = delta.view(delta.size(0), -1)

    delta_norm = torch.norm(
        delta_flat,
        p=2,
        dim=1,
        keepdim=True
    )

    
    direction = delta_flat / (delta_norm + 1e-12)

    #uniformly sample a point inside the L2 ball
    d = delta_flat.size(1)
    u = torch.rand(
        delta.size(0),
        1,
        device=images.device
    )

    radius = epsilon * u.pow(1.0 / d)

    delta_flat = direction * radius

    delta = delta_flat.view_as(original_images)

    
    adv_images = original_images + delta

   
    adv_images = torch.clamp(
        adv_images,
        0.0,
        1.0
    )


    for _ in range(num_iter):

        adv_images.requires_grad_(True)


        normalized_adv_images = (
            adv_images - 0.5
        ) / 0.5

        logits = model(normalized_adv_images)


        loss = criterion(logits, labels)

    

        grad = torch.autograd.grad(
            loss,
            adv_images
        )[0]

        grad_flat = grad.view(
            grad.size(0),
            -1
        )

        grad_norm = torch.norm(
            grad_flat,
            p=2,
            dim=1,
            keepdim=True
        )

        normalized_grad = (
            grad_flat
            / (grad_norm + 1e-12)
        )

        normalized_grad = normalized_grad.view_as(grad)


        adv_images = (
            adv_images.detach()
            + alpha * normalized_grad
        )


        delta = adv_images - original_images

        delta_flat = delta.view(
            delta.size(0),
            -1
        )

        delta_norm = torch.norm(
            delta_flat,
            p=2,
            dim=1,
            keepdim=True
        )

        # If norm > epsilon, scale it back
        scale = torch.minimum(
            torch.ones_like(delta_norm),
            epsilon / (delta_norm + 1e-12)
        )

        delta_flat = delta_flat * scale

        delta = delta_flat.view_as(adv_images)


        adv_images = original_images + delta


        adv_images = torch.clamp(
            adv_images,
            0.0,
            1.0
        )


    adv_images = (
        adv_images - 0.5
    ) / 0.5

    return adv_images.detach()

def accuracy_under_attack(
    model,
    data_loader,
    criterion,
    epsilon=0.3,
    alpha=0.01,
    num_iter=40
):
    """
    Accuracy under an untargeted L2-PGD attack.
    """

    model.eval()

    correct = 0
    total = 0

    for images, labels in data_loader:

        # Generate L2-PGD adversarial examples
        adv_images = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            criterion=criterion,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter
        )

        # Evaluate adversarial examples
        with torch.no_grad():
            logits = model(adv_images)
            preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total



class LipConstEstimatorL1():
    def __init__(self, model):
        """
        Extract weights directly from PyTorch model (no extract_model_info needed).
        Assumes sequential fully-connected layers: Linear -> optional activation.
        """
        self.weights = []
        self.num_layers = 0
        
        
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
            col_sums = torch.sum(torch.abs(w), dim=0) 
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
        
        
        if hidden_layers > 0:
            layers.append(nn.Linear(input_size, hidden_units))
            layers.append(nn.ReLU())
            
            
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(nn.ReLU())
            
           
            layers.append(nn.Linear(hidden_units, output_size))
        else:
            
            layers.append(nn.Linear(input_size, output_size))
        
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output