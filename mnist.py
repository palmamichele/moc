import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from utils import (
    NeuralNet,
    accuracy_under_attack,
    box_clipping,
    pgd_m_x,
    pgd_moc,
    make_t_grid,
    compute_robustness_auc,
    l2_distance,
    linf_distance
)

from pgdmoc.pgd_moc import moc_regularizer

num_experiments = 5

# MNIST images are in [0, 1]
epsilon = 0.3

num_attacks = 5

# PGD step size
alpha = 0.01

# Number of PGD iterations
num_iter = 10

num_classes = 10

# Number of training epochs
num_epochs = 100

num_restarts = 1

seed = 0

# Number of points used for MOC
nbins = 5

learning_rate = alpha

# Attack norm
norm = "Linf"

# Since MNIST is already in [0, 1], use box clipping
clipping = box_clipping

# MOC coefficient
lambda_k = 1.0

# Spectral regularization coefficient
lambda_spectral = 0.01

model_types = [
    "vanilla",
    "adv",
    "mocreg",
    "spectral",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="MNIST classification robustness experiment"
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="Batch size for training/evaluation (default: 1)",
)

args = parser.parse_args()

batch_size = args.batch_size

if batch_size <= 0:
    parser.error("--batch-size must be a positive integer")


# ============================================================
# PATHS
# ============================================================

base_path = Path("data")

save_path = base_path / "mnist_with_clip"
save_path.mkdir(parents=True, exist_ok=True)

folder_path = Path("experiments") / "mnist"
folder_path.mkdir(parents=True, exist_ok=True)

file_ext = (
    "_"
    + str(batch_size)
    + "_"
    + norm
    + "_"
    + str(lambda_k)
)


# ============================================================
# DATA
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
])


train_dataset = MNIST(
    root=str(save_path),
    train=True,
    download=True,
    transform=transform,
)


test_dataset = MNIST(
    root=str(save_path),
    train=False,
    download=True,
    transform=transform,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
)


# ============================================================
# ATTACK / MOC GRIDS
# ============================================================

# MOC t values
t_values = np.linspace(
    0,
    epsilon,
    nbins,
)


# Attack budgets for robustness curve.
#
# We stay inside [0, epsilon] rather than using 10*epsilon,
# since epsilon is the maximum perturbation budget for this
# experiment.
attacks_budget = np.linspace(
    epsilon / 10,
    10*epsilon,
    num_attacks,
)


# ============================================================
# CLASSIFICATION LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


def per_example_cross_entropy(adv_output, y):
    return nn.functional.cross_entropy(
        adv_output,
        y.long(),
        reduction="none"
    )

# ============================================================
# ACCURACY FUNCTION
# ============================================================

def classification_accuracy(model, X, y):
    """
    Compute classification accuracy on a tensor.
    """

    model.eval()

    with torch.no_grad():

        logits = model(X)

        predictions = logits.argmax(dim=1)

        correct = (
            predictions == y
        ).sum().item()

        total = y.numel()

    return correct / total


# ============================================================
# EVALUATE CLEAN DATA
# ============================================================

def evaluate_clean(model, loader, device):
    """
    Evaluate clean classification loss and accuracy.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            # 1 x 28 x 28 -> 784
            xb = xb.view(
                xb.size(0),
                -1,
            )

            logits = model(xb)

            loss = criterion(
                logits,
                yb,
            )

            total_loss += (
                loss.item()
                * xb.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            total_correct += (
                predictions == yb
            ).sum().item()

            total_samples += xb.size(0)

    avg_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    return avg_loss, accuracy


# ============================================================
# LOAD ENTIRE TEST SET
# ============================================================

def get_full_dataset(loader, device):
    """
    Convert a DataLoader into tensors.

    Useful for the adversarial robustness evaluation.
    """

    X_list = []
    y_list = []

    for xb, yb in loader:

        xb = xb.view(
            xb.size(0),
            -1,
        )

        X_list.append(
            xb
        )

        y_list.append(
            yb
        )

    X = torch.cat(
        X_list,
        dim=0,
    ).to(device)

    y = torch.cat(
        y_list,
        dim=0,
    ).to(device)

    return X, y


X_test_tensor, y_test_tensor = get_full_dataset(
    test_loader,
    device,
)


# ============================================================
# OUTPUT STORAGE
# ============================================================

all_experiment_results = []

all_experiment_mocs = {
    model_type: []
    for model_type in model_types
}

all_learning_curves = {
    model_type: []
    for model_type in model_types
}


# ============================================================
# EXPERIMENT LOOP
# ============================================================

for experiment_idx in range(num_experiments):

    print()
    print("=" * 70)
    print(
        f"EXPERIMENT {experiment_idx + 1}"
        f"/{num_experiments}"
    )
    print("=" * 70)

    all_mocs = {}

    all_accuracy = {}

    all_adv_accuracy = {}

    all_gen_gap = {}


    # --------------------------------------------------------
    # SAME ARCHITECTURE AS BEFORE
    # --------------------------------------------------------
    #
    # Previous:
    #
    # input_size=8
    # hidden_layers=2
    # hidden_units=16
    # output_size=1
    #
    # MNIST:
    #
    # 784 -> 16 -> 16 -> 10
    #

    base_model = NeuralNet(
        hidden_layers=2,
        hidden_units=16,
        input_size=784,
        output_size=10,
    ).to(device)


    # ========================================================
    # MODEL TYPE LOOP
    # ========================================================

    for model_type in model_types:

        print()
        print("-" * 70)
        print(
            f"Training model: {model_type}"
        )
        print("-" * 70)


        history = {
            "epoch": [],
            "train_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
        }


        # Same initial model for each model type
        model = copy.deepcopy(
            base_model
        ).to(device)


        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.0,
        )


        # ====================================================
        # VANILLA
        # ====================================================

        if model_type == "vanilla":

            for epoch in range(num_epochs):

                model.train()

                total_loss = 0.0
                total_correct = 0
                total_samples = 0


                for xb, yb in train_loader:

                    xb = xb.to(device)
                    yb = yb.to(device)

                    xb = xb.view(
                        xb.size(0),
                        -1,
                    )


                    optimizer.zero_grad()


                    logits = model(xb)

                    loss = criterion(
                        logits,
                        yb,
                    )


                    loss.backward()

                    optimizer.step()


                    total_loss += (
                        loss.item()
                        * xb.size(0)
                    )

                    predictions = logits.argmax(
                        dim=1
                    )

                    total_correct += (
                        predictions == yb
                    ).sum().item()

                    total_samples += (
                        xb.size(0)
                    )


                train_loss = (
                    total_loss
                    / total_samples
                )

                train_accuracy = (
                    total_correct
                    / total_samples
                )


                test_loss, test_accuracy = (
                    evaluate_clean(
                        model,
                        test_loader,
                        device,
                    )
                )


                history["epoch"].append(
                    epoch + 1
                )

                history["train_loss"].append(
                    train_loss
                )

                history["test_loss"].append(
                    test_loss
                )

                history["train_accuracy"].append(
                    train_accuracy
                )

                history["test_accuracy"].append(
                    test_accuracy
                )


                if (
                    epoch + 1
                ) % 10 == 0:

                    print(
                        f"Epoch "
                        f"{epoch + 1:3d}/{num_epochs} | "
                        f"train loss = "
                        f"{train_loss:.6f} | "
                        f"train acc = "
                        f"{train_accuracy:.4f} | "
                        f"test acc = "
                        f"{test_accuracy:.4f}"
                    )


        # ====================================================
        # ADVERSARIAL TRAINING
        # ====================================================

        elif model_type == "adv":

            for epoch in range(num_epochs):

                model.train()

                total_adv_loss = 0.0

                total_clean_loss = 0.0

                total_correct = 0

                total_samples = 0


                for xb, yb in train_loader:

                    xb = xb.to(device)
                    yb = yb.to(device)

                    xb = xb.view(
                        xb.size(0),
                        -1,
                    )


                    # ------------------------------------------------
                    # Generate adversarial perturbation
                    # ------------------------------------------------

                    _, deltas = pgd_m_x(
                        model,
                        xb,
                        yb,
                        epsilon,
                        per_example_cross_entropy,
                        norm,
                        clipping,
                        num_iter,
                        alpha,
                    )


                    adv_input = (
                        xb + deltas
                    )



                    # ------------------------------------------------
                    # Train on adversarial examples
                    # ------------------------------------------------

                    optimizer.zero_grad()


                    adv_outputs = model(
                        adv_input
                    )

                    adv_loss = criterion(
                        adv_outputs,
                        yb,
                    )


                    adv_loss.backward()

                    optimizer.step()


                    # ------------------------------------------------
                    # Clean training statistics
                    # ------------------------------------------------

                    with torch.no_grad():

                        clean_outputs = model(
                            xb
                        )

                        clean_loss = criterion(
                            clean_outputs,
                            yb,
                        )

                        predictions = (
                            clean_outputs.argmax(
                                dim=1
                            )
                        )

                        total_correct += (
                            predictions == yb
                        ).sum().item()


                    total_adv_loss += (
                        adv_loss.item()
                        * xb.size(0)
                    )

                    total_clean_loss += (
                        clean_loss.item()
                        * xb.size(0)
                    )

                    total_samples += (
                        xb.size(0)
                    )


                train_adv_loss = (
                    total_adv_loss
                    / total_samples
                )

                train_clean_loss = (
                    total_clean_loss
                    / total_samples
                )

                train_accuracy = (
                    total_correct
                    / total_samples
                )


                test_loss, test_accuracy = (
                    evaluate_clean(
                        model,
                        test_loader,
                        device,
                    )
                )


                history["epoch"].append(
                    epoch + 1
                )

                history["train_loss"].append(
                    train_clean_loss
                )

                history["test_loss"].append(
                    test_loss
                )

                history["train_accuracy"].append(
                    train_accuracy
                )

                history["test_accuracy"].append(
                    test_accuracy
                )


                if (
                    epoch + 1
                ) % 10 == 0:

                    print(
                        f"Epoch "
                        f"{epoch + 1:3d}/{num_epochs} | "
                        f"adv loss = "
                        f"{train_adv_loss:.6f} | "
                        f"clean train acc = "
                        f"{train_accuracy:.4f} | "
                        f"test acc = "
                        f"{test_accuracy:.4f}"
                    )


        # ====================================================
        # MOC REGULARIZATION
        # ====================================================

        elif model_type == "mocreg":

            for epoch in range(num_epochs):

                model.train()

                total_clean_loss = 0.0

                total_moc_loss = 0.0

                total_samples = 0

                total_correct = 0


                for xb, yb in train_loader:

                    xb = xb.to(device)
                    yb = yb.to(device)

                    xb = xb.view(
                        xb.size(0),
                        -1,
                    )


                    optimizer.zero_grad()


                    # ------------------------------------------------
                    # Clean classification loss
                    # ------------------------------------------------

                    outputs = model(xb)

                    clean_loss = criterion(
                        outputs,
                        yb,
                    )


                    # ------------------------------------------------
                    # MOC regularizer
                    # ------------------------------------------------

                    moc_loss, omega_values, t_grid = (
                        moc_regularizer(
                            model=model,
                            X=xb,
                            dy_f=linf_distance,
                            norm=norm,
                            t_values=t_values,
                            clipping_f=clipping,
                            step_size=alpha,
                            num_iter=num_iter,
                            num_restarts=num_restarts,
                        )
                    )


                    # ------------------------------------------------
                    # Combined objective
                    # ------------------------------------------------

                    loss = (
                        clean_loss
                        + lambda_k * moc_loss
                    )


                    loss.backward()

                    optimizer.step()


                    # ------------------------------------------------
                    # Statistics
                    # ------------------------------------------------

                    with torch.no_grad():

                        predictions = (
                            outputs.argmax(
                                dim=1
                            )
                        )

                        total_correct += (
                            predictions == yb
                        ).sum().item()


                    total_clean_loss += (
                        clean_loss.item()
                        * xb.size(0)
                    )

                    total_moc_loss += (
                        moc_loss.item()
                        * xb.size(0)
                    )

                    total_samples += (
                        xb.size(0)
                    )


                train_clean_loss = (
                    total_clean_loss
                    / total_samples
                )

                train_moc_loss = (
                    total_moc_loss
                    / total_samples
                )

                train_accuracy = (
                    total_correct
                    / total_samples
                )


                test_loss, test_accuracy = (
                    evaluate_clean(
                        model,
                        test_loader,
                        device,
                    )
                )


                history["epoch"].append(
                    epoch + 1
                )

                history["train_loss"].append(
                    train_clean_loss
                )

                history["test_loss"].append(
                    test_loss
                )

                history["train_accuracy"].append(
                    train_accuracy
                )

                history["test_accuracy"].append(
                    test_accuracy
                )


                if (
                    epoch + 1
                ) % 10 == 0:

                    print(
                        f"Epoch "
                        f"{epoch + 1:3d}/{num_epochs} | "
                        f"CE = "
                        f"{train_clean_loss:.6f} | "
                        f"MOC = "
                        f"{train_moc_loss:.6f} | "
                        f"train acc = "
                        f"{train_accuracy:.4f} | "
                        f"test acc = "
                        f"{test_accuracy:.4f}"
                    )


        # ====================================================
        # SPECTRAL REGULARIZATION
        # ====================================================

        elif model_type == "spectral":

            # ------------------------------------------------
            # Initialize power iteration vectors
            # ------------------------------------------------

            v_vectors = {}

            for name, param in model.named_parameters():

                if (
                    "weight" in name
                    and param.dim() == 2
                ):

                    v = torch.randn(
                        param.size(1),
                        1,
                        device=param.device,
                    )

                    v = (
                        v
                        / (
                            v.norm()
                            + 1e-12
                        )
                    )

                    v_vectors[name] = v


            # ------------------------------------------------
            # Training
            # ------------------------------------------------

            for epoch in range(num_epochs):

                model.train()

                total_loss = 0.0

                total_samples = 0

                total_correct = 0


                for xb, yb in train_loader:

                    xb = xb.to(device)
                    yb = yb.to(device)

                    xb = xb.view(
                        xb.size(0),
                        -1,
                    )


                    optimizer.zero_grad()


                    outputs = model(xb)

                    loss = criterion(
                        outputs,
                        yb,
                    )


                    loss.backward()


                    # ------------------------------------------------
                    # Spectral norm regularization
                    # ------------------------------------------------

                    for name, param in model.named_parameters():

                        if (
                            "weight" not in name
                            or param.grad is None
                            or name not in v_vectors
                        ):
                            continue


                        W = param.data

                        v = v_vectors[name]


                        # Power iteration

                        u = torch.matmul(
                            W,
                            v,
                        )

                        u_norm = u.norm()

                        if u_norm > 0:

                            u = (
                                u
                                / u_norm
                            )


                        v = torch.matmul(
                            W.t(),
                            u,
                        )

                        v_norm = v.norm()

                        if v_norm > 0:

                            v = (
                                v
                                / v_norm
                            )


                        v_vectors[name] = (
                            v.detach()
                        )


                        # Approximate largest singular value

                        sigma = torch.matmul(
                            u.t(),
                            torch.matmul(
                                W,
                                v,
                            ),
                        )


                        # Gradient of spectral norm

                        spectral_grad = (
                            lambda_spectral
                            * sigma
                            * torch.matmul(
                                u,
                                v.t(),
                            )
                        )


                        param.grad.data.add_(
                            spectral_grad
                        )


                    optimizer.step()


                    # ------------------------------------------------
                    # Statistics
                    # ------------------------------------------------

                    total_loss += (
                        loss.item()
                        * xb.size(0)
                    )

                    predictions = (
                        outputs.argmax(
                            dim=1
                        )
                    )

                    total_correct += (
                        predictions == yb
                    ).sum().item()

                    total_samples += (
                        xb.size(0)
                    )


                train_loss = (
                    total_loss
                    / total_samples
                )

                train_accuracy = (
                    total_correct
                    / total_samples
                )


                test_loss, test_accuracy = (
                    evaluate_clean(
                        model,
                        test_loader,
                        device,
                    )
                )


                history["epoch"].append(
                    epoch + 1
                )

                history["train_loss"].append(
                    train_loss
                )

                history["test_loss"].append(
                    test_loss
                )

                history["train_accuracy"].append(
                    train_accuracy
                )

                history["test_accuracy"].append(
                    test_accuracy
                )


                if (
                    epoch + 1
                ) % 10 == 0:

                    print(
                        f"Epoch "
                        f"{epoch + 1:3d}/{num_epochs} | "
                        f"CE = "
                        f"{train_loss:.6f} | "
                        f"train acc = "
                        f"{train_accuracy:.4f} | "
                        f"test acc = "
                        f"{test_accuracy:.4f}"
                    )


        else:

            raise ValueError(
                f"Unknown model type: {model_type}"
            )


        # ========================================================
        # SAVE LEARNING CURVE
        # ========================================================

        all_learning_curves[
            model_type
        ].append(
            pd.DataFrame(history)
        )


        # ========================================================
        # FINAL CLEAN PERFORMANCE
        # ========================================================

        train_loss, train_accuracy = (
            evaluate_clean(
                model,
                train_loader,
                device,
            )
        )

        test_loss, test_accuracy = (
            evaluate_clean(
                model,
                test_loader,
                device,
            )
        )


        print()
        print(
            f"{model_type} clean accuracy: "
            f"{test_accuracy:.4f}"
        )


        all_accuracy[
            model_type
        ] = test_accuracy


        # ========================================================
        # GENERALIZATION GAP
        # ========================================================

        all_gen_gap[
            model_type
        ] = abs(
            train_accuracy
            - test_accuracy
        )


        # ========================================================
        # ADVERSARIAL EVALUATION
        # ========================================================

        all_adv_accuracy[
            model_type
        ] = {}


        model.eval()


        for budget in attacks_budget:

            print(
                f"Evaluating "
                f"{model_type} at "
                f"epsilon={budget:.4f}"
            )


            # --------------------------------------------
            # Generate adversarial examples
            # --------------------------------------------

            _, deltas = pgd_m_x(
                model,
                X_test_tensor,
                y_test_tensor,
                budget,
                per_example_cross_entropy,
                norm,
                clipping,
                num_iter,
                alpha,
            )


            adv_input = (
                X_test_tensor
                + deltas
            )


            # --------------------------------------------
            # Evaluate adversarial accuracy
            # --------------------------------------------

            with torch.no_grad():

                logits = model(
                    adv_input
                )

                predictions = (
                    logits.argmax(
                        dim=1
                    )
                )

                correct = (
                    predictions
                    == y_test_tensor
                ).sum().item()


                adv_accuracy = (
                    correct
                    / y_test_tensor.numel()
                )


            all_adv_accuracy[
                model_type
            ][budget] = adv_accuracy


            print(
                f"  adversarial accuracy = "
                f"{adv_accuracy:.4f}"
            )


        # ========================================================
        # COMPUTE MOC
        # ========================================================

        print(
            f"Computing MOC for "
            f"{model_type}..."
        )


        with torch.no_grad():

            model_outputs = model(
                X_test_tensor
            )


        moc, _ = pgd_moc(
            model,
            X_test_tensor,
            model_outputs,
            linf_distance,
            clipping_f=clipping,
            norm=norm,
            t_values=t_values,
            step_size=alpha,
            numiter=num_iter,
            numrestarts=num_restarts,
            batch_size=batch_size,
        )


        if isinstance(
            moc,
            torch.Tensor,
        ):

            moc = (
                moc
                .detach()
                .cpu()
                .numpy()
            )

        else:

            moc = np.asarray(moc)


        moc = moc.reshape(-1)


        all_mocs[
            model_type
        ] = moc


        all_experiment_mocs[
            model_type
        ].append(
            moc.copy()
        )


    # ========================================================
    # EXPERIMENT RESULTS
    # ========================================================

    results_data = []


    for model_type in model_types:

        # ----------------------------------------------------
        # Robustness AUC
        # ----------------------------------------------------
        #
        # compute_robustness_auc was written for the original
        # robustness dictionaries, so we use accuracy here.
        #

        auc = compute_robustness_auc(
            all_adv_accuracy[
                model_type
            ],
            attacks_budget,
        )


        row = {
            "Experiment": experiment_idx,
            "Model": model_type,

            "Clean Accuracy":
                all_accuracy[
                    model_type
                ],

            "Generalization Gap":
                all_gen_gap[
                    model_type
                ],

            "AUC":
                auc,
        }


        # ----------------------------------------------------
        # Accuracy at each attack budget
        # ----------------------------------------------------

        for budget in attacks_budget:

            row[
                f"{budget}"
            ] = all_adv_accuracy[
                model_type
            ][budget]


        results_data.append(
            row
        )


    df_results = pd.DataFrame(
        results_data
    )


    # ========================================================
    # CUMULATIVE ROBUSTNESS
    # ========================================================

    df_results[
        "Cumulative Robustness"
    ] = 0.0


    for budget in attacks_budget:

        col_name = f"{budget}"


        # Best accuracy at this epsilon
        best_accuracy = (
            df_results[
                col_name
            ].max()
        )


        # Larger difference = worse
        #
        # Since we want a robustness penalty,
        # models below the best model get penalized.
        #
        # A smaller value is therefore better.

        df_results[
            "Cumulative Robustness"
        ] += (
            best_accuracy
            - df_results[
                col_name
            ]
        )


    all_experiment_results.append(
        df_results
    )


# ============================================================
# COMBINE EXPERIMENTS
# ============================================================

df_all = pd.concat(
    all_experiment_results,
    ignore_index=True,
)


# ============================================================
# SUMMARY METRICS
# ============================================================

metrics = [
    "Clean Accuracy",
    "Generalization Gap",
    "Cumulative Robustness",
    "AUC",
]


for budget in attacks_budget:

    metrics.append(
        f"{budget}"
    )


df_summary = (
    df_all
    .groupby("Model")[metrics]
    .agg(
        [
            "mean",
            "std",
        ]
    )
)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

out_dir = (
    Path("experiments")
    / "mnist"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)


df_all.to_csv(
    out_dir
    / f"experiments_results{file_ext}.csv",
    index=False,
)


df_summary.to_csv(
    out_dir
    / f"summary{file_ext}.csv",
)


# ============================================================
# LATEX TABLE
# ============================================================

data = {}


for metric in [
    "Clean Accuracy",
    "Generalization Gap",
    "Cumulative Robustness",
    "AUC",
]:

    for model_type in (
        df_summary.index
    ):

        mean = df_summary.loc[
            model_type,
            (metric, "mean"),
        ]

        std = df_summary.loc[
            model_type,
            (metric, "std"),
        ]


        data[
            (
                metric,
                model_type,
            )
        ] = (
            rf"{mean:.4f} "
            rf"$\pm$ "
            rf"{std:.4f}"
        )


df_latex = pd.DataFrame(
    [data],
    index=[
        f"mnist{file_ext}"
    ],
)


df_latex.columns = (
    pd.MultiIndex.from_tuples(
        df_latex.columns,
        names=[
            "Metric",
            "Model type",
        ],
    )
)


df_transposed = (
    df_latex.T
)


df_transposed.columns = [
    "mnist"
    + file_ext
]


df_transposed.index.names = [
    "Metric",
    "Model type",
]


df_transposed.to_pickle(
    folder_path
    / f"df_transposed{file_ext}.pkl"
)


# ============================================================
# ADVERSARIAL CURVES
# ============================================================

adv_summary = {}


for model_type in model_types:

    adv_mean = np.array(
        [
            df_summary.loc[
                model_type,
                (f"{budget}", "mean"),
            ]
            for budget in attacks_budget
        ]
    )


    adv_std = np.array(
        [
            df_summary.loc[
                model_type,
                (f"{budget}", "std"),
            ]
            for budget in attacks_budget
        ]
    )


    adv_summary[
        model_type
    ] = {
        "mean": adv_mean,
        "std": adv_std,
    }


    # --------------------------------------------
    # CSV for PGFPlots
    # --------------------------------------------

    df_adv = pd.DataFrame(
        {
            "epsilon":
                attacks_budget,

            "mean":
                adv_mean,

            "std":
                adv_std,
        }
    )


    df_adv.to_csv(
        out_dir
        / (
            f"adv_{model_type}"
            f"{file_ext}.csv"
        ),
        index=False,
    )


# ============================================================
# MOC SUMMARY
# ============================================================

moc_summary = {}


for model_type in model_types:

    moc_array = np.stack(
        all_experiment_mocs[
            model_type
        ],
        axis=0,
    )


    moc_mean = (
        moc_array.mean(
            axis=0
        )
    )


    if moc_array.shape[0] > 1:

        moc_std = (
            moc_array.std(
                axis=0,
                ddof=1,
            )
        )

    else:

        moc_std = np.zeros_like(
            moc_mean
        )


    moc_summary[
        model_type
    ] = {
        "mean": moc_mean,
        "std": moc_std,
        "values": moc_array,
    }


    # --------------------------------------------
    # CSV for PGFPlots
    # --------------------------------------------

    df_moc = pd.DataFrame(
        {
            "t": t_values,

            "mean": moc_mean,

            "std": moc_std,
        }
    )


    df_moc.to_csv(
        out_dir
        / (
            f"moc_{model_type}"
            f"{file_ext}.csv"
        ),
        index=False,
    )


# ============================================================
# LEARNING CURVES
# ============================================================

for model_type in model_types:

    train_loss_array = np.stack(
        [
            df[
                "train_loss"
            ].to_numpy()

            for df
            in all_learning_curves[
                model_type
            ]
        ],
        axis=0,
    )


    test_loss_array = np.stack(
        [
            df[
                "test_loss"
            ].to_numpy()

            for df
            in all_learning_curves[
                model_type
            ]
        ],
        axis=0,
    )


    train_acc_array = np.stack(
        [
            df[
                "train_accuracy"
            ].to_numpy()

            for df
            in all_learning_curves[
                model_type
            ]
        ],
        axis=0,
    )


    test_acc_array = np.stack(
        [
            df[
                "test_accuracy"
            ].to_numpy()

            for df
            in all_learning_curves[
                model_type
            ]
        ],
        axis=0,
    )


    if num_experiments > 1:

        train_loss_std = (
            train_loss_array.std(
                axis=0,
                ddof=1,
            )
        )

        test_loss_std = (
            test_loss_array.std(
                axis=0,
                ddof=1,
            )
        )

        train_acc_std = (
            train_acc_array.std(
                axis=0,
                ddof=1,
            )
        )

        test_acc_std = (
            test_acc_array.std(
                axis=0,
                ddof=1,
            )
        )

    else:

        train_loss_std = (
            np.zeros(
                train_loss_array.shape[1]
            )
        )

        test_loss_std = (
            np.zeros(
                test_loss_array.shape[1]
            )
        )

        train_acc_std = (
            np.zeros(
                train_acc_array.shape[1]
            )
        )

        test_acc_std = (
            np.zeros(
                test_acc_array.shape[1]
            )
        )


    df_curve = pd.DataFrame(
        {
            "epoch":
                all_learning_curves[
                    model_type
                ][0]["epoch"],

            "train_loss_mean":
                train_loss_array.mean(
                    axis=0
                ),

            "train_loss_std":
                train_loss_std,

            "test_loss_mean":
                test_loss_array.mean(
                    axis=0
                ),

            "test_loss_std":
                test_loss_std,

            "train_accuracy_mean":
                train_acc_array.mean(
                    axis=0
                ),

            "train_accuracy_std":
                train_acc_std,

            "test_accuracy_mean":
                test_acc_array.mean(
                    axis=0
                ),

            "test_accuracy_std":
                test_acc_std,
        }
    )


    df_curve.to_csv(
        out_dir
        / (
            f"learning_curve_{model_type}"
            f"{file_ext}.csv"
        ),
        index=False,
    )


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print()

for model_type in model_types:

    mean_clean = df_summary.loc[
        model_type,
        (
            "Clean Accuracy",
            "mean",
        ),
    ]

    std_clean = df_summary.loc[
        model_type,
        (
            "Clean Accuracy",
            "std",
        ),
    ]

    mean_auc = df_summary.loc[
        model_type,
        (
            "AUC",
            "mean",
        ),
    ]

    std_auc = df_summary.loc[
        model_type,
        (
            "AUC",
            "std",
        ),
    ]

    print(
        f"{model_type:10s} | "
        f"clean acc = "
        f"{mean_clean:.4f} "
        f"+/- {std_clean:.4f} | "
        f"AUC = "
        f"{mean_auc:.4f} "
        f"+/- {std_auc:.4f}"
    )


print()
print(
    "Results saved to:",
    out_dir,
)