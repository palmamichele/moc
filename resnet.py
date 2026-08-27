from pathlib import Path
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

from utils import save_moc


# ============================================================
# FMCA
# ============================================================

sys.path.append(str(Path("fmca") / "build" / "py"))
import FMCA


# ============================================================
# Reproducibility
# ============================================================

SEED = 0

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ============================================================
# Configuration
# ============================================================

# ------------------------------------------------------------
# CIFAR-10 training
# ------------------------------------------------------------

TRAIN_BATCH_SIZE = 256
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4

# Set this to True if you want to load an existing trained
# model instead of training from scratch.
LOAD_CHECKPOINT = False

CHECKPOINT_PATH = (
    Path("experiments")
    / "cifar10_resnet18"
    / "resnet18_cifar10.pth"
)


# ------------------------------------------------------------
# MOC / FMCA
# ------------------------------------------------------------

BATCH_SIZES = [10, 100, 1000]

qX = 0.0001
TX = 1000
nbins = 10000

norms = ["EUCLIDEAN"]


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

save_path = Path("experiments") / "cifar10_resnet18"
save_path.mkdir(parents=True, exist_ok=True)


# ============================================================
# CIFAR-10 transforms
# ============================================================
#
# We use ImageNet preprocessing because the pretrained
# ResNet-18 weights were trained on ImageNet.
#
# The official ResNet-18 preprocessing:
#
#   resize -> 256
#   center crop -> 224
#   normalize with ImageNet mean/std
#
# For training we additionally use standard augmentation.
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


# ============================================================
# CIFAR-10 datasets
# ============================================================

print("\nLoading CIFAR-10...")

train_ds = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)

test_ds = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform,
)

print(f"Training samples: {len(train_ds)}")
print(f"Test samples:     {len(test_ds)}")


# ============================================================
# Training DataLoader
# ============================================================

train_loader = DataLoader(
    train_ds,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
)


test_loader_training = DataLoader(
    test_ds,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
)


# ============================================================
# Create pretrained ResNet-18
# ============================================================
#
# This loads ImageNet-pretrained weights and then replaces
# the 1000-class ImageNet classifier with a 10-class CIFAR-10
# classifier.
# ============================================================

print("\nCreating ImageNet-pretrained ResNet-18...")

weights = ResNet18_Weights.DEFAULT

tr_model = resnet18(weights=weights)

# Original:
#
#     tr_model.fc -> 512 -> 1000
#
# Replace with:
#
#     tr_model.fc -> 512 -> 10

tr_model.fc = nn.Linear(
    tr_model.fc.in_features,
    10
)

tr_model = tr_model.to(device)


# ============================================================
# Create randomly initialized ResNet-18
# ============================================================
#
# Same architecture as the trained model, but NO ImageNet
# pretrained weights.
# ============================================================

print("Creating randomly initialized ResNet-18...")

un_model = resnet18(weights=None)

un_model.fc = nn.Linear(
    un_model.fc.in_features,
    10
)

un_model = un_model.to(device)


# ============================================================
# Print model information
# ============================================================

print("\nModels:")
print(f"Trained model output dimension:   {tr_model.fc.out_features}")
print(f"Untrained model output dimension: {un_model.fc.out_features}")


# ============================================================
# Train ResNet-18 on CIFAR-10
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    tr_model.parameters(),
    lr=LEARNING_RATE,
)


def evaluate_model(model, loader):
    """
    Evaluate classification accuracy.
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total

    return accuracy


# ============================================================
# Train
# ============================================================

if LOAD_CHECKPOINT:

    print(
        f"\nLoading trained model from:\n"
        f"{CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    tr_model.load_state_dict(checkpoint)

else:

    print("\nStarting CIFAR-10 fine-tuning...")
    print(f"Epochs:       {NUM_EPOCHS}")
    print(f"Batch size:   {TRAIN_BATCH_SIZE}")
    print(f"Learning rate:{LEARNING_RATE}")

    for epoch in range(NUM_EPOCHS):

        start_epoch = time.time()

        tr_model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = tr_model(imgs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predicted = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        train_accuracy = (
            100.0 * correct / total
        )

        test_accuracy = evaluate_model(
            tr_model,
            test_loader_training
        )

        elapsed = time.time() - start_epoch

        avg_loss = (
            running_loss /
            len(train_loader)
        )

        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Test Acc: {test_accuracy:.2f}% "
            f"Time: {elapsed:.1f}s"
        )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    torch.save(
        tr_model.state_dict(),
        CHECKPOINT_PATH
    )

    print(
        f"\nSaved trained model to:\n"
        f"{CHECKPOINT_PATH}"
    )


# ============================================================
# Evaluation mode
# ============================================================

tr_model.eval()
un_model.eval()


# ============================================================
# Sanity check
# ============================================================

print("\nChecking model outputs...")

with torch.no_grad():

    imgs, labels = next(iter(test_loader_training))

    imgs = imgs.to(device)

    trained_output = tr_model(imgs)
    untrained_output = un_model(imgs)

print(
    f"Input shape:             {tuple(imgs.shape)}"
)

print(
    f"Trained output shape:    "
    f"{tuple(trained_output.shape)}"
)

print(
    f"Untrained output shape:  "
    f"{tuple(untrained_output.shape)}"
)

assert trained_output.shape[1] == 10
assert untrained_output.shape[1] == 10

print("Sanity check passed.")


# ============================================================
# MOC computation
# ============================================================

def minibatchmoc(
    loader,
    tr_model,
    un_model,
    TX,
    qX,
    nbins,
    norm,
    split_name,
    C,
):

    start = time.time()

    print("\n")
    print("=" * 70)
    print(
        f"MOC computation: "
        f"split={split_name}, "
        f"batch_size={C}, "
        f"norm={norm[0]}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Determine grid.
    #
    # The grid is shared between all batches.
    # --------------------------------------------------------

    dmoc = FMCA.DiscreteModulusOfContinuity()

    dmoc.init(
        np.empty((1, 1), dtype=np.float64),
        np.empty((1, 1), dtype=np.float64),
        TX,
        qX,
        nbins,
        norm,
        norm,
    )

    t_values = dmoc.tgrid()

    NT = len(t_values)

    save_moc(
        t_values,
        save_path,
        f"batchdeltas_dmoc_{C}_{norm[0]}"
    )

    # --------------------------------------------------------
    # Final MOCs
    # --------------------------------------------------------

    final_moc_tr = np.zeros(NT)
    final_moc_un = np.zeros(NT)
    final_moc_data = np.zeros(NT)

    # --------------------------------------------------------
    # Iterate over batches
    # --------------------------------------------------------

    for j, (imgs, labels) in enumerate(
        loader,
        start=1
    ):

        print(
            f"batch i={j}/{len(loader)}"
        )

        imgs = imgs.to(device)

        # ----------------------------------------------------
        # Model outputs
        # ----------------------------------------------------

        with torch.no_grad():

            F_trained = torch.softmax(
                tr_model(imgs),
                dim=1
            )

            F_untrained = torch.softmax(
                un_model(imgs),
                dim=1
            )

        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        F_trained = (
            F_trained
            .cpu()
            .numpy()
        )

        F_untrained = (
            F_untrained
            .cpu()
            .numpy()
        )

        labels_np = (
            labels
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Input points P
        #
        # FMCA expects points as columns.
        #
        # Before:
        #
        #     imgs = (B, C, H, W)
        #
        # After flatten:
        #
        #     P = (B, C*H*W)
        #
        # Transpose:
        #
        #     P = (C*H*W, B)
        # ----------------------------------------------------

        P = (
            imgs
            .view(imgs.size(0), -1)
            .cpu()
            .numpy()
        )

        P = np.ascontiguousarray(
            P.T,
            dtype=np.float64
        )

        # ----------------------------------------------------
        # Trained model output
        #
        # (B, 10)
        #     ↓ transpose
        # (10, B)
        # ----------------------------------------------------

        F_trained = np.ascontiguousarray(
            F_trained.T,
            dtype=np.float64
        )

        # ----------------------------------------------------
        # Untrained model output
        # ----------------------------------------------------

        F_untrained = np.ascontiguousarray(
            F_untrained.T,
            dtype=np.float64
        )

        # ----------------------------------------------------
        # CIFAR-10 data function
        #
        # y ∈ {0,...,9}
        #
        # Convert each label to a one-hot vector:
        #
        # class 3 -> [0,0,0,1,0,...,0]
        #
        # Result:
        #
        # F_data = (10, B)
        # ----------------------------------------------------

        num_classes = 10

        F_data = np.eye(
            num_classes,
            dtype=np.float64
        )[labels_np]

        F_data = np.ascontiguousarray(
            F_data.T,
            dtype=np.float64
        )

        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        assert P.shape[1] == imgs.size(0)

        assert (
            F_trained.shape[0] == 10
        )

        assert (
            F_untrained.shape[0] == 10
        )

        assert (
            F_data.shape[0] == 10
        )

        assert (
            P.shape[1]
            == F_trained.shape[1]
            == F_untrained.shape[1]
            == F_data.shape[1]
        )

        # ----------------------------------------------------
        # Trained model MOC
        # ----------------------------------------------------

        dmoc_tr = (
            FMCA.DiscreteModulusOfContinuity()
        )

        dmoc_tr.init(
            P,
            F_trained,
            TX,
            qX,
            nbins,
            norm,
            norm
        )

        batch_tr = dmoc_tr.omegat()

        # ----------------------------------------------------
        # Untrained model MOC
        # ----------------------------------------------------

        dmoc_un = (
            FMCA.DiscreteModulusOfContinuity()
        )

        dmoc_un.init(
            P,
            F_untrained,
            TX,
            qX,
            nbins,
            norm,
            norm
        )

        batch_un = dmoc_un.omegat()

        # ----------------------------------------------------
        # Data / label MOC
        # ----------------------------------------------------

        dmoc_data = (
            FMCA.DiscreteModulusOfContinuity()
        )

        dmoc_data.init(
            P,
            F_data,
            TX,
            qX,
            nbins,
            norm,
            norm
        )

        batch_data = dmoc_data.omegat()

        # ----------------------------------------------------
        # Union over batches
        #
        # We want the maximum MOC over all batches.
        # ----------------------------------------------------

        final_moc_tr = np.maximum(
            final_moc_tr,
            batch_tr
        )

        final_moc_un = np.maximum(
            final_moc_un,
            batch_un
        )

        final_moc_data = np.maximum(
            final_moc_data,
            batch_data
        )

    # ========================================================
    # Save results
    # ========================================================

    elapsed = time.time() - start

    save_moc(
        final_moc_tr,
        save_path,
        f"trained_batch{split_name}_dmoc_{C}_{norm[0]}"
    )

    save_moc(
        final_moc_un,
        save_path,
        f"untrained_batch{split_name}_dmoc_{C}_{norm[0]}"
    )

    save_moc(
        final_moc_data,
        save_path,
        f"data_batch{split_name}_dmoc_{C}_{norm[0]}"
    )

    print(
        f"MOC computation took {elapsed:.2f} seconds"
    )


# ============================================================
# MOC experiment
# ============================================================

print("\n")
print("=" * 70)
print("STARTING MOC EXPERIMENT")
print("=" * 70)


for C in BATCH_SIZES:

    print("\n")
    print("#" * 70)
    print(f"BATCH SIZE = {C}")
    print("#" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # shuffle=False here.
    #
    # This matches your original experiment and means that
    # the maximum over batches can be used as the union MOC.
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=C,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=C,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    for norm in norms:

        # ====================================================
        # TEST
        # ====================================================

        minibatchmoc(
            test_loader,
            tr_model,
            un_model,
            TX,
            qX,
            nbins,
            norm,
            "test",
            C,
        )

        # ====================================================
        # TRAIN
        # ====================================================

        minibatchmoc(
            train_loader,
            tr_model,
            un_model,
            TX,
            qX,
            nbins,
            norm,
            "train",
            C,
        )

        # ====================================================
        # Union train + test
        # ====================================================

        for model_type in [
            "trained",
            "untrained",
            "data"
        ]:

            train_file = (
                save_path
                / f"{model_type}_batchtrain_dmoc_"
                  f"{C}_{norm[0]}.csv"
            )

            test_file = (
                save_path
                / f"{model_type}_batchtest_dmoc_"
                  f"{C}_{norm[0]}.csv"
            )

            train_moc = np.loadtxt(
                train_file,
                delimiter=","
            )

            test_moc = np.loadtxt(
                test_file,
                delimiter=","
            )

            # Maximum over train/test batches
            union_moc = np.maximum(
                test_moc,
                train_moc
            )

            save_moc(
                union_moc,
                save_path,
                f"{model_type}_batchunion_dmoc_"
                f"{C}_{norm[0]}"
            )

            print(
                f"Saved union MOC: "
                f"{model_type}, C={C}, norm={norm[0]}"
            )


# ============================================================
# Finished
# ============================================================

print("\n")
print("=" * 70)
print("ALL EXPERIMENTS FINISHED")
print("=" * 70)

print(
    f"Results saved in:\n{save_path}"
)