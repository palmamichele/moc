
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torch.utils.data import ConcatDataset


# ============================================================
# REPRODUCIBILITY
# ============================================================

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE AND PATHS
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

base_path = Path("data")
save_path = base_path / "mnist_with_clip"
save_path.mkdir(parents=True, exist_ok=True)

output_path = (
    Path("experiments")
    / "mnist"
    / "mnist_distance_distribution"
)

output_path.mkdir(parents=True, exist_ok=True)


# ============================================================
# MNIST DATASETS — ORIGINAL TRANSFORM
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

full_dataset = ConcatDataset([
    train_dataset,
    test_dataset,
])


# ============================================================
# PREPARE MNIST POINTS
# ============================================================

X_mnist = torch.stack([
    full_dataset[i][0].flatten()
    for i in range(len(full_dataset))
])

n, mnist_dim = X_mnist.shape
uniform_dim = 10


print("Number of MNIST points:", n)
print("Number of unordered pairs:", n * (n - 1) // 2)


# ============================================================
# GENERATE UNIFORM RANDOM POINTS
# ============================================================

generator = torch.Generator(device="cpu")
generator.manual_seed(seed)

X_uniform = torch.rand(
    n,
    uniform_dim,
    generator=generator,
)

print(f"Generated {n} uniform points in [0, 1]^{uniform_dim}.")



# ============================================================
# FULL PAIRWISE L2 DISTANCE MATRIX — NO BATCHING
# ============================================================
def pairwise_distance_distribution(
    X,
    device,
    bin_edges,
):
    """
    Compute the full pairwise L2 distance matrix.
    Use shared bin edges for comparing datasets.
    """

    X = X.to(device)
    n, dim = X.shape

    with torch.no_grad():
        D = torch.cdist(X, X, p=2)

    print(f"Input shape: {n} points in {dim}D")
    print(f"Full distance matrix shape: {tuple(D.shape)}")

    idx = torch.triu_indices(
        n, n, offset=1, device=device
    )

    distances = D[idx[0], idx[1]].cpu().numpy()

    counts, _ = np.histogram(distances, bins=bin_edges)

    total_pairs = n * (n - 1) // 2

    if counts.sum() != total_pairs:
        raise ValueError(
            f"Histogram captured {counts.sum():,} of "
            f"{total_pairs:,} pairs. Expand bin edges."
        )

    proportions = counts / total_pairs
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    del D, idx, distances, X

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return bin_centers, proportions, counts

# ============================================================
# COMPUTE MNIST DISTRIBUTION
# ============================================================

n_bins = 200

max_distance = np.sqrt(784)

bin_edges = np.linspace(
    0.0,
    max_distance,
    n_bins + 1,
)

print("\nComputing MNIST distances without batching...")

(
    mnist_centers,
    mnist_prob,
    mnist_counts,
) = pairwise_distance_distribution(
    X_mnist,
    device=device,
    bin_edges=bin_edges,
)

np.savez_compressed(
    output_path / "mnist_histogram.npz",
    bin_centers=mnist_centers,
    proportions=mnist_prob,
    counts=mnist_counts,
    bin_edges=bin_edges,
)

del X_mnist


# ============================================================
# COMPUTE UNIFORM DISTRIBUTION
# ============================================================

print("\nComputing uniform distances without batching...")

(
    uniform_centers,
    uniform_prob,
    uniform_counts,
) = pairwise_distance_distribution(
    X_uniform,
    device=device,
    bin_edges=bin_edges,
)
np.savez_compressed(
    output_path / "uniform_histogram.npz",
    bin_centers=uniform_centers,
    proportions=uniform_prob,
    counts=uniform_counts,
    bin_edges=bin_edges,
)

del X_uniform


# ============================================================
# PLOT DISTRIBUTIONS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    mnist_centers,
    mnist_prob,
    label="MNIST",
    linewidth=2,
)

plt.plot(
    uniform_centers,
    uniform_prob,
    label="Uniform random [0, 1]^784",
    linewidth=2,
)

plt.xlabel("Pairwise L2 distance")
plt.ylabel("Proportion of pairs per bin")

plt.title(
    "Pairwise L2 Distance Distributions: MNIST vs Uniform"
)

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    output_path / "mnist_vs_uniform.png",
    dpi=200,
)

plt.show()


# ============================================================
# SUMMARY
# ============================================================

mnist_total = int(mnist_counts.sum())
uniform_total = int(uniform_counts.sum())

mnist_mean_approx = np.average(
    mnist_centers,
    weights=mnist_counts,
)

uniform_mean_approx = np.average(
    uniform_centers,
    weights=uniform_counts,
)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print("Number of points:", n)
print("Dimension:", dim)

print("MNIST pairs:", f"{mnist_total:,}")
print("Uniform pairs:", f"{uniform_total:,}")

print(
    "MNIST mean distance (histogram approximation):",
    f"{mnist_mean_approx:.4f}",
)

print(
    "Uniform mean distance (histogram approximation):",
    f"{uniform_mean_approx:.4f}",
)

print("\nSaved results to:", output_path)


