import numpy as np
from pathlib import Path


def max_intra_class_distance(X, labels, norm="L2"):
    """
    Computes exact max pairwise intra-class distance.

    X: (N, d)
    labels: (N,)
    norm: "L2" or "L1"
    """

    classes = np.unique(labels)
    max_dists = {}

    for c in classes:
        Xc = X[labels == c]
        n = Xc.shape[0]

        if n < 2:
            max_dists[c] = 0.0
            continue

        if norm == "L2":
            # squared distance trick for efficiency
            norms = np.sum(Xc ** 2, axis=1, keepdims=True)
            dist_sq = norms + norms.T - 2 * (Xc @ Xc.T)
            dist_sq = np.maximum(dist_sq, 0.0)
            dist_matrix = np.sqrt(dist_sq)

        elif norm == "L1":
            # Manhattan distance (no shortcut; exact)
            diff = Xc[:, None, :] - Xc[None, :, :]
            dist_matrix = np.sum(np.abs(diff), axis=2)

        else:
            raise ValueError("norm must be 'L2' or 'L1'")

        max_dists[c] = np.max(dist_matrix)

    return max_dists



def onehot_to_labels(Y):
    return np.argmax(Y, axis=1)



def load_split(base_path, split):
    X = np.loadtxt(base_path / f"X_{split}.csv", delimiter=",", ndmin=2)
    Y = np.loadtxt(base_path / f"Y_{split}.csv", delimiter=",", ndmin=2)

    # fix orientation if needed
    if X.shape[0] < X.shape[1]:
        X = X.T
    if Y.shape[0] < Y.shape[1]:
        Y = Y.T

    return X, Y



def main():

    base_path = Path("data/MNIST")

    norms = ["L2", "L1"]
    splits = ["train", "test", "union"]

    results = {}

    # Load once
    X_train, Y_train = load_split(base_path, "train")
    X_test, Y_test = load_split(base_path, "test")

    X_union = np.vstack([X_train, X_test])
    Y_union = np.vstack([Y_train, Y_test])

    datasets = {
        "train": (X_train, Y_train),
        "test": (X_test, Y_test),
        "union": (X_union, Y_union),
    }

    for norm in norms:
        results[norm] = {}

        for split in splits:

            X, Y = datasets[split]
            labels = onehot_to_labels(Y)

            print(f"\nProcessing: {split} | norm: {norm} | N={X.shape[0]}")

            max_dists = max_intra_class_distance(X, labels, norm=norm)

            results[norm][split] = max_dists

            # print per class
            print("Class-wise max intra-class distances:")
            for c in sorted(max_dists.keys()):
                print(f"  Class {c}: {max_dists[c]:.6f}")

    # ---------------------------------------------------
    # Summary comparison
    # ---------------------------------------------------
    print("\n================ SUMMARY ================")

    for norm in norms:
        for split in splits:
            vals = np.array(list(results[norm][split].values()))
            print(f"{norm} | {split}: mean={vals.mean():.6f}, max={vals.max():.6f}, min={vals.min():.6f}")


# ---------------------------------------------------
if __name__ == "__main__":
    main()
