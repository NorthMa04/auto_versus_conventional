import numpy as np
import time
from pathlib import Path
import argparse

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# =========================
# I/O: load matrix (MatrixMarket or dense txt)
# =========================
def load_matrix(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()

    if first.startswith("%%MatrixMarket"):
        from scipy.io import mmread
        A = mmread(path)
        A = A.toarray() if not isinstance(A, np.ndarray) else A
        return A.astype(float)
    else:
        return np.loadtxt(path).astype(float)


# =========================
# Normalization
# =========================
def minmax_01(M, eps=1e-12):
    mn, mx = M.min(), M.max()
    if abs(mx - mn) < eps:
        return np.zeros_like(M)
    return (M - mn) / (mx - mn)


# =========================
# Paper: X similarity (no gating, no learning)
# Sim(u,v) = alpha * W_uv + beta * sum_{m in N(u)∩N(v)} (W_{u m} + W_{m v})
# =========================
def compute_X(W, A, alpha=0.5, beta=0.5):
    n = W.shape[0]
    X = np.zeros((n, n), dtype=float)
    neighbors = [set(np.where(A[i] > 0)[0]) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            common = neighbors[i] & neighbors[j]
            W1 = sum(W[i, m] + W[m, j] for m in common)
            x = alpha * W[i, j] + beta * W1
            X[i, j] = X[j, i] = x

    return X


# =========================
# Modularity score (weighted, undirected)
# =========================
def modularity_score(W, labels):
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    m = W.sum() / 2.0
    if m <= 0:
        return 0.0

    deg = W.sum(axis=1)
    Qv = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Qv += W[i, j] - deg[i] * deg[j] / (2.0 * m)
    return Qv / (2.0 * m)


# =========================
# Main: X -> KMeans (no learning)
# Output spec compatible with your pipeline
# =========================
def main():
    parser = argparse.ArgumentParser("X-direct KMeans (no learning), fixed output")
    parser.add_argument("-i", "--input", default="football.txt", help="Adjacency matrix (MatrixMarket or dense txt)")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--kmax", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not np.isclose(args.alpha + args.beta, 1.0, atol=1e-8):
        raise ValueError(f"alpha+beta must be 1. Got {args.alpha + args.beta}")

    dataset_name = Path(args.input).stem
    out_dir = Path(f"fixed_paper_{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ---- load + standardize W ----
    W = load_matrix(Path(args.input))
    if not (W.ndim == 2 and W.shape[0] == W.shape[1]):
        raise ValueError(f"Input must be NxN adjacency matrix, got {W.shape}")

    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)

    n = W.shape[0]
    A = (W > 0).astype(int)

    # ---- build X ----
    X = compute_X(W, A, alpha=args.alpha, beta=args.beta)
    X = minmax_01(X)

    # ---- samples for clustering: each node i uses column X[:, i] (consistent with your WCD code)
    H = X.T  # [n, n]

    # ---- KMeans scan ----
    best_mod, best_k, best_labels = -1.0, None, None
    k_max = min(args.kmax, n - 1)

    for k in range(2, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=20, random_state=args.seed).fit_predict(H)
        Qv = modularity_score(W, labels)
        if Qv > best_mod:
            best_mod, best_k, best_labels = Qv, k, labels

    t1 = time.perf_counter()

    # ---- save outputs (same spec style) ----
    np.savetxt(out_dir / "labels.txt", best_labels, fmt="%d")

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset {dataset_name}\n")
        f.write("method x_direct_kmeans\n")
        f.write(f"n {n}\n")
        f.write(f"seed {args.seed}\n")
        f.write(f"alpha {args.alpha}\n")
        f.write(f"beta {args.beta}\n")
        f.write("T 0\n")
        f.write(f"h {n}\n")
        f.write("epochs_per_layer 0\n")
        f.write(f"dims_trace {n}->{n}\n")
        f.write(f"best_k {best_k}\n")
        f.write(f"modularity {best_mod:.6f}\n")
        f.write(f"time_seconds {t1 - t0:.6f}\n")
        f.write("X_normalization minmax_01\n")
        f.write("clustering_input X_columns_as_samples\n")

    # ---- visualization (PCA on H) ----
    H2 = PCA(n_components=2, random_state=args.seed).fit_transform(H)
    plt.figure(figsize=(7, 6))
    plt.scatter(H2[:, 0], H2[:, 1], c=best_labels, cmap="tab10", s=60)
    plt.title(f"X-direct KMeans ({dataset_name}) | Q={best_mod:.3f} | k={best_k}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "community.png", dpi=300)
    plt.close()

    print(f"✅ X-direct KMeans done: dataset={dataset_name}, Q={best_mod:.4f}, k={best_k}, time={t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
