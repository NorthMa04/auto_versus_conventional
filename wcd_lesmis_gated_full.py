import numpy as np
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# =========================
# Reproducibility
# =========================
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# I/O
# =========================
def load_matrix(path: Path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()

    if first.startswith("%%MatrixMarket"):
        from scipy.io import mmread
        A = mmread(path)
        return A.toarray().astype(float)
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


def scale_signed(M, eps=1e-12):
    s = np.max(np.abs(M))
    return M / (s + eps)


# =========================
# Algorithm 1: X (gated), Z, Q
# =========================
def compute_X_gated(W, A, alpha=0.5, beta=0.5, gate_common_min=2):
    """
    Lesmis special optimization: gate the 2-hop/common-neighbor term.

    If |N(i) ∩ N(j)| >= gate_common_min:
        X_ij = alpha * W_ij + beta * sum_{m in common}(W_im + W_mj)
    else:
        X_ij = alpha * W_ij
    """
    n = W.shape[0]
    X = np.zeros((n, n), dtype=float)
    neighbors = [set(np.where(A[i] > 0)[0]) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            common = neighbors[i] & neighbors[j]
            if len(common) >= gate_common_min:
                W1 = sum(W[i, m] + W[m, j] for m in common)
                x = alpha * W[i, j] + beta * W1
            else:
                x = alpha * W[i, j]
            X[i, j] = X[j, i] = x

    return X


def compute_Z(A):
    B = A @ A
    Z = 0.5 * A + B
    np.fill_diagonal(Z, 0)
    return Z


def compute_modularity_matrix(W):
    k = W.sum(axis=1)
    twoW = W.sum()
    Q = W - np.outer(k, k) / twoW
    np.fill_diagonal(Q, 0)
    return Q


# =========================
# Modularity score
# =========================
def modularity_score(W, labels):
    n = W.shape[0]
    m = W.sum() / 2.0
    deg = W.sum(axis=1)

    Qv = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Qv += W[i, j] - deg[i] * deg[j] / (2.0 * m)
    return Qv / (2.0 * m)


# =========================
# Sparse Autoencoders
# =========================
class SparseAE_Sigmoid(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.enc = nn.Linear(d_in, d_h)
        self.dec = nn.Linear(d_h, d_in)
        self.act = nn.Sigmoid()

    def forward(self, x):
        h = self.act(self.enc(x))
        x_hat = self.act(self.dec(h))
        return x_hat, h


class SparseAE_Tanh(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.enc = nn.Linear(d_in, d_h)
        self.dec = nn.Linear(d_h, d_in)
        self.act_h = nn.Sigmoid()
        self.act_out = nn.Tanh()

    def forward(self, x):
        h = self.act_h(self.enc(x))
        x_hat = self.act_out(self.dec(h))
        return x_hat, h


def kl_div(rho, rho_hat):
    eps = 1e-8
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))


# =========================
# Algorithm 2: one layer
# =========================
def train_one_layer(X, Q, Z, d_h, epochs=400, batch_size=32, lr=1e-3, rho=0.05):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_in, _ = X.shape
    Xc = torch.tensor(X.T, dtype=torch.float32, device=device)
    Qc = torch.tensor(Q.T, dtype=torch.float32, device=device)
    Zc = torch.tensor(Z.T, dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(Xc, Qc, Zc), batch_size=batch_size, shuffle=True)

    ae_x = SparseAE_Sigmoid(d_in, d_h).to(device)
    ae_q = SparseAE_Tanh(d_in, d_h).to(device)
    ae_z = SparseAE_Sigmoid(d_in, d_h).to(device)

    opt = torch.optim.Adam(
        list(ae_x.parameters()) + list(ae_q.parameters()) + list(ae_z.parameters()),
        lr=lr
    )
    mse = nn.MSELoss()

    for _ in range(epochs):
        for xb, qb, zb in loader:
            xh, hx = ae_x(xb)
            qh, hq = ae_q(qb)
            zh, hz = ae_z(zb)

            loss = (
                mse(xh, xb) +
                mse(qh, qb) +
                mse(zh, zb) +
                kl_div(rho, hx.mean(0)).sum() +
                kl_div(rho, hq.mean(0)).sum() +
                kl_div(rho, hz.mean(0)).sum()
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        _, HX = ae_x(Xc)
        _, HQ = ae_q(Qc)
        _, HZ = ae_z(Zc)

    return HX.T.cpu().numpy(), HQ.T.cpu().numpy(), HZ.T.cpu().numpy()


# =========================
# Main
# =========================
def main():
    # ---- fixed config (VS run friendly) ----
    set_seed(0)

    input_path = Path("lesmis.txt")
    dataset_name = input_path.stem
    out_dir = Path(f"fixed_paper_{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # original-wcd defaults
    alpha, beta = 0.5, 0.5
    T = 3
    h = 64
    epochs = 400

    # Lesmis special: gating threshold
    gate_common_min = 2  # try 2, then 3 if still smooth

    t0 = time.perf_counter()

    W = load_matrix(input_path)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)

    A = (W > 0).astype(int)

    # ---- Algorithm 1 (gated X) ----
    X = compute_X_gated(W, A, alpha=alpha, beta=beta, gate_common_min=gate_common_min)
    Z = compute_Z(A)
    Qm = compute_modularity_matrix(W)

    # normalization consistent with original-wcd
    X = minmax_01(X)
    Z = minmax_01(Z)
    Qm = scale_signed(Qm)

    # ---- Algorithm 2 ----
    X_t, Q_t, Z_t = X, Qm, Z
    dims_trace = [W.shape[0]]
    for _ in range(T):
        X_t, Q_t, Z_t = train_one_layer(X_t, Q_t, Z_t, d_h=h, epochs=epochs)
        dims_trace.append(h)

    H = X_t.T  # [n, h]

    # ---- KMeans scan ----
    best_Q, best_k, best_labels = -1, None, None
    for k in range(2, 15):
        labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(H)
        Qv = modularity_score(W, labels)
        if Qv > best_Q:
            best_Q, best_k, best_labels = Qv, k, labels

    t1 = time.perf_counter()

    # ---- Save outputs (same spec) ----
    np.savetxt(out_dir / "labels.txt", best_labels, fmt="%d")

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset {dataset_name}\n")
        f.write("method original_wcd_lesmis_gated\n")
        f.write(f"n {W.shape[0]}\n")
        f.write("seed 0\n")
        f.write(f"alpha {alpha}\n")
        f.write(f"beta {beta}\n")
        f.write(f"T {T}\n")
        f.write(f"h {h}\n")
        f.write(f"epochs_per_layer {epochs}\n")
        f.write(f"dims_trace {'->'.join(map(str, dims_trace))}\n")
        f.write(f"best_k {best_k}\n")
        f.write(f"modularity {best_Q:.6f}\n")
        f.write(f"time_seconds {t1 - t0:.6f}\n")
        f.write(f"gate_common_min {gate_common_min}\n")

    # ---- Visualization ----
    H2 = PCA(n_components=2, random_state=0).fit_transform(H)
    plt.figure(figsize=(7, 6))
    plt.scatter(H2[:, 0], H2[:, 1], c=best_labels, cmap="tab10", s=60)
    plt.title(f"Gated WCD (Lesmis) | Q={best_Q:.3f} | k={best_k} | gate>={gate_common_min}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "community.png", dpi=300)
    plt.close()

    print(f"✅ gated WCD done: Q={best_Q:.4f}, k={best_k}, time={t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
