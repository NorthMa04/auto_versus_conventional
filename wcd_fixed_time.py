import numpy as np
import os
import time
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# =========================
# 读取 Matrix / txt
# =========================
def load_matrix(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, 'r') as f:
        first = f.readline().strip()

    if first.startswith('%%MatrixMarket'):
        from scipy.io import mmread
        A = mmread(path)
        A = A.toarray() if not isinstance(A, np.ndarray) else A
        return A.astype(float)
    else:
        return np.loadtxt(path)


# =========================
# 相似性矩阵 X（论文公式 1）
# =========================
def compute_X(W, A, alpha=0.5, beta=0.5):
    n = W.shape[0]
    neighbors = [set(np.where(A[i] > 0)[0]) for i in range(n)]
    X = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            common = neighbors[i] & neighbors[j]
            W1 = sum(W[i, m] + W[m, j] for m in common)
            X_ij = alpha * W[i, j] + beta * W1
            X[i, j] = X_ij
            X[j, i] = X_ij

    return X


# =========================
# SAE 模型（论文 3.4）
# =========================
class SharedSparseAE(nn.Module):
    def __init__(self, n, h_branch=32, h_shared=16):
        super().__init__()
        self.act = nn.Sigmoid()
        self.enc_x = nn.Linear(n, h_branch)
        self.enc_q = nn.Linear(n, h_branch)
        self.enc_z = nn.Linear(n, h_branch)
        self.fuse = nn.Linear(3 * h_branch, h_shared)
        self.dec_x = nn.Linear(h_shared, n)
        self.dec_q = nn.Linear(h_shared, n)
        self.dec_z = nn.Linear(h_shared, n)

    def forward(self, x, q, z):
        hx = self.act(self.enc_x(x))
        hq = self.act(self.enc_q(q))
        hz = self.act(self.enc_z(z))
        h = self.act(self.fuse(torch.cat([hx, hq, hz], dim=1)))
        return self.act(self.dec_x(h)), self.act(self.dec_q(h)), self.act(self.dec_z(h)), h


def kl_div(rho, rho_hat):
    eps = 1e-8
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
def scale_signed(M,eps=1e-12):
    s=np.max(np.abs(M))
    return M/(s+eps)

# =========================
# 模块度
# =========================
def modularity(W, labels):
    m = W.sum() / 2
    k = W.sum(axis=1)
    Q = 0.0
    for i in range(len(W)):
        for j in range(len(W)):
            if labels[i] == labels[j]:
                Q += W[i, j] - k[i] * k[j] / (2 * m)
    return Q / (2 * m)


# =========================
# 主流程（固定参数 + 计时）
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="lesmis.txt", help="Input adjacency matrix")
    args = parser.parse_args()

    dataset_name = Path(args.input).stem
    out_dir = Path(f"fixed_{dataset_name}")
    out_dir.mkdir(exist_ok=True)

    t_start = time.perf_counter()

    # ---------- 预处理 ----------
    W = load_matrix(Path(args.input))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)

    A = (W > 0).astype(int)

    # Z：论文公式 (4)
    B = A @ A
    Z = 0.5 * A + B
    np.fill_diagonal(Z, 0)

    # X：论文公式 (1)
    X = compute_X(W, A, alpha=0.5, beta=0.5)

    # Q：论文公式 (3)
    k = W.sum(axis=1)
    twoW = W.sum()
    Q = W - np.outer(k, k) / twoW
    np.fill_diagonal(Q, 0)

    # 归一化
    def minmax(M):
        mn, mx = M.min(), M.max()
        return np.zeros_like(M) if mx - mn < 1e-12 else (M - mn) / (mx - mn)

    X,Z = minmax(X), minmax(Z)
    Q=scale_signed(Q)

    # ---------- SAE ----------
    n = W.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xc = torch.tensor(X.T, dtype=torch.float32).to(device)
    Qc = torch.tensor(Q.T, dtype=torch.float32).to(device)
    Zc = torch.tensor(Z.T, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(Xc, Qc, Zc), batch_size=32, shuffle=True)

    model = SharedSparseAE(n, h_branch=128, h_shared=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    rho = 0.05

    for _ in range(1000):
        for xb, qb, zb in loader:
            xh, qh, zh, h = model(xb, qb, zb)
            loss = mse(xh, xb) + mse(qh, qb) + mse(zh, zb) + kl_div(rho, h.mean(dim=0)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        _, _, _, H = model(Xc, Qc, Zc)
        H = H.cpu().numpy()

    # ---------- 聚类 ----------
    best_Q, best_k, best_labels = -1, None, None
    for k in range(2, int(np.sqrt(n)) + 20):
        labels = KMeans(k, n_init=20).fit_predict(H)
        Qv = modularity(W, labels)
        if Qv > best_Q:
            best_Q, best_k, best_labels = Qv, k, labels

    t_end = time.perf_counter()
    total_time = t_end - t_start

    # ---------- 输出 ----------
    np.savetxt(out_dir / "labels.txt", best_labels, fmt="%d")

    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"dataset {dataset_name}\n")
        f.write("method fixed\n")
        f.write(f"n {n}\n")
        f.write(f"modularity {best_Q:.6f}\n")
        f.write(f"time_seconds {total_time:.6f}\n")

    pca = PCA(2)
    H2 = pca.fit_transform(H)
    plt.figure(figsize=(7, 6))
    plt.scatter(H2[:, 0], H2[:, 1], c=best_labels, cmap="tab10", s=60)
    plt.title(f"Fixed WCD ({dataset_name})")
    plt.savefig(out_dir / "community.png", dpi=300)
    plt.close()

    print(f"✅ Fixed WCD finished: Q={best_Q:.4f}, time={total_time:.2f}s")


if __name__ == "__main__":
    main()
