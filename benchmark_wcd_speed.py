import os
import gc
import time
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans


# =========================
# Benchmark 配置
# =========================
CONFIG = {
    "seed": 0,

    # 换成你三台机器都有的数据集
    "input_path": "lfr_like_2_normalized.txt",

    # 固定一组中等强度参数，适合测速
    "alpha": 0.5,
    "T": 8,              # 实际训练 T-1 层
    "h": 32,
    "rho": 0.05,
    "lr": 0.01,
    "lam_sparse": 0.00033,
    "epochs": 300,
    "batch_size": 32,

    # KMeans 测试
    "k_min": 4,
    "k_max": 18,
    "k_n_init": 10,

    # 测试次数：建议 3 次，取平均
    "repeat": 3,

    # 是否强制 CPU
    "force_cpu": False,
}


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device():
    if CONFIG["force_cpu"]:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_hardware_info(device):
    print("=" * 100)
    print("Hardware Info")
    print("=" * 100)
    print(f"Platform: {platform.platform()}")
    print(f"CPU count logical: {os.cpu_count()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU total memory: {prop.total_memory / 1024 ** 3:.2f} GB")
        print(f"GPU multiprocessors: {prop.multi_processor_count}")
    print("=" * 100)


def load_matrix(path: Path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()

    if first.startswith("%%MatrixMarket"):
        from scipy.io import mmread
        A = mmread(path)
        return A.toarray().astype(float)
    else:
        return np.loadtxt(path).astype(float)


def minmax_01(M, eps=1e-12):
    mn, mx = M.min(), M.max()
    if abs(mx - mn) < eps:
        return np.zeros_like(M)
    return (M - mn) / (mx - mn)


def compute_X(W, A, alpha=0.5, beta=0.5):
    n = W.shape[0]
    X = np.zeros((n, n), dtype=float)
    B = A @ A
    neighbors = [set(np.where(A[i] > 0)[0]) for i in range(n)]

    for i in range(n):
        for j in range(n):
            x_ij = alpha * W[i, j]
            if B[i, j] != 0:
                common = neighbors[i] & neighbors[j]
                W1 = sum(W[i, m] + W[m, j] for m in common)
                x_ij += beta * W1
            X[i, j] = x_ij

    return X


def compute_Z(A):
    B = A @ A
    return (0.5 * A + B).astype(float)


def compute_modularity_matrix(W):
    k = W.sum(axis=1)
    twoW = W.sum()
    Q = W - np.outer(k, k) / (twoW + 1e-12)
    np.fill_diagonal(Q, 0)
    return Q


def modularity_score(W, labels):
    m = W.sum() / 2.0
    deg = W.sum(axis=1)

    Qv = 0.0
    n = W.shape[0]
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Qv += W[i, j] - deg[i] * deg[j] / (2.0 * m + 1e-12)

    return Qv / (2.0 * m + 1e-12)


class SparseAE(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.enc = nn.Linear(d_in, d_h)
        self.dec = nn.Linear(d_h, d_in)
        self.act = nn.Sigmoid()

    def forward(self, x):
        h = self.act(self.enc(x))
        x_hat = self.act(self.dec(h))
        return x_hat, h


def kl_div(rho, rho_hat):
    eps = 1e-8
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    return rho * torch.log(rho / rho_hat) + \
           (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))


def train_one_layer(X, Q, Z, d_h, device, epochs, batch_size, lr, rho, lam_sparse):
    d_in, n = X.shape

    Xc = torch.tensor(X.T, dtype=torch.float32, device=device)
    Qc = torch.tensor(Q.T, dtype=torch.float32, device=device)
    Zc = torch.tensor(Z.T, dtype=torch.float32, device=device)

    train_data = torch.cat([Xc, Qc, Zc], dim=0)

    loader = DataLoader(
        TensorDataset(train_data),
        batch_size=batch_size,
        shuffle=True
    )

    ae = SparseAE(d_in, d_h).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    mse = nn.MSELoss()

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for _ in range(epochs):
        for (batch,) in loader:
            x_hat, h = ae(batch)
            recon_loss = mse(x_hat, batch)
            kl_term = kl_div(rho, h.mean(0)).sum()
            total_loss = recon_loss + lam_sparse * kl_term

            opt.zero_grad()
            total_loss.backward()
            opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    train_time = time.perf_counter() - t0

    with torch.no_grad():
        _, HX = ae(Xc)
        _, HQ = ae(Qc)
        _, HZ = ae(Zc)

        HX_np = HX.T.detach().cpu().numpy()
        HQ_np = HQ.T.detach().cpu().numpy()
        HZ_np = HZ.T.detach().cpu().numpy()

    del Xc, Qc, Zc, train_data, loader
    cleanup_memory()

    return HX_np, HQ_np, HZ_np, ae.cpu(), train_time


def prepare_dataset(path):
    W = load_matrix(path)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)

    A = (W > 0).astype(int)
    Z = minmax_01(compute_Z(A))
    Qm = minmax_01(compute_modularity_matrix(W))

    n = W.shape[0]
    edges = int(np.count_nonzero(np.triu(W > 0, k=1)))
    density = 0.0 if n <= 1 else 2.0 * edges / (n * (n - 1))

    return W, A, Z, Qm, n, edges, density


def run_once(device):
    set_seed(CONFIG["seed"])

    path = Path(CONFIG["input_path"])
    W, A, Z, Qm, n, edges, density = prepare_dataset(path)

    print(f"Dataset: {path.name}")
    print(f"Nodes: {n}, Edges: {edges}, Density: {density:.6f}")

    alpha = CONFIG["alpha"]
    beta = 1.0 - alpha

    timing = {}

    t0 = time.perf_counter()
    X = compute_X(W, A, alpha=alpha, beta=beta)
    X = minmax_01(X)
    timing["compute_X_seconds"] = time.perf_counter() - t0

    encoders = []
    X_t, Q_t, Z_t = X, Qm, Z

    layer_times = []

    for layer in range(CONFIG["T"] - 1):
        print(f"Training layer {layer + 1}/{CONFIG['T'] - 1}")

        HX, HQ, HZ, ae, layer_time = train_one_layer(
            X_t, Q_t, Z_t,
            d_h=CONFIG["h"],
            device=device,
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            lr=CONFIG["lr"],
            rho=CONFIG["rho"],
            lam_sparse=CONFIG["lam_sparse"],
        )

        layer_times.append(layer_time)
        encoders.append(ae)
        X_t, Q_t, Z_t = HX, HQ, HZ

        cleanup_memory()

    timing["ae_train_seconds"] = sum(layer_times)

    t0 = time.perf_counter()

    H = X.T
    for ae in encoders:
        ae.eval()
        ae.to(device)
        H_tensor = torch.tensor(H, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, H_hidden = ae(H_tensor)
        H = H_hidden.detach().cpu().numpy()
        ae.to("cpu")
        del H_tensor, H_hidden
        cleanup_memory()

    if device.type == "cuda":
        torch.cuda.synchronize()

    timing["feature_extract_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()

    best_Q, best_k = -1e18, None
    for k in range(CONFIG["k_min"], CONFIG["k_max"] + 1):
        kmeans = KMeans(
            n_clusters=k,
            n_init=CONFIG["k_n_init"],
            random_state=CONFIG["seed"]
        )
        labels = kmeans.fit_predict(H)
        Qv = modularity_score(W, labels)

        if Qv > best_Q:
            best_Q = float(Qv)
            best_k = int(k)

    timing["kmeans_seconds"] = time.perf_counter() - t0
    timing["total_time_seconds"] = sum(timing.values())

    return {
        "best_k": best_k,
        "best_modularity": best_Q,
        **timing,
        "layer_times": layer_times,
    }


def main():
    device = get_device()
    print_hardware_info(device)

    results = []

    # 预热一次，避免第一次 CUDA 初始化影响结果
    if device.type == "cuda":
        print("CUDA warmup...")
        x = torch.randn(1024, 1024, device=device)
        y = x @ x
        torch.cuda.synchronize()
        del x, y
        cleanup_memory()

    for i in range(CONFIG["repeat"]):
        print("\n" + "=" * 100)
        print(f"Repeat {i + 1}/{CONFIG['repeat']}")
        print("=" * 100)

        result = run_once(device)
        results.append(result)

        print("\nCurrent result:")
        for k, v in result.items():
            if k != "layer_times":
                print(f"{k}: {v}")

    print("\n" + "=" * 100)
    print("Final Benchmark Summary")
    print("=" * 100)

    keys = [
        "compute_X_seconds",
        "ae_train_seconds",
        "feature_extract_seconds",
        "kmeans_seconds",
        "total_time_seconds",
    ]

    for key in keys:
        values = [r[key] for r in results]
        print(f"{key}: avg={np.mean(values):.4f}s, std={np.std(values):.4f}s")

    print("\nBest result info:")
    print(f"best_k: {results[-1]['best_k']}")
    print(f"best_modularity: {results[-1]['best_modularity']:.6f}")


if __name__ == "__main__":
    main()