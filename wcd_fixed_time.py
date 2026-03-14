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
# 可重复性设置
# =========================
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 矩阵加载（支持 Matrix Market 和文本格式）
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
# 归一化（Min‑Max 缩放到 [0,1]）
# =========================
def minmax_01(M, eps=1e-12):
    mn, mx = M.min(), M.max()
    if abs(mx - mn) < eps:
        return np.zeros_like(M)
    return (M - mn) / (mx - mn)


# =========================
# 算法 1：计算相似性矩阵 X、Z 和模块度矩阵 Q
# =========================
def compute_X(W, A, alpha=0.5, beta=0.5):
    """
    根据公式 (1) 计算考虑二阶邻居的相似性矩阵 X。
    W : 加权邻接矩阵
    A : 未加权邻接矩阵（0/1）
    alpha, beta : 权重因子，且 alpha + beta = 1
    """
    n = W.shape[0]
    X = np.zeros((n, n), dtype=float)

    # 每个节点的邻居集合
    neighbors = [set(np.where(A[i] > 0)[0]) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            # 公共邻居
            common = neighbors[i] & neighbors[j]
            # 公式 (1) 中的第二项：通过公共邻居的路径权重和
            W1 = sum(W[i, m] + W[m, j] for m in common)
            # 直接边的贡献 + 二阶邻居贡献
            X[i, j] = X[j, i] = alpha * W[i, j] + beta * W1
    return X


def compute_Z(A):
    """
    计算未加权网络的二阶邻接矩阵 Z。
    Z = 0.5 * A + A^2
    """
    B = A @ A
    Z = 0.5 * A + B
    np.fill_diagonal(Z, 0)
    return Z


def compute_modularity_matrix(W):
    """
    计算加权网络的模块度矩阵 Q (公式 3)。
    """
    k = W.sum(axis=1)                 # 节点强度
    twoW = W.sum()                     # 总权重
    Q = W - np.outer(k, k) / (twoW + 1e-12)
    np.fill_diagonal(Q, 0)
    return Q


# =========================
# 模块度 Q 值计算（评价指标）
# =========================
def modularity_score(W, labels):
    """
    计算给定划分的模块度 Q (公式 10)。
    """
    n = W.shape[0]
    m = W.sum() / 2.0                   # 总边权的一半
    deg = W.sum(axis=1)                 # 节点强度

    Qv = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Qv += W[i, j] - deg[i] * deg[j] / (2.0 * m + 1e-12)
    return Qv / (2.0 * m + 1e-12)


# =========================
# 稀疏自编码器（Sigmoid 激活）
# =========================
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
    """
    KL 散度用于稀疏约束 (公式 9)。
    """
    eps = 1e-8
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))


def train_one_layer(X, Q, Z, d_h, epochs=400, batch_size=32, lr=1e-2,
                    rho=0.05, lam_sparse=1e-4):
    """
    训练一层稀疏自编码器（三个并行的 AE）。
    返回：
        HX, HQ, HZ : 当前层输出的低维特征矩阵 (形状: d_h × n)
        ae_x, ae_q, ae_z : 训练好的自编码器对象（已在 CPU 上）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_in, n = X.shape                     # X 是 d_in × n，每一列是一个样本
    Xc = torch.tensor(X.T, dtype=torch.float32, device=device)   # 转换为 n × d_in
    Qc = torch.tensor(Q.T, dtype=torch.float32, device=device)
    Zc = torch.tensor(Z.T, dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(Xc, Qc, Zc),
                        batch_size=batch_size, shuffle=True)

    ae_x = SparseAE(d_in, d_h).to(device)
    ae_q = SparseAE(d_in, d_h).to(device)
    ae_z = SparseAE(d_in, d_h).to(device)

    opt = torch.optim.Adam(
        list(ae_x.parameters()) +
        list(ae_q.parameters()) +
        list(ae_z.parameters()),
        lr=lr
    )

    mse = nn.MSELoss()

    for _ in range(epochs):
        for xb, qb, zb in loader:
            xh, hx = ae_x(xb)
            qh, hq = ae_q(qb)
            zh, hz = ae_z(zb)

            # 稀疏惩罚项：KL 散度
            kl_term = (
                kl_div(rho, hx.mean(0)).sum() +
                kl_div(rho, hq.mean(0)).sum() +
                kl_div(rho, hz.mean(0)).sum()
            )

            loss = mse(xh, xb) + mse(qh, qb) + mse(zh, zb) + lam_sparse * kl_term
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 获得压缩后的特征矩阵
    with torch.no_grad():
        _, HX = ae_x(Xc)
        _, HQ = ae_q(Qc)
        _, HZ = ae_z(Zc)

    # 返回的是 d_h × n 的矩阵（论文中的特征矩阵格式），同时返回 CPU 上的编码器
    return HX.T.cpu().numpy(), HQ.T.cpu().numpy(), HZ.T.cpu().numpy(), ae_x.cpu(), ae_q.cpu(), ae_z.cpu()


# =========================
# 主程序：完整 WCD 算法
# =========================
def main():
    set_seed(0)

    # 输入数据路径（请根据实际情况修改）
    input_path = Path("football.txt")          # 示例数据集
    dataset_name = input_path.stem
    out_dir = Path(f"wcd_paper_{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ---------- 加载并预处理原始加权网络 ----------
    W = load_matrix(input_path)
    W = 0.5 * (W + W.T)                       # 确保对称
    np.fill_diagonal(W, 0)                     # 对角线置 0

    A = (W > 0).astype(int)                    # 未加权邻接矩阵

    # ---------- 算法 1：计算三个特征矩阵 ----------
    X = compute_X(W, A, alpha=0.5, beta=0.5)   # 相似性矩阵（公式 1）
    Z = compute_Z(A)                            # 二阶邻接矩阵
    Qm = compute_modularity_matrix(W)           # 模块度矩阵（公式 3）

    # 归一化到 [0,1]（论文未明确要求，但有利于训练）
    X = minmax_01(X)
    Z = minmax_01(Z)
    Qm = minmax_01(Qm)

    # ---------- 深度稀疏自编码器堆叠训练（算法 2） ----------
    T = 8               # 深度自编码器层数
    h = 32            # 隐藏层维度（压缩后的大小）

    # 保存每一层训练好的编码器（用于最终前向传播）
    encoders_x = []       # 只保留 X 路径的编码器

    # 初始输入
    X_t, Q_t, Z_t = X, Qm, Z

    for layer in range(T):
        print(f"Training layer {layer+1}/{T} ...")
        HX, HQ, HZ, ae_x, ae_q, ae_z = train_one_layer(X_t, Q_t, Z_t, d_h=h)
        encoders_x.append(ae_x)          # 保存当前层 X 的编码器
        # 下一层的输入为当前层输出的特征
        X_t, Q_t, Z_t = HX, HQ, HZ

    # ---------- 最终特征提取：原始相似性矩阵 X 依次通过所有保存的编码器 ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = X.T                               # 原始 X，形状 n × d_in

    for i, ae in enumerate(encoders_x):
        ae.eval()                         # 设置为评估模式（影响 dropout / BN，此处无，但习惯）
        ae.to(device)
        H_tensor = torch.tensor(H, dtype=torch.float32, device=device)
        with torch.no_grad():
            # 注意：ae.forward 返回 (x_hat, h)，我们取隐藏层 h 作为下一层的输入
            _, H_hidden = ae(H_tensor)
        H = H_hidden.cpu().numpy()        # 转换为 numpy 供下一轮使用

    # 此时 H 的形状为 n × h（每行是一个节点的低维特征）

    # ---------- K‑means 聚类 ----------
    best_Q, best_k, best_labels = -1, None, None
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
        labels = kmeans.fit_predict(H)
        Qv = modularity_score(W, labels)
        if Qv > best_Q:
            best_Q, best_k, best_labels = Qv, k, labels

    t1 = time.perf_counter()

    # ---------- 保存结果 ----------
    np.savetxt(out_dir / "labels.txt", best_labels, fmt="%d")

    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"dataset {dataset_name}\n")
        f.write("method wcd_paper\n")
        f.write(f"n {W.shape[0]}\n")
        f.write("alpha 0.5\n")
        f.write("beta 0.5\n")
        f.write(f"T {T}\n")
        f.write(f"h {h}\n")
        f.write(f"best_k {best_k}\n")
        f.write(f"modularity {best_Q:.6f}\n")
        f.write(f"time_seconds {t1 - t0:.6f}\n")

    # ---------- 可视化（PCA 降维到 2D）----------
    H2 = PCA(n_components=2).fit_transform(H)
    plt.figure(figsize=(8, 6))
    plt.scatter(H2[:, 0], H2[:, 1], c=best_labels, cmap="tab10", s=60, edgecolor='k')
    plt.title(f"WCD (paper) | Q = {best_Q:.3f} | k = {best_k}")
    plt.savefig(out_dir / "community.png", dpi=300)
    plt.close()

    print(f"Finished. Q = {best_Q:.4f}, k = {best_k}")
    print(f"Results saved in {out_dir}")


if __name__ == "__main__":
    main()