import numpy as np
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pandas as pd


# =========================
# 全部配置统一放在这里
# =========================
CONFIG = {
    # ----- experiment identity -----
    "experiment_tag": "h_Test",
    # 当前实验标签。
    # 会直接写入输出目录名、文件名和汇总表中，
    # 用于显式区分这是一轮针对 h 的参数测试。

    # ----- reproducibility -----
    "seed": 0,
    # 随机种子。固定后可尽量保证：
    # 1) 自编码器初始化一致
    # 2) DataLoader 打乱顺序一致
    # 3) KMeans 初始化一致

    # ----- datasets -----
    "input_paths": [
        "lesmis.txt",
        "football.txt",
        "celegans_edges.txt",
        # "hep-th.txt",   # 当前按你的要求先不跑
    ],
    # 待测试的数据集列表。
    # 当前 main() 会对列表中的每个数据集都完整跑一遍 h 扫描。

    # ----- output -----
    "output_root": "wcd_experiments",
    # 所有实验结果的根目录。

    # ----- feature construction -----
    "alpha": 0.5,
    "beta": 0.5,
    # 相似性矩阵 X 的构造参数。
    # 当前固定为统一配置，避免这轮 h 实验被 alpha/beta 干扰。

    # ----- one-layer sparse AE training -----
    "epochs": 600,
    "batch_size": 16,
    "lr": 1e-3,
    "rho": 0.10,
    "lam_sparse": 1e-3,
    # 当前固定为较稳健的训练配置，尽量避免训练本身成为瓶颈。

    # ----- stacked deep sparse AE -----
    "T": 5,
    # 最终层编号。
    # 实际训练层数仍为 T - 1，保持你当前代码逻辑不变。

    "h_list": [8, 16, 24, 32, 48, 64, 80, 96],
    # 当前实验的核心扫描参数。
    # main() 会依次测试这些 h。

    # ----- kmeans -----
    "k_min": 2,
    "k_max": 14,
    "k_n_init": 50,

    # ----- visualization -----
    "figsize": (8, 6),
    "dpi": 300,
    "scatter_size": 60,

    # ----- logging -----
    "save_epoch_history": True,
    # 是否保存每一层每个 epoch 的训练历史。
    # 建议开启，这对排查“某个 h 为什么效果差”很有帮助。

    "save_excel": True,
    # 是否额外导出 Excel 汇总表。
}


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
# 归一化（Min-Max 缩放到 [0,1]）
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
    根据论文算法 1 计算考虑二阶邻居的相似性矩阵 X。
    这里严格按算法 1 的写法实现：
    1) 先计算 B = A^2
    2) 对所有 i, j 遍历
    3) 仅当 B[i, j] != 0 时，才计算 X[i, j]
    """
    n = W.shape[0]
    X = np.zeros((n, n), dtype=float)
    B = A @ A

    neighbors = [set(np.where(A[i] > 0)[0]) for i in range(n)]

    for i in range(n):
        for j in range(n):
            if B[i, j] != 0:
                common = neighbors[i] & neighbors[j]
                W1 = sum(W[i, m] + W[m, j] for m in common)
                X[i, j] = alpha * W[i, j] + beta * W1

    return X


def compute_Z(A):
    """
    根据论文算法 1 计算未加权网络的二阶邻接矩阵 Z。
    Z = 0.5 * A + A^2
    这里不额外清零对角线，保持与论文伪代码一致。
    """
    B = A @ A
    Z = 0.5 * A + B
    return Z.astype(float)


def compute_modularity_matrix(W):
    """
    计算加权网络的模块度矩阵 Q (公式 3)。
    """
    k = W.sum(axis=1)
    twoW = W.sum()
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
    m = W.sum() / 2.0
    deg = W.sum(axis=1)

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


def train_one_layer(
    X, Q, Z, d_h, epochs=400, batch_size=32, lr=1e-2,
    rho=0.05, lam_sparse=1e-4, layer_index=1,
    save_epoch_history=True
):
    """
    训练一层稀疏自编码器。

    返回：
        HX, HQ, HZ : 当前层输出的低维特征矩阵 (形状: d_h × n)
        ae : 当前层训练好的自编码器（已在 CPU 上）
        layer_summary : 当前层训练摘要(dict)
        epoch_history : 当前层每个 epoch 的训练历史(list[dict])
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_in, n = X.shape
    Xc = torch.tensor(X.T, dtype=torch.float32, device=device)
    Qc = torch.tensor(Q.T, dtype=torch.float32, device=device)
    Zc = torch.tensor(Z.T, dtype=torch.float32, device=device)

    train_data = torch.cat([Xc, Qc, Zc], dim=0)
    loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)

    ae = SparseAE(d_in, d_h).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    mse = nn.MSELoss()

    epoch_history = []
    layer_t0 = time.perf_counter()

    for epoch in range(epochs):
        batch_total_losses = []
        batch_recon_losses = []
        batch_kl_losses = []
        batch_mean_acts = []

        for (batch,) in loader:
            x_hat, h = ae(batch)
            recon_loss = mse(x_hat, batch)
            kl_term = kl_div(rho, h.mean(0)).sum()
            total_loss = recon_loss + lam_sparse * kl_term

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            batch_total_losses.append(float(total_loss.item()))
            batch_recon_losses.append(float(recon_loss.item()))
            batch_kl_losses.append(float(kl_term.item()))
            batch_mean_acts.append(float(h.mean().item()))

        if save_epoch_history:
            epoch_history.append({
                "layer": layer_index,
                "epoch": epoch + 1,
                "total_loss": float(np.mean(batch_total_losses)),
                "recon_loss": float(np.mean(batch_recon_losses)),
                "kl_loss": float(np.mean(batch_kl_losses)),
                "mean_activation": float(np.mean(batch_mean_acts)),
            })

    layer_t1 = time.perf_counter()

    with torch.no_grad():
        _, HX = ae(Xc)
        _, HQ = ae(Qc)
        _, HZ = ae(Zc)

        x_hat_all, h_all = ae(train_data)
        final_recon = float(mse(x_hat_all, train_data).item())
        final_kl = float(kl_div(rho, h_all.mean(0)).sum().item())
        final_total = float(final_recon + lam_sparse * final_kl)

        h_mean = float(h_all.mean().item())
        h_std = float(h_all.std().item())
        h_min = float(h_all.min().item())
        h_max = float(h_all.max().item())

    layer_summary = {
        "layer": layer_index,
        "d_in": d_in,
        "d_h": d_h,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "rho": rho,
        "lam_sparse": lam_sparse,
        "final_total_loss": final_total,
        "final_recon_loss": final_recon,
        "final_kl_loss": final_kl,
        "final_mean_activation": h_mean,
        "final_std_activation": h_std,
        "final_min_activation": h_min,
        "final_max_activation": h_max,
        "layer_time_seconds": layer_t1 - layer_t0,
        "has_nan_in_HX": bool(np.isnan(HX.cpu().numpy()).any()),
        "has_nan_in_HQ": bool(np.isnan(HQ.cpu().numpy()).any()),
        "has_nan_in_HZ": bool(np.isnan(HZ.cpu().numpy()).any()),
    }

    return HX.T.cpu().numpy(), HQ.T.cpu().numpy(), HZ.T.cpu().numpy(), ae.cpu(), layer_summary, epoch_history


def build_run_name(dataset_name, h):
    return f"{CONFIG['experiment_tag']}_{dataset_name}_h{int(h):03d}"


def save_metrics_txt(path, metrics_dict):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in metrics_dict.items():
            f.write(f"{k} {v}\n")


def run_single_experiment(input_path: Path, h_value: int):
    set_seed(CONFIG["seed"])

    dataset_name = input_path.stem
    run_name = build_run_name(dataset_name, h_value)

    out_root = Path(CONFIG["output_root"])
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ---------- 加载并预处理原始加权网络 ----------
    W = load_matrix(input_path)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)

    A = (W > 0).astype(int)

    n = W.shape[0]
    edges = int(np.count_nonzero(np.triu(W > 0, k=1)))
    density = 0.0 if n <= 1 else (2.0 * edges) / (n * (n - 1))

    # ---------- 算法 1：计算三个特征矩阵 ----------
    feat_t0 = time.perf_counter()
    X = compute_X(W, A, alpha=CONFIG["alpha"], beta=CONFIG["beta"])
    Z = compute_Z(A)
    Qm = compute_modularity_matrix(W)

    X = minmax_01(X)
    Z = minmax_01(Z)
    Qm = minmax_01(Qm)
    feat_t1 = time.perf_counter()

    # ---------- 深度稀疏自编码器堆叠训练 ----------
    T = CONFIG["T"]
    h = h_value

    encoders = []
    X_t, Q_t, Z_t = X, Qm, Z

    layer_summaries = []
    epoch_histories = []

    sae_t0 = time.perf_counter()
    for layer in range(T - 1):
        print(f"[{run_name}] Training layer {layer + 1}/{T - 1} ...")
        HX, HQ, HZ, ae, layer_summary, epoch_history = train_one_layer(
            X_t, Q_t, Z_t,
            d_h=h,
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            lr=CONFIG["lr"],
            rho=CONFIG["rho"],
            lam_sparse=CONFIG["lam_sparse"],
            layer_index=layer + 1,
            save_epoch_history=CONFIG["save_epoch_history"],
        )
        encoders.append(ae)
        X_t, Q_t, Z_t = HX, HQ, HZ
        layer_summaries.append(layer_summary)
        epoch_histories.extend(epoch_history)
    sae_t1 = time.perf_counter()

    # ---------- 最终特征提取 ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = X.T

    forward_t0 = time.perf_counter()
    for ae in encoders:
        ae.eval()
        ae.to(device)
        H_tensor = torch.tensor(H, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, H_hidden = ae(H_tensor)
        H = H_hidden.cpu().numpy()
    forward_t1 = time.perf_counter()

    H_nan = bool(np.isnan(H).any())
    H_inf = bool(np.isinf(H).any())

    # ---------- K-means 聚类 ----------
    best_Q, best_k, best_labels = -1, None, None
    kmeans_records = []

    kmeans_t0 = time.perf_counter()
    for k in range(CONFIG["k_min"], CONFIG["k_max"] + 1):
        kmeans = KMeans(
            n_clusters=k,
            n_init=CONFIG["k_n_init"],
            random_state=CONFIG["seed"]
        )
        labels = kmeans.fit_predict(H)
        Qv = modularity_score(W, labels)

        inertia = float(kmeans.inertia_)
        unique_clusters = int(len(np.unique(labels)))

        kmeans_records.append({
            "dataset": dataset_name,
            "run_name": run_name,
            "h": h,
            "k": k,
            "modularity": float(Qv),
            "inertia": inertia,
            "unique_clusters": unique_clusters,
        })

        if Qv > best_Q:
            best_Q, best_k, best_labels = Qv, k, labels
    kmeans_t1 = time.perf_counter()

    t1 = time.perf_counter()

    # ---------- 保存标签 ----------
    np.savetxt(out_dir / f"{run_name}_labels.txt", best_labels, fmt="%d")

    # ---------- 保存层摘要 ----------
    df_layers = pd.DataFrame(layer_summaries)
    df_layers.insert(0, "dataset", dataset_name)
    df_layers.insert(1, "run_name", run_name)
    df_layers.insert(2, "h", h)
    df_layers.to_csv(out_dir / f"{run_name}_layer_summary.csv", index=False)

    # ---------- 保存 epoch 训练历史 ----------
    if CONFIG["save_epoch_history"]:
        df_epochs = pd.DataFrame(epoch_histories)
        if not df_epochs.empty:
            df_epochs.insert(0, "dataset", dataset_name)
            df_epochs.insert(1, "run_name", run_name)
            df_epochs.insert(2, "h", h)
            df_epochs.to_csv(out_dir / f"{run_name}_epoch_history.csv", index=False)
    else:
        df_epochs = pd.DataFrame()

    # ---------- 保存 KMeans 扫描 ----------
    df_kmeans = pd.DataFrame(kmeans_records)
    df_kmeans.to_csv(out_dir / f"{run_name}_kmeans_scan.csv", index=False)

    # ---------- 汇总指标 ----------
    metrics = {
        "experiment_tag": CONFIG["experiment_tag"],
        "run_name": run_name,
        "dataset": dataset_name,
        "method": "wcd_paper",
        "n": n,
        "edges": edges,
        "density": density,
        "seed": CONFIG["seed"],
        "alpha": CONFIG["alpha"],
        "beta": CONFIG["beta"],
        "T": CONFIG["T"],
        "trained_layers": max(CONFIG["T"] - 1, 0),
        "h": h,
        "epochs": CONFIG["epochs"],
        "batch_size": CONFIG["batch_size"],
        "lr": CONFIG["lr"],
        "rho": CONFIG["rho"],
        "lam_sparse": CONFIG["lam_sparse"],
        "k_min": CONFIG["k_min"],
        "k_max": CONFIG["k_max"],
        "k_n_init": CONFIG["k_n_init"],
        "best_k": best_k,
        "modularity": float(best_Q),
        "feature_time_seconds": feat_t1 - feat_t0,
        "sae_train_time_seconds": sae_t1 - sae_t0,
        "forward_time_seconds": forward_t1 - forward_t0,
        "kmeans_time_seconds": kmeans_t1 - kmeans_t0,
        "time_seconds": t1 - t0,
        "H_shape_rows": H.shape[0],
        "H_shape_cols": H.shape[1],
        "H_mean": float(np.mean(H)),
        "H_std": float(np.std(H)),
        "H_min": float(np.min(H)),
        "H_max": float(np.max(H)),
        "H_has_nan": H_nan,
        "H_has_inf": H_inf,
        "last_layer_final_total_loss": float(df_layers["final_total_loss"].iloc[-1]) if not df_layers.empty else np.nan,
        "last_layer_final_recon_loss": float(df_layers["final_recon_loss"].iloc[-1]) if not df_layers.empty else np.nan,
        "last_layer_final_kl_loss": float(df_layers["final_kl_loss"].iloc[-1]) if not df_layers.empty else np.nan,
        "last_layer_final_mean_activation": float(df_layers["final_mean_activation"].iloc[-1]) if not df_layers.empty else np.nan,
    }

    save_metrics_txt(out_dir / f"{run_name}_metrics.txt", metrics)

    # ---------- 可视化 ----------
    H2 = PCA(n_components=2).fit_transform(H)
    plt.figure(figsize=CONFIG["figsize"])
    plt.scatter(
        H2[:, 0], H2[:, 1],
        c=best_labels,
        cmap="tab10",
        s=CONFIG["scatter_size"],
        edgecolor="k"
    )
    plt.title(f"{CONFIG['experiment_tag']} | {dataset_name} | h={h} | Q={best_Q:.3f} | k={best_k}")
    plt.savefig(out_dir / f"{run_name}_community.png", dpi=CONFIG["dpi"])
    plt.close()

    # ---------- 单次实验 Excel ----------
    if CONFIG["save_excel"]:
        excel_path = out_dir / f"{run_name}_results.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame([metrics]).to_excel(writer, sheet_name="summary", index=False)
            df_layers.to_excel(writer, sheet_name="layer_summary", index=False)
            df_kmeans.to_excel(writer, sheet_name="kmeans_scan", index=False)
            if not df_epochs.empty:
                df_epochs.to_excel(writer, sheet_name="epoch_history", index=False)

    print(f"[{run_name}] Finished. Q = {best_Q:.4f}, k = {best_k}")
    print(f"[{run_name}] Results saved in {out_dir}")

    return {
        "metrics": metrics,
        "df_layers": df_layers,
        "df_kmeans": df_kmeans,
        "df_epochs": df_epochs,
    }


# =========================
# 主程序：多数据集 × 多 h 的批量实验
# =========================
def main():
    out_root = Path(CONFIG["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_layers = []
    all_kmeans = []
    all_epochs = []

    global_t0 = time.perf_counter()

    for input_str in CONFIG["input_paths"]:
        input_path = Path(input_str)
        dataset_name = input_path.stem

        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)

        for h in CONFIG["h_list"]:
            result = run_single_experiment(input_path=input_path, h_value=h)

            all_metrics.append(result["metrics"])
            all_layers.append(result["df_layers"])
            all_kmeans.append(result["df_kmeans"])

            if not result["df_epochs"].empty:
                all_epochs.append(result["df_epochs"])

    global_t1 = time.perf_counter()

    # ---------- 全局汇总 ----------
    df_all_metrics = pd.DataFrame(all_metrics)
    df_all_layers = pd.concat(all_layers, ignore_index=True) if all_layers else pd.DataFrame()
    df_all_kmeans = pd.concat(all_kmeans, ignore_index=True) if all_kmeans else pd.DataFrame()
    df_all_epochs = pd.concat(all_epochs, ignore_index=True) if all_epochs else pd.DataFrame()

    summary_prefix = CONFIG["experiment_tag"]

    df_all_metrics.to_csv(out_root / f"{summary_prefix}_all_runs_summary.csv", index=False)
    if not df_all_layers.empty:
        df_all_layers.to_csv(out_root / f"{summary_prefix}_all_layers_summary.csv", index=False)
    if not df_all_kmeans.empty:
        df_all_kmeans.to_csv(out_root / f"{summary_prefix}_all_kmeans_scan.csv", index=False)
    if not df_all_epochs.empty:
        df_all_epochs.to_csv(out_root / f"{summary_prefix}_all_epoch_history.csv", index=False)

    # 每个数据集找最优 h
    if not df_all_metrics.empty:
        idx = df_all_metrics.groupby("dataset")["modularity"].idxmax()
        df_best_by_dataset = df_all_metrics.loc[idx].sort_values("dataset")
        df_best_by_dataset.to_csv(out_root / f"{summary_prefix}_best_h_by_dataset.csv", index=False)
    else:
        df_best_by_dataset = pd.DataFrame()

    # 全局 Excel 汇总
    if CONFIG["save_excel"]:
        excel_path = out_root / f"{summary_prefix}_global_summary.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_all_metrics.to_excel(writer, sheet_name="all_runs_summary", index=False)
            if not df_best_by_dataset.empty:
                df_best_by_dataset.to_excel(writer, sheet_name="best_h_by_dataset", index=False)
            if not df_all_layers.empty:
                df_all_layers.to_excel(writer, sheet_name="all_layers_summary", index=False)
            if not df_all_kmeans.empty:
                df_all_kmeans.to_excel(writer, sheet_name="all_kmeans_scan", index=False)
            if not df_all_epochs.empty:
                df_all_epochs.to_excel(writer, sheet_name="all_epoch_history", index=False)

    print("=" * 80)
    print(f"All {CONFIG['experiment_tag']} experiments finished.")
    print(f"Total time: {global_t1 - global_t0:.2f} seconds")
    print(f"Global summary saved in: {out_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()