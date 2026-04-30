import gc
import itertools
import numpy as np
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans

import pandas as pd


# =========================
# 全部配置统一放在这里
# =========================
CONFIG = {
    # ----- experiment identity -----
    "experiment_tag": "full_param_grid",

    # ----- reproducibility -----
    "seed": 0,

    # ----- datasets -----
    "input_paths": [
        #"lesmis.txt",
        #"football.txt",
        #"arenas-jazz_normalized.txt",
        #"polbooks_normalized.txt",
        #"soc-dolphins_normalized.txt",
        #"ucidata-zachary_normalized.txt",
        #"ia-primary-school-proximity.txt",
        #"ia-workplace-contacts.txt",
        #"ia-enron-only.txt",
        #"ia-infect-hyper.txt",
        #"eco-foodweb-baywet.txt",
        #"ca-netscience.txt",
        #"adjnoun.txt",
        # synthetic
        #"er_dense_normalized.txt",
        #"er_sparse_normalized.txt",
        #"lfr_like_1_normalized.txt",
        #"lfr_like_2_normalized.txt",
        #"sbm_blurry_normalized.txt",
        #"sbm_clear_normalized.txt"
        #"brock200-3.txt",
        #"CAG_mat72.txt",
        #"ca-sandi_auths.txt",
        #"chesapeake.txt",
        #"eco-florida.txt",
        #"eco-mangwet.txt",
        #"econ-wm3.txt",
        #"ENZYMES123.txt",
        #"inf-USAir97.txt",
        #"insecta-ant-colony2.txt",
        #"johnson8-4-4.txt",
        #"reptilia-tortoise-network-bsv.txt",
        #"sociopatterns-hypertext.txt",
        #"SW-100-6-0d1-trial2.txt"
        "synthetic_50_clear_four_blocks.txt",
        #"synthetic_50_core_periphery_bridge.txt",
        #"synthetic_100_fuzzy_mixed.txt",
        #"synthetic_100_hierarchical.txt",
        #"synthetic_150_hub_modular.txt",
        #"synthetic_150_ring_bottleneck.txt",
    ],
    # ----- output -----
    "output_root": "wcd_experiments",

    # =========================================================
    # 六参数联调
    # 细扫：alpha, T, lam_sparse
    # 粗扫：h, rho, lr
    # 另外再联调：epochs, batch_size, k_min, k_max, k_n_init
    # =========================================================

    # ----- fine search 1: alpha -----
    "alpha_list": [0.1,0.3,0.5,0.7,0.9],
    # beta 自动取 1 - alpha

    # ----- fine search 2: compression layers -----
    # 注意：代码里实际训练层数是 T - 1
    "T_list": [6,8,10],

    # ----- fine search 3: sparse penalty -----
    "lam_sparse_list": [0.0001,0.00033,0.00066,0.01],

    # ----- coarse search 1: hidden size -----
    "h_list": [16,32,48],

    # ----- coarse search 2: rho -----
    "rho_list": [0.05],

    # ----- coarse search 3: learning rate -----
    "lr_list": [0.01,0.02],

    # ----- training params -----
    "epochs_list": [600],
    "batch_size_list": [16,32],

    # ----- kmeans -----
    "k_min_list": [2],
    "k_max_list": [9],
    "k_n_init_list": [10],

    # ----- device / stability -----
    "force_cpu": False,
    "print_every_epoch": False,
    "save_excel": True,
}


# =========================
# 可重复性设置
# =========================
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================
# 设备选择
# =========================
def get_device():
    if CONFIG["force_cpu"]:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================
# 内存清理
# =========================
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


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
    修正点：
    1) 直接边 alpha * W[i,j] 应始终保留
    2) 若存在共同邻居，再叠加 beta * W1
    """
    n = W.shape[0]
    X = np.zeros((n, n), dtype=float)
    B = A @ A

    neighbors = [set(np.where(A[i] > 0)[0]) for i in range(n)]

    for i in range(n):
        for j in range(n):
            # 直接边信息始终保留
            x_ij = alpha * W[i, j]

            # 若存在二阶路径，再加共同邻居项
            if B[i, j] != 0:
                common = neighbors[i] & neighbors[j]
                W1 = sum(W[i, m] + W[m, j] for m in common)
                x_ij += beta * W1

            X[i, j] = x_ij

    return X


def compute_Z(A):
    """
    根据论文算法 1 计算未加权网络的二阶邻接矩阵 Z。
    """
    B = A @ A
    Z = 0.5 * A + B
    return Z.astype(float)


def compute_modularity_matrix(W):
    """
    计算加权网络的模块度矩阵 Q。
    """
    k = W.sum(axis=1)
    twoW = W.sum()
    Q = W - np.outer(k, k) / (twoW + 1e-12)
    np.fill_diagonal(Q, 0)
    return Q


# =========================
# 模块度 Q 值计算
# =========================
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
    eps = 1e-7
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))


# =========================
# 单层训练（不改原始逻辑）
# =========================
def train_one_layer(
    X, Q, Z, d_h, device,
    epochs=400, batch_size=32, lr=1e-2,
    rho=0.05, lam_sparse=1e-4, layer_index=1
):
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

    for epoch in range(epochs):
        for (batch,) in loader:
            x_hat, h = ae(batch)
            recon_loss = mse(x_hat, batch)
            kl_term = kl_div(rho, h.mean(0)).sum()
            total_loss = recon_loss + lam_sparse * kl_term

            opt.zero_grad()
            total_loss.backward()
            opt.step()

        if CONFIG["print_every_epoch"] and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(f"      layer {layer_index} | epoch {epoch + 1}/{epochs}")

    with torch.no_grad():
        _, HX = ae(Xc)
        _, HQ = ae(Qc)
        _, HZ = ae(Zc)

        HX_np = HX.T.detach().cpu().numpy()
        HQ_np = HQ.T.detach().cpu().numpy()
        HZ_np = HZ.T.detach().cpu().numpy()

    del Xc, Qc, Zc, train_data, loader
    cleanup_memory()

    return HX_np, HQ_np, HZ_np, ae.cpu()


# =========================
# 命名
# =========================
def build_run_name(dataset_name, alpha, T, h, rho, lr, lam_sparse, batch_size, epochs):
    return (
        f"{CONFIG['experiment_tag']}_{dataset_name}"
        f"_a{alpha:.1f}"
        f"_T{int(T):02d}"
        f"_h{int(h):03d}"
        f"_rho{rho:.3f}"
        f"_lr{lr:.4g}"
        f"_lam{lam_sparse:.6f}"
        f"_bs{int(batch_size)}"
        f"_ep{int(epochs)}"
    )


# =========================
# 数据集预处理（只做一次）
# =========================
def prepare_dataset(input_path: Path):
    dataset_name = input_path.stem

    W = load_matrix(input_path)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)

    A = (W > 0).astype(int)
    Z = minmax_01(compute_Z(A))
    Qm = minmax_01(compute_modularity_matrix(W))

    n = W.shape[0]
    edges = int(np.count_nonzero(np.triu(W > 0, k=1)))
    density = 0.0 if n <= 1 else (2.0 * edges) / (n * (n - 1))

    return {
        "dataset_name": dataset_name,
        "W": W,
        "A": A,
        "Z": Z,
        "Qm": Qm,
        "n": n,
        "edges": edges,
        "density": density,
        "X_cache": {},
    }


# =========================
# 单组参数实验
# =========================
def run_single_experiment(
    prepared,
    alpha,
    T,
    h,
    rho,
    lr,
    lam_sparse,
    batch_size,
    epochs,
    k_min,
    k_max,
    k_n_init,
    combo_index,
    total_combos,
    device
):
    set_seed(CONFIG["seed"])

    dataset_name = prepared["dataset_name"]
    W = prepared["W"]
    A = prepared["A"]
    Z = prepared["Z"]
    Qm = prepared["Qm"]

    beta = 1.0 - alpha
    run_name = build_run_name(dataset_name, alpha, T, h, rho, lr, lam_sparse, batch_size, epochs)

    start_time = time.perf_counter()

    print(
        f"[{combo_index}/{total_combos}] {dataset_name} | "
        f"alpha={alpha:.1f}, beta={beta:.1f}, T={T}, h={h}, "
        f"rho={rho:.3f}, lr={lr:.4g}, lam_sparse={lam_sparse:.6f}, "
        f"batch_size={batch_size}, epochs={epochs}, "
        f"k_range=[{k_min},{k_max}], k_n_init={k_n_init}"
    )

    # ---------- 计算 X（Qm 和 Z 已预计算） ----------
    if alpha not in prepared["X_cache"]:
        X = compute_X(W, A, alpha=alpha, beta=beta)
        X = minmax_01(X)
        prepared["X_cache"][alpha] = X
    else:
        X = prepared["X_cache"][alpha]
    # ---------- 深度稀疏自编码器堆叠训练 ----------
    encoders = []
    X_t, Q_t, Z_t = X, Qm, Z

    for layer in range(T - 1):
        print(f"   -> training layer {layer + 1}/{T - 1}")
        HX, HQ, HZ, ae = train_one_layer(
            X_t, Q_t, Z_t,
            d_h=h,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            rho=rho,
            lam_sparse=lam_sparse,
            layer_index=layer + 1,
        )
        encoders.append(ae)
        X_t, Q_t, Z_t = HX, HQ, HZ

        del HX, HQ, HZ
        cleanup_memory()

    # ---------- 最终特征提取 ----------
    H = X.T
    H_tensor = None
    try:
        for ae in encoders:
            ae.eval()
            ae.to(device)
            H_tensor = torch.tensor(H, dtype=torch.float32, device=device)
            with torch.no_grad():
                _, H_hidden = ae(H_tensor)
            H = H_hidden.detach().cpu().numpy()

            del H_tensor, H_hidden
            ae.to("cpu")
            cleanup_memory()
            H_tensor = None
    finally:
        if H_tensor is not None:
            del H_tensor
        cleanup_memory()

    # ---------- K-means 聚类 ----------
    best_Q, best_k = -1e18, None
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(
            n_clusters=k,
            n_init=k_n_init,
            random_state=CONFIG["seed"]
        )
        labels = kmeans.fit_predict(H)
        Qv = modularity_score(W, labels)
        if Qv > best_Q:
            best_Q, best_k = float(Qv), int(k)

    elapsed = time.perf_counter() - start_time

    # ---------- 清理 ----------
    del X, X_t, Q_t, Z_t, H, encoders
    cleanup_memory()

    print(
        f"   -> done | best_k={best_k}, modularity={best_Q:.6f}, "
        f"time={elapsed:.2f}s"
    )

    return {
        "run_name": run_name,
        "dataset": dataset_name,
        "alpha": alpha,
        "beta": beta,
        "T": T,
        "h": h,
        "rho": rho,
        "lr": lr,
        "lam_sparse": lam_sparse,
        "epochs": epochs,
        "batch_size": batch_size,
        "k_min": k_min,
        "k_max": k_max,
        "k_n_init": k_n_init,
        "best_k": best_k,
        "modularity": best_Q,
        "time_seconds": elapsed,
    }


# =========================
# 主程序：联调
# =========================
def main():
    set_seed(CONFIG["seed"])
    device = get_device()
    print(f"Using device: {device}")

    out_root = Path(CONFIG["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print(f"Experiment: {CONFIG['experiment_tag']}")
    print(f"Device: {device}")
    print(f"Datasets: {CONFIG['input_paths']}")
    print(f"alpha_list: {CONFIG['alpha_list']}")
    print(f"T_list: {CONFIG['T_list']}")
    print(f"lam_sparse_list: {[float(x) for x in CONFIG['lam_sparse_list']]}")
    print(f"h_list: {CONFIG['h_list']}")
    print(f"rho_list: {CONFIG['rho_list']}")
    print(f"lr_list: {CONFIG['lr_list']}")
    print(f"epochs_list: {CONFIG['epochs_list']}")
    print(f"batch_size_list: {CONFIG['batch_size_list']}")
    print(f"k_min_list: {CONFIG['k_min_list']}")
    print(f"k_max_list: {CONFIG['k_max_list']}")
    print(f"k_n_init_list: {CONFIG['k_n_init_list']}")
    print("=" * 120)

    all_results = []
    global_t0 = time.perf_counter()

    search_space = list(itertools.product(
        CONFIG["alpha_list"],
        CONFIG["T_list"],
        CONFIG["lam_sparse_list"],
        CONFIG["h_list"],
        CONFIG["rho_list"],
        CONFIG["lr_list"],
        CONFIG["batch_size_list"],
        CONFIG["epochs_list"],
        CONFIG["k_min_list"],
        CONFIG["k_max_list"],
        CONFIG["k_n_init_list"],
    ))

    combos_per_dataset = len(search_space)

    for input_str in CONFIG["input_paths"]:
        dataset_t0 = time.perf_counter()
        input_path = Path(input_str)

        prepared = prepare_dataset(input_path)
        dataset_name = prepared["dataset_name"]

        print("\n" + "=" * 120)
        print(f"Dataset: {dataset_name}")
        print(f"Nodes: {prepared['n']} | Edges: {prepared['edges']} | Density: {prepared['density']:.6f}")
        print(f"Total combos for this dataset: {combos_per_dataset}")
        print("=" * 120)

        dataset_results = []

        for idx, (
            alpha, T, lam_sparse, h, rho, lr,
            batch_size, epochs, k_min, k_max, k_n_init
        ) in enumerate(search_space, start=1):

            result = run_single_experiment(
                prepared=prepared,
                alpha=float(alpha),
                T=int(T),
                h=int(h),
                rho=float(rho),
                lr=float(lr),
                lam_sparse=float(lam_sparse),
                batch_size=int(batch_size),
                epochs=int(epochs),
                k_min=int(k_min),
                k_max=int(k_max),
                k_n_init=int(k_n_init),
                combo_index=idx,
                total_combos=combos_per_dataset,
                device=device,
            )
            dataset_results.append(result)
            all_results.append(result)

        df_dataset = pd.DataFrame(dataset_results)
        if not df_dataset.empty:
            best_idx = df_dataset["modularity"].idxmax()
            df_dataset["is_best_for_dataset"] = False
            df_dataset.loc[best_idx, "is_best_for_dataset"] = True
        else:
            best_idx = None

        dataset_elapsed = time.perf_counter() - dataset_t0

        if best_idx is not None:
            best_row = df_dataset.loc[best_idx]
            print(
                f"\n*** BEST for {dataset_name} *** "
                f"alpha={best_row['alpha']:.1f}, beta={best_row['beta']:.1f}, "
                f"T={int(best_row['T'])}, h={int(best_row['h'])}, "
                f"rho={best_row['rho']:.3f}, lr={best_row['lr']:.4g}, "
                f"lam_sparse={best_row['lam_sparse']:.6f}, "
                f"batch_size={int(best_row['batch_size'])}, epochs={int(best_row['epochs'])}, "
                f"k_range=[{int(best_row['k_min'])},{int(best_row['k_max'])}], "
                f"k_n_init={int(best_row['k_n_init'])}, "
                f"best_k={int(best_row['best_k'])}, modularity={best_row['modularity']:.6f}"
            )
        print(f"Dataset {dataset_name} finished in {dataset_elapsed:.2f}s")

        # 单个数据集 Excel
        if CONFIG["save_excel"]:
            dataset_excel = out_root / f"{CONFIG['experiment_tag']}_{dataset_name}_summary.xlsx"
            with pd.ExcelWriter(dataset_excel, engine="openpyxl") as writer:
                df_dataset.to_excel(writer, sheet_name="results", index=False)

        del prepared, dataset_results, df_dataset
        cleanup_memory()

    global_t1 = time.perf_counter()

    print("\n" + "=" * 120)
    print(f"All {CONFIG['experiment_tag']} experiments finished.")
    print(f"Total time: {global_t1 - global_t0:.2f} seconds")
    print(f"Results saved in: {out_root}")
    print("=" * 120)


if __name__ == "__main__":
    main()