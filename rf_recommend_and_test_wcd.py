"""
Random Forest structure-aware WCD parameter recommendation + actual WCD test.

功能：
1. 从 ./model_input/model_input.xlsx 读取融合后的建模表；
2. 显式指定 TEST_DATASET；
3. 自动屏蔽该数据集对应的全部 720 行作为测试集；
4. 使用其他数据集训练 RandomForestRegressor；
5. 预测测试数据集所有候选参数组合的 score_sigmoid_signed；
6. 输出推荐参数集合；
7. 从推荐集合中选择一组“稳又好”的参数；
8. 读取根目录下 TEST_DATASET.txt；
9. 使用推荐参数实际运行 WCD，计算 modularity；
10. 随机抽取若干组参数实际运行 WCD，作为随机初始化/随机搜索对照；
11. 输出 Excel 汇总和 CSV 明细。

依赖：
    pip install pandas numpy scikit-learn torch openpyxl scipy
"""

from __future__ import annotations

import gc
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


# ============================================================
# 配置区：主要改这里
# ============================================================

CONFIG = {
    # 显式指定一个或多个测试数据集名称，不带 .txt。
    # 脚本会对列表中的每个数据集分别执行：
    # 屏蔽该数据集 720 行 -> 训练 RF -> 推荐参数 -> 实际 WCD 测试 -> 随机对照。
    "TEST_DATASETS": [
        #"celegans_edges",
        "lesmis",
        "football",
        "arenas-jazz_normalized",
        "polbooks_normalized",
        "soc-dolphins_normalized",
        "ucidata-zachary_normalized",
        "ia-primary-school-proximity",
        "ia-workplace-contacts",
        "ia-enron-only",
        "ia-infect-hyper",
        "eco-foodweb-baywet",
        "adjnoun",
        "er_dense_normalized",
        "er_sparse_normalized",
        "lfr_like_1_normalized",
        "lfr_like_2_normalized",
        "sbm_blurry_normalized",
        "sbm_clear_normalized",
        "brock200-3",
        "ca-netscience",
        "CAG_mat72",
        "ca-sandi_auths",
        "chesapeake",
        "eco-florida",
        "eco-mangwet",
        #"econ-wm3",
        "ENZYMES123",
        #"inf-USAir97",
        "insecta-ant-colony2",
        "johnson8-4-4",
        "reptilia-tortoise-network-bsv",
        "sociopatterns-hypertext",
        "SW-100-6-0d1-trial2",
        "synthetic_50_clear_four_blocks",
        "synthetic_50_core_periphery_bridge",
        "synthetic_100_fuzzy_mixed",
        "synthetic_100_hierarchical",
        "synthetic_150_hub_modular",
        #"synthetic_150_ring_bottleneck",
    ],

    # 兼容旧写法：如果 TEST_DATASETS 留空，则使用 TEST_DATASET。
    "TEST_DATASET": "football",

    # 建模输入文件。
    # 如果你的融合 xlsx 在根目录下的 fuckingresults 文件夹中，改成：
    # "fuckingresults/model_input.xlsx"
    "MODEL_INPUT_FILE": "model_input/model_input.xlsx",
    "MODEL_INPUT_SHEET": 0,

    # 测试集 txt 默认在根目录下。
    # 实际读取路径为 ./<dataset>.txt
    "DATASET_TXT_DIR": ".",

    # 输出文件夹：脚本会在根目录下自动新建这个文件夹。
    "OUTPUT_DIR": "rf_multi_dataset_recommendation",

    # 使用 PC1-PC8
    "PC_FEATURES": [f"PC{i}" for i in range(1, 9)],

    # 使用的 WCD 参数特征。
    # 不纳入 epochs、time_seconds、best_k、modularity、k_min、k_max、k_n_init。
    "PARAM_FEATURES": [
        "alpha",
        "beta",
        "T",
        "h",
        "rho",
        "lr",
        "lam_sparse",
        "batch_size",
    ],

    # 目标变量
    "TARGET_COL": "score_sigmoid_signed",

    # 数据集列名
    "DATASET_COL": "dataset",

    # 推荐比例：推荐预测得分最高的前 10%。
    "RECOMMEND_RATIO": 0.10,

    # 冒险指数：
    # 0.0 极稳健：更偏向推荐集合的参数共识；
    # 1.0 极激进：更偏向预测得分最高的参数；
    # 0.3~0.5 通常比较适合作为论文实验默认值。
    "RISK_INDEX": 0.3,

    # 随机实际运行 WCD 的次数。
    # 每次都会真正训练 WCD，次数太大就很慢；论文验证建议 5~20。
    "RANDOM_WCD_TRIALS": 5,

    # 随机种子
    "SEED": 0,

    # Random Forest 参数
    "RF_PARAMS": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 0,
        "n_jobs": -1,
    },

    # WCD 固定训练设置。
    # 如果 model_input 里面有 epochs/k_min/k_max/k_n_init，也会优先读取行内值；
    # 没有就用这里的默认值。
    "DEFAULT_EPOCHS": 600,
    "DEFAULT_K_MIN": 2,
    "DEFAULT_K_MAX": 14,
    "DEFAULT_K_N_INIT": 10,

    # 设备设置
    "FORCE_CPU": False,
    "PRINT_EVERY_EPOCH": False,

    # 是否保存每个推荐/随机实际运行结果
    "SAVE_EXCEL": True,

    # 是否生成多数据集汇总图
    "SAVE_PLOTS": True,
    "DPI": 300,
}


# ============================================================
# 可重复性设置
# ============================================================

def set_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if CONFIG["FORCE_CPU"]:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# ============================================================
# 矩阵加载：按你的 WCD 代码原封不动保留
# ============================================================

def load_matrix(path: Path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()

    if first.startswith("%%MatrixMarket"):
        from scipy.io import mmread
        A = mmread(path)
        return A.toarray().astype(float)
    else:
        return np.loadtxt(path).astype(float)


# ============================================================
# WCD 核心函数
# ============================================================

def minmax_01(M, eps=1e-12):
    mn, mx = M.min(), M.max()
    if abs(mx - mn) < eps:
        return np.zeros_like(M)
    return (M - mn) / (mx - mn)


def compute_X(W, A, alpha=0.5, beta=0.5):
    """
    根据论文算法 1 计算考虑二阶邻居的相似性矩阵 X。
    """
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
    Z = 0.5 * A + B
    return Z.astype(float)


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
    eps = 1e-7
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))


def train_one_layer(
    X,
    Q,
    Z,
    d_h,
    device,
    epochs=400,
    batch_size=32,
    lr=1e-2,
    rho=0.05,
    lam_sparse=1e-4,
    layer_index=1,
):
    d_in, n = X.shape

    Xc = torch.tensor(X.T, dtype=torch.float32, device=device)
    Qc = torch.tensor(Q.T, dtype=torch.float32, device=device)
    Zc = torch.tensor(Z.T, dtype=torch.float32, device=device)

    train_data = torch.cat([Xc, Qc, Zc], dim=0)
    loader = DataLoader(
        TensorDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
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

        if CONFIG["PRINT_EVERY_EPOCH"] and ((epoch + 1) % 50 == 0 or epoch == 0):
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


def run_wcd_once(
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
    device,
    tag="run",
):
    """
    使用一组参数实际运行一次 WCD。
    """
    set_seed(CONFIG["SEED"])

    dataset_name = prepared["dataset_name"]
    W = prepared["W"]
    A = prepared["A"]
    Z = prepared["Z"]
    Qm = prepared["Qm"]

    beta = 1.0 - alpha
    start_time = time.perf_counter()

    print(
        f"[{tag}] {dataset_name} | "
        f"alpha={alpha:.1f}, beta={beta:.1f}, T={T}, h={h}, "
        f"rho={rho:.3f}, lr={lr:.4g}, lam_sparse={lam_sparse:.6f}, "
        f"batch_size={batch_size}, epochs={epochs}, "
        f"k_range=[{k_min},{k_max}], k_n_init={k_n_init}"
    )

    if alpha not in prepared["X_cache"]:
        X = compute_X(W, A, alpha=alpha, beta=beta)
        X = minmax_01(X)
        prepared["X_cache"][alpha] = X
    else:
        X = prepared["X_cache"][alpha]

    encoders = []
    X_t, Q_t, Z_t = X, Qm, Z

    for layer in range(T - 1):
        print(f"   -> training layer {layer + 1}/{T - 1}")
        HX, HQ, HZ, ae = train_one_layer(
            X_t,
            Q_t,
            Z_t,
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

    best_Q, best_k = -1e18, None
    best_labels = None

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(
            n_clusters=k,
            n_init=k_n_init,
            random_state=CONFIG["SEED"],
        )
        labels = kmeans.fit_predict(H)
        Qv = modularity_score(W, labels)
        if Qv > best_Q:
            best_Q, best_k = float(Qv), int(k)
            best_labels = labels.copy()

    elapsed = time.perf_counter() - start_time

    del X, X_t, Q_t, Z_t, H, encoders
    cleanup_memory()

    print(
        f"   -> done | best_k={best_k}, modularity={best_Q:.6f}, "
        f"time={elapsed:.2f}s"
    )

    return {
        "dataset": dataset_name,
        "alpha": float(alpha),
        "beta": float(beta),
        "T": int(T),
        "h": int(h),
        "rho": float(rho),
        "lr": float(lr),
        "lam_sparse": float(lam_sparse),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "k_n_init": int(k_n_init),
        "best_k": int(best_k),
        "modularity": float(best_Q),
        "time_seconds": float(elapsed),
        "labels": best_labels,
    }


# ============================================================
# Random Forest 推荐部分
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_model_input(path: Path, sheet_name=0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到建模输入文件：{path}")

    if path.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"不支持的建模输入文件类型：{path}")


def check_columns(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "model_input 缺少必要列：\n"
            + "\n".join(f" - {c}" for c in missing)
            + "\n\n当前列：\n"
            + "\n".join(map(str, df.columns))
        )


def to_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def train_rf_and_predict_for_dataset(df: pd.DataFrame, test_dataset: str) -> tuple[pd.DataFrame, RandomForestRegressor, dict]:
    dataset_col = CONFIG["DATASET_COL"]
    target_col = CONFIG["TARGET_COL"]
    feature_cols = CONFIG["PC_FEATURES"] + CONFIG["PARAM_FEATURES"]

    required = [dataset_col, target_col] + feature_cols
    check_columns(df, required)

    df = df.copy()
    df[dataset_col] = df[dataset_col].astype(str).str.strip()

    df = to_numeric_cols(df, feature_cols + [target_col])

    before = len(df)
    df = df.dropna(subset=[dataset_col, target_col] + feature_cols).copy()
    after = len(df)

    if before != after:
        print(f"删除关键字段缺失样本：{before - after} 行")

    test_mask = df[dataset_col] == test_dataset
    test_df = df[test_mask].copy()
    train_df = df[~test_mask].copy()

    if test_df.empty:
        available = sorted(df[dataset_col].unique())
        raise ValueError(
            f"在 model_input 中找不到测试数据集：{test_dataset}\n"
            f"可用数据集示例：{available[:20]}"
        )

    if train_df.empty:
        raise ValueError("训练集为空。请确认 model_input 中包含多个数据集。")

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)

    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)

    rf = RandomForestRegressor(**CONFIG["RF_PARAMS"])
    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    test_pred = test_df.copy()
    test_pred["pred_score"] = pred
    test_pred["pred_rank_desc"] = test_pred["pred_score"].rank(method="first", ascending=False)

    metrics = {
        "test_dataset": test_dataset,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "pred_mae": mean_absolute_error(y_test, pred),
        "pred_rmse": rmse(y_test, pred),
        "pred_r2": r2_score(y_test, pred) if len(np.unique(y_test)) > 1 else np.nan,
        "pred_spearman": pd.Series(y_test).rank().corr(pd.Series(pred).rank()),
    }

    return test_pred, rf, metrics


def choose_stable_good_params(recommended_df: pd.DataFrame) -> pd.Series:
    """
    从推荐 Top-r 集合中选择一组参数。

    使用 RISK_INDEX 控制“稳健性”和“上限追求”之间的权衡：
    - RISK_INDEX = 0.0：完全偏稳健，优先选择最接近推荐集合参数共识的组合；
    - RISK_INDEX = 1.0：完全偏激进，优先选择预测得分最高的组合；
    - 0.0 < RISK_INDEX < 1.0：同时考虑预测得分和参数稳定性。
    """
    param_cols = CONFIG["PARAM_FEATURES"]
    risk = float(CONFIG.get("RISK_INDEX", 0.5))

    if not (0.0 <= risk <= 1.0):
        raise ValueError(f"RISK_INDEX 必须位于 [0, 1]，当前为：{risk}")

    temp = recommended_df.copy().reset_index(drop=True)

    # 1. 计算推荐集合中的参数共识，即每个参数的众数。
    consensus = {}
    for c in param_cols:
        consensus[c] = temp[c].mode(dropna=True).iloc[0]

    # 2. 计算每组参数与共识参数的匹配程度。
    match_count = np.zeros(len(temp), dtype=float)
    for c, v in consensus.items():
        match_count += (temp[c] == v).to_numpy(dtype=float)

    stability_score = match_count / max(len(param_cols), 1)

    # 3. 预测得分归一化到 [0, 1]。
    pred = pd.to_numeric(temp["pred_score"], errors="coerce").to_numpy(dtype=float)
    pred_min = np.nanmin(pred)
    pred_max = np.nanmax(pred)

    if abs(pred_max - pred_min) < 1e-12:
        pred_score_norm = np.ones_like(pred) * 0.5
    else:
        pred_score_norm = (pred - pred_min) / (pred_max - pred_min)

    # 4. 综合选择分数。
    final_select_score = risk * pred_score_norm + (1.0 - risk) * stability_score

    temp["consensus_match_count"] = match_count
    temp["stability_score"] = stability_score
    temp["pred_score_norm"] = pred_score_norm
    temp["risk_index"] = risk
    temp["final_select_score"] = final_select_score

    chosen = (
        temp.sort_values(
            by=["final_select_score", "pred_score", "stability_score"],
            ascending=[False, False, False],
        )
        .iloc[0]
    )

    return chosen


def row_to_wcd_params(row: pd.Series) -> dict:
    def get_int(name: str, default: int) -> int:
        if name in row and not pd.isna(row[name]):
            return int(row[name])
        return int(default)

    def get_float(name: str, default: float) -> float:
        if name in row and not pd.isna(row[name]):
            return float(row[name])
        return float(default)

    alpha = get_float("alpha", 0.5)

    return {
        "alpha": alpha,
        "T": get_int("T", 6),
        "h": get_int("h", 16),
        "rho": get_float("rho", 0.05),
        "lr": get_float("lr", 0.01),
        "lam_sparse": get_float("lam_sparse", 0.0001),
        "batch_size": get_int("batch_size", 16),
        "epochs": get_int("epochs", CONFIG["DEFAULT_EPOCHS"]),
        "k_min": get_int("k_min", CONFIG["DEFAULT_K_MIN"]),
        "k_max": get_int("k_max", CONFIG["DEFAULT_K_MAX"]),
        "k_n_init": get_int("k_n_init", CONFIG["DEFAULT_K_N_INIT"]),
    }


def run_actual_wcd_tests(
    test_dataset: str,
    test_pred: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    txt_path = Path(CONFIG["DATASET_TXT_DIR"]) / f"{test_dataset}.txt"

    if not txt_path.exists():
        raise FileNotFoundError(f"找不到测试集 txt 文件：{txt_path}")

    print(f"\nUsing device: {device}")
    print(f"Loading test graph: {txt_path}")

    prepared = prepare_dataset(txt_path)
    print(
        f"Dataset prepared: {prepared['dataset_name']} | "
        f"nodes={prepared['n']} | edges={prepared['edges']} | density={prepared['density']:.6f}"
    )

    recommend_ratio = float(CONFIG["RECOMMEND_RATIO"])
    rec_count = max(1, int(math.ceil(len(test_pred) * recommend_ratio)))

    recommended_df = (
        test_pred
        .sort_values("pred_score", ascending=False)
        .head(rec_count)
        .copy()
        .reset_index(drop=True)
    )

    chosen_row = choose_stable_good_params(recommended_df)
    chosen_params = row_to_wcd_params(chosen_row)

    print("\n" + "=" * 100)
    print("结构感知推荐集合参数共识：")
    for c in CONFIG["PARAM_FEATURES"]:
        mode_val = recommended_df[c].mode(dropna=True).iloc[0]
        print(f"  {c}: {mode_val}")

    print("\n最终选择的一组风险控制推荐参数：")
    print(f"  risk_index: {float(CONFIG.get('RISK_INDEX', 0.5))}")
    for k, v in chosen_params.items():
        print(f"  {k}: {v}")
    print("=" * 100)

    actual_rows = []

    # 推荐参数实际运行
    rec_result = run_wcd_once(
        prepared=prepared,
        device=device,
        tag="recommended_risk_controlled",
        **chosen_params,
    )

    rec_record = {
        "selection_type": "recommended_risk_controlled",
        "risk_index": float(CONFIG.get("RISK_INDEX", 0.5)),
        "pred_score": float(chosen_row["pred_score"]),
        "pred_score_norm": float(chosen_row.get("pred_score_norm", np.nan)),
        "stability_score": float(chosen_row.get("stability_score", np.nan)),
        "final_select_score": float(chosen_row.get("final_select_score", np.nan)),
        "consensus_match_count": float(chosen_row.get("consensus_match_count", np.nan)),
        "pred_rank_desc": float(chosen_row["pred_rank_desc"]),
        "true_score_in_grid": float(chosen_row[CONFIG["TARGET_COL"]]) if CONFIG["TARGET_COL"] in chosen_row else np.nan,
    }
    rec_record.update({k: v for k, v in rec_result.items() if k != "labels"})
    actual_rows.append(rec_record)

    # 随机参数实际运行
    rng = np.random.default_rng(CONFIG["SEED"])
    random_trials = int(CONFIG["RANDOM_WCD_TRIALS"])

    all_indices = np.arange(len(test_pred))
    random_indices = rng.choice(all_indices, size=min(random_trials, len(all_indices)), replace=False)

    random_selected_df = test_pred.iloc[random_indices].copy().reset_index(drop=True)

    for t, (_, rand_row) in enumerate(random_selected_df.iterrows(), start=1):
        rand_params = row_to_wcd_params(rand_row)

        rand_result = run_wcd_once(
            prepared=prepared,
            device=device,
            tag=f"random_{t}",
            **rand_params,
        )

        rand_record = {
            "selection_type": f"random_{t}",
            "risk_index": np.nan,
            "pred_score": float(rand_row["pred_score"]),
            "pred_rank_desc": float(rand_row["pred_rank_desc"]),
            "true_score_in_grid": float(rand_row[CONFIG["TARGET_COL"]]) if CONFIG["TARGET_COL"] in rand_row else np.nan,
        }
        rand_record.update({k: v for k, v in rand_result.items() if k != "labels"})
        actual_rows.append(rand_record)

    actual_results_df = pd.DataFrame(actual_rows)

    # 推荐集合明细
    recommended_detail_path = output_dir / f"{test_dataset}_recommended_top_{int(recommend_ratio * 100)}pct.csv"
    recommended_df.to_csv(recommended_detail_path, index=False, encoding="utf-8-sig")

    return actual_results_df, recommended_df


def build_dataset_summary(test_dataset: str, actual_results_df: pd.DataFrame) -> pd.DataFrame:
    recommended_mask = actual_results_df["selection_type"] == "recommended_risk_controlled"
    if not recommended_mask.any():
        raise ValueError(f"{test_dataset} 中找不到 recommended_risk_controlled 结果。")

    recommended_row = actual_results_df.loc[recommended_mask].iloc[0]
    recommended_mod = float(recommended_row["modularity"])

    random_df = actual_results_df[actual_results_df["selection_type"].str.startswith("random_")].copy()

    summary = {
        "test_dataset": test_dataset,
        "risk_index": float(CONFIG.get("RISK_INDEX", 0.5)),
        "recommend_ratio": float(CONFIG["RECOMMEND_RATIO"]),
        "random_trials": int(len(random_df)),
        "recommended_modularity": recommended_mod,
        "random_mean_modularity": float(random_df["modularity"].mean()) if not random_df.empty else np.nan,
        "random_max_modularity": float(random_df["modularity"].max()) if not random_df.empty else np.nan,
        "random_min_modularity": float(random_df["modularity"].min()) if not random_df.empty else np.nan,
        "random_std_modularity": float(random_df["modularity"].std(ddof=1)) if len(random_df) > 1 else 0.0,
        "recommended_minus_random_mean": float(recommended_mod - random_df["modularity"].mean()) if not random_df.empty else np.nan,
        "recommended_minus_random_max": float(recommended_mod - random_df["modularity"].max()) if not random_df.empty else np.nan,
        "recommended_rank_in_actual_tests": int(
            actual_results_df["modularity"].rank(method="min", ascending=False).loc[recommended_mask].iloc[0]
        ),
        "actual_test_count": int(len(actual_results_df)),
        "recommended_alpha": float(recommended_row["alpha"]),
        "recommended_beta": float(recommended_row["beta"]),
        "recommended_T": int(recommended_row["T"]),
        "recommended_h": int(recommended_row["h"]),
        "recommended_rho": float(recommended_row["rho"]),
        "recommended_lr": float(recommended_row["lr"]),
        "recommended_lam_sparse": float(recommended_row["lam_sparse"]),
        "recommended_batch_size": int(recommended_row["batch_size"]),
        "recommended_best_k": int(recommended_row["best_k"]),
        "recommended_pred_score": float(recommended_row.get("pred_score", np.nan)),
        "recommended_stability_score": float(recommended_row.get("stability_score", np.nan)),
        "recommended_final_select_score": float(recommended_row.get("final_select_score", np.nan)),
    }

    return pd.DataFrame([summary])


def save_outputs_for_one_dataset(
    test_dataset: str,
    output_dir: Path,
    test_pred: pd.DataFrame,
    rf: RandomForestRegressor,
    metrics: dict,
    actual_results_df: pd.DataFrame,
    recommended_df: pd.DataFrame,
) -> pd.DataFrame:
    feature_cols = CONFIG["PC_FEATURES"] + CONFIG["PARAM_FEATURES"]

    # 排序预测明细
    pred_sorted = test_pred.sort_values("pred_score", ascending=False).copy()
    pred_sorted.to_csv(output_dir / f"{test_dataset}_all_predicted_params.csv", index=False, encoding="utf-8-sig")

    # 特征重要性
    feature_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    feature_importance_df.to_csv(output_dir / f"{test_dataset}_feature_importance.csv", index=False, encoding="utf-8-sig")

    # 预测指标
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / f"{test_dataset}_rf_prediction_metrics.csv", index=False, encoding="utf-8-sig")

    # 实际 WCD 结果
    actual_results_df.to_csv(output_dir / f"{test_dataset}_actual_wcd_comparison.csv", index=False, encoding="utf-8-sig")

    # 简要对比
    summary_df = build_dataset_summary(test_dataset, actual_results_df)
    summary_df.to_csv(output_dir / f"{test_dataset}_actual_comparison_summary.csv", index=False, encoding="utf-8-sig")

    if CONFIG["SAVE_EXCEL"]:
        excel_path = output_dir / f"{test_dataset}_rf_recommendation_and_wcd_test.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            metrics_df.to_excel(writer, sheet_name="rf_prediction_metrics", index=False)
            actual_results_df.to_excel(writer, sheet_name="actual_wcd_comparison", index=False)
            recommended_df.to_excel(writer, sheet_name="recommended_top", index=False)
            pred_sorted.to_excel(writer, sheet_name="all_predicted_params", index=False)
            feature_importance_df.to_excel(writer, sheet_name="feature_importance", index=False)

            for ws in writer.book.worksheets:
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
                for col_cells in ws.columns:
                    max_len = 0
                    col_letter = col_cells[0].column_letter
                    for cell in col_cells[:2000]:
                        if cell.value is None:
                            continue
                        max_len = max(max_len, len(str(cell.value)))
                    ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 32)

        print(f"\nExcel 汇总已保存：{excel_path}")

    print("\n" + "=" * 100)
    print(f"{test_dataset} 实际 WCD 对比摘要")
    print("=" * 100)
    for k, v in summary_df.iloc[0].to_dict().items():
        print(f"{k}: {v}")
    print("=" * 100)

    return summary_df


def plot_overall_results(overall_summary_df: pd.DataFrame, output_root: Path) -> None:
    if overall_summary_df.empty or not CONFIG.get("SAVE_PLOTS", True):
        return

    plot_dir = ensure_dir(output_root / "plots")
    dpi = int(CONFIG.get("DPI", 300))

    df = overall_summary_df.copy()
    df = df.sort_values("test_dataset").reset_index(drop=True)

    # 图 1：推荐参数 vs 随机均值 modularity 折线图
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.75), 5.2))
    ax.plot(df["test_dataset"], df["recommended_modularity"], marker="o", label="Structure-aware recommendation")
    ax.plot(df["test_dataset"], df["random_mean_modularity"], marker="o", label="Random mean")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Modularity")
    ax.set_title("Recommended vs Random Modularity")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(plot_dir / "recommended_vs_random_modularity.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 图 2：推荐相对随机均值提升量柱状图
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.75), 5.2))
    ax.bar(df["test_dataset"], df["recommended_minus_random_mean"])
    ax.axhline(0, linewidth=1.0)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Recommended - Random Mean")
    ax.set_title("Improvement over Random Mean")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(plot_dir / "improvement_over_random_mean.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\n汇总图已保存到：{plot_dir.resolve()}")


def get_test_datasets() -> list[str]:
    datasets = CONFIG.get("TEST_DATASETS", None)
    if datasets:
        return [str(x).strip() for x in datasets if str(x).strip()]

    one = str(CONFIG.get("TEST_DATASET", "")).strip()
    if not one:
        raise ValueError("请在 CONFIG 中设置 TEST_DATASETS 或 TEST_DATASET。")
    return [one]


def run_one_dataset(df: pd.DataFrame, test_dataset: str, output_root: Path, device: torch.device) -> pd.DataFrame:
    dataset_output_dir = ensure_dir(output_root / test_dataset)

    print("\n" + "#" * 110)
    print(f"Start dataset: {test_dataset}")
    print(f"Dataset output dir: {dataset_output_dir.resolve()}")
    print("#" * 110)

    test_pred, rf, metrics = train_rf_and_predict_for_dataset(df, test_dataset)

    print("\nRF 预测指标：")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    actual_results_df, recommended_df = run_actual_wcd_tests(
        test_dataset=test_dataset,
        test_pred=test_pred,
        output_dir=dataset_output_dir,
        device=device,
    )

    summary_df = save_outputs_for_one_dataset(
        test_dataset=test_dataset,
        output_dir=dataset_output_dir,
        test_pred=test_pred,
        rf=rf,
        metrics=metrics,
        actual_results_df=actual_results_df,
        recommended_df=recommended_df,
    )

    return summary_df


def main():
    set_seed(CONFIG["SEED"])

    test_datasets = get_test_datasets()
    model_input_path = Path(CONFIG["MODEL_INPUT_FILE"])
    output_root = ensure_dir(Path(CONFIG["OUTPUT_DIR"]))
    device = get_device()

    print("=" * 110)
    print("Random Forest WCD Parameter Recommendation + Multi-Dataset Actual WCD Test")
    print(f"TEST_DATASETS: {test_datasets}")
    print(f"MODEL_INPUT_FILE: {model_input_path}")
    print(f"OUTPUT_ROOT: {output_root.resolve()}")
    print(f"DEVICE: {device}")
    print(f"RISK_INDEX: {CONFIG.get('RISK_INDEX', 0.5)}")
    print(f"RANDOM_WCD_TRIALS: {CONFIG['RANDOM_WCD_TRIALS']}")
    print("=" * 110)

    df = read_model_input(model_input_path, CONFIG["MODEL_INPUT_SHEET"])

    all_summaries = []
    failed_rows = []

    for test_dataset in test_datasets:
        dataset_t0 = time.perf_counter()
        try:
            summary_df = run_one_dataset(df, test_dataset, output_root, device)
            summary_df["elapsed_seconds_for_dataset"] = time.perf_counter() - dataset_t0
            all_summaries.append(summary_df)
        except Exception as exc:
            failed_rows.append({
                "test_dataset": test_dataset,
                "error": repr(exc),
            })
            print("\n" + "!" * 110)
            print(f"Dataset failed: {test_dataset}")
            print(repr(exc))
            print("!" * 110)

    overall_summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    failed_df = pd.DataFrame(failed_rows)

    if not overall_summary_df.empty:
        overall_csv = output_root / "overall_actual_comparison_summary.csv"
        overall_summary_df.to_csv(overall_csv, index=False, encoding="utf-8-sig")

        overall_excel = output_root / "overall_rf_recommendation_summary.xlsx"
        with pd.ExcelWriter(overall_excel, engine="openpyxl") as writer:
            overall_summary_df.to_excel(writer, sheet_name="overall_summary", index=False)
            if not failed_df.empty:
                failed_df.to_excel(writer, sheet_name="failed", index=False)

            for ws in writer.book.worksheets:
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
                for col_cells in ws.columns:
                    max_len = 0
                    col_letter = col_cells[0].column_letter
                    for cell in col_cells[:2000]:
                        if cell.value is None:
                            continue
                        max_len = max(max_len, len(str(cell.value)))
                    ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 32)

        plot_overall_results(overall_summary_df, output_root)

        print("\n" + "=" * 110)
        print("多数据集验证完成。")
        print(f"Overall CSV: {overall_csv.resolve()}")
        print(f"Overall Excel: {overall_excel.resolve()}")
        print("=" * 110)

        key_cols = [
            "test_dataset",
            "recommended_modularity",
            "random_mean_modularity",
            "recommended_minus_random_mean",
            "recommended_minus_random_max",
        ]
        print("\n核心结果预览：")
        print(overall_summary_df[[c for c in key_cols if c in overall_summary_df.columns]].to_string(index=False))

    if not failed_df.empty:
        failed_path = output_root / "failed_datasets.csv"
        failed_df.to_csv(failed_path, index=False, encoding="utf-8-sig")
        print(f"\n部分数据集失败，已保存失败信息：{failed_path.resolve()}")

    if overall_summary_df.empty and not failed_df.empty:
        raise RuntimeError("所有测试数据集均失败，请查看 failed_datasets.csv。")

    print("\nDone.")


if __name__ == "__main__":
    main()
