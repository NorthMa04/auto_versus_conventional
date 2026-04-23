from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans


# =========================================================
# PARAMETERS
# 你主要改这里
# =========================================================
CONFIG = {
    # 一次性跑多个数据集
    # 你自己通过注释控制，不再检查是否在 DATASETS 里
    "dataset_names": [
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
        #"sbm_clear_normalized.txt",
        "celegans_edges.txt",
    ],

    # k 搜索范围
    "k_min": 2,
    "k_max": 35,

    # KMeans 参数
    "k_n_init": 10,
    "random_state": 0,

    # 是否打印每个 k 的 modularity
    "verbose": True,
}


ROOT_DIR = Path(".")
RESULT_DIR = ROOT_DIR / "baseline_results"


# =========================================================
# MatrixMarket 读取
# =========================================================
def load_matrix_market(path: Path) -> np.ndarray:
    """
    读取标准 MatrixMarket coordinate 格式：

    例如：
    %%MatrixMarket matrix coordinate integer symmetric
    %%MatrixMarket matrix coordinate real general
    % 注释
    n n m
    i j w

    也兼容 pattern：
    i j
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"{path.name} 文件为空")

    header = lines[0].strip().lower()
    if not header.startswith("%%matrixmarket"):
        raise ValueError(f"{path.name} 不是标准 MatrixMarket 文件")

    is_symmetric = "symmetric" in header
    is_pattern = "pattern" in header

    # 跳过注释，找到尺寸行
    idx = 1
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line.startswith("%"):
            idx += 1
            continue
        break

    if idx >= len(lines):
        raise ValueError(f"{path.name} 缺少尺寸行")

    parts = lines[idx].split()
    if len(parts) != 3:
        raise ValueError(f"{path.name} 尺寸行格式错误，应为 n n m")

    nrows, ncols, nnz = map(int, parts)
    n = max(nrows, ncols)

    idx += 1
    W = np.zeros((n, n), dtype=float)

    # 读取边
    for line in lines[idx:]:
        line = line.strip()
        if not line or line.startswith("%"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        i = int(parts[0]) - 1
        j = int(parts[1]) - 1

        if not (0 <= i < n and 0 <= j < n):
            continue

        if is_pattern:
            w = 1.0
        else:
            if len(parts) < 3:
                raise ValueError(f"{path.name} 中存在缺失权重的边: {line}")
            w = float(parts[2])

        W[i, j] += w

        if is_symmetric and i != j:
            W[j, i] += w

    # general 统一转成无向图
    if not is_symmetric:
        W = 0.5 * (W + W.T)

    np.fill_diagonal(W, 0.0)
    return W.astype(float)


def prepare_W(path: Path) -> np.ndarray:
    W = load_matrix_market(path)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


# =========================================================
# 模块度
# =========================================================
def modularity_score(W: np.ndarray, labels: np.ndarray) -> float:
    m = W.sum() / 2.0
    if m <= 1e-12:
        return -1e18

    deg = W.sum(axis=1)
    n = W.shape[0]

    Qv = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Qv += W[i, j] - deg[i] * deg[j] / (2.0 * m + 1e-12)

    return Qv / (2.0 * m + 1e-12)


# =========================================================
# KMeans baseline
# =========================================================
def run_kmeans_baseline(
    W: np.ndarray,
    k_min: int,
    k_max: int,
    k_n_init: int,
    random_state: int,
    verbose: bool = True,
):
    """
    baseline:
    直接在原始图表示 W 上做 KMeans
    每个节点 = 一行特征
    """
    X = W.copy()

    best_Q = -1e18
    best_k = None
    best_labels = None
    per_k_results = []

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(
            n_clusters=k,
            n_init=k_n_init,
            random_state=random_state,
        )

        labels = kmeans.fit_predict(X)
        Qv = modularity_score(W, labels)

        per_k_results.append((k, float(Qv)))

        if verbose:
            print(f"k = {k:2d} | modularity = {Qv:.6f}")

        if Qv > best_Q:
            best_Q = float(Qv)
            best_k = int(k)
            best_labels = labels.copy()

    return best_k, best_Q, best_labels, per_k_results


# =========================================================
# 保存结果
# =========================================================
def save_result(
    dataset_name: str,
    n: int,
    edges: int,
    density: float,
    best_k: int,
    best_Q: float,
    best_labels: np.ndarray,
    per_k_results,
):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    stem = Path(dataset_name).stem
    out_path = RESULT_DIR / f"baseline_{stem}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("KMeans Baseline on Original Graph Representation\n")
        f.write(f"dataset = {dataset_name}\n")
        f.write(f"nodes = {n}\n")
        f.write(f"edges = {edges}\n")
        f.write(f"density = {density:.6f}\n")
        f.write(f"best_k = {best_k}\n")
        f.write(f"best_modularity = {best_Q:.6f}\n")
        f.write("\n")

        f.write("[k_search_results]\n")
        for k, q in per_k_results:
            f.write(f"k = {k}, modularity = {q:.6f}\n")

        f.write("\n")
        f.write("[best_labels]\n")
        for idx, label in enumerate(best_labels, start=1):
            f.write(f"{idx}\t{label}\n")

    return out_path


# =========================================================
# 单个数据集运行
# =========================================================
def run_one_dataset(dataset_name: str):
    data_path = ROOT_DIR / dataset_name
    if not data_path.exists():
        print(f"[SKIP] 未找到数据文件: {data_path}")
        return

    print("=" * 72)
    print("KMeans Baseline on Original Graph Representation")
    print(f"Dataset : {dataset_name}")
    print(f"k range : [{CONFIG['k_min']}, {CONFIG['k_max']}]")
    print("=" * 72)

    # 读取图
    W = prepare_W(data_path)

    n = W.shape[0]
    edges = int(np.count_nonzero(np.triu(W > 0, k=1)))
    density = 0.0 if n <= 1 else 2.0 * edges / (n * (n - 1))

    print(f"Nodes   : {n}")
    print(f"Edges   : {edges}")
    print(f"Density : {density:.6f}")
    print("-" * 72)

    best_k, best_Q, best_labels, per_k_results = run_kmeans_baseline(
        W=W,
        k_min=CONFIG["k_min"],
        k_max=CONFIG["k_max"],
        k_n_init=CONFIG["k_n_init"],
        random_state=CONFIG["random_state"],
        verbose=CONFIG["verbose"],
    )

    print("-" * 72)
    print("BEST RESULT")
    print(f"best_k          = {best_k}")
    print(f"best_modularity = {best_Q:.6f}")

    out_path = save_result(
        dataset_name=dataset_name,
        n=n,
        edges=edges,
        density=density,
        best_k=best_k,
        best_Q=best_Q,
        best_labels=best_labels,
        per_k_results=per_k_results,
    )

    print(f"saved_to        = {out_path}")
    print("=" * 72)
    print()


# =========================================================
# 主程序
# =========================================================
def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_names = CONFIG["dataset_names"]
    if not dataset_names:
        raise ValueError("CONFIG['dataset_names'] 不能为空")

    for dataset_name in dataset_names:
        try:
            run_one_dataset(dataset_name)
        except Exception as e:
            print(f"[ERROR] {dataset_name} 运行失败: {e}")
            print()


if __name__ == "__main__":
    main()