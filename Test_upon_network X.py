import numpy as np
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import networkx as nx
from community import community_louvain


# =========================
# 全部配置统一放在这里
# =========================
CONFIG = {
    # ----- experiment identity -----
    "experiment_tag": "louvain_Baseline",
    # 当前实验标签，会体现在输出目录名和汇总文件名中。

    # ----- reproducibility -----
    "seed": 0,
    # Louvain 的随机种子，用于固定随机初始化顺序。

    # ----- datasets -----
    "input_paths": [
        #"lesmis.txt",
        #"football.txt",
        #"celegans_edges.txt",
        #"hep-th.txt",   # 当前先不跑
        #"arenas-jazz_normalized.txt",
        #"email-Eu-core_normalized.txt",
        #"polbooks_normalized.txt"
        #"soc-dolphins_normalized.txt",
        #"ucidata-zachary_normalized.txt",
        #"lfr_like_2_normalized.txt",
        #"adjnoun.txt"
        #"ca-netscience.txt"
        #"eco-foodweb-baywet.txt"
        #"er_dense_normalized.txt",
        #"er_sparse_normalized.txt",
        #"ia-enron-only.txt"
        #"ia-infect-hyper.txt",
        #"ia-primary-school-proximity.txt",
        #"ia-workplace-contacts.txt"
        #"lfr_like_2_normalized.txt",
        #"sbm_blurry_normalized.txt",
        #"er_dense_normalized.txt",
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
        #"road-chesapeake.txt",
        #"sociopatterns-hypertext.txt",
        #"SW-100-6-0d1-trial2.txt"
        "synthetic_50_clear_four_blocks.txt",
        "synthetic_50_core_periphery_bridge.txt",
        "synthetic_100_fuzzy_mixed.txt",
        "synthetic_100_hierarchical.txt",
        "synthetic_150_hub_modular.txt",
        "synthetic_150_ring_bottleneck.txt",
    ],
    # 待测试的数据集列表。

    # ----- output -----
    "output_root": "louvain_baseline_experiments",
    # Louvain 对照实验输出根目录。
    # 这是一个新的文件夹，不和 WCD 的结果混在一起。

    # ----- Louvain params -----
    "resolution": 1.0,
    # Louvain 分辨率参数。
    # 默认 1.0。值更大通常会倾向于得到更多社区，
    # 值更小通常会倾向于得到更少社区。

    "weight_key": "weight",
    # 传给 Louvain 的边权属性名。
    # 当前构图时会把矩阵中的权值写到边属性 "weight" 中。

    # ----- visualization -----
    "figsize": (8, 6),
    "dpi": 300,
    "scatter_size": 60,

    # ----- export -----
    "save_excel": True,
}


# =========================
# 可重复性设置
# =========================
def set_seed(seed=0):
    np.random.seed(seed)


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
# 从加权邻接矩阵构图
# =========================
def matrix_to_weighted_graph(W):
    """
    将对称加权邻接矩阵 W 转为无向加权图。
    只添加上三角中的非零边，避免重复加边。
    """
    n = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            w = float(W[i, j])
            if w > 0:
                G.add_edge(i, j, weight=w)

    return G


# =========================
# 模块度 Q 值计算（与你当前 WCD 代码保持一致）
# =========================
def modularity_score(W, labels):
    """
    计算给定划分的模块度 Q。
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
# 输出辅助函数
# =========================
def build_run_name(dataset_name):
    return f"{CONFIG['experiment_tag']}_{dataset_name}"


def save_metrics_txt(path, metrics_dict):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in metrics_dict.items():
            f.write(f"{k} {v}\n")


# =========================
# 单个数据集实验
# =========================
def run_single_experiment(input_path: Path):
    set_seed(CONFIG["seed"])

    dataset_name = input_path.stem
    run_name = build_run_name(dataset_name)

    out_root = Path(CONFIG["output_root"])
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ---------- 加载并预处理 ----------
    W = load_matrix(input_path)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)

    n = W.shape[0]
    edges = int(np.count_nonzero(np.triu(W > 0, k=1)))
    density = 0.0 if n <= 1 else (2.0 * edges) / (n * (n - 1))

    # ---------- 构图 ----------
    graph_t0 = time.perf_counter()
    G = matrix_to_weighted_graph(W)
    graph_t1 = time.perf_counter()

    # ---------- Louvain ----------
    louvain_t0 = time.perf_counter()
    partition = community_louvain.best_partition(
        G,
        weight=CONFIG["weight_key"],
        resolution=CONFIG["resolution"],
        random_state=CONFIG["seed"],
    )
    louvain_t1 = time.perf_counter()

    # 将 partition(dict) 转成按节点顺序排列的 labels
    labels = np.array([partition[i] for i in range(n)], dtype=int)

    # 为了让标签更紧凑，重新映射成 0,1,2,...
    unique_labels = sorted(np.unique(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[x] for x in labels], dtype=int)

    num_communities = int(len(np.unique(labels)))

    # 用和 WCD 同样的 modularity_score 再算一次，便于横向对比
    modularity_custom = float(modularity_score(W, labels))

    # 同时也记录 python-louvain 自带模块度
    partition_dict = {i: int(labels[i]) for i in range(n)}
    modularity_louvain_api = float(
        community_louvain.modularity(
            partition_dict,
            G,
            weight=CONFIG["weight_key"]
        )
    )

    t1 = time.perf_counter()

    # ---------- 保存标签 ----------
    np.savetxt(out_dir / f"{run_name}_labels.txt", labels, fmt="%d")

    # ---------- 保存指标 ----------
    metrics = {
        "experiment_tag": CONFIG["experiment_tag"],
        "run_name": run_name,
        "dataset": dataset_name,
        "method": "louvain",
        "n": n,
        "edges": edges,
        "density": density,
        "seed": CONFIG["seed"],
        "resolution": CONFIG["resolution"],
        "weight_key": CONFIG["weight_key"],
        "num_communities": num_communities,
        "modularity_custom": modularity_custom,
        "modularity_louvain_api": modularity_louvain_api,
        "graph_build_time_seconds": graph_t1 - graph_t0,
        "louvain_time_seconds": louvain_t1 - louvain_t0,
        "time_seconds": t1 - t0,
    }

    save_metrics_txt(out_dir / f"{run_name}_metrics.txt", metrics)

    # ---------- 可视化 ----------
    # 用邻接矩阵做 PCA，仅作为一个简单展示方式，和你之前的风格尽量统一
    H2 = PCA(n_components=2).fit_transform(W)

    plt.figure(figsize=CONFIG["figsize"])
    plt.scatter(
        H2[:, 0], H2[:, 1],
        c=labels,
        cmap="tab10",
        s=CONFIG["scatter_size"],
        edgecolor="k"
    )
    plt.title(
        f"Louvain | {dataset_name} | Q={modularity_custom:.3f} | communities={num_communities}"
    )
    plt.savefig(out_dir / f"{run_name}_community.png", dpi=CONFIG["dpi"])
    plt.close()

    # ---------- 单次实验 Excel ----------
    if CONFIG["save_excel"]:
        excel_path = out_dir / f"{run_name}_results.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame([metrics]).to_excel(writer, sheet_name="summary", index=False)
            pd.DataFrame({
                "node": np.arange(n, dtype=int),
                "label": labels
            }).to_excel(writer, sheet_name="labels", index=False)

    print(f"[{run_name}] Finished. Q = {modularity_custom:.4f}, communities = {num_communities}")
    print(f"[{run_name}] Results saved in {out_dir}")

    return metrics


# =========================
# 主程序：多数据集批量实验
# =========================
def main():
    out_root = Path(CONFIG["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    global_t0 = time.perf_counter()

    for input_str in CONFIG["input_paths"]:
        input_path = Path(input_str)
        dataset_name = input_path.stem

        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)

        metrics = run_single_experiment(input_path)
        all_metrics.append(metrics)

    global_t1 = time.perf_counter()

    # ---------- 全局汇总 ----------
    df_all_metrics = pd.DataFrame(all_metrics)
    summary_prefix = CONFIG["experiment_tag"]

    df_all_metrics.to_csv(out_root / f"{summary_prefix}_summary.csv", index=False)

    if CONFIG["save_excel"]:
        with pd.ExcelWriter(out_root / f"{summary_prefix}_summary.xlsx", engine="openpyxl") as writer:
            df_all_metrics.to_excel(writer, sheet_name="summary", index=False)

    print("=" * 80)
    print(f"All {CONFIG['experiment_tag']} experiments finished.")
    print(f"Total time: {global_t1 - global_t0:.2f} seconds")
    print(f"Global summary saved in: {out_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()