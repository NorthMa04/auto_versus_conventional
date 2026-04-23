
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


# =========================================================
# PCA for dataset_profile.csv
# 作用：
# 1. 自动读取 dataset_profile.csv
# 2. 仅对结构特征做 PCA（自动排除 dataset 列）
# 3. 标准化 + 缺失值填补
# 4. 输出：
#    - explained_variance.csv
#    - pca_scores.csv
#    - pca_loadings.csv
#    - feature_summary.csv
#    - scree_plot.png
#    - cumulative_explained_variance.png
#    - pca_scatter_pc1_pc2.png
# =========================================================


CONFIG = {
    "input_csv": "dataset_profile.csv",
    "output_dir": "pca_results",

    # 主成分个数：
    # None -> 自动取 min(样本数, 特征数)
    # 也可以改成固定整数，比如 5
    "n_components": None,

    # 是否删除方差几乎为 0 的列
    "drop_low_variance": True,
    "low_variance_threshold": 1e-12,

    # 是否在图上标注数据集名称
    "annotate_points": True,

    # 画图 DPI
    "dpi": 180,
}


def ensure_output_dir(path_str: str) -> Path:
    out_dir = Path(path_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def detect_feature_columns(df: pd.DataFrame):
    """
    自动识别可用于 PCA 的数值结构特征列。
    默认排除 dataset。
    """
    exclude_cols = {"dataset"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    return feature_cols


def drop_low_variance_features(X: pd.DataFrame, threshold: float):
    variances = X.var(axis=0, ddof=0)
    keep_cols = variances[variances > threshold].index.tolist()
    drop_cols = variances[variances <= threshold].index.tolist()
    return X[keep_cols].copy(), keep_cols, drop_cols, variances


def build_feature_summary(df_features: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "feature": df_features.columns,
        "missing_count": df_features.isna().sum().values,
        "mean": df_features.mean(numeric_only=True).values,
        "std": df_features.std(numeric_only=True).values,
        "min": df_features.min(numeric_only=True).values,
        "max": df_features.max(numeric_only=True).values,
    })
    return summary


def run_pca(df: pd.DataFrame, feature_cols, n_components=None):
    """
    执行：
    缺失值填补 -> 标准化 -> PCA
    """
    X_raw = df[feature_cols].copy()

    # 记录原始特征概况
    feature_summary = build_feature_summary(X_raw)

    # 缺失值填补
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_raw),
        columns=feature_cols,
        index=df.index
    )

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # PCA 主成分数自动确定
    max_components = min(X_scaled.shape[0], X_scaled.shape[1])
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X_scaled)

    # 得分矩阵（每个数据集在各主成分上的坐标）
    score_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    scores_df = pd.DataFrame(X_pca, columns=score_cols, index=df.index)

    # 载荷矩阵（原始特征对各主成分的贡献）
    # sklearn 的 components_ 行是主成分，列是原始特征
    loadings = pca.components_.T
    loadings_df = pd.DataFrame(loadings, index=feature_cols, columns=score_cols)

    # 方差解释
    evr_df = pd.DataFrame({
        "component": score_cols,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_)
    })

    return {
        "feature_summary": feature_summary,
        "scores_df": scores_df,
        "loadings_df": loadings_df,
        "evr_df": evr_df,
        "pca_model": pca,
        "imputer": imputer,
        "scaler": scaler,
        "X_imputed": X_imputed,
    }


def plot_scree(evr_df: pd.DataFrame, out_path: Path, dpi=180):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(evr_df) + 1), evr_df["explained_variance_ratio"], marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_cumulative_variance(evr_df: pd.DataFrame, out_path: Path, dpi=180):
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(evr_df) + 1),
        evr_df["cumulative_explained_variance_ratio"],
        marker="o"
    )
    plt.axhline(0.80, linestyle="--")
    plt.axhline(0.90, linestyle="--")
    plt.axhline(0.95, linestyle="--")
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Cumulative Explained Variance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_pc1_pc2(scores_with_name: pd.DataFrame, out_path: Path, annotate=True, dpi=180):
    if "PC1" not in scores_with_name.columns or "PC2" not in scores_with_name.columns:
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(scores_with_name["PC1"], scores_with_name["PC2"])

    if annotate:
        for _, row in scores_with_name.iterrows():
            label = str(row["dataset"])
            plt.annotate(label, (row["PC1"], row["PC2"]), fontsize=8, alpha=0.85)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Scatter: PC1 vs PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main():
    input_path = Path(CONFIG["input_csv"])
    out_dir = ensure_output_dir(CONFIG["output_dir"])

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path.resolve()}")

    df = pd.read_csv(input_path)

    if "dataset" not in df.columns:
        raise ValueError("dataset_profile.csv 中必须包含 'dataset' 列")

    print("=" * 80)
    print("Loaded file:", input_path.resolve())
    print("Shape:", df.shape)
    print("Columns:")
    for c in df.columns:
        print(" -", c)
    print("=" * 80)

    feature_cols = detect_feature_columns(df)
    print(f"自动识别到 {len(feature_cols)} 个数值结构特征。")

    X = df[feature_cols].copy()

    dropped_low_variance = []
    variances = X.var(axis=0, ddof=0)

    if CONFIG["drop_low_variance"]:
        X_filtered, kept_cols, dropped_low_variance, variances = drop_low_variance_features(
            X, threshold=CONFIG["low_variance_threshold"]
        )
        feature_cols = kept_cols
        print(f"保留 {len(feature_cols)} 个特征，删除 {len(dropped_low_variance)} 个低方差特征。")
        if dropped_low_variance:
            print("低方差特征：", dropped_low_variance)

    result = run_pca(
        df=df,
        feature_cols=feature_cols,
        n_components=CONFIG["n_components"]
    )

    scores_df = result["scores_df"].copy()
    scores_df.insert(0, "dataset", df["dataset"].values)

    loadings_df = result["loadings_df"].copy()
    evr_df = result["evr_df"].copy()
    feature_summary = result["feature_summary"].copy()

    # 保存文件
    feature_summary.to_csv(out_dir / "feature_summary.csv", index=False)
    evr_df.to_csv(out_dir / "explained_variance.csv", index=False)
    scores_df.to_csv(out_dir / "pca_scores.csv", index=False)
    loadings_df.to_csv(out_dir / "pca_loadings.csv", index=True)

    # 保存“每个主成分贡献最大的特征”
    top_loading_rows = []
    for pc in loadings_df.columns:
        temp = loadings_df[pc].abs().sort_values(ascending=False)
        for rank, (feat, val) in enumerate(temp.head(10).items(), start=1):
            top_loading_rows.append({
                "component": pc,
                "rank": rank,
                "feature": feat,
                "abs_loading": float(val),
                "signed_loading": float(loadings_df.loc[feat, pc]),
            })
    pd.DataFrame(top_loading_rows).to_csv(out_dir / "top10_features_per_pc.csv", index=False)

    # 画图
    plot_scree(evr_df, out_dir / "scree_plot.png", dpi=CONFIG["dpi"])
    plot_cumulative_variance(evr_df, out_dir / "cumulative_explained_variance.png", dpi=CONFIG["dpi"])
    plot_pc1_pc2(
        scores_with_name=scores_df,
        out_path=out_dir / "pca_scatter_pc1_pc2.png",
        annotate=CONFIG["annotate_points"],
        dpi=CONFIG["dpi"]
    )

    # 简要打印
    print("\nPCA 完成。")
    print("\n各主成分方差解释率：")
    print(evr_df.to_string(index=False))

    cum = evr_df["cumulative_explained_variance_ratio"].values
    for target in [0.80, 0.90, 0.95]:
        idx = np.where(cum >= target)[0]
        if len(idx) > 0:
            print(f"达到 {int(target*100)}% 累积解释率所需主成分数: {idx[0] + 1}")
        else:
            print(f"未达到 {int(target*100)}% 累积解释率")

    print("\nPC1 贡献最大的前5个特征：")
    pc1_top = loadings_df["PC1"].abs().sort_values(ascending=False).head(5)
    print(pc1_top.to_string())

    if "PC2" in loadings_df.columns:
        print("\nPC2 贡献最大的前5个特征：")
        pc2_top = loadings_df["PC2"].abs().sort_values(ascending=False).head(5)
        print(pc2_top.to_string())

    print(f"\n结果已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
