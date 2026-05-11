
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

    "n_components": None,

    "drop_low_variance": True,
    "low_variance_threshold": 1e-12,

    # 图像设置
    "annotate_points": True,
    "dpi": 300,

    # 新增：PCA 散点图美化设置
    # label_mode:
    #   "outliers" 只标注离群点，推荐论文正文使用
    #   "all"      标注所有点，适合检查
    #   "none"     不标注
    "label_mode": "outliers",

    # 标注 PC1-PC2 平面中距离中心最远的若干个点
    "top_label_count": 10,

    # 是否去掉数据集名末尾的 .txt
    "remove_txt_suffix": True,

    # 是否额外保存 PDF，适合插入 Word 后保持清晰
    "save_pdf": True,
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

def beautify_dataset_name(name: str, remove_txt_suffix: bool = True) -> str:
    """
    美化数据集名称：
    1. 去掉 .txt 后缀；
    2. 名称太长时适当截断，避免图中标签过长。
    """
    name = str(name)

    if remove_txt_suffix and name.endswith(".txt"):
        name = name[:-4]

    max_len = 28
    if len(name) > max_len:
        name = name[:max_len - 3] + "..."

    return name


def choose_labels_for_scatter(
    scores_with_name: pd.DataFrame,
    label_mode: str = "outliers",
    top_label_count: int = 10,
) -> pd.Series:
    """
    决定哪些点需要标注。

    outliers 模式：
    计算每个点到 PC1-PC2 平面中心的距离，只标注距离最大的若干个点。
    """
    if label_mode == "none":
        return pd.Series(False, index=scores_with_name.index)

    if label_mode == "all":
        return pd.Series(True, index=scores_with_name.index)

    if label_mode == "outliers":
        pc1 = scores_with_name["PC1"]
        pc2 = scores_with_name["PC2"]

        center_pc1 = pc1.median()
        center_pc2 = pc2.median()

        dist = np.sqrt((pc1 - center_pc1) ** 2 + (pc2 - center_pc2) ** 2)
        label_indices = dist.sort_values(ascending=False).head(top_label_count).index

        mask = pd.Series(False, index=scores_with_name.index)
        mask.loc[label_indices] = True
        return mask

    raise ValueError(f"未知 label_mode: {label_mode}")
def plot_pc1_pc2(
    scores_with_name: pd.DataFrame,
    out_path: Path,
    annotate=True,
    dpi=300,
    label_mode="outliers",
    top_label_count=10,
    remove_txt_suffix=True,
    save_pdf=True,
):
    """
    绘制 PC1-PC2 散点图。

    改进点：
    1. 默认只标注离群点，避免文字重叠；
    2. 使用更适合论文的尺寸、字体和网格；
    3. 去掉 .txt 后缀；
    4. 同时保存 PNG 和 PDF；
    5. 对中心密集点保持低干扰显示。
    """
    if "PC1" not in scores_with_name.columns or "PC2" not in scores_with_name.columns:
        return

    df_plot = scores_with_name.copy()

    # 基础绘图风格
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(8.2, 4.1))

    pc1 = df_plot["PC1"]
    pc2 = df_plot["PC2"]

    # 根据到中心的距离区分普通点和离群点
    center_pc1 = pc1.median()
    center_pc2 = pc2.median()
    dist = np.sqrt((pc1 - center_pc1) ** 2 + (pc2 - center_pc2) ** 2)

    outlier_threshold = dist.quantile(0.85)
    is_outlier = dist >= outlier_threshold

    # 普通点
    ax.scatter(
        pc1[~is_outlier],
        pc2[~is_outlier],
        s=42,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.6,
        label="General datasets",
    )

    # 离群点
    ax.scatter(
        pc1[is_outlier],
        pc2[is_outlier],
        s=70,
        alpha=0.92,
        edgecolors="black",
        linewidths=0.7,
        label="Structurally distinctive datasets",
    )

    # 坐标轴零线
    ax.axhline(0, linewidth=1.0, linestyle="--", alpha=0.45)
    ax.axvline(0, linewidth=1.0, linestyle="--", alpha=0.45)

    # 标注策略
    if annotate:
        label_mask = choose_labels_for_scatter(
            df_plot,
            label_mode=label_mode,
            top_label_count=top_label_count,
        )

        # 几组偏移，避免所有文字都贴在点右上角
        offsets = [
            (6, 6), (6, -10), (-8, 8), (-8, -12),
            (10, 0), (-10, 0), (0, 10), (0, -14),
        ]

        label_rows = df_plot[label_mask].copy()
        for idx, (_, row) in enumerate(label_rows.iterrows()):
            label = beautify_dataset_name(
                row["dataset"],
                remove_txt_suffix=remove_txt_suffix,
            )
            dx, dy = offsets[idx % len(offsets)]

            ax.annotate(
                label,
                xy=(row["PC1"], row["PC2"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9.5,
                alpha=0.92,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                arrowprops=dict(
                    arrowstyle="-",
                    linewidth=0.6,
                    alpha=0.45,
                    shrinkA=0,
                    shrinkB=4,
                ),
            )

    # 标题和标签：论文里建议中文标题放图题，这里图内标题简洁一点
    ax.set_title("PCA Projection of Network Structure Features", pad=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # 网格和边框
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 图例
    ax.legend(frameon=False, loc="best")

    # 留白
    x_margin = (pc1.max() - pc1.min()) * 0.08
    y_margin = (pc2.max() - pc2.min()) * 0.08
    ax.set_xlim(pc1.min() - x_margin, pc1.max() + x_margin)
    ax.set_ylim(pc2.min() - y_margin, pc2.max() + y_margin)

    fig.tight_layout()

    # 保存 PNG
    fig.savefig(out_path, bbox_inches="tight")

    # 保存 PDF
    if save_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)


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
        dpi=CONFIG["dpi"],
        label_mode=CONFIG["label_mode"],
        top_label_count=CONFIG["top_label_count"],
        remove_txt_suffix=CONFIG["remove_txt_suffix"],
        save_pdf=CONFIG["save_pdf"],
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
