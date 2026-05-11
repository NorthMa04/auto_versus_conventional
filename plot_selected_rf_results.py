from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 配置区
# ============================================================

CONFIG = {
    # 输入汇总结果
    "input_file": "finalresults/overall_rf_recommendation_summary.xlsx",

    # 输出文件夹：会在根目录下自动新建
    "output_dir": "final_selected_plots",

    # Excel sheet，通常是 overall_summary
    # 如果报错，可以改成 0 读取第一个 sheet
    "sheet_name": "overall_summary",

    # 你最终挑选的数据集，顺序就是作图顺序
    "selected_datasets": [
        "sociopatterns-hypertext",
        "adjnoun",
        "synthetic_150_hub_modular",
        "ia-workplace-contacts",
        "chesapeake",
        "lfr_like_1_normalized",
        "er_sparse_normalized",
        "soc-dolphins_normalized",
        "ca-sandi_auths",
        "lesmis",
        "football",
    ],

    # 是否按推荐 - 随机平均提升量重新排序
    # False：按 selected_datasets 的顺序画
    # True：按 recommended_minus_random_mean 从高到低画
    "sort_by_improvement": False,

    # 图像设置
    "dpi": 300,
    "figsize_compare": (12.5, 5.8),
    "figsize_improve": (12.5, 5.4),

    # 是否同时保存 PDF
    "save_pdf": True,

    # 是否在柱子上标数值
    "annotate_values": True,
}


# ============================================================
# 工具函数
# ============================================================

def short_name(name: str) -> str:
    """
    缩短部分过长数据集名称，避免横轴太挤。
    """
    mapping = {
        "sociopatterns-hypertext": "sociopatterns",
        "synthetic_150_hub_modular": "syn_150_hub",
        "ia-workplace-contacts": "workplace",
        "lfr_like_1_normalized": "lfr_like_1",
        "er_sparse_normalized": "er_sparse",
        "soc-dolphins_normalized": "dolphins",
        "ca-sandi_auths": "ca-sandi",
    }
    return mapping.get(name, name)


def read_summary(input_path: Path, sheet_name):
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path.resolve()}")

    if input_path.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
        try:
            df = pd.read_excel(input_path, sheet_name=sheet_name)
        except ValueError:
            print(f"读取 sheet={sheet_name!r} 失败，改为读取第一个 sheet。")
            df = pd.read_excel(input_path, sheet_name=0)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"不支持的文件类型：{input_path}")

    return df


def check_columns(df: pd.DataFrame):
    required_cols = [
        "test_dataset",
        "recommended_modularity",
        "random_mean_modularity",
        "recommended_minus_random_mean",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "输入表缺少必要列：\n"
            + "\n".join(f" - {c}" for c in missing)
            + "\n\n当前列为：\n"
            + "\n".join(map(str, df.columns))
        )


def filter_selected(df: pd.DataFrame, selected_datasets: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["test_dataset"] = df["test_dataset"].astype(str).str.strip()

    selected_set = set(selected_datasets)

    missing = [x for x in selected_datasets if x not in set(df["test_dataset"])]
    if missing:
        print("\n警告：以下数据集没有在汇总表中找到，将跳过：")
        for x in missing:
            print(f" - {x}")

    out = df[df["test_dataset"].isin(selected_set)].copy()

    # 按 selected_datasets 的顺序排列
    order_map = {name: i for i, name in enumerate(selected_datasets)}
    out["plot_order"] = out["test_dataset"].map(order_map)
    out = out.sort_values("plot_order").reset_index(drop=True)

    return out


def setup_matplotlib():
    """
    尽量使用论文友好的字体和风格。
    """
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.unicode_minus": False,
    })


# ============================================================
# 作图函数 1：推荐 vs 随机平均
# ============================================================

def plot_recommended_vs_random(df: pd.DataFrame, out_dir: Path):
    labels = [short_name(x) for x in df["test_dataset"]]
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=CONFIG["figsize_compare"])

    bars1 = ax.bar(
        x - width / 2,
        df["recommended_modularity"],
        width,
        label="Recommended",
    )
    bars2 = ax.bar(
        x + width / 2,
        df["random_mean_modularity"],
        width,
        label="Random mean",
    )

    ax.set_ylabel("Modularity")
    ax.set_xlabel("Dataset")
    ax.set_title("Recommended Strategy vs Random Search Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)

    if CONFIG["annotate_values"]:
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.tight_layout()

    png_path = out_dir / "fig_recommended_vs_random_mean.png"
    fig.savefig(png_path, dpi=CONFIG["dpi"], bbox_inches="tight")

    if CONFIG["save_pdf"]:
        pdf_path = out_dir / "fig_recommended_vs_random_mean.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"已保存：{png_path}")


# ============================================================
# 作图函数 2：提升量柱状图
# ============================================================

def plot_improvement(df: pd.DataFrame, out_dir: Path):
    labels = [short_name(x) for x in df["test_dataset"]]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=CONFIG["figsize_improve"])

    bars = ax.bar(
        x,
        df["recommended_minus_random_mean"],
        width=0.62,
    )

    ax.axhline(0, linewidth=1.0)
    ax.set_ylabel("Recommended - Random mean")
    ax.set_xlabel("Dataset")
    ax.set_title("Improvement over Random Search Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if CONFIG["annotate_values"]:
        for bar in bars:
            height = bar.get_height()
            offset = 3 if height >= 0 else -12
            va = "bottom" if height >= 0 else "top"
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=8,
            )

    fig.tight_layout()

    png_path = out_dir / "fig_improvement_over_random_mean.png"
    fig.savefig(png_path, dpi=CONFIG["dpi"], bbox_inches="tight")

    if CONFIG["save_pdf"]:
        pdf_path = out_dir / "fig_improvement_over_random_mean.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"已保存：{png_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    setup_matplotlib()

    input_path = Path(CONFIG["input_file"])
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_summary(input_path, CONFIG["sheet_name"])
    check_columns(df)

    selected_df = filter_selected(df, CONFIG["selected_datasets"])

    if selected_df.empty:
        raise RuntimeError("筛选后没有任何数据，请检查 selected_datasets 和表中的 test_dataset 是否一致。")

    # 转为数值，防止 Excel 读入成字符串
    numeric_cols = [
        "recommended_modularity",
        "random_mean_modularity",
        "recommended_minus_random_mean",
    ]
    for c in numeric_cols:
        selected_df[c] = pd.to_numeric(selected_df[c], errors="coerce")

    selected_df = selected_df.dropna(subset=numeric_cols).copy()

    if CONFIG["sort_by_improvement"]:
        selected_df = selected_df.sort_values(
            "recommended_minus_random_mean",
            ascending=False,
        ).reset_index(drop=True)

    # 保存一份筛选后的数据，便于核对和写论文表格
    selected_csv = out_dir / "selected_plot_data.csv"
    selected_df.to_csv(selected_csv, index=False, encoding="utf-8-sig")
    print(f"筛选后的作图数据已保存：{selected_csv}")

    print("\n作图数据预览：")
    preview_cols = [
        "test_dataset",
        "recommended_modularity",
        "random_mean_modularity",
        "recommended_minus_random_mean",
    ]
    print(selected_df[preview_cols].to_string(index=False))

    plot_recommended_vs_random(selected_df, out_dir)
    plot_improvement(selected_df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()