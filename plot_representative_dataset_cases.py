from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 配置区
# ============================================================

CONFIG = {
    # 输入文件夹：你的代表性数据集 actual_wcd_comparison.csv 所在位置
    "input_dir": "finalresults",

    # 输出文件夹：会自动创建
    "output_dir": "final_representative_case_plots",

    # 代表性数据集
    "selected_datasets": [
        "lesmis",
        "sociopatterns-hypertext",
        "adjnoun",
        "soc-dolphins_normalized",
    ],

    # 文件命名规则
    "file_pattern": "{dataset}_actual_wcd_comparison.csv",

    # 输出图片名
    "output_name": "fig_representative_actual_wcd_comparison",

    # 图像设置
    "dpi": 300,
    "figsize": (12.0, 8.0),
    "save_pdf": True,

    # 是否在柱子上标数值
    "annotate_values": True,
}


# ============================================================
# 名称美化
# ============================================================

def short_dataset_name(name: str) -> str:
    mapping = {
        "lesmis": "Lesmis",
        "sociopatterns-hypertext": "Sociopatterns",
        "adjnoun": "Adjnoun",
        "soc-dolphins_normalized": "Dolphins",
    }
    return mapping.get(name, name)


def short_selection_name(name: str) -> str:
    name = str(name)

    if name == "recommended_risk_controlled":
        return "Recommended"

    if name.startswith("random_"):
        return name.replace("random_", "Random ")

    return name


# ============================================================
# 读取数据
# ============================================================

def read_one_dataset(input_dir: Path, dataset: str) -> pd.DataFrame:
    path = input_dir / CONFIG["file_pattern"].format(dataset=dataset)

    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path.resolve()}")

    df = pd.read_csv(path)

    required_cols = ["selection_type", "modularity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} 缺少必要列：{missing}\n"
            f"当前列为：{list(df.columns)}"
        )

    df = df.copy()
    df["dataset"] = dataset
    df["modularity"] = pd.to_numeric(df["modularity"], errors="coerce")
    df = df.dropna(subset=["modularity"])

    return df


def load_all_selected() -> pd.DataFrame:
    input_dir = Path(CONFIG["input_dir"])
    if not input_dir.exists():
        raise FileNotFoundError(f"找不到输入文件夹：{input_dir.resolve()}")

    dfs = []
    for dataset in CONFIG["selected_datasets"]:
        df_one = read_one_dataset(input_dir, dataset)
        dfs.append(df_one)

    return pd.concat(dfs, ignore_index=True)


# ============================================================
# 绘图
# ============================================================

def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.unicode_minus": False,
    })


def plot_representative_cases(all_df: pd.DataFrame):
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = CONFIG["selected_datasets"]

    fig, axes = plt.subplots(2, 2, figsize=CONFIG["figsize"])
    axes = axes.flatten()

    for ax, dataset in zip(axes, datasets):
        df = all_df[all_df["dataset"] == dataset].copy()

        # 推荐策略排第一，random 按编号排序
        def sort_key(x):
            x = str(x)
            if x == "recommended_risk_controlled":
                return 0
            if x.startswith("random_"):
                try:
                    return int(x.replace("random_", "")) + 1
                except ValueError:
                    return 99
            return 100

        df["sort_order"] = df["selection_type"].map(sort_key)
        df = df.sort_values("sort_order").reset_index(drop=True)

        x_labels = [short_selection_name(x) for x in df["selection_type"]]
        x = np.arange(len(df))
        y = df["modularity"].to_numpy(dtype=float)

        bars = ax.bar(x, y, width=0.62)

        # 推荐策略柱子加边框，便于突出
        for i, bar in enumerate(bars):
            if df.loc[i, "selection_type"] == "recommended_risk_controlled":
                bar.set_linewidth(1.6)
                bar.set_edgecolor("black")

        # 随机均值参考线
        random_df = df[df["selection_type"].astype(str).str.startswith("random_")]
        if not random_df.empty:
            random_mean = random_df["modularity"].mean()
            ax.axhline(
                random_mean,
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                label="Random mean",
            )

        # 数值标注
        if CONFIG["annotate_values"]:
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

        ax.set_title(short_dataset_name(dataset))
        ax.set_ylabel("Modularity")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if not random_df.empty:
            ax.legend(frameon=False, loc="best")

    # 如果少于 4 个数据集，隐藏多余子图
    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.suptitle(
        "Actual WCD Modularity on Representative Datasets",
        fontsize=15,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.965])

    png_path = output_dir / f"{CONFIG['output_name']}.png"
    fig.savefig(png_path, dpi=CONFIG["dpi"], bbox_inches="tight")

    if CONFIG["save_pdf"]:
        pdf_path = output_dir / f"{CONFIG['output_name']}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"已保存：{png_path}")


# ============================================================
# 额外输出汇总表，方便写论文
# ============================================================

def save_case_summary(all_df: pd.DataFrame):
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset in CONFIG["selected_datasets"]:
        df = all_df[all_df["dataset"] == dataset].copy()

        rec = df[df["selection_type"] == "recommended_risk_controlled"]
        rand = df[df["selection_type"].astype(str).str.startswith("random_")]

        if rec.empty:
            continue

        rec_mod = float(rec["modularity"].iloc[0])
        rand_mean = float(rand["modularity"].mean()) if not rand.empty else np.nan
        rand_max = float(rand["modularity"].max()) if not rand.empty else np.nan

        rows.append({
            "dataset": dataset,
            "recommended_modularity": rec_mod,
            "random_mean_modularity": rand_mean,
            "random_max_modularity": rand_max,
            "recommended_minus_random_mean": rec_mod - rand_mean if not np.isnan(rand_mean) else np.nan,
            "recommended_minus_random_max": rec_mod - rand_max if not np.isnan(rand_max) else np.nan,
            "random_trials": len(rand),
        })

    summary_df = pd.DataFrame(rows)
    out_path = output_dir / "representative_case_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"代表数据集汇总已保存：{out_path}")
    print("\n汇总预览：")
    print(summary_df.to_string(index=False))


# ============================================================
# 主函数
# ============================================================

def main():
    setup_matplotlib()

    all_df = load_all_selected()

    # 保存合并后的原始作图数据，便于检查
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "representative_actual_wcd_plot_data.csv"
    all_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"合并后的作图数据已保存：{merged_path}")

    save_case_summary(all_df)
    plot_representative_cases(all_df)

    print("\nDone.")


if __name__ == "__main__":
    main()