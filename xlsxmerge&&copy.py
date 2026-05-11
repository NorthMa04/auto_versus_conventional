"""
Merge scored WCD results with PCA structural features.

功能：
1. 读取 score_results/<dataset>_score.xlsx；
2. 将多个数据集的 score 结果纵向合并；
3. 读取 pca_results/pca_scores.csv；
4. 按 dataset 字段拼接 PCA 主成分；
5. 输出：
   - merged_results/all_score_results.xlsx
   - merged_results/all_score_results.csv
   - model_input/model_input.xlsx
   - model_input/model_input.csv
   - model_input/merge_report.xlsx

最终推荐用于后续建模的文件：
    model_input/model_input.xlsx
    model_input/model_input.csv

依赖：
    pip install pandas openpyxl
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


# ============================================================
# 你最常改的配置区
# ============================================================

# ============================================================
# 你最常改的配置区
# ============================================================

ROOT_DIR = "."

# 处理好的非线性赋分结果所在文件夹
# 里面应该是：
#   <dataset>_score.xlsx
SCORE_INPUT_DIR_NAME = "score_results"

# PCA 结果所在文件夹
# 里面应该有：
#   pca_scores.csv
PCA_INPUT_DIR_NAME = "pca_results"

# 输出文件夹
MERGED_OUTPUT_DIR_NAME = "merged_results"
MODEL_INPUT_DIR_NAME = "model_input"

PCA_SCORES_FILE = "pca_scores.csv"

# score 文件命名规则
SCORE_PATTERN = "{dataset}_score.xlsx"

SCORE_SHEET_NAME: str | int = "results_score"

STRICT_MISSING_SCORE_FILES = False
STRICT_MISSING_PCA = False

SAVE_CSV = True
SAVE_EXCEL = True


# ============================================================
# 显式指定要合并的数据集
# 你要跑哪些，就取消哪些注释。
# ============================================================

DATASET_NAMES: list[str] = [
        #"celegans_edges",
        "ca-netscience",
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
]


# 如果 DATASET_NAMES 为空，是否自动合并 score_results 下所有 *_score.xlsx
AUTO_USE_ALL_SCORE_FILES_WHEN_EMPTY = True


# ============================================================
# 工具函数
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_dataset_name(x) -> str:
    """
    统一 dataset 名称，防止 Excel/CSV 中出现首尾空格导致 merge 失败。
    """
    if pd.isna(x):
        return ""
    return str(x).strip()


def extract_dataset_from_score_filename(path: Path) -> str:
    """
    从 <dataset>_score.xlsx 提取 dataset 名称。
    """
    stem = path.stem
    suffix = "_score"
    if stem.endswith(suffix):
        return stem[:-len(suffix)]
    return stem


def resolve_score_files(
    score_dir: Path,
    dataset_names: Iterable[str],
) -> tuple[list[Path], list[str]]:
    """
    根据 DATASET_NAMES 找到 score 文件。
    返回：
    - existing_files
    - missing_dataset_names
    """
    names = [normalize_dataset_name(x) for x in dataset_names if normalize_dataset_name(x)]

    if names:
        files = [score_dir / SCORE_PATTERN.format(dataset=name) for name in names]
    else:
        if not AUTO_USE_ALL_SCORE_FILES_WHEN_EMPTY:
            raise ValueError("DATASET_NAMES 为空，且 AUTO_USE_ALL_SCORE_FILES_WHEN_EMPTY=False。")
        files = sorted(score_dir.glob("*_score.xlsx"))

    existing_files = []
    missing_names = []

    for path in files:
        if path.exists():
            existing_files.append(path)
        else:
            # 从期望文件名反推 dataset
            name = extract_dataset_from_score_filename(path)
            missing_names.append(name)

    return existing_files, missing_names


def read_one_score_file(path: Path, sheet_name: str | int) -> pd.DataFrame:
    """
    读取单个 score 文件。
    如果表中没有 dataset 列，则根据文件名补一个。
    """
    dataset_from_filename = extract_dataset_from_score_filename(path)

    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except ValueError:
        # 有些文件 sheet 名可能不是 results_score，兜底读取第一个 sheet
        print(f"    ! sheet {sheet_name!r} 不存在，改读第一个 sheet：{path.name}")
        df = pd.read_excel(path, sheet_name=0)

    if df.empty:
        raise ValueError(f"{path} 是空表。")

    if "dataset" not in df.columns:
        df.insert(0, "dataset", dataset_from_filename)
    else:
        df["dataset"] = df["dataset"].apply(normalize_dataset_name)
        # 如果 dataset 列为空，则用文件名补
        df.loc[df["dataset"] == "", "dataset"] = dataset_from_filename

    # 保证文件来源可追踪
    if "source_score_file" not in df.columns:
        df["source_score_file"] = path.name

    return df


def load_all_score_results(
    score_files: list[Path],
    sheet_name: str | int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    纵向合并所有 score 文件。
    返回：
    - all_score_df
    - score_summary_df
    """
    frames = []
    summary_rows = []

    for i, path in enumerate(score_files, start=1):
        print(f"[{i}/{len(score_files)}] 读取 score 文件：{path.name}")
        df = read_one_score_file(path, sheet_name=sheet_name)

        dataset_name = extract_dataset_from_score_filename(path)

        row = {
            "dataset_from_filename": dataset_name,
            "file_name": path.name,
            "rows": len(df),
            "has_dataset_col": "dataset" in df.columns,
        }

        if "dataset" in df.columns:
            row["dataset_values"] = ", ".join(sorted(df["dataset"].dropna().astype(str).unique()))

        if "modularity" in df.columns:
            row["modularity_min"] = pd.to_numeric(df["modularity"], errors="coerce").min()
            row["modularity_max"] = pd.to_numeric(df["modularity"], errors="coerce").max()

        if "score_sigmoid_signed" in df.columns:
            row["score_signed_min"] = pd.to_numeric(df["score_sigmoid_signed"], errors="coerce").min()
            row["score_signed_max"] = pd.to_numeric(df["score_sigmoid_signed"], errors="coerce").max()

        frames.append(df)
        summary_rows.append(row)

        print(f"    -> rows={len(df)}")

    if not frames:
        raise FileNotFoundError("没有读取到任何 score 文件。")

    all_score_df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    all_score_df["dataset"] = all_score_df["dataset"].apply(normalize_dataset_name)

    score_summary_df = pd.DataFrame(summary_rows)
    return all_score_df, score_summary_df


def load_pca_scores(pca_path: Path) -> pd.DataFrame:
    """
    读取 PCA 得分表。
    要求至少包含 dataset 和 PC1。
    """
    if not pca_path.exists():
        raise FileNotFoundError(f"找不到 PCA 文件：{pca_path}")

    pca_df = pd.read_csv(pca_path)

    if "dataset" not in pca_df.columns:
        raise ValueError(f"{pca_path} 中必须包含 dataset 列。当前列：{list(pca_df.columns)}")

    pca_df["dataset"] = pca_df["dataset"].apply(normalize_dataset_name)

    pc_cols = [c for c in pca_df.columns if c.startswith("PC")]
    if not pc_cols:
        raise ValueError(f"{pca_path} 中没有发现 PC 列，例如 PC1、PC2。当前列：{list(pca_df.columns)}")

    # 检查 PCA 表 dataset 是否重复
    dup = pca_df[pca_df["dataset"].duplicated(keep=False)]
    if not dup.empty:
        dup_names = sorted(dup["dataset"].unique())
        raise ValueError(f"pca_scores.csv 中 dataset 存在重复：{dup_names}")

    return pca_df


def reorder_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    把建模最常用的列放到前面：
    dataset, PC1..PCn, 参数列, 结果列, score列, 其他列
    """
    cols = list(df.columns)

    pc_cols = [c for c in cols if c.startswith("PC")]

    param_cols = [
        "alpha",
        "beta",
        "T",
        "h",
        "rho",
        "lr",
        "lam_sparse",
        "epochs",
        "batch_size",
        "k_min",
        "k_max",
        "k_n_init",
    ]

    result_cols = [
        "best_k",
        "modularity",
        "time_seconds",
    ]

    score_cols = [
        "rank_ascending",
        "rank_descending",
        "rank_pct",
        "modularity_minmax",
        "relative_gap_to_best",
        "score_sigmoid_01",
        "score_sigmoid_signed",
        "label_top_q",
        "label_bottom_q",
        "score_extreme_01",
        "score_extreme_signed",
    ]

    front = (
        ["dataset"]
        + pc_cols
        + [c for c in param_cols if c in cols]
        + [c for c in result_cols if c in cols]
        + [c for c in score_cols if c in cols]
    )

    # 去重并保序
    seen = set()
    front_unique = []
    for c in front:
        if c in cols and c not in seen:
            front_unique.append(c)
            seen.add(c)

    rest = [c for c in cols if c not in seen]

    return df[front_unique + rest]


def build_merge_report(
    all_score_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    missing_score_names: list[str],
    model_input_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    生成一些检查表，方便你确认 merge 没翻车。
    """
    score_datasets = sorted(all_score_df["dataset"].dropna().unique())
    pca_datasets = sorted(pca_df["dataset"].dropna().unique())

    score_set = set(score_datasets)
    pca_set = set(pca_datasets)

    missing_pca = sorted(score_set - pca_set)
    unused_pca = sorted(pca_set - score_set)

    rows_per_dataset = (
        all_score_df
        .groupby("dataset", as_index=False)
        .size()
        .rename(columns={"size": "score_rows"})
    )

    model_rows_per_dataset = (
        model_input_df
        .groupby("dataset", as_index=False)
        .size()
        .rename(columns={"size": "model_input_rows"})
    )

    row_check = rows_per_dataset.merge(
        model_rows_per_dataset,
        on="dataset",
        how="outer"
    )

    pca_match_rows = pd.DataFrame({
        "type": [
            "missing_score_file",
            "score_dataset_missing_in_pca",
            "pca_dataset_not_used",
        ],
        "count": [
            len(missing_score_names),
            len(missing_pca),
            len(unused_pca),
        ],
        "items": [
            ", ".join(missing_score_names),
            ", ".join(missing_pca),
            ", ".join(unused_pca),
        ],
    })

    basic_info = pd.DataFrame([
        {"item": "all_score_rows", "value": len(all_score_df)},
        {"item": "model_input_rows", "value": len(model_input_df)},
        {"item": "score_dataset_count", "value": len(score_datasets)},
        {"item": "pca_dataset_count", "value": len(pca_datasets)},
        {"item": "model_input_column_count", "value": model_input_df.shape[1]},
    ])

    return {
        "basic_info": basic_info,
        "dataset_row_check": row_check,
        "merge_warnings": pca_match_rows,
    }


def save_excel_with_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """
    保存多 sheet Excel，并做轻量格式优化。
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_sheet_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

        # 轻量美化
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for col_cells in ws.columns:
                col_letter = col_cells[0].column_letter
                max_len = 0

                # 只取前 1000 行估列宽，避免大表过慢
                for cell in col_cells[:1000]:
                    if cell.value is None:
                        continue
                    max_len = max(max_len, len(str(cell.value)))

                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 32)


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    root = Path(ROOT_DIR).resolve()

    score_dir = root / SCORE_INPUT_DIR_NAME
    pca_dir = root / PCA_INPUT_DIR_NAME
    merged_out_dir = ensure_dir(root / MERGED_OUTPUT_DIR_NAME)
    model_out_dir = ensure_dir(root / MODEL_INPUT_DIR_NAME)

    print("=" * 100)
    print("Merge scored WCD results with PCA features")
    print(f"Root: {root}")
    print(f"Score dir: {score_dir}")
    print(f"PCA dir: {pca_dir}")
    print("=" * 100)

    if not score_dir.exists():
       raise FileNotFoundError(f"找不到 score 输入文件夹：{score_dir}")

    # 1. 找到 score 文件
    score_files, missing_score_names = resolve_score_files(score_dir, DATASET_NAMES)

    if missing_score_names:
        msg = "\n".join(f" - {name}" for name in missing_score_names)
        print("\n以下数据集没有找到对应 score 文件：")
        print(msg)

        if STRICT_MISSING_SCORE_FILES:
            raise FileNotFoundError("存在缺失 score 文件，且 STRICT_MISSING_SCORE_FILES=True。")
        else:
            print("当前 STRICT_MISSING_SCORE_FILES=False，将跳过这些数据集。\n")

    if not score_files:
        raise FileNotFoundError("没有可合并的 score 文件。")

    print(f"将合并 {len(score_files)} 个 score 文件。")

    # 2. 纵向合并 score 结果
    all_score_df, score_summary_df = load_all_score_results(
        score_files=score_files,
        sheet_name=SCORE_SHEET_NAME,
    )

    # 3. 保存 all_score_results
    all_score_xlsx = merged_out_dir / "all_score_results.xlsx"
    all_score_csv = merged_out_dir / "all_score_results.csv"

    if SAVE_EXCEL:
        save_excel_with_sheets(
            all_score_xlsx,
            {
                "all_score_results": all_score_df,
                "score_file_summary": score_summary_df,
            }
        )

    if SAVE_CSV:
        all_score_df.to_csv(all_score_csv, index=False, encoding="utf-8-sig")

    print("\n已生成合并后的 score 总表：")
    if SAVE_EXCEL:
        print(f" - {all_score_xlsx}")
    if SAVE_CSV:
        print(f" - {all_score_csv}")

    # 4. 读取 PCA 得分
    pca_path = pca_dir / PCA_SCORES_FILE
    pca_df = load_pca_scores(pca_path)

    print("\nPCA scores loaded:")
    print(f" - {pca_path}")
    print(f" - shape={pca_df.shape}")

    # 5. 检查 score 数据集是否都有 PCA
    score_dataset_set = set(all_score_df["dataset"].dropna().unique())
    pca_dataset_set = set(pca_df["dataset"].dropna().unique())
    missing_pca = sorted(score_dataset_set - pca_dataset_set)

    if missing_pca:
        print("\n以下 score 数据集没有在 pca_scores.csv 中找到：")
        for name in missing_pca:
            print(f" - {name}")

        if STRICT_MISSING_PCA:
            raise ValueError("存在缺失 PCA 的数据集，且 STRICT_MISSING_PCA=True。")
        else:
            print("当前 STRICT_MISSING_PCA=False，将继续合并，对应 PC 列为空。\n")

    # 6. 按 dataset 拼接 PCA
    # many_to_one 表示：
    # score 表中一个 dataset 有很多参数实验行；
    # PCA 表中每个 dataset 只能有一行。
    model_input_df = all_score_df.merge(
        pca_df,
        on="dataset",
        how="left",
        validate="many_to_one",
        suffixes=("", "_pca"),
    )

    model_input_df = reorder_model_columns(model_input_df)

    # 7. 保存 model_input
    model_input_xlsx = model_out_dir / "model_input.xlsx"
    model_input_csv = model_out_dir / "model_input.csv"

    if SAVE_EXCEL:
        save_excel_with_sheets(
            model_input_xlsx,
            {
                "model_input": model_input_df,
            }
        )

    if SAVE_CSV:
        model_input_df.to_csv(model_input_csv, index=False, encoding="utf-8-sig")

    print("\n已生成最终建模输入表：")
    if SAVE_EXCEL:
        print(f" - {model_input_xlsx}")
    if SAVE_CSV:
        print(f" - {model_input_csv}")

    # 8. 保存合并报告
    report_sheets = build_merge_report(
        all_score_df=all_score_df,
        pca_df=pca_df,
        missing_score_names=missing_score_names,
        model_input_df=model_input_df,
    )

    report_sheets["score_file_summary"] = score_summary_df

    report_path = model_out_dir / "merge_report.xlsx"
    if SAVE_EXCEL:
        save_excel_with_sheets(report_path, report_sheets)
        print(f"\n合并检查报告已生成：{report_path}")

    # 9. 终端输出关键检查信息
    print("\n" + "=" * 100)
    print("Merge finished.")
    print(f"all_score_results shape: {all_score_df.shape}")
    print(f"model_input shape: {model_input_df.shape}")
    print(f"dataset count in score: {all_score_df['dataset'].nunique()}")
    print(f"dataset count in PCA: {pca_df['dataset'].nunique()}")

    pc_cols = [c for c in model_input_df.columns if c.startswith("PC")]
    print(f"PC columns: {pc_cols}")

    target_cols = [
        "rank_pct",
        "score_sigmoid_signed",
        "label_top_q",
        "label_bottom_q",
    ]
    existing_targets = [c for c in target_cols if c in model_input_df.columns]
    print(f"Target / label columns found: {existing_targets}")

    if missing_score_names:
        print(f"Missing score files: {len(missing_score_names)}")

    if missing_pca:
        print(f"Missing PCA matches: {len(missing_pca)}")

    print("=" * 100)


if __name__ == "__main__":
    main()