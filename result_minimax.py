"""
Batch score WCD parameter-search result Excel files.

功能：
1. 在根目录 root 下读取多个 full_param_grid_<dataset>_summary.xlsx 文件；
2. 对每个文件中的 modularity 按“数据集内部排序”计算多种得分；
3. 保留原始所有列，并在末尾追加得分列；
4. 在 root 下新建输出文件夹，将结果保存为 <dataset>_score.xlsx。

推荐主目标：score_sigmoid_signed
    - 取值范围 [-1, 1]
    - 越接近 1 表示越值得奖励
    - 越接近 -1 表示越应该惩罚

依赖：
    pip install pandas openpyxl numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ============================================================
# 你最常改的配置区
# ============================================================
ROOT_DIR = "."                      # 根目录，结果文件和输出文件夹都在这里
OUTPUT_DIR_NAME = "score_results"   # 输出文件夹名，会自动创建

# 显式指定要处理的数据集名称。
# 例如：DATASET_NAMES = ["brock200-3", "arenas-jazz_normalized", "football"]
# 留空 [] 则自动处理 ROOT_DIR 下所有匹配 PATTERN 的文件。
DATASET_NAMES: list[str] = [
    "lesmis",
]

# 输入文件命名规则：full_param_grid_<dataset>_summary.xlsx
PATTERN = "full_param_grid_{dataset}_summary.xlsx"
AUTO_GLOB = "full_param_grid_*_summary.xlsx"

# Excel 设置
SHEET_NAME: str | int | None = 0      # 0 表示第一个 sheet；如果你的 sheet 固定叫 results，可改成 "results"
MODULARITY_COL = "modularity"

# 非线性打分参数
TOP_Q = 0.05        # 前 5% 标记为高性能参数组合；可改 0.03
BOTTOM_Q = 0.05     # 后 5% 标记为低性能参数组合；可改 0.03
SIGMOID_SLOPE = 12  # 越大越强调头部/尾部；常用 8~15

# 是否生成一个总汇总表
SAVE_SUMMARY = True


# ============================================================
# 核心函数
# ============================================================
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def extract_dataset_name(path: Path) -> str:
    """从 full_param_grid_<dataset>_summary.xlsx 中提取 dataset 名称。"""
    stem = path.stem
    prefix = "full_param_grid_"
    suffix = "_summary"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix):-len(suffix)]
    return stem


def add_modularity_scores(
    df: pd.DataFrame,
    modularity_col: str = MODULARITY_COL,
    top_q: float = TOP_Q,
    bottom_q: float = BOTTOM_Q,
    sigmoid_slope: float = SIGMOID_SLOPE,
) -> pd.DataFrame:
    """
    保留原始列，并在末尾追加多种基于 modularity 排序的得分列。

    排名逻辑：
    - modularity 越大越好；
    - rank_desc = 1 表示最好；
    - rank_pct = 0 表示最差，1 表示最好；
    - score_sigmoid_signed 是推荐用于后续建模的非线性奖励/惩罚目标。
    """
    if modularity_col not in df.columns:
        raise ValueError(f"找不到列 {modularity_col!r}，当前列为：{list(df.columns)}")

    out = df.copy()
    q = pd.to_numeric(out[modularity_col], errors="coerce")

    if q.isna().any():
        bad_count = int(q.isna().sum())
        raise ValueError(f"列 {modularity_col!r} 中有 {bad_count} 个值无法转为数值，请先检查。")

    n = len(out)
    if n == 0:
        raise ValueError("输入表为空。")

    q_min = float(q.min())
    q_max = float(q.max())
    q_range = q_max - q_min
    eps = 1e-12

    # ascending rank: 最差接近 1，最好接近 n
    # method='average' 可以合理处理并列值。
    rank_ascending = q.rank(method="average", ascending=True)
    rank_descending = q.rank(method="average", ascending=False)

    if n > 1:
        rank_pct = (rank_ascending - 1.0) / (n - 1.0)
    else:
        rank_pct = pd.Series([0.5], index=out.index)

    # 线性 min-max，仅作对照。
    if abs(q_range) < eps:
        modularity_minmax = pd.Series([0.5] * n, index=out.index)
        relative_gap_to_best = pd.Series([0.0] * n, index=out.index)
    else:
        modularity_minmax = (q - q_min) / (q_range + eps)
        relative_gap_to_best = (q_max - q) / (q_range + eps)

    # 基于排名百分位的非线性得分。
    # 0.5 为中点，slope 越大越强调前部奖励和尾部惩罚。
    score_sigmoid_01 = sigmoid(sigmoid_slope * (rank_pct.to_numpy() - 0.5))
    score_sigmoid_signed = 2.0 * score_sigmoid_01 - 1.0

    # Top/Bottom 标签。注意 rank_pct 越大越好。
    top_threshold = 1.0 - top_q
    bottom_threshold = bottom_q
    label_top_q = (rank_pct >= top_threshold).astype(int)
    label_bottom_q = (rank_pct <= bottom_threshold).astype(int)

    # 极端强化版本：不建议一开始作为主目标，但可作为实验备用。
    # top 部分额外向 1 推，bottom 部分额外向 -1 推。
    score_extreme_signed = score_sigmoid_signed.copy()
    score_extreme_signed = np.asarray(score_extreme_signed, dtype=float)

    top_mask = label_top_q.to_numpy(dtype=bool)
    bottom_mask = label_bottom_q.to_numpy(dtype=bool)
    score_extreme_signed[top_mask] = 1.0 - 0.25 * (1.0 - score_extreme_signed[top_mask])
    score_extreme_signed[bottom_mask] = -1.0 + 0.25 * (score_extreme_signed[bottom_mask] + 1.0)
    score_extreme_01 = (score_extreme_signed + 1.0) / 2.0

    # 追加列：尽量用英文列名，后续建模更方便。
    out["rank_ascending"] = rank_ascending
    out["rank_descending"] = rank_descending
    out["rank_pct"] = rank_pct
    out["modularity_minmax"] = modularity_minmax
    out["relative_gap_to_best"] = relative_gap_to_best
    out["score_sigmoid_01"] = score_sigmoid_01
    out["score_sigmoid_signed"] = score_sigmoid_signed
    out["label_top_q"] = label_top_q
    out["label_bottom_q"] = label_bottom_q
    out["score_extreme_01"] = score_extreme_01
    out["score_extreme_signed"] = score_extreme_signed

    return out


def score_one_file(
    input_path: Path,
    output_dir: Path,
    sheet_name: str | int | None = SHEET_NAME,
    modularity_col: str = MODULARITY_COL,
    top_q: float = TOP_Q,
    bottom_q: float = BOTTOM_Q,
    sigmoid_slope: float = SIGMOID_SLOPE,
) -> dict:
    """处理单个 Excel/CSV 文件并输出 <dataset>_score.xlsx。"""
    dataset = extract_dataset_name(input_path)
    output_path = output_dir / f"{dataset}_score.xlsx"

    if input_path.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"不支持的文件类型：{input_path}")

    scored = add_modularity_scores(
        df,
        modularity_col=modularity_col,
        top_q=top_q,
        bottom_q=bottom_q,
        sigmoid_slope=sigmoid_slope,
    )

    # 排序后另存一个 sheet，便于人工检查 top/bottom 参数组合。
    sorted_view = scored.sort_values(modularity_col, ascending=False).reset_index(drop=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        scored.to_excel(writer, sheet_name="results_score", index=False)
        sorted_view.to_excel(writer, sheet_name="sorted_by_modularity", index=False)

        # 轻量美化：冻结首行、自动筛选、设置列宽。
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for col_cells in sheet.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells[:2000]:  # 避免超大文件时慢，前 2000 行足够估计列宽
                    val = cell.value
                    if val is None:
                        continue
                    max_len = max(max_len, len(str(val)))
                sheet.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 28)

    best_row = scored.loc[scored[modularity_col].idxmax()]
    worst_row = scored.loc[scored[modularity_col].idxmin()]

    return {
        "dataset": dataset,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "rows": len(scored),
        "modularity_min": float(scored[modularity_col].min()),
        "modularity_max": float(scored[modularity_col].max()),
        "top_q_count": int(scored["label_top_q"].sum()),
        "bottom_q_count": int(scored["label_bottom_q"].sum()),
        "best_run_name": best_row.get("run_name", ""),
        "worst_run_name": worst_row.get("run_name", ""),
    }


def resolve_input_files(root: Path, dataset_names: Iterable[str]) -> list[Path]:
    """根据 DATASET_NAMES 或自动 glob 获取输入文件列表。"""
    names = [x.strip() for x in dataset_names if str(x).strip()]
    if names:
        files = [root / PATTERN.format(dataset=name) for name in names]
    else:
        files = sorted(root.glob(AUTO_GLOB))

    missing = [p for p in files if not p.exists()]
    if missing:
        msg = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"以下输入文件不存在：\n{msg}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch score WCD modularity result Excel files.")
    parser.add_argument("--root", default=ROOT_DIR, help="根目录，默认当前目录。")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_NAME, help="输出文件夹名。")
    parser.add_argument(
        "--datasets",
        default=None,
        help="逗号分隔的数据集名称，如 brock200-3,football。为空则使用代码中的 DATASET_NAMES；若二者都空，则自动处理所有匹配文件。",
    )
    parser.add_argument("--sheet", default=SHEET_NAME, help="Excel sheet 名称；默认读取第一个 sheet。")
    parser.add_argument("--modularity-col", default=MODULARITY_COL, help="modularity 列名。")
    parser.add_argument("--top-q", type=float, default=TOP_Q, help="Top q 标签阈值，如 0.05。")
    parser.add_argument("--bottom-q", type=float, default=BOTTOM_Q, help="Bottom q 标签阈值，如 0.05。")
    parser.add_argument("--slope", type=float, default=SIGMOID_SLOPE, help="Sigmoid 斜率，越大越强调头尾。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets is not None:
        dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    else:
        dataset_names = DATASET_NAMES

    # argparse 的 --sheet 传进来都是字符串；"0" 转为 0，方便读取第一个 sheet。
    sheet_name: str | int | None
    if args.sheet is None or str(args.sheet).lower() == "none":
        sheet_name = 0
    elif str(args.sheet).isdigit():
        sheet_name = int(args.sheet)
    else:
        sheet_name = args.sheet

    files = resolve_input_files(root, dataset_names)
    print(f"Root: {root}")
    print(f"Output dir: {output_dir}")
    print(f"Files to process: {len(files)}")

    summary_rows = []
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Processing: {path.name}")
        info = score_one_file(
            input_path=path,
            output_dir=output_dir,
            sheet_name=sheet_name,
            modularity_col=args.modularity_col,
            top_q=args.top_q,
            bottom_q=args.bottom_q,
            sigmoid_slope=args.slope,
        )
        summary_rows.append(info)
        print(f"    -> saved: {Path(info['output_file']).name}")
        print(f"    -> Q range: {info['modularity_min']:.6f} ~ {info['modularity_max']:.6f}")

    if SAVE_SUMMARY and summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / "score_processing_summary.xlsx"
        summary_df.to_excel(summary_path, index=False)
        print(f"Summary saved: {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()
