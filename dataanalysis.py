import pandas as pd

# =========================================================
# 1. 读取并合并两个 Excel
# =========================================================
file1 = "full_param_grid_football_summary1.xlsx"
file2 = "full_param_grid_football_summary2.xlsx"

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

df = pd.concat([df1, df2], ignore_index=True)

# =========================================================
# 2. 阈值设定
# =========================================================
THRESHOLD = 0.59

df_good = df[df["modularity"] >= THRESHOLD].copy()

print("=" * 80)
print(f"总样本数: {len(df)}")
print(f"高性能样本数 (modularity >= {THRESHOLD}): {len(df_good)}")
if len(df) > 0:
    print(f"整体成功率: {len(df_good) / len(df):.4f}")
print("=" * 80)

# =========================================================
# 3. 需要统计的全部参数
#    这里把你实验里常见的参数都列上了
# =========================================================
params = [
    "alpha",
    "T",
    "h",
    "lam_sparse",
    "batch_size",
    "rho",
    "lr",
    "epochs",
    "k_min",
    "k_max",
    "k_n_init",
]

# 如果某些列在你的 Excel 里不存在，就自动跳过
params = [p for p in params if p in df.columns]

print("将统计的参数列：", params)
print("=" * 80)

# =========================================================
# 4. 单参数成功率统计函数
# =========================================================
def calc_success_table(df_all, df_good, column):
    """
    输出：
    - total_count: 该参数取值在所有实验中的出现次数
    - success_count: 该参数取值在高性能实验中的出现次数
    - success_rate: success_count / total_count
    - good_share: 该参数取值在所有高性能样本中的占比
    """
    total_counts = df_all[column].value_counts(dropna=False).sort_index()
    good_counts = df_good[column].value_counts(dropna=False).sort_index()

    all_values = sorted(set(total_counts.index).union(set(good_counts.index)))

    rows = []
    total_good = len(df_good)

    for val in all_values:
        total_count = int(total_counts.get(val, 0))
        success_count = int(good_counts.get(val, 0))
        success_rate = success_count / total_count if total_count > 0 else 0.0
        good_share = success_count / total_good if total_good > 0 else 0.0

        rows.append({
            "parameter": column,
            "value": val,
            "total_count": total_count,
            "success_count": success_count,
            "success_rate": round(success_rate, 4),
            "good_share": round(good_share, 4),
        })

    result = pd.DataFrame(rows)
    result = result.sort_values(
        by=["success_rate", "success_count", "value"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    return result


# =========================================================
# 5. 打印单参数统计
# =========================================================
all_param_tables = []

for p in params:
    table = calc_success_table(df, df_good, p)
    all_param_tables.append(table)

    print(f"\n{'=' * 80}")
    print(f"参数: {p}")
    print(f"{'=' * 80}")
    print(table.to_string(index=False))


# =========================================================
# 6. 汇总所有参数统计到一个总表
# =========================================================
df_param_summary = pd.concat(all_param_tables, ignore_index=True)

print("\n" + "=" * 80)
print("所有参数成功率汇总（前 50 行）")
print("=" * 80)
print(df_param_summary.head(50).to_string(index=False))

# =========================================================
# 7. 高频高性能参数组合统计
#    这里给一个常用组合视角：lam_sparse + batch_size + h
#    你也可以按需改成别的组合
# =========================================================
combo_cols = [c for c in ["lam_sparse", "batch_size", "h"] if c in df_good.columns]

if len(combo_cols) >= 1:
    df_combo = (
        df_good.groupby(combo_cols)
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 80)
    print(f"参数组合频率（高性能区域）: {combo_cols}")
    print("=" * 80)
    print(df_combo.to_string(index=False))
else:
    df_combo = pd.DataFrame()

# =========================================================
# 8. 更进一步：完整参数组合的成功率
#    这个会比较细，但很有用
# =========================================================
full_combo_cols = [c for c in params if c in df.columns]

df_total_combo = (
    df.groupby(full_combo_cols)
      .size()
      .reset_index(name="total_count")
)

df_good_combo = (
    df_good.groupby(full_combo_cols)
           .size()
           .reset_index(name="success_count")
)

df_full_combo = pd.merge(
    df_total_combo,
    df_good_combo,
    on=full_combo_cols,
    how="left"
)

df_full_combo["success_count"] = df_full_combo["success_count"].fillna(0).astype(int)
df_full_combo["success_rate"] = df_full_combo["success_count"] / df_full_combo["total_count"]

df_full_combo = df_full_combo.sort_values(
    by=["success_rate", "success_count"],
    ascending=[False, False]
).reset_index(drop=True)

print("\n" + "=" * 80)
print("完整参数组合成功率（前 30 行）")
print("=" * 80)
print(df_full_combo.head(30).to_string(index=False))

# =========================================================
# 9. 保存到 Excel
# =========================================================
output_file = "football_success_rate_analysis.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    # 原始数据
    df.to_excel(writer, sheet_name="all_data", index=False)
    df_good.to_excel(writer, sheet_name="good_data", index=False)

    # 单参数汇总
    df_param_summary.to_excel(writer, sheet_name="param_success_summary", index=False)

    # 每个参数单独一个 sheet
    for p in params:
        table = calc_success_table(df, df_good, p)
        sheet_name = f"{p}_success"[:31]
        table.to_excel(writer, sheet_name=sheet_name, index=False)

    # 高频组合
    if not df_combo.empty:
        df_combo.to_excel(writer, sheet_name="good_combo_freq", index=False)

    # 完整参数组合成功率
    df_full_combo.to_excel(writer, sheet_name="full_combo_success", index=False)

print("\n" + "=" * 80)
print(f"分析完成，结果已保存到: {output_file}")
print("=" * 80)