# -*- coding: utf-8 -*-
"""逐步复现 30 的加载/合并流程，定位重复标签出现的位置"""
import importlib.util

import pandas as pd

P = r"F:\43_华泰证券预处理\10_year_bond_predict\model_new\30_新版原始数据_特征构建.py"
spec = importlib.util.spec_from_file_location("m30", P)
m30 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m30)

L = []
xl = pd.ExcelFile(m30.SRC)
Y = "中国:国债到期收益率:10年"

behavior = m30.dedupe_columns(m30.load_sheet(xl, "滚动稳健标准化日频+买卖行为"))
plain = m30.dedupe_columns(m30.load_sheet(xl, "滚动稳健标准化日频"))
L.append(f"behavior: {behavior.shape}, 列重复={behavior.columns.duplicated().sum()}, "
         f"索引重复={behavior.index.duplicated().sum()}")
L.append(f"plain:    {plain.shape}, 列重复={plain.columns.duplicated().sum()}, "
         f"索引重复={plain.index.duplicated().sum()}")

shared = [c for c in behavior.columns if c in plain.columns]
comb = behavior.combine_first(plain[shared])
L.append(f"combined: {comb.shape}, 列重复={comb.columns.duplicated().sum()}, "
         f"索引重复={comb.index.duplicated().sum()}")

daily, dead = m30.zero_to_nan(comb)
L.append(f"zero_to_nan: {daily.shape}, 列重复={daily.columns.duplicated().sum()}, "
         f"索引重复={daily.index.duplicated().sum()}, 死列={len(dead)}")

y = daily[Y].dropna()
grid = y.index
L.append(f"grid: {len(grid)}, 重复={grid.duplicated().sum()}")
try:
    daily_g = daily.reindex(grid)
    L.append("reindex OK")
except ValueError as e:
    L.append(f"reindex 失败: {e}")
    dup_cols = daily.columns[daily.columns.duplicated()]
    L.append(f"  重复列: {list(dup_cols)[:10]}")

open(r"F:\43_华泰证券预处理\10_year_bond_predict\_debug30.txt", "w",
     encoding="utf-8").write("\n".join(L))
print("written")
