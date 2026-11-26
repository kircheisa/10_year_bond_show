# -*- coding: utf-8 -*-
import pandas as pd

p = r"F:\43_华泰证券预处理\10_year_bond_predict\data_ori\2026_ori_data_new.xlsx"
raw = pd.read_excel(p, sheet_name="滚动稳健标准化日频", header=None)
hi = raw.index[raw[0] == "指标名称"][0]
cols = [str(c) for c in raw.iloc[hi, 1:]]
df = raw.iloc[hi + 1:].copy()
df.columns = ["date"] + cols
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
df = df.apply(pd.to_numeric, errors="coerce")

names = ["DR001", "DR001#79", "中国:逆回购数量:7天", "中国:逆回购数量:隔夜",
         "中国:逆回购数量:14天", "中国:逆回购数量:28天", "中国:逆回购数量:21天",
         "中国:逆回购利率:7天", "GC001:收盘价", "DR007", "利率互换:FR007:1年"]
lines = []
for c in names:
    if c not in df.columns:
        lines.append(f"{c} 缺失")
        continue
    s = df[c].mask(df[c] == 0).dropna()
    tail = [(t.date().isoformat(), round(float(v), 4)) for t, v in s.tail(3).items()]
    lines.append(f"{c}: 有效{len(s)} 最新{tail}")
out = Path(r"F:\43_华泰证券预处理\10_year_bond_predict\_cols.txt")
out.write_text("\n".join(lines), encoding="utf-8")
print("written", out)
