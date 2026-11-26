# -*- coding: utf-8 -*-
"""核查：新原始列的数据覆盖、政策锚数值、F14 特征是否生成及最新取值"""
import pandas as pd

B = r"F:\43_华泰证券预处理\10_year_bond_predict"
L = []

# ---- 1) 原始列覆盖 ----
raw = pd.read_excel(B + r"\data_ori\2026_ori_data_new.xlsx",
                    sheet_name="滚动稳健标准化日频", header=None)
hi = raw.index[raw[0] == "指标名称"][0]
names = list(raw.iloc[hi, 1:])
for i, n in enumerate(names):            # 重复列名加编号
    if names[: i + 1].count(n) > 1:
        names[i] = f"{n}#{i}"
d = raw.iloc[hi + 1:].copy()
d.columns = ["date"] + names
d["date"] = pd.to_datetime(d["date"])
d = d.dropna(subset=["date"]).sort_values("date").set_index("date")
d = d.apply(pd.to_numeric, errors="coerce")
L.append("== 原始列覆盖（非零有效值）==")
for c in ["DR001", "DR001#79", "中国:逆回购利率:7天", "GC001:收盘价",
          "中国:逆回购数量:隔夜", "中国:逆回购数量:7天", "中国:逆回购数量:14天",
          "中国:逆回购数量:21天", "中国:逆回购数量:28天"]:
    if c not in d.columns:
        L.append(f"  {c}: 列不存在")
        continue
    s = d[c].mask(d[c] == 0).dropna()
    if s.empty:
        L.append(f"  {c}: 全为空/0")
        continue
    by_year = s.groupby(s.index.year).size()
    L.append(f"  {c}: 有效 {len(s)}  {s.index.min().date()} ~ {s.index.max().date()}")
    L.append(f"      逐年有效天数: {dict(by_year.tail(6))}")
    L.append(f"      最新值: {[(t.date().isoformat(), round(float(v), 4)) for t, v in s.tail(4).items()]}")

# ---- 2) 政策锚 vs DR001 ----
L.append("== DR001 与 7天逆回购利率 近期对照 ==")
a = d["DR001"].mask(d["DR001"] == 0)
b = d["中国:逆回购利率:7天"].mask(d["中国:逆回购利率:7天"] == 0)
cmp = pd.concat([a, b], axis=1).dropna().tail(8)
L.append(cmp.to_string())
L.append("7天逆回购利率 历史取值(近3年去重): " +
         str(sorted(set(b.dropna().tail(750).round(4)))))

# ---- 3) 特征文件 ----
f3 = pd.read_csv(B + r"\data_new\43v3_features.csv", index_col=0, nrows=2)
f4 = pd.read_csv(B + r"\data_new\43v4_features.csv", index_col=0, nrows=2)
fn = pd.read_csv(B + r"\data_new\43v4nb_features.csv", index_col=0, nrows=2)
L.append("== 特征文件列数 ==")
L.append(f"  43v3={f3.shape[1]}  43v4={f4.shape[1]}  43v4nb={fn.shape[1]}")
L.append("  43v3 中含'逆回购数量'的列: " +
         str([c for c in f3.columns if "逆回购数量" in c]))
cd = [c for c in f4.columns if c.startswith("传导__")]
L.append(f"  43v4 中 '传导__' 列共 {len(cd)} 个:")
for c in cd:
    L.append("    " + c)

# ---- 4) 新特征最新取值与缺失率 ----
full = pd.read_csv(B + r"\data_new\43v4_features.csv", index_col=0, parse_dates=True)
L.append("== 传导族 最新值 / 全样本缺失率 / 2024年起缺失率 ==")
recent = full[full.index >= "2024-01-01"]
for c in cd:
    s = full[c]
    L.append(f"  {c:<28} 最新={s.iloc[-1]:>10.4f}  缺失率={s.isna().mean():.3f}  "
             f"近三年缺失率={recent[c].isna().mean():.3f}")
open(B + r"\_check.txt", "w", encoding="utf-8").write("\n".join(L))
print("written")
