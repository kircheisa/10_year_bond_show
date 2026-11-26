# -*- coding: utf-8 -*-
"""临时验证：传导族(F14)前向AUC衰减诊断（查实现错误，如窗口方向写反）。
因果特征对未来k日方向的区分度不应随k拉长而明显上升；对照 y__mom20。"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = Path(r"F:\43_华泰证券预处理\10_year_bond_predict\data_new")
f = pd.read_csv(D / "43v4_features.csv", index_col=0, parse_dates=True)
y = pd.read_csv(D / "43v3_labels_bin.csv", index_col=0, parse_dates=True)["rate_delta_5d"]
cols = [c for c in f.columns if c.startswith("传导__")] + ["y__mom20"]
ks = [1, 5, 10, 21, 63, 126]
fwd = {k: y.shift(-k) - y for k in ks}
rows = []
for c in cols:
    s = f[c]
    rec = {"feature": c}
    for k in ks:
        d, m = fwd[k], s.notna() & fwd[k].notna()
        try:
            a = roc_auc_score((d[m] < 0).astype(int), s[m])
            rec[f"k{k}"] = round(max(a, 1 - a), 3)
        except ValueError:
            rec[f"k{k}"] = np.nan
    rows.append(rec)
res = pd.DataFrame(rows)
(D.parent / "_f14_leak.txt").write_text(res.to_string(index=False), encoding="utf-8")
print(res.to_string(index=False))
