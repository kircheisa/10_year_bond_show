# -*- coding: utf-8 -*-
"""临时验证：DR001传导族(F14)的加入是否改善样本外表现。

配置（均用无净买入版特征 43v4nb，协议与 21 脚本一致：窗口756/步长63/2018起）:
  A 基线          不含传导族，也不含新原始列(逆回购操作量)的基础特征
  B +传导族       A + 传导族
  C +传导族+新列基础特征   全量
输出: _f14_backtest.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

B = Path(r"F:\43_华泰证券预处理\10_year_bond_predict")
D = B / "data_new"
PARAMS = dict(max_depth=3, n_estimators=600, learning_rate=0.05,
              min_child_weight=8.0, reg_alpha=0.5, reg_lambda=2.0,
              subsample=0.8, colsample_bytree=0.8, tree_method="hist",
              random_state=42, eval_metric="logloss", n_jobs=4)

feats = pd.read_csv(D / "43v4nb_features.csv", index_col=0, parse_dates=True)
is_f14 = feats.columns.str.startswith("传导__")
is_newraw = feats.columns.str.contains("逆回购数量")
CONFIGS = {
    "A基线": [c for c in feats.columns if not (is_f14[list(feats.columns).index(c)]
              or is_newraw[list(feats.columns).index(c)])],
    "B+传导族": [c for c in feats.columns
               if not is_newraw[list(feats.columns).index(c)]],
    "C+传导族+新列基础特征": list(feats.columns),
}
L = [f"特征数: " + "  ".join(f"{k}={len(v)}" for k, v in CONFIGS.items())]
rows = []
for h in (10, 20):
    lab = pd.read_csv(D / f"43v3_labels_bin_h{h}.csv", index_col=0,
                      parse_dates=True)
    for name, cols in CONFIGS.items():
        df = feats[cols].join(lab, how="inner")
        df = df.loc[df["label"].notna() & df["sample_weight"].notna()]
        X, y, w = df[cols], df["label"].astype(int), df["sample_weight"]
        tidx = df.index[df.index >= "2018-01-01"]
        folds = [tidx[i:i + 63] for i in range(0, len(tidx), 63)]
        preds, imps = [], []
        for seg in folds:
            pos = df.index.get_indexer(seg)
            tr = np.arange(max(0, pos[0] - 756), pos[0])
            if len(tr) < 300:
                continue
            m = xgb.XGBClassifier(**PARAMS)
            m.fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr])
            preds.append(pd.Series(m.predict_proba(X.iloc[pos])[:, 1], index=seg))
            imps.append(pd.Series(m.feature_importances_, index=cols))
        p = pd.concat(preds)
        yt = y.loc[p.index]
        yr = pd.Series(p.index.year, index=p.index)
        rec = {"周期": f"h{h}", "配置": name, "特征数": len(cols),
               "全局Acc": accuracy_score(yt, p >= 0.5),
               "全局AUC": roc_auc_score(yt, p)}
        for yy in (2022, 2023, 2024, 2025, 2026):
            g = yt[yr == yy], p[yr == yy]
            rec[f"{yy}Acc"] = accuracy_score(g[0], g[1] >= 0.5)
            rec[f"{yy}AUC"] = roc_auc_score(g[0], g[1])
        rows.append(rec)
        if name.startswith("C"):
            imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
            fam = imp.groupby(imp.index.str.split("__").str[0]).sum().sort_values(
                ascending=False)
            L.append(f"\n[{name} h{h}] 各特征族重要性占比:")
            L.extend(f"    {k:<24} {v:.3f}" for k, v in fam.head(10).items())
            L.append(f"  传导族内部重要性排名(全特征中的名次):")
            for k, v in imp.items():
                if k.startswith("传导__"):
                    L.append(f"    第{list(imp.index).index(k) + 1:>3}名  {v:.4f}  {k}")
        print(f"done h{h} {name}", flush=True)

r = pd.DataFrame(rows)
L.append("\n== 回测结果 (Acc / AUC) ==")
for c in CONFIGS:
    for _, x in r[r["配置"] == c].iterrows():
        L.append(f"{x['周期']:>4} {c:<22} 全局 {x['全局Acc']:.4f}/{x['全局AUC']:.4f} | "
                 + " ".join(f"{yy} {x[f'{yy}Acc']:.3f}/{x[f'{yy}AUC']:.3f}"
                            for yy in (2022, 2023, 2024, 2025, 2026)))
(B / "_f14_backtest.txt").write_text("\n".join(L), encoding="utf-8")
print("written")
