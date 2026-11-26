# -*- coding: utf-8 -*-
"""
信号对账：台账中已到期信号的预测值 vs 实际值 逐条比对

对每条台账信号:
  - 到期日收益率(实际) → 实际方向/实际变动bp
  - XGB / 融合 / LLM修正 各腿的预测方向是否命中
  - 隐含点位估算 vs 实际变动 的误差
输出: data_new/out/信号对账.csv          逐条明细
      data_new/out/信号对账汇总.txt      命中率统计（按腿×周期）
未到期信号标注"未到期"，待数据更新后重跑本脚本即可滚动对账。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data_new"
OUT = DATA / "out"
LEDGER = DATA / "预测记录.csv"
XLSX = BASE / "data_ori" / "2026_ori_data_new.xlsx"


def load_yields() -> pd.Series:
    raw = pd.read_excel(XLSX, sheet_name="滚动稳健标准化日频", header=None)
    hi = raw.index[raw[0] == "指标名称"][0]
    d = raw.iloc[hi + 1:].copy()
    d.columns = ["date"] + list(raw.iloc[hi, 1:])
    y = pd.to_numeric(d.set_index(pd.to_datetime(d["date"]))[
        "中国:国债到期收益率:10年"], errors="coerce").replace(0, np.nan).dropna()
    return y.sort_index()


def main() -> None:
    led = pd.read_csv(LEDGER, dtype={"发出日": str}).drop_duplicates(
        subset=["发出日", "周期"], keep="last")
    y = load_yields()
    grid = list(y.index)
    pos = {t: i for i, t in enumerate(grid)}
    last_data = y.index.max()

    rows = []
    for _, r in led.iterrows():
        a = pd.Timestamp(r["发出日"])
        mat = pd.Timestamp(r["到期日"])
        if a not in pos:
            continue
        i0 = pos[a]
        matured = mat in pos and pos[mat] > i0
        h = int(r["周期"].lstrip("h"))
        if matured:
            i1 = pos[mat]
            bp = round((y.iloc[i1] - y.iloc[i0]) * 100, 1)
            act = 1 if y.iloc[i1] < y.iloc[i0] else 0
        else:
            bp, act = np.nan, np.nan
        row = {"发出日": r["发出日"][:10], "周期": r["周期"], "到期日": r["到期日"],
               "发出日收益率": r.get("当日收益率", ""),
               "状态": "已到期" if matured else "未到期",
               "实际变动bp": bp, "实际方向": ("下行" if act == 1 else
                                            "上行" if matured else "未到期")}
        for leg, pcol in [("XGB", "下行概率"), ("融合", "融合概率"),
                          ("LLM修正", "修正概率")]:
            p = r.get(pcol, np.nan)
            if pd.isna(p):
                row[f"{leg}概率"], row[f"{leg}命中"] = None, None
            else:
                p = float(p)
                row[f"{leg}概率"] = round(p, 3)
                row[f"{leg}命中"] = (int(p >= 0.5) == act) if matured else None
        row["点位误差bp"] = (round(bp - (float(r["隐含点位估算(%)"]) - y.iloc[i0])
                                 * 100, 1)
                            if matured and pd.notna(r.get("隐含点位估算(%)"))
                            else None)
        rows.append(row)

    det = pd.DataFrame(rows)
    det.to_csv(OUT / "信号对账.csv", index=False, encoding="utf-8-sig")

    L = ["=" * 70, "信号对账：预测 vs 实际（已到期部分）", "=" * 70]
    for _, r in det.iterrows():
        if r["状态"] != "已到期":
            continue
        legs = "  ".join(
            f"{leg}:{r[f'{leg}概率']}({'✓' if r[f'{leg}命中'] else '✗'})"
            for leg in ("XGB", "融合", "LLM修正")
            if r.get(f"{leg}命中") is not None)
        L.append(f"{r['发出日']} {r['周期']:>3} → {r['到期日']}  "
                 f"实际{r['实际变动bp']:+.1f}bp({r['实际方向']})  {legs}")
    unmatured = det[det["状态"] == "未到期"]
    if len(unmatured):
        L.append(f"\n未到期 {len(unmatured)} 条（待数据更新后滚动对账）")

    L.append("\n" + "-" * 70)
    L.append("命中率汇总（已到期；|实际变动|<2bp 记为走平、不计命中）:")
    mg = det[det["状态"] == "已到期"].copy()
    mg["走平"] = mg["实际变动bp"].abs() < 2
    if mg.empty:
        L.append("  尚无已到期信号")
    else:
        valid = mg[~mg["走平"]]
        L.append(f"  有效样本 {len(valid)}/{len(mg)}（走平剔除 {int(mg['走平'].sum())} 条）")
        for leg in ("XGB", "融合", "LLM修正"):
            col = f"{leg}命中"
            g = valid[valid[col].notna()]
            if len(g):
                L.append(f"  {leg:<6} n={len(g):<3} 命中率={g[col].mean():.3f} "
                         f"({int(g[col].sum())}/{len(g)})")
        for hz, g in mg.groupby("周期"):
            gv = g[~g["走平"] & g["XGB命中"].notna()]
            seg = f"{hz}: 到期{len(g)} 有效{len(gv)}"
            if len(gv):
                seg += f" XGB {gv['XGB命中'].mean():.3f}"
                fuh = gv[gv["融合命中"].notna()]
                if len(fuh):
                    seg += f" 融合 {fuh['融合命中'].mean():.3f}"
            L.append(f"    {seg}")
        flat = mg[mg["走平"]]
        if len(flat):
            L.append("  走平样本: " + "、".join(
                f"{r['发出日']}{r['周期']}({r['实际变动bp']:+.1f}bp)"
                for _, r in flat.iterrows()))

    L += ["", "优化建议（基于当前样本，样本尚小仅作方向参考）:",
          "  1. 样本积累到 20+ 条后再评估各腿优劣，现在不下结论；",
          "  2. 每次运行 40 后重跑本脚本，台账自动滚动对账；",
          "  3. 观察点：LLM修正与融合在 h5 短端的分歧（当前互有胜负）。",
          "=" * 70]
    report = "\n".join(L)
    print(report)
    (OUT / "信号对账汇总.txt").write_text(report, encoding="utf-8")
    print(f"\n输出: {OUT / '信号对账.csv'} / 信号对账汇总.txt")


if __name__ == "__main__":
    main()
