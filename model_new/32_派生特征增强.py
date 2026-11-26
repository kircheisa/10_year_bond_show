# -*- coding: utf-8 -*-
"""
派生特征增强：在 43v3 基础上构建 F6~F12 七族信息增量特征 -> 43v4

设计原则：只提取"原始列组合/滚动统计"才能表达、而逐列三件套(z/chg/pct)
无法直接给出的信息；全部因果（trailing 窗口，不含未来）；周/月频不涉及。

族（对应旧库无泄漏Top16与43v2新库Top35的证据）：
  F6 资金面预期:  IRS1Y-DR007(政策预期) / DR007-DR001(期限溢价) /
                  隔夜占比(旧库Top9) 及其5日变化 / DR001波动
  F7 汇率政策:    逆周期强度 = 中间价-即期汇率（z/chg20）
  F8 股债联动:    沪深300 20/60日累计回报(由涨跌幅cumsum)、股债swing20
                  (旧库Top8)、利率-权益滚动相关
  F9 机构行为深化: 交易盘(基金+券商+理财) vs 配置盘(保险+货基)净买入差、
                  五机构分歧度、基金净买入×利率变化滚动相关（2022-07起）
  F10 曲线形态:    蝶式利差 3Y-(1Y+5Y)/2 与 5Y-(3Y+10Y)/2、斜率5日变化
  F11 波动/联动:   vol_of_vol(43v2 Top5)、rate_std_60(Top4)、
                  利率×美元/铜/标普滚动相关 + corr_break_dollar(旧库Top1)
  F12 趋势回归:    trailing OLS 斜率/R²/偏离z（43v2 Top35占5席的lr__族）
  F13 日历:        距月末/季末天数、月末/季末临近哑变量（旧库Top2）
  F14 DR001传导链: 政策锚→DR001→套息空间→长端，五个互补维度——
                  ①套息利差(国债-资金中枢20日均值)及一年分位数 ②DR001相对政策锚
                  的偏离/趋势/连续天数 ③交易所与银行间隔夜利差(资金分层，GC001代理)
                  ④DR001已实现波动率及突变 ⑤央行逆回购操作量/频率/估算净投放

输入: data_ori/2026_ori_data_new.xlsx + data_new/43v3_features.csv（复用其
z 列做机构合成，避免重复计算；网格 index 与 43v3 完全一致）
输出: data_new/43v4_features.csv（=43v3全部 + 新族），h10 版本内容相同，
      标签直接复用 43v3_labels_bin{,_h10}.csv
      data_new/43v4_build_meta.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data_ori" / "2026_ori_data_new.xlsx"
OUT_DIR = BASE / "data_new"

SHEET_DAILY_B = "滚动稳健标准化日频+买卖行为"
SHEET_DAILY = "滚动稳健标准化日频"

YIELD_COL = "中国:国债到期收益率:10年"
NETBUY = ["净买入：券商自营", "净买入：保险", "净买入：基金",
          "净买入：理财", "净买入：货币基金"]
Z_WIN, Z_MIN = 252, 120


# ---------- 与 30 脚本一致的装载 ----------
def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """重复列名保留非零有效值最多的一列；无指标名的列(nan)整列丢弃。
    新版 Wind 导出会出现 同名列新旧并存(如市净率旧列止于2022) 与 无名列。
    必须在加 #编号 后缀之前调用，否则新旧两列会被后缀拆成两个名字都保留。"""
    unnamed = [c for c in df.columns if isinstance(c, str) and c == "nan"]
    df = df.drop(columns=unnamed)
    keep = []
    for name in dict.fromkeys(df.columns):          # 按首次出现顺序去重
        same = [c for c in df.columns if c == name]
        if len(same) == 1:
            keep.append(name)
            continue
        best = max(same, key=lambda c: int((df[c] != 0).sum()))
        keep.append(best)
    return df[keep]


def load_sheet(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """定位'指标名称'行，返回数值矩阵（index=日期升序）"""
    raw = xl.parse(sheet, header=None)
    hi = raw.index[raw[0] == "指标名称"][0]
    names = list(raw.iloc[hi, 1:])
    for i, n in enumerate(names):              # 重复指标名加编号
        if names[: i + 1].count(n) > 1:
            names[i] = f"{n}#{i}"
    df = raw.iloc[hi + 1:].copy()
    df.columns = ["date"] + names
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])          # 新导出的分组/空日期分隔行
    df = df.sort_values("date").set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")
    return dedupe_columns(df)

def drop_exact_dups(d: pd.DataFrame) -> pd.DataFrame:
    return d.loc[:, ~d.T.duplicated()]


def robust_z(s: pd.Series, win: int = Z_WIN, minp: int = Z_MIN) -> pd.Series:
    med = s.rolling(win, min_periods=minp).median()
    q1 = s.rolling(win, min_periods=minp).quantile(0.25)
    q3 = s.rolling(win, min_periods=minp).quantile(0.75)
    iqr = (q3 - q1).clip(lower=1e-9)
    return (s - med) / iqr


def rolling_corr(a: pd.Series, b: pd.Series, win: int) -> pd.Series:
    mp = max(10, win // 3)
    return a.rolling(win, min_periods=mp).corr(b)


def trailing_ols(s: pd.Series, win: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """因果趋势回归：斜率/R²/残差z（只回归 t-win+1..t）"""
    mp = max(60, win // 2)
    x = np.arange(win, dtype=float)
    x_c = x - x.mean()
    sxx = (x_c ** 2).sum()

    def f(v: np.ndarray) -> tuple[float, float, float]:
        m = np.isfinite(v)
        if m.sum() < mp:
            return np.nan, np.nan, np.nan
        vv = v[m]
        xx = x_c[m]
        sxy = (xx * (vv - vv.mean())).sum()
        slope = sxy / (xx ** 2).sum()
        yhat = vv.mean() + slope * xx
        ssr = ((vv - yhat) ** 2).sum()
        sst = ((vv - vv.mean()) ** 2).sum()
        r2 = 1 - ssr / sst if sst > 1e-12 else np.nan
        resid = v[-1] - (vv.mean() + slope * x_c[m][-1])
        return slope, r2, resid

    out = np.array([f(v) for v in
                    np.lib.stride_tricks.sliding_window_view(
                        s.to_numpy(dtype=float), win)])
    idx = s.index[win - 1:]
    return (pd.Series(out[:, 0], idx), pd.Series(out[:, 1], idx),
            pd.Series(out[:, 2], idx))


# ---------- F14 参数 ----------
ON_QTY_COL = "中国:逆回购数量:隔夜"       # 隔夜逆回购操作量（只在日频表里）
ON_RATE_COL = "中国:逆回购利率:隔夜"      # 数据表目前没有这一列；补充后自动优先使用
OVERNIGHT_OMO_RATE = 1.25                 # 隔夜逆回购操作利率(%)，按业务口径设定，遇调整需更新
OMO_LAG = {"隔夜": 1, "7天": 5, "14天": 10, "21天": 15, "28天": 20}   # 期限 -> 到期所需交易日数


def policy_anchor(plain: pd.DataFrame,
                  grid: pd.DatetimeIndex) -> pd.Series | None:
    """政策锚利率：隔夜逆回购成为日常工具之前取7天逆回购利率，之后取隔夜逆回购利率。
    政策利率是阶梯函数，对齐到交易日网格后无限前向填充。"""
    if ON_RATE_COL in plain.columns:
        a = plain[ON_RATE_COL].mask(plain[ON_RATE_COL] == 0).dropna()
        return a.reindex(grid).ffill() if len(a) else None
    r7 = plain.get("中国:逆回购利率:7天")
    if r7 is None:
        return None
    a = r7.mask(r7 == 0).copy()
    if ON_QTY_COL in plain.columns:
        on = plain[ON_QTY_COL].mask(plain[ON_QTY_COL] == 0).dropna()
        if len(on):
            a.loc[a.index >= on.index.min()] = OVERNIGHT_OMO_RATE
    return a.reindex(grid).ffill()


def signed_streak(s: pd.Series) -> pd.Series:
    """连续同向天数：连续n天为正记+n，连续n天为负记-n，缺失记0"""
    sign = np.sign(s).fillna(0.0)
    run = (sign != sign.shift()).cumsum()
    return sign * sign.abs().groupby(run).cumsum()


def main() -> None:
    base = pd.read_csv(OUT_DIR / "43v3_features.csv", index_col=0,
                       parse_dates=True)
    grid = base.index

    xl = pd.ExcelFile(SRC)
    behavior = drop_exact_dups(load_sheet(xl, SHEET_DAILY_B))
    plain = drop_exact_dups(load_sheet(xl, SHEET_DAILY))
    shared = [c for c in behavior.columns if c in plain.columns]
    daily_raw = behavior.combine_first(plain[shared])
    daily_raw = daily_raw.mask(daily_raw == 0)
    C = daily_raw.reindex(grid).ffill(limit=20)

    y = C[YIELD_COL]
    feat = pd.DataFrame(index=grid)
    notes: list[str] = []

    def put(name: str, s: pd.Series, z: bool = True, extra: tuple = ()) -> None:
        s = s.astype(float)
        if z:
            feat[f"{name}__z"] = robust_z(s)
        for k, v in extra:
            feat[f"{name}__{k}"] = v

    # ---- F6 资金面预期 ----
    if "利率互换:FR007:1年" in C and "DR007" in C:
        put("资金__IRS1Y-DR007", C["利率互换:FR007:1年"] - C["DR007"],
            extra=[("chg20", (C["利率互换:FR007:1年"] - C["DR007"]).diff(20))])
    if "DR007" in C and "DR001" in C:
        put("资金__DR007-DR001", C["DR007"] - C["DR001"],
            extra=[("chg20", (C["DR007"] - C["DR001"]).diff(20))])
    if "成交量:R001" in C and "成交量:R007" in C:
        ratio = C["成交量:R001"] / (C["成交量:R001"] + C["成交量:R007"])
        put("资金__隔夜占比", ratio,
            extra=[("chg5", ratio.diff(5)), ("chg20", ratio.diff(20))])
    if "DR001" in C:
        put("资金__DR001波动20", C["DR001"].diff().rolling(20, 10).std())

    # ---- F7 汇率政策 ----
    if "中间价:美元兑人民币" in C and "CFETS:即期汇率:美元兑人民币" in C:
        pmid = C["中间价:美元兑人民币"] - C["CFETS:即期汇率:美元兑人民币"]
        put("汇率__逆周期强度", pmid, extra=[("chg20", pmid.diff(20))])

    # ---- F8 股债联动 ----
    if "沪深300指数:涨跌幅" in C:
        ret1 = C["沪深300指数:涨跌幅"]
        ret20 = ret1.rolling(20, 15).sum()
        ret60 = ret1.rolling(60, 40).sum()
        feat["股债__股20日回报"] = ret20
        feat["股债__股60日回报"] = ret60
        # 债近似回报 = -久期8 × Δy(bp)；swing>0 = 股强债弱
        dy20bp = y.diff(20) * 100
        feat["股债__swing20"] = ret20 + 8 * dy20bp / 100
        feat["股债__corr_利率权益20"] = rolling_corr(y.diff(), ret1, 20)
        feat["股债__corr_利率权益60"] = rolling_corr(y.diff(), ret1, 60)

    # ---- F9 机构行为深化（复用 43v3 的净买入 z 列保证口径一致）----
    zb = {c: base[f"{c}__z"] for c in NETBUY if f"{c}__z" in base.columns}
    if len(zb) == len(NETBUY):
        trad = (zb["净买入：基金"].fillna(0) + zb["净买入：券商自营"].fillna(0)
                + zb["净买入：理财"].fillna(0))
        alloc = (zb["净买入：保险"].fillna(0) + zb["净买入：货币基金"].fillna(0))
        feat["机构__交易盘净买入"] = trad
        feat["机构__配置盘净买入"] = alloc
        feat["机构__交易配置差额"] = trad - alloc
        div = pd.DataFrame(zb).std(axis=1)
        feat["机构__分歧度"] = div.where(pd.DataFrame(zb).notna().any(axis=1))
        fund_raw = C["净买入：基金"]
        feat["机构__corr_基金净买入利率20"] = rolling_corr(
            fund_raw.rolling(5, 1).sum(), y.diff(), 20)
        notes.append("F9机构族 2022-07 起（净买入数据起点+预热）")
    else:
        notes.append(f"F9缺净买入z列，仅{len(zb)}/5，跳过机构深化族")

    # ---- F10 曲线形态 ----
    y1, y3, y5, y10 = (C.get("中国:国债到期收益率:1年"),
                       C.get("中国:国债到期收益率:3年"),
                       C.get("中国:国债到期收益率:5年"), y)
    if y1 is not None and y3 is not None and y5 is not None:
        fly135 = y3 - (y1 + y5) / 2
        put("曲线__蝶式1-3-5", fly135, extra=[("chg20", fly135.diff(20))])
    if y3 is not None and y5 is not None:
        fly3510 = y5 - (y3 + y10) / 2
        put("曲线__蝶式3-5-10", fly3510, extra=[("chg20", fly3510.diff(20))])
    if y1 is not None:
        slope = y10 - y1
        feat["曲线__斜率chg5"] = slope.diff(5)
        feat["曲线__斜率chg20"] = slope.diff(20)

    # ---- F11 波动/联动 ----
    dy = y.diff()
    std20 = dy.rolling(20, 10).std()
    feat["波动__rate_std_60"] = dy.rolling(60, 30).std()
    feat["波动__vol_of_vol_60"] = std20.rolling(60, 30).std()
    feat["波动__rate_std_ratio"] = std20 / dy.rolling(60, 30).std().clip(
        lower=1e-9)
    if "美元指数" in C:
        cd20 = rolling_corr(C["美元指数"].diff(), dy, 20)
        cd60 = rolling_corr(C["美元指数"].diff(), dy, 60)
        feat["联动__corr_利率美元20"] = cd20
        feat["联动__corr_利率美元60"] = cd60
        feat["联动__corr_break_dollar"] = (cd20 - cd60).abs()
    if "期货结算价(活跃合约):阴极铜" in C:
        feat["联动__corr_利率铜20"] = rolling_corr(
            C["期货结算价(活跃合约):阴极铜"].diff(), dy, 20)
        feat["联动__corr_利率铜60"] = rolling_corr(
            C["期货结算价(活跃合约):阴极铜"].diff(), dy, 60)
    if "美国:标准普尔500指数" in C:
        feat["联动__corr_利率标普20"] = rolling_corr(
            C["美国:标准普尔500指数"].diff(), dy, 20)
        feat["联动__corr_利率标普60"] = rolling_corr(
            C["美国:标准普尔500指数"].diff(), dy, 60)

    # ---- F12 趋势回归（trailing OLS，替代泄漏的 STL 族）----
    slope120, r2_120, dev120 = trailing_ols(y, 120)
    slope252, r2_252, dev252 = trailing_ols(y, 252)
    feat["趋势__slope_120"] = slope120.reindex(grid)
    feat["趋势__r2_120"] = r2_120.reindex(grid)
    feat["趋势__slope_252"] = slope252.reindex(grid)
    feat["趋势__r2_252"] = r2_252.reindex(grid)
    feat["趋势__dev_z_252"] = robust_z(dev252.reindex(grid))

    # ---- F13 日历 ----
    dim = grid.to_series().dt.days_in_month
    dom = grid.day
    days_to_me = (dim - dom).astype(float)
    feat["日历__距月末天数"] = days_to_me
    feat["日历__月末3日内"] = (days_to_me <= 2).astype(int)
    q_end = grid.to_period("Q").to_timestamp(how="end").normalize()
    days_to_qe = (q_end - grid).days.astype(float)
    feat["日历__距季末天数"] = days_to_qe
    feat["日历__季末3日内"] = (days_to_qe <= 2).astype(int)

    # ---- F14 DR001传导链（政策锚→DR001→负债/杠杆成本→短端→套息空间→长端）----
    # 五个互补维度，全部 trailing 窗口。资金中枢用20日均值而非HP滤波：
    # HP滤波是双边平滑，会带入未来信息。
    dr001 = C.get("DR001")
    if dr001 is not None:
        hub = dr001.rolling(20, 10).mean()
        feat["传导__dr001_ma20"] = hub
        feat["传导__dr001_chg5"] = dr001.diff(5)
        # 维度1 估值保护：套息利差=国债收益率-资金中枢。保留符号（1年常为负），附一年分位数
        for tenor, yn in [("1年", y1), ("5年", y5), ("10年", y10)]:
            if yn is not None:
                cs = yn - hub
                feat[f"传导__carry_{tenor}"] = cs
                feat[f"传导__carry_{tenor}_pct250"] = cs.rolling(
                    250, 120).rank(pct=True)
        # 维度2 政策态度：DR001相对政策锚的偏离。水平、趋势、连续同向天数
        anchor = policy_anchor(plain, grid)
        if anchor is not None:
            dev = dr001 - anchor
            feat["传导__dev_anchor"] = dev
            put("传导__dev_anchor", dev,
                extra=[("ma5", dev.rolling(5, 3).mean()),
                       ("chg5", dev.diff(5)), ("chg10", dev.diff(10)),
                       ("streak", signed_streak(dev))])
        # 维度3 传导摩擦：交易所与银行间隔夜利差。数据表没有R001利率，用GC001作非银成本代理
        if "GC001:收盘价" in C:
            tier = C["GC001:收盘价"] - dr001
            feat["传导__tier_gc"] = tier
            put("传导__tier_gc", tier,
                extra=[("ma5", tier.rolling(5, 3).mean())])
        # 维度4 套息确定性：DR001已实现波动率、一年分位数、短长比（突变）
        dchg = dr001.diff()
        dvol20 = dchg.rolling(20, 10).std()
        feat["传导__dr001_vol20"] = dvol20
        feat["传导__dr001_vol20_pct250"] = dvol20.rolling(250, 120).rank(pct=True)
        feat["传导__dr001_vol20_ratio"] = dvol20 / dchg.rolling(
            60, 30).std().clip(lower=1e-9)
    # 维度5 央行即时操作。操作量只在日频表里，且0=当日无操作，取原始值不做0->NaN
    qty = {c: plain[c] for c in plain.columns if "逆回购数量:" in c}
    if qty:
        omo = pd.DataFrame(qty).reindex(grid).fillna(0.0)
        total = omo.sum(axis=1)
        # 净投放=操作量-到期量。表里没有到期量，按期限顺延估算（隔夜次日到期、
        # 7天≈5个交易日、14天≈10个…），长假前后会有误差，所以命名带 est
        net = pd.Series(0.0, index=grid)
        for c in omo.columns:
            lag = OMO_LAG.get(c.split(":")[-1])
            if lag:
                net = net + (omo[c] - omo[c].shift(lag))
        gross5 = total.rolling(5, 1).sum()
        feat["传导__omo_gross_5d"] = gross5
        feat["传导__omo_gross_20d"] = total.rolling(20, 1).sum()
        feat["传导__omo_freq_5d"] = (total > 0).rolling(5, 1).sum()
        feat["传导__omo_freq_20d"] = (total > 0).rolling(20, 1).sum()
        feat["传导__omo_net_est"] = net
        feat["传导__omo_net_est_5d"] = net.rolling(5, 1).sum()
        feat["传导__omo_net_est_20d"] = net.rolling(20, 1).sum()
        if ON_QTY_COL in omo.columns:
            feat["传导__omo_隔夜占比_5d"] = (
                omo[ON_QTY_COL].rolling(5, 1).sum()
                / gross5.where(gross5 > 0))
        first_op = total.index[total > 0].min()
        notes.append(f"F14传导族：央行操作量取日频表{len(qty)}个期限列"
                     + (f"（{first_op.date()}起有操作）" if pd.notna(first_op) else ""))

    # ---- 合并输出 ----
    new_cols = [c for c in feat.columns if c not in base.columns]
    out = base.join(feat[new_cols], how="inner")
    assert len(out) == len(base), "网格不一致"
    out.to_csv(OUT_DIR / "43v4_features.csv", encoding="utf-8-sig")
    out.to_csv(OUT_DIR / "43v4_features_h10.csv", encoding="utf-8-sig")

    nan_rate = out[new_cols].isna().mean()
    lines = ["=" * 60,
             f"43v4 = 43v3({base.shape[1]}) + 新族{len(new_cols)}列 "
             f"共{out.shape[1]}列  网格{out.index.min().date()}~"
             f"{out.index.max().date()} {len(out)}行",
             f"新族NaN率: min={nan_rate.min():.3f} 中位={nan_rate.median():.3f} "
             f"max={nan_rate.max():.3f}",
             *notes,
             "标签复用 43v3_labels_bin{,_h10}.csv",
             "=" * 60]
    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "43v4_build_meta.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
