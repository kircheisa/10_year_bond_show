# 10年期国债收益率方向预测（精简生产版）

从 43_华泰证券预处理 主项目梳理出的必备代码与数据，自包含可运行。

## 目录结构

```
├── data_ori/2026_ori_data_new.xlsx   唯一原始数据源（Wind导出，4 sheet）
├── data_new/
│   ├── 43v3_labels_bin{,_h10,_h15,_h20}.csv   四周期标签（下行=1）
│   ├── 预测记录.csv                  预测台账（40脚本追加）
│   ├── llm_key.txt                   智谱API key（勿外传/勿入库）
│   └── out/                          时点预测曲线.html / 大模型修正.*
├── model_new/
│   ├── 30_新版原始数据_特征构建.py    43v3特征(351列)+四周期标签，--horizon
│   ├── 32_派生特征增强.py            43v4(396列, 资金/汇率/股债/曲线/波动/趋势/日历)
│   ├── 38_无买卖行为_特征构建.py     43v4nb(371列, 剔除净买入——当前生产口径)
│   ├── 21_新基础数据_XGB训练.py      滚动回测协议（窗口756/步长63/测试2018起）
│   ├── 40_最新时点预测.py            ★生产入口：自动判最新数据日→选口径→
│   │                                 XGB+NN融合预测→台账→曲线HTML
│   └── 41_大模型信息修正.py          新闻搜集→GLM-5.3-Flash→有界修正(±0.08)
├── model_nn/17_BiGRU训练_新基础数据_二分类.py   BiGRU模型与训练协议（40调用）
└── bond_show/                        本地研究仪表盘（总览含前向隐含点位射线）
```

## 日常运行

```bash
PY=D:/soft/Anaconda3/python.exe
$PY model_new/40_最新时点预测.py       # 数据更新后跑（自动重建特征/训练/预测）
$PY model_new/41_大模型信息修正.py     # 可选：LLM修正
$PY bond_show/export_ori_data.py      # 可选：刷新面板数据
bond_show/start_dashboard.bat         # 启动研究面板 (localhost:8000)
```

## 特征口径（40 自动选择）

- 买卖行为表覆盖发出日 → 含净买入 43v4；否则无净买入 43v4nb（当前常用）
- 特征文件缺失/过期时自动重建链 30(×4周期) → 32 → 38

## 评估基线（诚实口径，2026-11-10实验说明）

- 无净买入 XGB h20：2026 Acc 0.766 / AUC 0.693（多数类基线 0.777）
- 信号分级：≥0.75强烈看下行 / 0.60-0.75看下行 / 0.40-0.60观望 / 其余上行
- 模型缓存：data_new/models/{发出日}/，当日复用，新日自动重建
- 详细实验记录见 data_new/新基础数据_实验说明.md

## 环境依赖

Python (Anaconda D:/soft/Anaconda3 已验证)：pandas numpy openpyxl xgboost
torch(sklearn requests)；GPU 可选（NN训练加速）。
