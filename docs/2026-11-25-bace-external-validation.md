# BACE 跨数据集外部验证：BBBP v2 协议在第二个 MoleculeNet 二分类集上的复现

日期：2026-11-25
性质：外部验证实验（新增脚本与产物；未改动 README.md、BME-DEEP-DIVE.md 或任何既有 git-tracked 文件）
复现命令：`python3 -u bench_mol_bace.py --selftest && python3 -u bench_mol_bace.py`

## 实测表

| 项 | 值 |
|---|---|
| 数据源 | `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv`（小写路径 HTTP 200；题面给的大写 `BACE.csv` 实测 HTTP 403） |
| 落盘 | `data/BACE.csv`，3,897,767 bytes，sha256 `f3fb9ce90bada3e2bd6148b0df13f8f8145a357bf87df0dd5b391ede974fc737` |
| 原始表 | 1513 行 × 595 列；关键列 `mol`（SMILES）、`Class`；Class 0=822 / 1=691 |
| canonical 去重 | 1513 → 1513；无效 0、重复 0、重复标签冲突 0（对照 BBBP：2050 → 1975，丢弃 75） |
| scaffold | 671 个 Bemis–Murcko scaffold；DeepChem 式按组大小降序填 80/10/10 |
| split 样本数 | train/valid/test = 1210/151/152；正例 584/53/54；两两 scaffold 重叠均为 0 |
| 特征维度 | ECFP4-2048=2048；MACCS-167=167；ECFP4+desc=2056；ECFP4+MACCS+desc=2223 |
| 模型 | RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight=None) |
| CI | 2000× 分层 bootstrap（正负例分别重采样），百分位 95% CI，固定 bootstrap seed 42 |
| 环境 | Python 3.12.3；rdkit 2026.09.1；scikit-learn 1.9.1；numpy 2.5.2；pandas 2.3.3 |
| 自检 | PASS：canonical 去重、无效剔除、标签冲突检测、零重叠 split、重叠检测器、bootstrap CI |

### AUC-ROC（点估计 + 95% CI）

| 特征 | random split | scaffold split |
|---|---|---|
| ECFP4-2048 | 0.8805 [0.843, 0.914] | 0.8909 [0.834, 0.940] |
| MACCS-167 | 0.8674 [0.824, 0.907] | 0.8654 [0.797, 0.921] |
| ECFP4+desc | 0.8862 [0.850, 0.919] | **0.9027 [0.846, 0.950]** |
| ECFP4+MACCS+desc | **0.8912 [0.856, 0.923]** | 0.8998 [0.843, 0.947] |

配对 bootstrap（同一 test 重采样上融合减单 ECFP4）：random `+0.0108 [-0.000, +0.022]`；scaffold `+0.0088 [-0.005, +0.024]`。两个区间都跨 0。

### 5-seed scaffold 稳定性

| 设置 | seed 42 / 0 / 7 / 2024 / 123 | mean | std(ddof=1) | range |
|---|---|---|---|---|
| 同时重切分 + 换 RF seed | 0.8998 / 0.8768 / 0.8893 / 0.8620 / 0.9128 | 0.8882 | 0.0197 | [0.8620, 0.9128] |
| 固定本轮 scaffold split，仅换 RF seed | 0.8998 / 0.8967 / 0.8981 / 0.8976 / 0.8976 | 0.8980 | 0.0012 | [0.8967, 0.8998] |

split 扰动造成的波动约为 RF seed 扰动的 16 倍（0.0197 / 0.0012），说明本数据集上 scaffold 切分比树模型随机性更主导。

## 逐条判断

1. **协议迁移成立**：BACE 走完与 BBBP v2 相同的 canonical 去重、BM scaffold 分组填充分割、四种特征、RF500、2000× bootstrap 流程；三集合 scaffold 重叠为 0，分割并非随机 split 冒充。随机 split 只作为对照，主结论以 scaffold split 为准。
2. **融合特征的优势在 BACE 收窄**：random split 上完整融合最高（0.8912），scaffold split 上 `ECFP4+desc`（0.9027）略高于完整融合（0.8998）。融合相对单 ECFP4 在两个 split 上都为正（+0.0108 / +0.0088），与 BBBP 的 +0.0166 / +0.0313 同号；但 BACE 的配对 CI 跨 0，且 MACCS 的加入没有在 scaffold 上带来增益。因此“融合跨数据集都优于单 ECFP4”应表述为**点估计同号、统计证据不足、且并非每个 split 都排第一**。
3. **BACE 与 BBBP 的 scaffold 表现差异**：BBBP 融合 scaffold 0.9413 高于 random 0.9288；BACE 融合 scaffold 0.8998 高于 random 0.8912，方向一致但幅度更小（+0.0086 vs +0.0125）。BACE test 正例率 0.355，低于 train 0.483，类分布漂移可能是难度来源之一；未额外重加权，以保持与 BBBP 协议一致。
4. **MoleculeNet 表格数字证伪**：web_fetch 核实 `https://arxiv.org/abs/1703.00564` 题名与版本（arXiv:1703.00564v3），并从 `https://arxiv.org/html/1703.00564v3#S6.T5` 读出 BACE 位于 **Table 5**，不是题面所说的 Table 6。Table 5 scaffold AUC-ROC：RF `0.867±0.008`、KernelSVM `0.862±0.000`、GC `0.783±0.014`、Weave `0.806±0.002`。题面给的 `RF 0.783 / KernelSVM 0.867 / GraphConv 0.806` 与原文不符：0.783 属 GC，0.867 属 RF，0.806 属 Weave，KernelSVM 为 0.862。
5. **官方对标只能定性**：本轮 scaffold 融合 0.8998 相对官方 RF +0.0328、KernelSVM +0.0378、GC +0.1168、Weave +0.0938。但官方数字是 GP 超参搜索后三次平均、默认 ECFP4（原文未钉死位数），本轮固定 RF500、无调参、2223 维融合特征、全量数据先去重且自实现 80/10/10 scaffold 填充；两边不是同口径复现，不写成“同条件超越 SOTA”。
6. **CI 的语义边界**：bootstrap 只量 test 抽样方差；5-seed 实验单独量 split 与 RF seed 方差。融合 scaffold 点估计 std 0.0012（固定 split）与 0.0197（重切分）说明只报 bootstrap CI 会低估整体不确定性。

## 证据等级

- A 级（直接实测）：`data/bace_rigorous_metrics.json` 中全部数字；`bench_mol_bace.py --selftest` PASS；`python3 -u bench_mol_bace.py` 完整 stdout；`data/BACE.csv` 字节数与 sha256。
- B 级（可复算产物）：`bench_mol_bace.py`（协议、随机种子、bootstrap 与 CI 计算均落代码）；1513 行原始表与 label 计数可从 CSV 重算。
- C 级（外部核验）：arXiv abs 页验号 + HTML Table 5 逐格读取，URL 与核验日期记入 JSON 的 `official_reference`。

## limitations

- 描述子 `StandardScaler` 在全量去重数据上拟合后再切分，忠实 BBBP v2，但不是严格 train-only 预处理。
- MoleculeNet 历史 scaffold split 的 fold membership 未逐字节复现；本轮用 BBBP v2 同款 DeepChem 式分组填充。
- 官方模型经过 GP 超参搜索，本轮 RF 固定 500 棵树且未搜索超参；两者只作定性参照。

## 产物

- `bench_mol_bace.py`（新增脚本）
- `data/BACE.csv`（新增数据，预计 gitignore）
- `data/bace_rigorous_metrics.json`（新增完整指标与协议记录）
- `notes/2026-11-25-bace-external-validation.md`（本文件）
