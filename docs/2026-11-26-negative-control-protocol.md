# 第九轮阴性对照（标签置换）统一协议 · 2026-11-26

## 目的
前八轮所有"阳性"数字（AUC 0.92~0.99）都在真实标签上得到。本轮做**阴性对照**：保持特征 X 与切分结构完全不变，只把标签 y 随机置换，重跑完整"特征→模型→评估"流水线。若流水线干净（无泄漏、无标签回灌），置换后的 AUC 应围绕 0.5（随机猜测）；若置换后仍得高 AUC，则前面所有阳性结论都不可信。

## 统一判据（三条流水线必须一致）
1. 每条流水线每个数据集做 **N≥100 次标签置换**（seed 固定，逐一记录）。
2. 置换对象：**全量标签**（train+val+test 一起打乱）；特征矩阵 X 与 scaffold/official split 索引保持不动。
3. 模型与超参与真实标签实验**完全一致**（不得为阴性对照调参）。
4. 每次置换算一次 test AUC；记录 null 分布的 mean±std、median、95th percentile、max、min。
5. 经验 p 值 = (null AUC ≥ 真实 AUC 的次数 + 1) / (N + 1)。
6. 假阳性计数：有多少次置换的 bootstrap CI 下界 > 0.5（应 ≈0，或 ≤ 名义 α×N）。
7. 判定：null mean 应落在 0.5±0.05；真实 AUC 的经验 p 值应 ≤0.01；两者同时成立 = 流水线干净。

## 三条流水线
- **分子**：BBBP / BACE / ClinTox。canonical 去重 + DeepChem scaffold split（照 bench_mol_rigorous_v2.py / bench_mol_bace.py）+ ECFP4+MACCS+desc + RF500 + bootstrap。特征/scaffold 分组只算一次，N 次置换只重训 RF。
- **EEG**：EEGBCI S001–S009 27 EDF，subject-specific 5-fold CSP+LR（照 bench_eeg_extended.py A 臂）。epochs 与 split 固定，N 次置换只重跑 CSP fit + LR。小样本（每人 45 trial），null 方差大本身是发现。
- **影像**：PneumoniaMNIST / DermaMNIST。冻结 Rad-DINO CLS 768d 特征（照 bench_frontier_raddino.py）提一次并缓存，StandardScaler+LR linear probe，N 次置换只重训 scaler+LR。

## 产物（各自独立文件，只新增不改既有 tracked）
- 脚本：bench_negative_{mol,eeg,imaging}.py
- 数字：data/negative/negative_{mol,eeg,imaging}_metrics.json
- 说明：notes/2026-11-26-negative-{mol,eeg,imaging}.md
- 图（可选）：figures/negative_*.png

## 纪律
- 数字全部实测落 JSON，可复算；不得凭估计填数。
- 只新增文件，不改 README.md / BME-DEEP-DIVE.md / 任何既有 git-tracked 文件。
- 脚本开头 `--selftest` 模式验证置换实现（置换后标签分布不变、null 可复现）。
- 完成用 send_message 汇报关键数字与判据结论。
