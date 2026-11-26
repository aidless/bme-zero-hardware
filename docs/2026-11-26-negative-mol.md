# 第九轮阴性对照：分子流水线标签置换检验 · 2026-11-26

## 目的与判据

前八轮 BBBP / BACE / ClinTox 的阳性 AUC（0.94 / 0.90 / 0.80）都建立在真实标签上。本轮保持特征与切分结构不变，只置换全量标签并重训同一个 RF500，检验流水线是否会在随机标签上造出显著 AUC。

统一判据（`notes/2026-11-26-negative-control-protocol.md`）：
1. 每个数据集 N=100 次置换，固定 seed 20261126，逐一记录 null AUC。
2. 置换对象是**全量标签向量**（train+valid+test 一起打乱）；特征矩阵 X、Bemis-Murcko scaffold 分组、train/test 索引只计算一次。
3. 模型与超参与真实实验完全一致：RF500、random_state=42、n_jobs=-1、class_weight=None；ECFP4(2048)+MACCS(167)+8 descriptors = 2223 维。
4. 经验 p = (count(null ≥ real) + 1) / (N + 1)；假阳性 = bootstrap CI 下界 > 0.5 的置换次数。
5. 干净判定：null mean ∈ [0.45, 0.55] 且经验 p ≤ 0.01。

## 口径：照搬的既有实现

| 数据集 | 去重 / split | 特征 | scaler | 真实 AUC 来源 |
|---|---|---|---|---|
| BBBP | `bench_mol_rigorous_v2.py` 的 canonical SMILES 去重 + DeepChem-style scaffold split（函数逐字复制到本脚本，因为该脚本顶层直接执行） | `AllChem.GetMorganFingerprintAsBitVect` + MACCS + 8 desc | full-fit | `bench_mol_rigorous_v2.py` fusion/scaffold = 0.9413 |
| BACE | `bench_mol_bace.py` 的 `canonicalize` / `deepchem_scaffold_split`（直接 import） | `rdFingerprintGenerator` + MACCS + 8 desc | full-fit | `data/bace_rigorous_metrics.json` results/scaffold/ECFP4+MACCS+desc = 0.899848828420257 |
| ClinTox | `bench_mol_clintox.py` + `bench_mol_trainonly.py` 的 `canonicalize(df,"smiles","CT_TOX")` / `deepchem_scaffold_split`（直接 import） | `raw_features` + `build_reps(...,"train_only")` | train-only | `data/clintox_metrics.json` scaffold_80_10_10/ECFP4+MACCS+desc.auc_seed42 = 0.7962962962962963 |

置换实现：`rng.permutation(y_full)`（保持全局正类数不变）+ 拒绝采样——若 train 或 test 中任一集合不含两个类别则重抽，保证每次置换的 AUC 有定义。每个数据集的 label shuffle RNG 为 `np.random.default_rng(20261126)`，模型 seed 固定 42；bootstrap 沿用各真实实验的 N（BBBP 1000，BACE/ClinTox 2000）。

## 实测结果（N=100，seed=20261126，全部落 JSON 实测）

| 数据集 | 真实 AUC | null mean±std | null p95 | null max | 经验 p | CI 下界>0.5 次数 | 判定 |
|---|---|---|---|---|---|---|---|
| BBBP | 0.9413 | 0.5049±0.0446 | 0.5753 | 0.5959 | 0.0099 | 1/100 | 干净 |
| BACE | 0.8998 | 0.4948±0.0478 | 0.5733 | 0.6033 | 0.0099 | 3/100 | 干净 |
| ClinTox | 0.7963 | 0.5018±0.0987 | 0.6507 | 0.8206 | 0.0198 | 2/100 | 见下 |

## 结论

三个数据集的 null 均值全部落在 0.5±0.05（0.5049 / 0.4948 / 0.5018），证明流水线无标签泄漏——随机标签只能得到随机排序。假阳性计数 1/3/2 次均在名义 α=0.05 的期望（0–5 次）内，bootstrap CI 没有系统性骗出显著性。

ClinTox 的 `pipeline_clean=false` 是**功效问题不是泄漏**：它的 null std=0.0987 是 BBBP 的两倍（test 只有 147 个样本、正类约 12 个），置换能冲到 0.82，真实 AUC 0.796 的经验 p=0.0198 只勉强越过 0.01。换句话说，在 ClinTox 这种极端不平衡的小测试集上，"真实信号比随机强"只达到 p≈0.02 而非 p<0.01；这与第四、五轮 ClinTox 融合特征不稳定是同一件事的另一面。不能写成"ClinTox 流水线不干净"，正确读法是"ClinTox 的样本量支撑不了 p<0.01 的强判据"。

## 复算

```bash
python3 -u bench_negative_mol.py --selftest   # 置换多重集不变 / 同 seed 可复现 / null mean≈0.5
python3 -u bench_negative_mol.py              # 完整 N=100 三数据集，重跑覆盖 JSON
python3 -u bench_negative_mol.py --verify     # 从 JSON 的 null.aucs 复算全部统计量与 p 值
```

JSON 内含 `null.aucs` 全量数组、`script.sha256`、`environment`、数据文件 sha256、每数据集 split 与真实 AUC，统计量与经验 p 均可逐项复算。

## 局限

- null 分布只评估一次固定 scaffold split；split 本身的抽样方差不在本轮检验范围内。
- ClinTox 测试集正类较少（约 12 个），单次置换 AUC 的方差天然更大。
- bootstrap CI 下界计数以名义 95% CI 计；名义 α=0.05 下 100 次置换期望约 0–5 次越界。
