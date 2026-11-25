# ClinTox 第三数据集融合增益实测（2026-11-26，mol-clintox）

> 起因：BBBP/BACE 上"融合"相对 ECFP4 的增益方向为正但显著性不稳。加第三个数据集 ClinTox 检验。
> 下载文件 `clintox.csv.gz` 的列名是 `smiles,FDA_APPROVED,CT_TOX`，**没有 HIV 列**；
> 本笔记明确用 **CT_TOX（1=临床毒性阳性）** 作终点，FDA_APPROVED 只是监管状态，不参与建模。

## 摘要

ClinTox scaffold split 上融合相对 ECFP4 的 5-seed 平均 AUC 增益为 +0.1268608024691358，
配对 bootstrap 95% CI [0.050115740740740745, 0.20562345679012345]，显著；相对 ECFP4+desc 为
+0.12255154320987655，CI [0.05085956790123455, 0.1953719135802469]，也显著。
但**融合不是最优**：MACCS-only 平均 0.8949，高于融合 0.8036，融合−MACCS = −0.0901，
CI [−0.1508, −0.0349]，显著为负。增益来自 MACCS，desc 加在 MACCS 上反而拖累。
分层随机 80/10/10 sensitivity 同向（融合−ECFP4 +0.0567 显著，融合−MACCS −0.0254 显著为负）。

## 装置与判据

- 脚本：`bench_mol_clintox.py`；产物：`data/clintox_metrics.json`；`--selftest` 通过。
- 数据：`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz`，
  本地 `data/ClinTox.csv.gz`，gzip SHA256 `7324a84a180ad9068d3847677c85d58e8432e82b3ae6ac95cf0e98b3a58d9697`。
- 1484 行 → canonical 去重 1461（invalid 4、dup 19、冲突 19）；CT_TOX pos=103 neg=1358，
  positive rate=0.070500；FDA_APPROVED 计数 0:94 / 1:1390（未用）。
- 主 split：DeepChem 风格 scaffold 80/10/10（等大组洗牌 seed42）；sensitivity：分层随机 80/10/10 seed42。
- 4 表征：ECFP4-2048、MACCS、ECFP4+desc、ECFP4+MACCS+desc（融合）；含 desc 的表征严格 train-only fit scaler。
- RF500、5 seeds 42/0/7/2024/123；seed42 的 2000 次分层 bootstrap CI；融合差值为同 resample 配对、
  跨 5 seed 平均；CI 不跨 0 判显著。

## 实测表（主 split，scaffold）

CT_TOX 折别比例：train n=1168 pos=80 neg=1088（rate 0.0684931506849315）；
valid n=146 pos=11 neg=135（rate 0.07534246575342465）；
test n=147 pos=12 neg=135（rate 0.08163265306122448）——类别极不平衡且 scaffold split 未强制分层。

| 表征 | dim | 5-seed mean | std | seed42 AUC [CI] |
|---|---:|---:|---:|---|
| ECFP4-2048 | 2048 | 0.6760 | 0.0160 | 0.6864 [0.553,0.820] |
| MACCS | 167 | 0.8949 | 0.0047 | 0.8988 [0.841,0.947] |
| ECFP4+desc | 2056 | 0.6802 | 0.0150 | 0.6861 [0.552,0.819] |
| 融合 | 2223 | 0.8036 | 0.0165 | 0.7963 [0.696,0.888] |

配对融合增益（mean，95% CI，显著）：
- vs ECFP4-2048：+0.1268608024691358 [0.050115740740740745, 0.20562345679012345]，True
- vs ECFP4+desc：+0.12255154320987655 [0.05085956790123455, 0.1953719135802469]，True
- vs MACCS：−0.0901 [−0.1508, −0.0349]，True（融合显著更差）

分层随机 80/10/10 sensitivity：train n=1167 pos=83 neg=1084、valid n=147 pos=10 neg=137、
test n=147 pos=10 neg=137。mean AUC：ECFP4 0.8623、MACCS 0.9444、ECFP4+desc 0.8708、融合 0.9188；
融合−ECFP4 +0.0567 CI [0.0154,0.1066]；融合−MACCS −0.0254 CI [−0.0577,−0.0028]。

## 逐条判断

1. 第三个数据集上融合相对 ECFP4 **仍为正且显著**；相对 ECFP4+desc 也正且显著。
2. 但融合相对 MACCS-only 显著为负：所谓"融合增益"不是普适的"越多表征越好"，而是被 MACCS 主导。
3. 跨三数据集一句话：融合相对 ECFP4 的方向在 BBBP（+0.031）、BACE（+0.008，CI 跨 0）、ClinTox（+0.127，显著）均为正，
   但显著性只在 ClinTox 成立，且 ClinTox 上 MACCS-only 反超融合，稳健性有限。

证据等级：A（数据哈希、split 比例、5 seed、配对 bootstrap、JSON 逐字）。复算：
`python3 -u bench_mol_clintox.py --selftest && python3 -u bench_mol_clintox.py`。
