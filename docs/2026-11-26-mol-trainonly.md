# BBBP/BACE scaler leakage 修复实测（2026-11-26，mol-trainonly）

> 起因：既有 `bench_mol_rigorous_v2.py`（BBBP）与 `bench_mol_bace.py`（BACE）把 8 个 RDKit 描述符的
> `StandardScaler` 在**全量去重数据**上 fit，然后才做 scaffold split。要回答：严格 train-only fit 后 AUC 掉多少？
> 这个 leakage 能否解释 BBBP 0.9413 与 MoleculeNet Table 6 RF 0.714 的 0.23 gap？

## 摘要

严格 train-only fit 后，BBBP 融合 AUC 从 0.9412603150787697 → 0.9410352588147036（−0.00022505626406610357），
BACE 融合从 0.8998488284202570 → 0.8989984882842026（−0.0008503401360544505）。
两个变化都被 paired bootstrap CI 覆盖 0（BBBP [−0.001650412603150797, 0.0010502625656413384]；
BACE 见 JSON）。**leakage 修复解释不了 0.23 gap，0.9413 的开放性保持。**

## 装置与判据

- 脚本：`bench_mol_trainonly.py`；产物：`data/trainonly_metrics.json`；自检 `--selftest` 通过。
- 数据：BBBP `data/BBBP.csv`，BACE `data/BACE.csv`；canonical SMILES 去重、DeepChem 风格 scaffold split
  （seed 42、等大 scaffold 组洗牌、按组大小降序填 80/10/10）；RF500 seed42；2000 次分层 bootstrap。
- 严格口径：`StandardScaler` 只在 scaffold **train 折** fit，valid/test 只 transform，不 refit。
- 非空性反证（三类同时成立，见 `scaler_proof`）：
  1. `scaler.mean_/scale_` 等于 train 折矩；
  2. 给 held-out 描述施行 +1000 扰动后 train-only 参数不变；
  3. 同样扰动会改变 full-fit 参数（负对照变红）。
  BBBP 证明细节：`train_rows_fit=1580`、`heldout_rows_not_fit=198`、
  `train_mean_l2_vs_full_mean=7.234718801891121`。

## 实测表（seed42；CI 为 2000 次分层 bootstrap 的 95% 百分位区间）

| 数据集 | 表征 | full-fit AUC | train-only AUC | Δ |
|---|---|---:|---:|---:|
| BBBP | ECFP4-2048 | 0.9100 [0.857,0.956] | 0.9100 [0.857,0.956] | 0.0000 |
| BBBP | ECFP4+desc | 0.9385 [0.900,0.971] | 0.9389 [0.901,0.971] | +0.0003750937734433579 |
| BBBP | 融合 | 0.9413 [0.906,0.971] | 0.9410 [0.906,0.970] | −0.00022505626406610357 |
| BACE | ECFP4-2048 | 0.8909 [0.834,0.940] | 0.8909 [0.834,0.940] | 0.0000 |
| BACE | ECFP4+desc | 0.9027 [0.846,0.950] | 0.9015 [0.844,0.949] | −0.0011337868480726376 |
| BACE | 融合 | 0.8998 [0.843,0.947] | 0.8990 [0.843,0.947] | −0.0008503401360544505 |

样本与折别（JSON 原文）：BBBP 2050→1975（invalid 11、dup 64、冲突 10），train n=1580 pos=1208 neg=372、
test n=198 pos=155 neg=43；BACE 1513 unique valid，train n=1210 pos=584 neg=626、test n=152 pos=54 neg=98。

## 逐条判断

1. **AUC 掉多少**：核心融合口径 BBBP −0.000225、BACE −0.000850；ECFP4 本身无描述符故完全不变。
2. **0.9413 开放性是否被解释**：否。leakage 量级 ≤0.001，离 0.23 差两个数量级；根因另找（见 `2026-11-26-mol-gap.md`）。
3. **稳健性**：BBBP 融合 5 seed full-fit mean 0.9384246061515379 / train-only mean 0.9383945986496623，
   seed 扰动（std≈0.003）远大于 leakage 修复量级。

证据等级：A（脚本自证 + 数字落 JSON + 2000 次 bootstrap + 负对照）。复算命令：
`python3 -u bench_mol_trainonly.py --selftest && python3 -u bench_mol_trainonly.py`。
