# 预注册 2 · 制药 · scaffold split 实现差异的效应量

> 冻结时间：2026-11-26T00:40+08:00（跑之前写死，跑后不改）
> 冻结方式：本文件 sha256 记入 `data/prereg/prereg_mol.sha256`
> 数据：BBBP（https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv），canonical 去重后 n=1975，671 个 Bemis-Murcko scaffold
> 已有产物（历史对照）：`data/gap_metrics.json`（已测 deepchem_exact_split 单因素 −0.2188）

## 一、唯一主指标

**scaffold split 实现差异的效应量** = AUC(我们的 shuffle 平局组切分) − AUC(DeepChem 按 (len, first_index) 降序切分)，同一 RF500 + ECFP4-2048 配置，配对 bootstrap 95% CI（2000 次）。

## 二、判定阈值（跑之前写死）

- CI **排除 0** → 判「split 实现差异有可测效应」，并给出方向。
- CI 跨 0 → 判「无证据」，此前 0.23 gap 的归因需重述。

## 三、为什么值得再跑

前一轮 gap 归因是**单因素逐切**，未给该因素的配对 CI；本预注册补上区间估计，且把「0.23 gap 中 split 占多少」量成一个可引用的带 CI 数字。

## 四、多重性

单一判定，无多重性。

## 五、失败处理

DeepChem 不可导入 → 用自己实现的 (len, first_index) 降序切分复现（与 DeepChem 同规则），并在产物注明「自实现复现」。

## 六、复算

`python3 bench_prereg_mol.py` 输出 `data/prereg/prereg_mol_metrics.json`，含两切分 AUC、差值、CI、判定。