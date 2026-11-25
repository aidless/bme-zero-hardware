# 预注册 1 · EEG 跨被试 · 协方差对齐 vs CSP

> 冻结时间：2026-11-26T00:40+08:00（跑之前写死，跑后不改）
> 冻结方式：本文件 sha256 记入 `data/prereg/prereg_eeg.sha256`，任何改动即破盲
> 数据：EEGBCI PhysioNet v1.0.0，S001–S009 共 9 人 × 45 trials（runs 4/8/12，左/右拳运动想象），64 通道 160 Hz
> 已有产物（只可作历史对照，不参与本预注册判定）：`data/eeg/eeg_extended_metrics.json`

## 一、唯一主指标

**pooled LOSO accuracy**（留一人交叉验证，9 折，把 9 名被试的留出预测拼成 405 trial 级 OOF 后算一个 accuracy）。

- 理由：跨被试泛化的审稿级口径就是 LOSO；pooled 而非 macro 避免小样本被试权重失真。
- 次指标（只报告、不参与判定）：pooled LOSO AUC、per-subject macro acc、校准 ECE。

## 二、两臂定义（冻结）

- **A 臂 = CSP+LR**（对照）：FIR 7–30 Hz → epoch(-0.5,4s) → CSP 4 分量 → StandardScaler → LogisticRegression（默认参数，固定 seed 42）。
- **B 臂 = Tangent Space+LR**（实验）：Covariances(estimator='lwf') → TangentSpace(metric='riemann', ts=10) → StandardScaler → LogisticRegression（默认参数，seed 42）。
- 两臂的 epoch/通道/交叉验证折划分完全一致，唯一差异是特征提取。

## 三、判定阈值（跑之前写死）

- **主判定**：B 臂 pooled LOSO acc 的 95% bootstrap CI（2000 次，trial 级重采样）**下界 > 0.5** → 判「存在跨被试优势」；否则判「无证据」。
- **次判定**：B−A 的 pooled acc 差，配对 bootstrap 95% CI **排除 0** → 判「优于 CSP」。

## 四、多重性

本预注册只含 1 个主判定 + 1 个次判定，不做跨臂 BH（家族=2，Holm 校正后 α=0.025/0.05）。

## 五、失败处理

数据下载失败 → 如实报 unreachable，不用旧数据顶替主判定。两臂任一崩溃 → 报该臂失败，不换臂。

## 六、复算

`python3 bench_prereg_eeg.py` 输出 `data/prereg/prereg_eeg_metrics.json`，含两臂 acc/AUC/CI、判定布尔、两文件 sha256。