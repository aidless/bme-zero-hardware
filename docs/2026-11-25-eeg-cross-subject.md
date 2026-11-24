# 2026-11-25 EEGBCI 跨被试外部验证（subject 2–5）

## 结论（运行时实测，2026-11-24 UTC）

- 在 EEGBCI/EEGMMIDB v1.0.0 subject 2–5 的 runs 4/8/12（每人 45 trials）上，CSP(4, Ledoit-Wolf, log) + StandardScaler + LR 的 subject-specific 5-fold OOF acc 分别为 0.4889 / 0.5778 / 0.5556 / 0.4889，AUC 为 0.5099 / 0.6225 / 0.6364 / 0.5278（macro acc 0.5278，macro AUC 0.5741）。
- LOSO（每人被独立留出，另 3 人训练）ensemble acc 为 0.4889 / 0.4444 / 0.4222 / 0.6222，AUC 为 0.4802 / 0.4644 / 0.4407 / 0.6310；pooled LOSO acc 0.4944、AUC 0.4851。
- pooled LOSO 相对 subject-specific macro acc 掉 3.33 个百分点，AUC 掉 0.0891；acc bootstrap 95% CI [0.4222, 0.5667]，AUC CI [0.3979, 0.5701]（2000 次 trial 级有放回重采样）。CI 覆盖 0.5，本小样本协议下无跨被试优势的证据。
- S001 原有 0.7600 ± 0.0514 是单被试结果；本测试表明它不能外推到 subject 2–5。文献中 CTNet subject-specific 82–88% vs cross-subject 58–76% 的差距，在本复现的 CSP+LR 上表现为 subject-specific macro 0.5278 → LOSO pooled 0.4944，而非 0.76 → 0.58–0.76。

## 协议与口径

- 数据：`eegbci.load_data(subjects=[2,3,4,5], runs=[4,8,12], path='data/eeg/mne_data', update_path=False)`；12 个 EDF 的 SHA-256 见 `data/eeg/eeg_cross_subject_metrics.json` 的 `source_files`。
- 预处理：每被试独立 64ch、160 Hz、FIR 7–30 Hz（firwin）、epoch −0.5–4.0 s、baseline −0.5–0 s；T1=left、T2=right。每名被试每个 run 15 个 epoch（左/右 7/8、8/7、8/7 等），合计每人 45 个 epoch（左右按 subject 不同为 23/22 或 22/23）。
- Subject-specific：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`；每折 CSP + scaler + LR 在训练折 fit，OOF 后计算 acc/AUC。CI 对 OOF trial 有放回重采样。
- LOSO：留一名被试测试，其他三名被试的 135 个 trial 组成训练集；只在训练集上做 stratified shuffled 5-fold。每个 fold 模型对整名留出被试预测 `predict_proba[:,1]`，主口径是 5 个概率的算术平均；另报把全部 135 个训练 trial 重新 fit 的 secondary full-refit 结果。pooled 是 4 名留出被试的 180 个 trial 级 OOF 预测汇总（trial 加权，每人等样本量）。
- 泄漏控制：CSP、StandardScaler、LogisticRegression 全在 `sklearn.pipeline.make_pipeline` 内；`fit` 调用只接收训练被试/训练折索引，留出被试从未参与 fit、Scaler 统计或阈值选择。
- 标签阈值：y ∈ {1,2}，`predict_proba[:,1] >= 0.5` 必须映射为标签 2，否则为标签 1。首版误用 `(prob >= 0.5).astype(int)`，导致所有 y=2 判错、subject-specific acc 被系统性压到 0.2444；已在 `bench_eeg_cross_subject.py` 的 `metrics()` 与 `bootstrap_ci()` 同步改为 `np.where(prob >= 0.5, 2, 1)` 后完整重跑。最终 JSON/图均来自修复后的运行。
- 口径提醒：subject-specific 对照用四名被试的 macro 均值；pooled LOSO 是 trial 级加权。两者样本量相等，因此在这里等价，但换数据集时不能默认等价。

## 产物与复现

- 脚本：`bench_eeg_cross_subject.py`（sha256 见 JSON `artifacts.script_sha256`）。
- 指标：`data/eeg/eeg_cross_subject_metrics.json`（含 12 个 EDF sha256、每 fold 明细、LOSO 逐被试、bootstrap CI、环境版本）。
- 运行日志：`data/eeg/eeg_cross_subject_run.log`。
- 图：`figures/eeg_cross_subject.png`；左为 acc、右为 AUC，蓝=subject-specific、橙=LOSO，灰色虚线=0.5 机会水平。
- 命令：`python3 -u bench_eeg_cross_subject.py`；运行耗时 143.7 s；Python 3.12.3、MNE 1.13.2、scikit-learn 1.9.1、NumPy 2.5.2、SciPy 1.18.1。

## 关键教训

1. 标签编码一致性和阈值映射必须写成对 y={1,2} 显式的 `np.where(prob>=0.5,2,1)`；`astype(int)` 得到 0/1 会把正类全部判错，AUC 仍正常，因此 AUC 正常不能证明 acc 正确。
2. 跨被试验证必须真正把被试从训练数组里排除，并在该被试上独立评估；只引用文献的 subject-specific/cross-subject 差距不能替代实测。
3. 小样本（每人 45 trial、CSP 4 分量）下，4 名被试的 acc 均围绕 0.5，LOSO 掉幅 3.33 pp 且 CI 覆盖 0.5；S001 的高分更可能含被试内过拟合/个体差异，不能作为跨被试泛化证据。
