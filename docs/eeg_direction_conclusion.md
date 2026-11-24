# 脑机接口 / EEG 信号解码 —— 纯计算方向结论

> 本文所有数字均从 `mini_project_eeg.py` 的 stdout 抄录（同一脚本的 JSON 账本见
> `data/eeg/eeg_metrics.json`，原始 stdout 见 `data/eeg/eeg_run_log.txt`）。复现命令：
> `cd 到工作区根目录 && python3 mini_project_eeg.py`。

## 1 数据（真实公开数据，非合成）

- 来源：**EEGBCI / EEG Motor Movement/Imagery Dataset v1.0.0**（PhysioNet，Schalk et al. 2004），
  https://physionet.org/content/eegmmidb/1.0.0/
- 受试者 **S001**，**runs 4 / 8 / 12** = 左拳 vs 右拳运动想象；事件码 `T0=rest, T1=left-fist imagery,
  T2=right-fist imagery`（已在 dataset 页核对事件定义）。
- 64 通道、采样率 **160 Hz**，3 个 EDF 各 2,596,896 bytes（sha256 见 JSON），拼接时长 **374.99 s**。
- 事件计数：`T0=45, T1=23, T2=22`（另有 `BAD boundary=2`、`EDGE boundary=2`，未纳入 epoch）。
- 有效试次 **45**（左 23 / 右 22），epoch 形状 `(45, 64, 721)`，窗口 `-0.5~4.0 s`，基线 `(-0.5, 0)`。

未做任何数据增广，未使用合成数据；本方向全部结论基于真实录制。

## 2 流程与实测数字

处理链：FIR 带通 `7-30 Hz` → 事件切 epoch → 两组特征 → sklearn 逻辑回归 →
`StratifiedKFold(5, shuffle=True, random_state=42)` 交叉验证。随机基线 = **0.5000**。

| 特征 | 模型 | 5 折准确率（均值±std） | 逐折 | CV test accuracy | ROC-AUC |
|---|---|---|---|---|---|
| CSP（4 分量，Ledoit-Wolf 正则） | LogisticRegression(C=1.0) | **0.7778 ± 0.0703** | 0.7778 / 0.8889 / 0.7778 / 0.7778 / 0.6667 | **0.7778** | **0.8775** |
| μ(8–13 Hz)+β(13–30 Hz) log 功率（128 维 → SelectKBest k=20） | LogisticRegression(C=1.0) | 0.6667 ± 0.1988 | 0.7778 / 0.3333 / 0.7778 / 0.8889 / 0.5556 | 0.6667 | 0.6225 |

CSP 混淆矩阵（行=真实，列=预测，左→右）：`[[18, 5], [5, 17]]`。

解读：
- CSP + 逻辑回归在单被试 45 试次上得到 **77.8% 交叉验证准确率、AUC 0.877**，显著高于 0.5 随机水平，说明
  左/右运动想象的 μ/β 频段空间滤波 + 线性分类在公开数据上可复现。
- 频带功率基线只有 66.7%、AUC 0.62，逐折 std 0.199 说明 128 维特征 + 45 样本高维欠定，
  必须靠特征选择/降维；CSP 用协方差空间投影把维度压到 4 维，小样本更稳。
- 数字量级是**演示级**，不是基准成绩：单被试、45 试次、未做被试间泛化，CSP 的 0.0703 折间 std 已说明
  小样本不稳定，不能外推为跨被试或跨数据集性能。

## 3 这个纯计算方向能做什么

1. **解码算法研究**：预处理（带通/陷波/重参考/伪迹剔除）、特征（频带功率、CSP、小波、Riemannian
   协方差、共空间模式）、分类器（LDA/LR/SVM/RF/浅层 CNN）在同一离线数据上的对照与消融。
2. **评估协议研究**：交叉验证、留一试次、跨被试校准（subject transfer / domain adaptation）、
   混淆矩阵与 AUC、置信区间、小样本偏差——都是纯统计/纯计算工作。
3. **公开基准复现**：EEGBCI、BCI Competition IV 2a/2b 等公开集上的基线复现、超参扫描、
   CPU/GPU 上离线跑通全流程，不需要任何采集硬件。
4. **可复现工程**：固定种子、落盘 sha256 账本、脚本化画图；本文件即一个最小可复现样例。

## 4 这个纯计算方向不能做什么

1. **不能做实时采集与硬件在环**：没有放大器、电极、阻抗检查、参考/接地，无法产生新的真实 EEG；
   本方向只能消费已录制的数据，不能替代实验采集。
2. **不能验证硬件层延迟/抖动**：采集链路、无线传输、放大器滤波的真实延迟与抖动，离线仿真给不出真值；
   闭环 BCI 的反馈时序、用户训练与神经适应也无法用离线数据验证。
3. **不能做临床诊断/医学结论**：单被试 45 试次、无独立测试集、无伦理/临床金标准，任何“诊断/疗效”表述
   都超出数据支持范围。
4. **不能声称跨被试/跨设备泛化**：本脚本只跑 S001 三个 run；换被试、换导联、换采样率需重新验证。

## 5 产物清单

- `mini_project_eeg.py` —— 全流程脚本（真实数据，Agg 后端，17.8 s 跑完）
- `data/eeg/mne_data/.../S001/S001R04.edf`, `S001R08.edf`, `S001R12.edf` —— 真实 EDF（共约 7.8 MB）
- `data/eeg/eeg_metrics.json` —— 数字 + 文件 sha256 的溯源账本
- `data/eeg/eeg_run_log.txt` —— 本次 stdout 原文
- `figures/eeg_spectrum.png` —— C3/C4 按想象类别的 grand-average Welch PSD
- `figures/eeg_topomap.png` —— 右减左 log 频带功率头皮分布（μ / β 两图）
- `figures/eeg_confusion_matrix.png` —— CSP 解码器 5 折 CV 混淆矩阵

## 6 已知局限（必须随数字一起引用）

- 单被试、45 试次、单随机种子；未做被试间/跨数据集泛化，未做统计检验与多重比较校正。
- 7–30 Hz 带通在 PSD 之前施加，故频谱图 7 Hz 以下段不代表原始信号，仅作对照。
- `T0` 休息期与 boundary 事件被排除；基线校正用 `-0.5~0 s`。
- 频带功率臂的 128 维特征在 45 样本上高维欠定，结论只对“加 SelectKBest”的该实现成立。