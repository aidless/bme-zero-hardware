# 概率校准分析（第四轮深挖，2026-11-25）

## 为什么做这个
前三轮六个方向只报 AUC/acc，但审稿标配还要问：模型输出的概率可信吗？
一个 AUC 0.94 的模型，如果预测 0.9 的样本里只有 60% 真是阳性，概率就不可用于临床决策。
按 uncertainty-quantification skill 门禁做：ECE(15 bins)/MCE/Brier/LogLoss + sharpness，
ECE>0.05 用 temperature scaling，且校准集/测试集分开避免同批 fit+eval。

## 协议
- OOF 概率：5-fold StratifiedKFold(seed=42) 的 cross_val_predict predict_proba。
- ECE/MCE：Guo et al. 2017 等宽 15 bin。Brier = mean((p-y)^2)，LogLoss 标准式。
- sharpness = 预测概率的 std（恒报，避免只报 ECE 忽略锐度）。
- Temperature scaling：OOF 概率随机对半切，一半 fit T（最小化 LogLoss），另一半评估校准前后。

## 实测结果（全部 stdout 落 calibration_run.log）

| 方向 | 样本 | ECE(15b) | MCE | Brier | LogLoss | sharpness_std | mean_conf |
|---|---|---|---|---|---|---|---|
| 制药 BBBP ECFP4+RF | 2039 | **0.0409** | 0.2855 | 0.0853 | 0.2956 | 0.2798 | 0.7678 |
| EEG CSP+LR S001 | 45 | **0.1521** | 0.6758 | 0.1425 | 0.4294 | 0.3021 | 0.4883 |

Temperature scaling（OOF 对半切，后半评估）：
- BBBP: T=0.877，ECE 0.0365→0.0375（略升），Brier 0.0865→0.0860，LogLoss 0.2995→0.2980 → **本就校准良好，无需校准**
- EEG: T=0.891，ECE 0.2187→0.1694（降但仍>0.05），Brier 0.1251→0.1239，LogLoss 0.3918→0.3841 → **小样本下温度缩放只能部分缓解**

## 结论
1. BBBP 的 RF 概率基本校准（ECE 0.041 < 0.05），AUC 0.94 的概率可直接用作风险分。MCE 0.2855 高是因为某个低置信度 bin 样本少、经验频率抖动，不是系统性问题。
2. EEG 的 CSP+LR 概率严重失校准（ECE 0.152 远超 0.05），且 45 试次太小——temperature scaling 后 test-half ECE 仍 0.169。叠加第三轮的多 seed 0.76±0.05 / CI[0.64,0.89]，EEG 这个方向的两层警告都成立：点估计不稳 + 概率不可信。诚实结论：CSP+LR 在单被试上"有信号"（置换 p=0.016）但"概率不可用于决策"。
3. mean_conf：BBBP 0.768 vs 实际阳性率（BBBP p_np 阳性约 0.80 左右）接近；EEG 0.488 因两类平衡符合预期。

## 产物
- `bench_calibration.py`（`python3 -u bench_calibration.py` 可重跑，34s）
- `figures/calibration_bbbp.png` / `figures/calibration_eeg.png`（raw vs temperature-scaled 可靠性图）
- `data/calibration/calibration_metrics.json`

## 踩坑
RF `predict_proba` 会给出精确 0/1，temperature scaling 的 logit 变换 `log(p/(1-p))` 除零 → 变换前 clip 到 [1e-6, 1-1e-6]，警告消失、数字不变。
