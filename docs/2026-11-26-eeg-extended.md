# 2026-11-26 EEGBCI 扩被试 × 强模型对照 × 跨被试概率校准

## 结论（全部为运行时实测，2026-11-24 本机 CPU）

- 覆盖 S001–S009 共 9 名被试、runs 4/8/12、每人 45 trials、pooled 405 trials；27 个 EDF 的 SHA-256 全部落 JSON 并逐个复验通过。`eegbci.load_data(subjects=[6,7,8,9], ...)` 实测可用：S006 单文件 3.8 s，全量 27 文件解析 31.1 s。
- **CSP+LR（基线）**：subject-specific macro acc 0.575309、AUC 0.613696；LOSO pooled acc 0.518519、CI [0.4691358024691358, 0.5703703703703704]（覆盖 0.5），AUC 0.524418、CI [0.4686372104216991, 0.5835269968340707]。扩到 9 人后仍无跨被试优势证据。
- **Riemannian TangentSpace+LR（强模型）**：subject-specific macro acc 0.676543、AUC 0.699185；LOSO pooled acc 0.580247、CI [0.5333333333333333, 0.6271604938271605]（不含 0.5），AUC 0.620896、CI [0.5670845751427988, 0.6752914271351941]（不含 0.5）。
- **同被试同 split 对照**：acc 0.518519 → 0.580247（+6.1728 pp），配对 acc 差 CI [0.0, 0.12098765432098768]（下界恰为 0）；AUC 差 +0.096478、CI [0.02591424937015869, 0.169086890015464]（排除 0）；McNemar b=89、c=64、p=0.051988852826861894（边界）；逐被试 7 胜 2 平 0 负。**AUC 提升证据强于 acc 提升证据。**
- **校准（cross-subject，留一被试拟合）**：CSP+LR raw ECE 0.069649 → temperature 0.015750（ECE 最佳）、isotonic 0.045353；TangentSpace+LR raw ECE 0.250983 → temperature 0.073575、isotonic 0.050925（ECE 最佳，但 Brier/LogLoss 仍以 temperature 最佳：0.240083/0.672821 vs 0.243146/0.705214）。isotonic 在 CSP 上把 Brier 从 0.252083 抬到 0.260477、LogLoss 从 0.697470 抬到 0.749811 —— ECE 下降不等于 proper score 改善。
- **一句话结论**：CSP+LR 的跨被试泛化仍不成立（CI 覆盖 0.5）；Riemannian 强模型在同一批 9 名被试、同一 LOSO split 上把 pooled acc 提到 0.580247、AUC 配对提升 CI 排除 0，跨被试泛化得到弱—中等证据支持（McNemar p≈0.052 为边界值）。

## 逐被试数字（acc / AUC，6 位小数与 stdout 一致）

| 被试 | subject-specific CSP | subject-specific Tangent | LOSO CSP | LOSO Tangent |
|---|---|---|---|---|
| S001 | 0.777778 / 0.877470 | 0.777778 / 0.828063 | 0.533333 / 0.397233 | 0.533333 / 0.786561 |
| S002 | 0.488889 / 0.509881 | 0.800000 / 0.881423 | 0.555556 / 0.671937 | 0.622222 / 0.671937 |
| S003 | 0.577778 / 0.622530 | 0.666667 / 0.687747 | 0.444444 / 0.462451 | 0.577778 / 0.707510 |
| S004 | 0.555556 / 0.636364 | 0.777778 / 0.800395 | 0.511111 / 0.727273 | 0.555556 / 0.565217 |
| S005 | 0.488889 / 0.527778 | 0.355556 / 0.311508 | 0.555556 / 0.505952 | 0.577778 / 0.583333 |
| S006 | 0.444444 / 0.440476 | 0.555556 / 0.615079 | 0.533333 / 0.490079 | 0.555556 / 0.682540 |
| S007 | 0.933333 / 0.992095 | 0.888889 / 0.958498 | 0.555556 / 0.707510 | 0.755556 / 0.839921 |
| S008 | 0.466667 / 0.500000 | 0.622222 / 0.592885 | 0.488889 / 0.549407 | 0.488889 / 0.577075 |
| S009 | 0.444444 / 0.416667 | 0.644444 / 0.617063 | 0.488889 / 0.486111 | 0.555556 / 0.698413 |

- 两套 acc/AUC 的逐被试 bootstrap CI（2000 次 trial 级有放回重采样，seed 42）全在 `data/eeg/eeg_extended_metrics.json`：subject-specific 在 `subject_specific.<model>.<S>.bootstrap`，LOSO 在 `loso.<model>.<S>.bootstrap`（后者由落盘 OOF 预测经同一 bootstrap 程序后处理算出，stdout 打印并以 `[posthoc-bootstrap]` 前缀追加到运行日志）。
- 两模型 pooled 的 secondary full-refit：CSP acc 0.525926 / AUC 0.532395；Tangent acc 0.587654 / AUC 0.617163。主口径仍是 5 折概率平均。

## 协议与口径

- 数据：`eegbci.load_data(subjects=[1..9], runs=[4,8,12], path='data/eeg/mne_data', update_path=False)`；每被试独立 64ch、160 Hz、FIR 7–30 Hz（firwin）、epoch −0.5–4.0 s、baseline −0.5–0 s；T1=left、T2=right；每人每 run 15 trials、合计 45。
- 模型：CSP(4, Ledoit-Wolf, log, average_power) + StandardScaler + LR(C=1.0, max_iter=2000) 对照 Covariances(lwf) + TangentSpace(riemann) + StandardScaler + LR(C=1.0, max_iter=1000)；pyriemann 0.12，pip 安装带 `--break-system-packages`。
- subject-specific：StratifiedKFold(5, shuffle=True, random_state=42) OOF。LOSO：每次留一名被试，其他 8 人（360 trials）组成训练集，仅在训练集上做同样的 5 折，五个 fold 模型对留出被试预测后取概率均值；留出被试从不进入 fit/scaler/阈值选择。
- 标签阈值：y ∈ {1,2}，`np.where(prob >= 0.5, 2, 1)`；脚本里写成显式规则，避免 `astype(int)` 把正类全判错而 AUC 看起来正常的静默 bug。
- 比较：两模型用同一批被试、同一 CV fold、同一 trial 索引；配对 bootstrap 与 McNemar 均基于 405 个 pooled OOF trial。
- 校准：ECE/MCE 用 15 等宽 bin；temperature 在**其他被试**的 pooled LOSO OOF 上最小化 LogLoss（bound 0.2–10.0），isotonic 同样只在其他被试上 fit，再作用到留出被试；CSP 的 T 落在 2.58–4.17、Tangent 落在 5.38–7.10，均未触边界。所有校准预测在留一被试意义下无同被试 fit+eval 泄漏。
- 设备：CPU only（OMP/OPENBLAS/MKL/NUMEXPR threads=1），实际耗时 605.0 s。

## 产物与复现

- 脚本 `bench_eeg_extended.py`（sha256 `a66d9cde56ad4da923eed57298a75697668e835d1e51d7d67e9b6e6151702d0a`，与 JSON `artifacts.script_sha256` 一致）。
- 指标 `data/eeg/eeg_extended_metrics.json`（含 27 个 EDF sha256、每 fold 明细、两套 OOF 预测数组、bootstrap CI、McNemar、校准明细、环境版本）。
- 日志 `data/eeg/eeg_extended_run.log`（含 18 行 `[posthoc-bootstrap]` 与独立复核摘要；复核结论：36 组 subject/model/setting 的 acc+AUC 从落盘 OOF 预测重算最大误差 0.0、27 个 EDF sha256 全通过）。
- 图 `figures/eeg_extended_model_compare.png`、`figures/eeg_extended_reliability.png`（2 行 × 3 列：两模型 × raw/temperature/isotonic）。
- 命令：`python3 -u bench_eeg_extended.py 2>&1 | tee data/eeg/eeg_extended_run.log`；Python 3.12.3、MNE 1.13.2、scikit-learn 1.9.1、NumPy 2.5.2、SciPy 1.18.1、pyriemann 0.12。
- 复算核验：逐被试 subject-specific/LOSO acc/AUC、pooled acc/AUC、配对 diff、McNemar、6 组校准 ECE/MCE/Brier/LogLoss、脚本与 27 个 EDF 的 sha256 均从 JSON 存盘数组独立重算，最大误差 0.0（校准重算 <1e-10）。

## 关键教训

1. 换更强模型确实能把跨被试从"围绕 0.5"推到 0.58，但要把"提升"拆成两个口径报：acc 的配对 CI 下界恰为 0、McNemar p≈0.052，AUC 的配对 CI 才明确排除 0；只报其中一个会误导。
2. S007 是被试内和被试间的双重异常点（subject-specific CSP acc 0.933333、LOSO Tangent acc 0.755556），强模型的逐被试提升主要由少数被试拉动，7/2/0 的胜负统计比 pooled 均值更稳健。
3. 校准必须按"其他被试 fit、留出被试 eval"；ECE 下降可能与 Brier/LogLoss 上升同时发生（CSP 的 isotonic 即是），报告时要同时给 proper score 和 sharpness。
4. 只新增任务允许的产物；`bench_eeg_cross_subject.py`、`bench_calibration.py`、`README.md`、`BME-DEEP-DIVE.md` 均未改动。
