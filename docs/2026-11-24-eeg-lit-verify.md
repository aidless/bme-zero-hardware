# EEG / BCI 运动想象解码 2024–2026 前沿文献逐条核实（供审稿级深挖报告引用）

核实人：子代理 eeg-lit｜核实日期：2026-11-24
核实纪律：所有 arXiv id、标题、数字均经 `web_fetch` 实页「号-题一致」核对；期刊论文用 Crossref API / 出版社页核卷期页。凡未取到一手页者一律标「未核实」，不凭记忆填号填数。

## 本机基线（对比锚点，非 SOTA）
- 数据集：PhysioNet EEGBCI（EEGMMIDB）S001，run 4/8/12，左右手运动想象二分类。
- 方法：CSP + Logistic Regression（传统机器学习基线）。
- 结果：test accuracy **0.7778**，AUC **0.8775**。
- 口径：**subject-specific / within-subject（单被试内）**，不是跨被试。

---

## 一、逐条核实清单（8 条已验号 + 2 条里程碑）

### 1. LaBraM（EEG 通用基础模型）
- 标题：*Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI*
- 作者：Wei-Bang Jiang, Li-Ming Zhao, Bao-Liang Lu（上海交通大学 + Shanghai Emotionhelper）
- 出处：arXiv:2405.18765（cs.LG），Submitted 2024-05-29，v1；**Journal ref: The Twelfth International Conference on Learning Representations (ICLR 2024)**
- DOI：10.48550/arXiv.2405.18765
- 核心贡献：统一的 EEG 基础模型，用 EEG channel patches + 向量量化的神经频谱预测（neural tokenizer）+ masked patch 预测做大规模自监督预训练。
- 关键数字：约 **2,500 小时**、约 **20 个数据集** 预训练；下游异常检测/事件类型分类/情绪识别/步态预测均超当时 SOTA。**注意：摘要不含运动想象解码任务，不能称为「运动想象 SOTA」。**
- 核实状态：**已验号**（arXiv abs 页 HTTP 200，号-题一致）
- URL：https://arxiv.org/abs/2405.18765 ｜ OpenReview：https://openreview.net/forum?id=QzTpTRVtrP

### 2. BrainGPT / 原 EEGPT（Yue 等，自回归 EEG 基础模型）—— 关键改名纠正
- 标题（当前 v2 2025-08-29）：*BrainGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training*
- 标题（v1 2024-10-14）：*EEGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training*
- 作者：Tongtian Yue, Xuange Gao, Shuning Xue, Yepeng Tang, Longteng Guo, Jie Jiang, Jing Liu
- 出处：arXiv:2410.19779（eess.SP）；页内声明 "This work has been submitted to the IEEE for possible publication"
- DOI：10.48550/arXiv.2410.19779
- 核心贡献：electrode-wise 自回归 EEG 通用基础模型，首个 next-signal prediction 自回归预训练，支持失配/可变电极（最多 138 电极）。
- 关键数字：**37.5M** 预训练样本，最高 **1.1B** 参数，**5 任务 12 benchmark** 全面超过 specialist 模型。
- 核实状态：**已验号**（arXiv abs 页 HTTP 200，号-题一致；确认 v1/v2 标题变更）
- URL：https://arxiv.org/abs/2410.19779
- ⚠️ 与下文第 3 条 NeurIPS 2024 的 EEGPT（Wang 等）是**同名不同文**，切勿混为一篇或互安 id。

### 3. EEGPT（NeurIPS 2024，Wang 等）
- 标题：*EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals*
- 作者：Guagnyu Wang, Wenchao Liu, Yuhong He, Cong Xu, Lin Ma, Haifeng Li（哈尔滨工业大学）
- 出处：**NeurIPS 37 (2024) Main Conference Track**；DOI：10.52202/079017-1239
- 核心贡献：mask-based dual 自监督学习 + 时空表征对齐，做通用可靠 EEG 表征，linear-probing 达 SOTA。
- 关键数字：**10M** 参数。（linear-probing SOTA，为通用 EEG 表征，非专门 MI 解码）
- 核实状态：**已验号**（NeurIPS proceedings 页 HTTP 200，题/作者/年份一致）
- URL：https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html ｜ 代码：https://github.com/BINE022/EEGPT
- 留白：**其 arXiv id 未核实到**（arXiv 搜索页超时、搜索结果无对应记录）。不得凭记忆填号。

### 4. BIOT（跨数据集生物信号基础模型）—— 年份纠正
- 标题：*BIOT: Cross-data Biosignal Learning in the Wild*
- 作者：Chaoqi Yang, M. Brandon Westover, Jimeng Sun
- 出处：arXiv:2305.10351（eess.SP），Submitted **2023-05-10**；NeurIPS 2023（proceedings hash f6b30f3e…）
- DOI：10.48550/arXiv.2305.10351
- 核心贡献：Biosignal Transformer，统一处理跨数据集的通道失配、变长序列与缺失值。
- 关键数字：CHB-MIT 癫痫检测 vanilla **+3% balanced accuracy**，预训练最多 **+4%**。（非 MI 任务；**是 2023 工作，不属于 2024–2026 新进展**）
- 核实状态：**已验号**（arXiv abs 页 HTTP 200，号-题一致）
- URL：https://arxiv.org/abs/2305.10351

### 5. MIRepNet（首个运动想象专用 EEG 基础模型）
- 标题：*MIRepNet: A Pipeline and Foundation Model for EEG-Based Motor Imagery Classification*
- 作者：Dingkun Liu, Zhu Chen, Jingwei Luo, Shijie Lian, Dongrui Wu
- 出处：arXiv:2507.20254（cs.CV），Submitted **2025-07-27**，v1；期刊录用 **Knowledge-Based Systems, vol. 343, 2026-06-15, pp. 115966**
- DOI：10.48550/arXiv.2507.20254；期刊 DOI：10.1016/j.knosys.2026.115966（来自 arXiv Journal-ref 与 GitHub，出版社页未单独 fetch）
- 核心贡献：首个专为运动想象范式定制的 EEG 基础模型——神经生理通道模板预处理 + 混合预训练（自监督 masked token reconstruction + 监督 MI 分类），支持少样本快速适配。
- 关键数字：**<30 trials/class** 快速适配；**7 个数据集预训练、5 个下游数据集、47 downstream subjects** 上 SOTA（超 specialist 与 general 模型）；参数 **5.2M**（GitHub README 对照表：BIOT 3.2M / BENDR 4.0M / LaBraM 5.8M / CBraMod 4.0M / EEGPT 25.0M / EEGNet 1.9K）。
- 核实状态：**已验号**（arXiv abs 页 HTTP 200，号-题一致；GitHub README raw HTTP 200）
- URL：https://arxiv.org/abs/2507.20254 ｜ 代码：https://github.com/staraink/MIRepNet
- 留白：①逐数据集 accuracy 数值在图片 `asset/Main_Results.jpg` 中，未取到可引用数值；②项目页 https://dingkun0817.github.io/MIRepNet/ 为 **404**，勿引用；③GitHub README 正文自称 RepMI（命名不一致）；④期刊出版社页未独立核验，卷期页来自 arXiv Journal-ref。

### 6. CTNet（subject-specific vs cross-subject 差距的直接证据）
- 标题：*CTNet: a convolutional transformer network for EEG-based motor imagery classification*
- 作者：Wei Zhao, Xiaolu Jiang, Baocan Zhang, Shixiao Xiao, Sujun Weng（厦门大学等）
- 出处：**Scientific Reports 14(1): 20237**，published **2024-08-30**
- DOI：10.1038/s41598-024-71118-7
- 核心贡献：卷积 + Transformer 混合网络用于 EEG 运动想象分类，同时报告 subject-specific 与 cross-subject 两套评估。
- 关键数字（摘要原文，同一模型两种口径）：
  - **subject-specific（被试内）**：BCI IV-2a **82.52%**，BCI IV-2b **88.49%**
  - **cross-subject（跨被试）**：BCI IV-2a **58.64%**，BCI IV-2b **76.27%**
- 核实状态：**已验号**（Springer 网页被 Client Challenge 拦，改用 Crossref API HTTP 200，题名/作者/卷期/摘要一致）
- URL：https://doi.org/10.1038/s41598-024-71118-7 ｜ Crossref：https://api.crossref.org/works/10.1038/s41598-024-71118-7

### 7. MOABB benchmark（EEG BCI 可复现性基准）
- 标题：*The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark*
- 作者：Sylvain Chevallier, Igor Carrara, Bruno Aristimunha, Pierre Guetschel, Sara Sedlar, Bruna Lopes, Sebastien Velut, Salim Khazem, Thomas Moreau
- 出处：arXiv:2404.15319（eess.SP），Submitted **2024-04-03**，v1（43 pages / 13 figures / 5 tables）
- DOI：10.48550/arXiv.2404.15319
- 核心贡献：大规模标准化基准——**30 个 ML pipeline**（raw 11 / Riemannian 13 / deep learning 6）× **36 个公开数据集**（MI 14 / P300 15 / SSVEP 7），含统计 meta-analysis、执行时间与环境影响。
- 关键数字：结论为 **Riemannian（空间协方差矩阵）整体最优**，深度学习需大量数据才有竞争力。**摘要未给统一单一 accuracy 数字**（数据集异质，不能引用一个「SOTA 数」）。
- 核实状态：**已验号**（arXiv abs 页 HTTP 200，号-题一致）
- URL：https://arxiv.org/abs/2404.15319

### 8. PhysioNet EEGMMIDB（本机所用数据集官方页——无官方准确率基线）
- 标题：*EEG Motor Movement/Imagery Dataset* v1.0.0
- 作者/贡献者：Gerwin Schalk（BCI2000 团队）；published **2009-09-09**
- DOI：10.13026/C28G6P；RRID:SCR_007345
- 数据集事实（官方页原文）：**109 名被试**（S001–S109），64 通道 10-10 系统，160 Hz，EDF+；每人 **14 runs** = 2 baseline + 12 task（4 条件 ×3 重复）；单侧 fist run 中 **T1=左手、T2=右手**；run 4/8/12 为 real/imagined 左右手（每 run 30 trials，三 run 共 90 trials，左右各 45）。
- 核实状态：**已验号**（PhysioNet 官方页 HTTP 200，字段一致）
- 留白：**官方页不提供任何分类 accuracy 基线**。引用准确率须引论文（如 CTNet 等），**不得写「官方基线准确率」**。
- URL：https://physionet.org/content/eegmmidb/1.0.0/

### 9. 侵入式 Speech BCI 里程碑（Nature 正刊）
- 标题：*An instantaneous voice-synthesis neuroprosthesis*
- 作者：Maitreyee Wairagkar 等；通讯 David M. Brandman & Sergey D. Stavisky（Stanford 等）
- 出处：**Nature 644, 145–152 (2025)**，published **2025-06-12**
- DOI：10.1038/s41586-025-09127-3
- 核心贡献：向 ALS 患者腹侧中央前回植入微电极，闭环音频反馈即时合成语音，解码音素 + 副语言特征，可实时改语调、唱短旋律。
- 关键数字：**256 微电极**（摘要原文），单例。
- 核实状态：**已验号**（nature.com 页 HTTP 200）
- URL：https://www.nature.com/articles/s41586-025-09127-3 ｜ 数据 Dryad 10.5061/dryad.2280gb64f ｜ 代码 github.com/Neuroprosthetics-Lab/brain-to-voice-2025

### 10. 流式 brain-to-voice（Nature Neuroscience）
- 标题：*A streaming brain-to-voice neuroprosthesis to restore naturalistic communication*
- 作者：Kaylo T. Littlejohn 等；通讯 Edward F. Chang & Gopala K. Anumanchipalli（UCSF）
- 出处：**Nature Neuroscience 28, 902–912 (2025)**，published **2025-03-31**，Technical Report
- DOI：10.1038/s41593-025-01905-6
- 核心贡献：高密度 ECoG + RNN-T，连续流式大词表语音合成，个性化到患者伤前声音，离线隐式语音检测、可无限连续解码。
- 关键数字：**80 ms 增量**（摘要原文）。
- 核实状态：**已验号**（nature.com 页 HTTP 200；PubMed 40164740 返回精简页，改用 nature.com 核实）
- URL：https://www.nature.com/articles/s41593-025-01905-6

---

## 二、证伪 / 纠正（凭记忆易错点）

| 易错说法 | 核实后的事实 |
|---|---|
| 「EEGPT 是 arXiv:2410.19779」并把它与 NeurIPS 2024 EEGPT 当同一篇 | **两个同名不同工作**：arXiv:2410.19779 是 Yue 等（v2 已改名 **BrainGPT**）；NeurIPS 2024 EEGPT 是 Wang 等（DOI 10.52202/079017-1239，10M 参数，**arXiv id 未核实到**）。不得互安 id。 |
| 「BIOT 是 2024–2026 新基础模型」 | BIOT 是 **arXiv:2305.10351 / 2023（NeurIPS 2023）**，不在本时间窗。 |
| 「LaBraM 是运动想象 SOTA」 | LaBraM 号 2405.18765（ICLR 2024）属实，但摘要下游任务为异常检测/事件分类/情绪识别/步态预测，**不含 MI 解码**。 |
| 「首个 MI 专用基础模型是 EEGPT/LaBraM」 | 首个专为 MI 范式定制的是 **MIRepNet（arXiv:2507.20254 → KBS 2026, 343:115966）**。 |
| 任何未实页核对的 arXiv id（如给 NeurIPS EEGPT 硬填一个号） | 一律标「未核实到」，不凭记忆填。 |

---

## 三、本机 0.7778 accuracy 的口径归属与可比性判断

**结论：0.7778 属于 subject-specific / within-subject（单被试内）口径，不是跨被试。** 理由：数据全部来自 S001 单一被试，测试集与训练集同属该被试的 run 4/8/12，CSP 的协方差矩阵也是按单被试拟合的——这是典型的 subject-dependent 设置。

可比性判断（诚实口径）：
1. **该和谁比**：与文献的 subject-specific 数字放一栏。CTNet 同模型 subject-specific 为 IV-2a 82.52% / IV-2b 88.49%。本机 0.7778 低于该档，符合「传统 ML 基线、非 SOTA」的定位。
2. **绝不能和谁比**：不能与 cross-subject 数字（CTNet IV-2a 58.64% / IV-2b 76.27%）直接对比。跨被试那档反而更低，若把 0.7778 放到跨被试栏会被误读为「超越 SOTA」，属高估。
3. **数据集不可等值**：EEGMMIDB run4/8/12 单被试仅 90 trials（左右各 45），与 BCI IV-2a/2b（9 名被试、每被试 288/120 trials）在试次数、电极、任务设计上都不同，数字不可直接等值。
4. **AUC 不可换算**：AUC 0.8775 不能折成 accuracy 与文献对标，两个指标分别报告即可。
5. 如需最接近 EEGMMIDB 的跨被试参照，候选是 MDPI Sensors 26(22):7259（2026-11-17，摘要称 EEGMMIDB 四类 subject-independent 71.28% / κ 0.617），但 **mdpi.com 抓取 Access Denied、数字未独立验号，本报告不写入结论**。

---

## 四、无法核实的留白（如实列出，不填数）

1. NeurIPS 2024 EEGPT（Wang 等）的 arXiv id —— 未核实到。
2. MIRepNet 逐数据集 accuracy 数值 —— 在图片 asset/Main_Results.jpg 内，未取到；期刊出版社页未独立核验（卷期页来自 arXiv Journal-ref）。
3. PhysioNet EEGMMIDB 官方 accuracy 基线 —— 官方页不提供，属不存在（非抓取失败）。
4. Neuralink 同行评审论文 —— 未核实到可引用的同行评审论文；Bloomberg 2025-10-05 报道称将投稿；ClinicalTrials.gov NCT06429735（PRIME）注册页只返回页头；网传 90/40 WPM 等数字来自非权威聚合站（quillberg / babelwire / insidebci），**不作为已核实数字引用**，里程碑只标「公司公告 + 临床试验注册，同行评审论文未核实到」。
5. MDPI Sensors 26(22):7259 的 accuracy —— 页面 Access Denied，未验号，不引用。
6. 未深挖候选（仅搜索命中，勿当已核实）：PMC13259246（2026-05-23，multi-branch CNN-LSTM cross-subject）；arXiv:2606.24394（2026-06-23，Friedman-Nemenyi decoder benchmark）。

---

## 五、证据等级说明
- A 级（一手页号-题一致 + 关键数字出摘要/正文）：条目 1/2/4/5/6/7/9/10。
- B 级（一手页核实但次级信息来自作者自述或第三方元数据）：条目 3（NeurIPS 页核实、arXiv id 留白）、条目 5 期刊卷期页。
- 数据集事实：条目 8（PhysioNet 官方页）。
- 未核实项统一在第四节列出，正文结论不引用未验号数字。
