# 生物医学工程深挖报告 · 从入门基线到审稿级 benchmark

> 目的：把已开源作品集（github.com/aidless/bme-zero-hardware）的五个方向从"单数字 demo"升级到"经得起审稿的 benchmark"。核心动作有两层：① 联网逐条核实 2024–2026 真实前沿（号-题一致 + 数字溯源，杜绝编造 DOI）；② 用更严格的实验协议重估我们的基线，并诚实标注口径差异。
>
> 日期：2026-11-24（第三轮补齐类器官 & EEG；第四轮跨数据集、跨被试与概率校准） · 证据分级：🟢 已打开正文核对标题/年份/数字 ｜ 🟡 仅搜索摘要未验证（不得当事实）｜ ⚪ 自报未独立复现

---

## 零、一句话结论

深挖之后，**我们作品集的六个数字没有一个能直接写成"超过 SOTA"**——不是因为结果差，而是因为 SOTA 对比必须口径一致（切分方式、预训练与否、特征维度、是否去重、被试内还是跨被试）。本报告的真正产出是：把每个数字放回它真实的口径里，并给出审稿人会问的 bootstrap 置信区间、scaffold split、seed 稳定性、多重检验校正。结果反而更可信——融合特征在严格 scaffold split 下没有掉，反而略升；而 EEG 的 0.7778 经多 seed 重估实为 0.76±0.05，单 seed 数字偏乐观。文献核实覆盖全部五个方向，累计证伪 / 纠正 9 个易记错的文献 id，并对不存在官方榜单的方向如实留白。

**第四轮的新发现更尖锐**：三个方向的"外部验证 / 校准"把前面的乐观进一步收敛。① 制药换到第二个数据集 BACE，融合特征相对单 ECFP4 仍是正增量，但 bootstrap CI 跨 0、统计上不显著——"融合永远最优"不成立；② 影像换到 7 类 DermaMNIST，协议可迁移（macro-AUC 0.9343），但与二分类不同口径；③ EEG 换被试后从 0.76 塌到近随机（跨被试 0.4944，CI 覆盖 0.5），**S001 的高分是被试特异的幸运抽样，不是泛化能力**；④ 概率校准上 BBBP 可信（ECE 0.041），EEG 严重失校准（ECE 0.152）。一句话：**判别力、泛化力、概率可信是三件事，单数据集单被试的单数字三样都没证明。**

---

## 一、AI 医学影像：0.9741 不能写"超过 SOTA"

### 1.1 前沿核实（🟢 已核对正文）

| 工作 | 出处（已验号） | 核心数字 |
|---|---|---|
| **PanDerm** | Nature Medicine 2025, 31:2691–2702；arXiv:2410.15038；DOI 10.1039→见 nature.com/articles/s41591-025-03747-y | 2.1M 图/11 机构/4 成像模态；28 项 benchmark 全 SOTA；仅用常规标注量的 5–10%；读者研究早黑检出 +10.2%、皮肤癌诊断 +11%、128 种病鉴别 +16.5% |
| Rad-DINO | arXiv:2401.10815（注意论文标题里**没有** "Rad-DINO" 字样） | 纯图像自监督胸片编码器，论证不需文本监督即可达/超语言监督模型；摘要无具体 AUC |
| MedImageInsight | arXiv:2410.06542（2024-10） | 跨 9 类影像的嵌入模型；多数领域 AUC>0.9；骨龄达专家水平 |
| CheXzero | Nature BME 2022, 6:1399–1406 | 无显式标注胸片自监督；8 种病理中 3 种超全监督模型 |
| ELIXR | arXiv:2308.01317 | zero-shot mean AUC 0.850（13 findings）；1%/10% 数据 0.893/0.898 |
| EVA-X | npj Digital Medicine 2025-11, 8:678 | 20+ 病理、11+ 检测任务领先 |
| MedGemma | arXiv:2507.05201（含 MedSigLIP 编码器） | 医学多模态 QA +2.6–10%；胸片 finding 分类 +15.5–18.1% |

**术语纠正**：PanDerm 是"多**成像模态**视觉基础模型"（皮肤镜/临床照/皮肤病理/全身照），不是图文对话式多模态大模型——这是之前调研里被写错的地方。

### 1.2 SOTA 对比的口径陷阱

PneumoniaMNIST 没有官方 leaderboard（Papers with Code 已停服），能核到的公开数字来自第三方站 sota2.com：**ResNet-18 从头训练 ERM，test AUC 0.9556**。

我们的 **0.9741 是 ImageNet 预训练 ResNet18 迁移学习**——用了预训练权重，与从头训练口径不同，**不能相减声称超过 SOTA**。稳妥表述："迁移学习 test AUC 0.97 量级，与公开从头训练基线（≈0.956）同区间；二者协议不同，不做直接比较。"

### 1.3 我们重估的实测（新增实验 `bench_imaging_rigorous.py`）

同一 ResNet18 迁移学习协议重跑，加 1000× bootstrap：

```
[test] acc=0.8558 auc=0.9721 f1=0.8963
[test AUC 95% CI] 0.9721 [0.9569, 0.9833]  (bootstrap n=1000)
```

关键发现：上次跑 0.9741、这次 0.9721，**GPU 训练有 seed 抖动**。报告数字应写区间 0.957–0.983，而非小数点后四位的单值——这正是单数字 demo 最容易被审稿人挑刺的地方。

---

## 二、AI 制药：0.9405 在 random split 下成立，但不能对标 MoleculeNet

### 2.1 前沿核实（🟢 已核对正文）

- **MoleculeNet**（BBBP 权威出处）：arXiv:1703.00564v3，Chemical Science 2018, 9, 513–530，DOI 10.1039/C7SC02664A。原文 §3.1.12 明确写"Scaffold splitting is also recommended for this well-defined target"。§3.2/§4.1 论证 scaffold split 比 random split 更能测泛化。
- **MoleculeNet Table 6（scaffold split）BBBP**：RF 0.714 / KernelSVM 0.729 / GraphConv 0.690 / Weave 0.671。
- **DeepChem 官方 leaderboard（scaffold split）**：ECFP4+RF 0.7194±0.0068 / GCN 0.6760±0.0195。
- **Chemprop / D-MPNN**（arXiv:1904.01561）：同是 scaffold split 但不同实现，BBBP ensemble 达 0.925±0.036——说明"scaffold split"不是一个数，seed/数据处理/特征都会大幅影响。
- **基础模型年份纠正**：ChemBERTa-2 (2022)、MolCLR (2021)、GROVER (2020)、Uni-Mol (ICLR 2023) 都不是 2024–2026。真正核到的 2025 代表是 **Token-Mol 1.0**（Nature Communications 2025, 16:4416，token-only 3D 药物设计，比扩散模型快 35 倍）。
- Un-Mol "14/15 SOTA"、Token-Mol 的 +30%/35× 均为**作者自报摘要**，标注后方可引用。

### 2.2 我们自己发现的两个协议错误（这是深挖最有价值的部分）

复查 `data/BBBP.csv` 发现：
1. **2050 行里只有 1986 个唯一 canonical SMILES**，64 个是同一分子的不同写法——random split 下这些同分子会同时落入训练/测试，造成身份泄漏。
2. 我第一版 scaffold split（`bench_mol_rigorous.py`）按骨架簇大小升序切分、不分层，导致 test 正类率 0.668 vs train 0.788 的分布漂移，且不符合 DeepChem `ScaffoldSplitter` 算法——**不可复现的自定义口径**。

### 2.3 v2 修正实验（`bench_mol_rigorous_v2.py`，已实测）

按 RDKit canonical SMILES 去重（2050→1975 有效唯分子，75 丢弃），用 DeepChem 标准 scaffold split（骨架组按大小降序填 train/valid/test），验证 train/test 骨架重叠=0：

```
                        RANDOM split          SCAFFOLD split (DeepChem)
ECFP4-1024             AUC=0.9078             AUC=0.8961
ECFP4-2048             AUC=0.9122             AUC=0.9100
ECFP4-2048+desc        AUC=0.9172             AUC=0.9385
ECFP4+MACCS+desc       AUC=0.9288 [0.895,0.956]  AUC=0.9413 [0.904,0.972]
best random-vs-scaffold inflation: -0.0125 AUC  (scaffold 反而更高)
5-seed scaffold 稳定性 (v1 口径): mean=0.9364 std=0.0015
```

诚实结论：
- 去重 + 严格 scaffold split 后，融合表征 **AUC 0.9413 没有掉、反而略升**——说明原始 0.9405 不是靠同骨架泄漏堆出来的，融合特征有真实泛化能力。
- 但 **0.94 vs MoleculeNet 官方 0.71 的 0.23 差距必须作为开放问题写明**，不能假装解决。来源至少有三：特征维度（2223 vs 1024）、集成规模（500 树）、split 实现差异（MoleculeNet/Chemprop 自己也相差 0.21）。这也说明"BBBP SOTA"没有单一可信数字，引用时必须连同 split 与实现一起写。
- 正确表述："在 canonical 去重 + DeepChem scaffold split 下，ECFP4+MACCS+描述子 RF 的 test AUC=0.9413 [0.904,0.972]；该数字高于 MoleculeNet 2018 用 1024-bit 浅层模型的 0.714，两者特征/模型/实现均不同，仅作定性参考，不声称 SOTA。"

---

## 三、单细胞组学：聚类稳不稳、126 对显著经不经得起多重检验

### 3.1 前沿核实（🟢 已核对期刊 abs 页；两个高危 arXiv id 被证伪）

**重要纠错**：凭记忆常写的 scGPT=arXiv:2304.05360、scBERT=arXiv:2207.09583 **号题不符**（前者实为信息论论文，后者实为统计物理论文）——一律改用期刊 DOI + 卷页引用。

| 模型 | 出处（已验号） | 规模/数字 |
|---|---|---|
| scGPT | Nature Methods 2024, DOI 10.1038/s41592-024-02201-0 | >33M 细胞；含扰动响应预测、基因网络推断 |
| Geneformer | Nature 2023, 618:616–624, DOI 10.1038/s41586-023-06139-9 | ~30M 细胞 |
| scFoundation | Nature Methods 2024, 21:1481–1491, DOI 10.1038/s41592-024-02305-7 | 100M 参数、~20000 基因、>50M 细胞 |
| UCE | Nature 2026, 656:183–191, DOI 10.1038/s41586-026-10689-z | 36M 细胞、>1000 细胞类型、8 物种；zero-shot 无需微调 |
| scBERT | Nature Mach Intell 2022, 4:852–866, DOI 10.1038/s42256-022-00534-z | 注意 2024-12 有 Matters Arising（DOI 10.1038/s42256-024-00949-w） |

细胞通讯工具链：CellChat v2（Nature Protocols 2024, DOI 10.1038/s41596-024-01045-4）、CellPhoneDB v2.0（Nature Protocols 2020, 15:1484–1506）、LIANA+（Nature Cell Biology 2024, 26:1613–1622，同时支持单细胞+空间转录组）、NicheNet（Nature Methods 2019）。Dimitrov et al. 2022 benchmark（Nature Communications, DOI 10.1038/s41467-022-30755-0）指出"方法间资源重叠有限、有先验偏倚"——具体百分比未取到正文，不引用数值。

PBMC3k 官方范式核对：Scanpy 教程 resolution=0.7→8 群；Seurat 教程 resolution=0.5→9 群。**我们的 resolution=1.0 是社区常用值，不是官方默认**——报告里要明确声明这一点。

2025–2026 新方向：AI 虚拟细胞 / 扰动预测（State，bioRxiv DOI 10.1101/2025.06.26.661135，>100M 扰动细胞训练，Cell-Eval 框架；⚠️ 上游摘要有 "contexts" 文本污染，标题以 bioRxiv 原文为准）。

### 3.2 我们重估的实测（新增实验 `bench_scrna_rigorous.py`，已实测）

审稿人会问的两件事——聚类稳不稳、多重检验校正没有：

```
[clustering stability] ARI vs baseline leiden (res=1.0):
  res=0.6 mean=0.628 | res=0.8 mean=0.667 | res=1.0 mean=0.932 | res=1.2 mean=0.880
[multiple testing] 630 sender-receiver-LR tests:
  p<0.05 (uncorrected): 126 significant
  BH-FDR q<0.05:        83 significant
  expected false positives at p<0.05 under global null: 32
  top FDR survivor: B->cycling CD74-MIF score=6.090 p=0.000 q=0.000
```

诚实结论：
- **聚类在 res=1.0 附近稳定**（跨 seed 平均 ARI 0.932），但降到 res=0.6/0.8 时 ARI 只有 0.63/0.67——细胞数/分辨率变化会明显改变分组，**必须显式声明 resolution**。
- 原来的 126 对显著是 630 次检验未校正的结果，全局零假设下期望 32 个假阳性。**BH-FDR 校正后剩 83 对**，最强的 B↔cycling CD74-MIF（score 6.090） survived（q<0.001）——核心生物学结论成立，但"126"这个数字要降级为"83 after FDR correction"。
- 修脚本时抓到一个真 bug：BH 校正把 q 值写到排序后位置却按原始索引读取，出现 p=0.836 却 q=0.000 的荒谬行；改成按原始索引写回后结果才可信（呼应 AGENTS.md 第 8 条"门必须能失败"）。

## 四、类器官 & EEG（第三轮补齐前沿核实 + 审稿级重估）

> 前两轮这一节只写了"沿用旧结论"。第三轮把这两个方向也拉到与影像/制药/单细胞同一审稿标准：每个方向新增一个审稿级脚本，数字全部本机实测。

### 4.1 类器官：前沿核实 + 细胞图像分割 230 核稳不稳

**前沿核实（🟢 本轮新增 6 条，全部 web_fetch 验号、出版社/abs 页 + Crossref/NCBI 双源）**

| 文献 | 核心贡献与关键数字 |
|---|---|
| Nature 644:1010-1019 (2025) "Morphodynamics of human early brain organoid development"，Jain/Treutlein (ETH)，DOI 10.1038/s41586-025-09151-3 | 活体光片追踪脑类器官形态动力学：day0~500 iPSC、16 个并行成像、188h/30min；体积 4×；管腔数 day5→6 由 3.7±2.5→13.4±2.5 后融合降至 5.4；无基质峰值 day7.8 vs Matrigel day5.3 |
| Nat Rev Bioeng 2:588-608 (2024) "Integrating organoids and organ-on-a-chip devices"，Zhao/Radisic，DOI 10.1038/s44222-024-00207-z | 器官芯片综述（9 类器官），点名血管化/批次/验证三瓶颈 |
| Nat Commun 17:7096 (2026) 腔室样心脏类器官，Zou/Pei，DOI 10.1038/s41467-026-73822-6 | day11.5 成腔、>day90.5 稳定、成功率 95%(N=8,n>600)；心肌细胞 87.4%、~60 BPM、APD50 88.57±25.65 ms |
| npj Precis Oncol 10:228 (2026) GBM 类器官自动药筛，Jungwirth/Herold-Mende，DOI 10.1038/s41698-026-01552-5 | 11 患者/166 药/924 类器官、术后 8 天出结果；仅 0.048%(8/166) 药使活性≤50%；romidepsin IC50<3 nM；marizomib >100× 弱，复现 EORTC-1709 三期失败 |
| arXiv:2609.08696 "MorphoOrgaAgent"（MICCAI 2026 workshop） | 基础模型多智能体类器官分析：Cellpose+SAM3 零样本分割 + MorphoOrgaVQA 基准 |
| FDA 监管 (2025-04-10 / 2026-04-20) | 明确类器官/器官芯片+AI 毒理替代动物实验（IND 即实施）；一年进展：>90% 药过动物实验却未获批、鲎检测年免>100万动物、首个 AI 药研工具获批 |

**证伪/易记错 id**：MorphoOrgaAgent 正确号 arXiv:2609.08696，易错近似号 2609.08669 / 2509.08696 / 2609.08966 均错；"肠类器官纳入 FDA NAMs"检索命中 PMID 42687608 实为另一题名（Advances in NAMs at FDA... CPT 2026），已剔除避免张冠李戴。**留白**：无官方类器官 SOTA 榜单，"心脏/肿瘤 SOTA"仅为代表性里程碑标签，不写排名分数；FDA 一手 PDF (HTTP 402) 与 PMC 全文 (reCAPTCHA) 未直取，结论只引官方新闻稿+NCBI 元数据。

**分割重估（`bench_organoid_rigorous.py`，已实测）**

**诚实边界先行**：`skimage.data.human_mitosis()` 是 HeLa 单层细胞核的 2D 荧光图，**不是 3D 类器官**；且无人工分割金标准，算不出 Dice/IoU。纯计算能做的是"已有显微图的自动分割与形态定量"，不能培养类器官、不能验证生物学功能。这一点写进产物，不自吹。

三个审稿人会问的问题，逐一实测：

| 检验 | 结果 | 判断 |
|---|---|---|
| 参数敏感性网格（sigma 1.5–3.0 × min_dist 8–12 × peak_thr 2–4，共 36 组） | 核计数 225.8±16.9，范围 [197, 252]，**CV=0.075**；面积均值 155.5±21.8 | 计数对参数不算敏感（CV 7.5%），baseline(2.0,10,3.0)=230 落在均值附近 ✅ |
| bootstrap CI（2000 次重采样，baseline 形态） | 面积 142.7 [134.1, 153.0]；直径 13.15 [12.79, 13.55]；圆度 0.971 [0.954, 0.987]；离心率 0.574 [0.554, 0.595] | 区间窄、点估计稳 ✅ |
| 加噪鲁棒性（高斯噪声 σ=0–0.15，每档 5 次） | 计数保持率 1.000 → 0.988 → 0.943 → 0.876 → 0.826 | σ≤0.05 保持 >94%，σ=0.15 掉到 82.6% —— 强噪声下会漏检，如实标注 ⚠️ |

产物：`figures/organoid_sensitivity_heatmap.png`、`data/organoid/organoid_rigorous_metrics.json`。

### 4.2 EEG 解码：0.7778 放进什么口径

**前沿核实（🟢 本轮新增 8 条，全部 web_fetch 实页号-题一致验号）**

| 文献 | 核实结论与关键数字 |
|---|---|
| LaBraM — arXiv:2405.18765 (ICLR 2024) | ~2500h/~20 数据集预训练；**摘要无 MI 任务，不能称「MI SOTA」** |
| BrainGPT（原 EEGPT，Yue 等）— arXiv:2410.19779 | v1 2024 名 EEGPT，v2 2025-08 改名 BrainGPT；37.5M 样本、最高 1.1B 参数、5 任务 12 benchmark 超 specialist |
| EEGPT（Wang 等，哈工大）— NeurIPS 37(2024), DOI 10.52202/079017-1239 | 10M 参数，mask 双自监督+时空对齐；**arXiv id 未核实到** |
| BIOT — arXiv:2305.10351 | **2023 (NeurIPS 2023) 非 2024-26**；CHB-MIT +3%/+4% balanced acc，非 MI 任务 |
| MIRepNet — arXiv:2507.20254 → KBS 2026, 343:115966 | **首个 MI 专用基础模型**；<30 trials/class 适配、7 预训练/5 下游、47 被试 SOTA、5.2M 参数；逐数据集 acc 在图片里未取到 |
| CTNet — Sci Rep 14(1):20237 (2024), DOI 10.1038/s41598-024-71118-7 | **同一模型两口径**：subject-specific IV-2a 82.52%/IV-2b 88.49%；cross-subject 58.64%/76.27% |
| MOABB — arXiv:2404.15319 (2024) | 30 pipeline×36 数据集（MI14/P300 15/SSVEP 7），Riemannian 整体最优；**无统一单一 acc，不可引用「一个 SOTA 数」** |
| PhysioNet EEGMMIDB 官方页 DOI 10.13026/C28G6P | 109 被试、64 通道、160Hz、run4/8/12 为左右手 MI；**官方不提供任何 acc 基线** |

侵入式里程碑（Nature 系验号）：*An instantaneous voice-synthesis neuroprosthesis*，Nature 644:145-152 (2025)，DOI 10.1038/s41586-025-09127-3，256 微电极单例；*A streaming brain-to-voice neuroprosthesis*，Nat Neurosci 28:902-912 (2025)，DOI 10.1038/s41593-025-01905-6，80ms 增量。

**证伪/纠正**：① EEGPT 是两个同名不同工作——2410.19779（Yue，已改 BrainGPT）≠ NeurIPS2024 Wang 等（arXiv 号未核实到），互安 id 即错；② BIOT 是 2023 不是新工作；③ MIRepNet 才是首个 MI 基础模型。

**0.7778 的诚实口径**：属 **subject-specific / within-subject**（数据全来自 S001 单被试、CSP 按单被试拟合），应和 CTNet subject-specific 82-88% 一栏比——我们的 0.7778 低于该档，是**传统 ML 基线，不是 SOTA**。**绝不能与 cross-subject 58-76% 比**（那档更低，误放会高估我们）。EEGMMIDB 单被试仅 90 trials vs BCI IV-2a/2b 9 被试 288/120 trials，不可等值；AUC 0.8775 不可换算 acc 对标。**留白**：NeurIPS EEGPT arXiv id、MIRepNet 逐数据集 acc、PhysioNet 官方 acc（不存在）、Neuralink 同行评审论文（未核实到）——不填数。

**审稿级重估（`bench_eeg_rigorous.py`，已实测）**

单 seed 0.7778 其实乐观了。三个检验揭示数字的脆弱性：

| 检验 | 结果 | 判断 |
|---|---|---|
| 多 seed 5-fold（42/0/7/2024/123） | acc 均值 **0.7600±0.0514**，range [0.6889, 0.8222]；auc 均值 0.8134±0.0360 | seed 抖动幅度 0.13，单 seed 0.7778 只是其中一次抽样 ⚠️ |
| bootstrap CI（2000 次，seed42 的 OOF） | acc 0.7778 **[0.6444, 0.8889]**；auc 0.8775 [0.7640, 0.9585] | 区间很宽、下界 0.644 接近 chance，45 试次样本太小 |
| 跨 run 泛化（leave-one-run-out） | 留 run4=0.80、留 run8=0.80、留 run12=0.7333，均值 0.7778±0.0314 | 跨 session 掉到 0.73，session 间漂移存在 |
| 标签置换检验（60 次） | null 均值 0.4837±0.0776，95th=0.6011；真实 acc 0.76，**经验 p=0.0164** | 显著高于随机（p<0.05），但 60 次置换分辨率有限 |

产物：`data/eeg/eeg_rigorous_metrics.json`。结论：CSP+LR 在单被试 45 试次上确实有信号（置换 p=0.016），但 0.7778 这个点估计不稳——诚实的写法是 **acc 0.76±0.05（多 seed）、95% CI [0.64, 0.89]**，且明确是 subject-specific 基线、低于 CTNet 82-88% 档。

## 五、跨方向总表：数字的真实口径（深挖核心产出）

| 原作品集表述 | 深挖后的诚实表述 | 口径问题 |
|---|---|---|
| 影像 ResNet18 AUC 0.9741 超过 CNN | ✅ 保留，但写区间 **0.9721 [0.957, 0.983]**，注明迁移学习 vs 从头训练不可对标 SOTA | seed 抖动 + SOTA 口径 |
| 制药 ECFP4+MACCS+desc AUC 0.9405 | ✅ 保留，补 **去重+scaffold split 0.9413 [0.904, 0.972]**；注明 random/scaffold 不可对标 MoleculeNet 0.714 | 64 个重复 SMILES + split 实现 |
| 单细胞 126 对显著通讯 | ⚠️ 降级为 **83 对（BH-FDR q<0.05）**；核心 CD74-MIF survived | 630 次检验未校正 |
| 单细胞 res=1.0 得 10 群 | ⚠️ 补注：**res=1.0 是社区值，官方教程 0.5/0.7**；跨 res ARI 0.63–0.93 | 分辨率敏感 |
| 细胞图像分割 230 核 | ✅ 保留，补 **36 组参数 CV=0.075**、面积 142.7 [134.1,153.0]；注明是 2D HeLa 核非 3D 类器官、无金标准算不出 Dice | 参数敏感 + 对象口径 |
| EEG CSP test acc 0.7778 | ⚠️ 降级为 **0.76±0.05（多 seed）、95% CI [0.64,0.89]**；置换 p=0.016；明确 subject-specific、低于 CTNet 82-88% 档 | 单 seed 抖动 + 样本小 + 口径 |

## 六、交付与复现

六个审稿级脚本（均本机实测、stdout 数字落报告，非编造）：

| 方向 | 脚本 | 审稿级检验 |
|---|---|---|
| 影像 | `bench_imaging_rigorous.py` | ResNet18 迁移学习 + 1000× bootstrap CI（GPU） |
| 制药 | `bench_mol_rigorous.py` | v1 scaffold split（有协议缺陷，保留作对照） |
| 制药 | `bench_mol_rigorous_v2.py` | canonical 去重 + DeepChem scaffold split + bootstrap CI（**推荐**） |
| 单细胞 | `bench_scrna_rigorous.py` | Leiden 聚类 ARI 稳定性 + BH-FDR 多重检验校正 |
| 细胞图像 | `bench_organoid_rigorous.py` | 36 组参数敏感性 + 2000× bootstrap CI + 加噪鲁棒性（CPU） |
| EEG | `bench_eeg_rigorous.py` | 5-seed 稳定性 + bootstrap CI + 跨 run 泛化 + 60 次标签置换（CPU） |

文献核实覆盖全部五个方向，前沿引用全部经期刊 abs 页或 arXiv 号-题一致验号。证伪 / 纠正的条目：
- 上一轮的 4 个 id：scGPT 2304.05360 / scBERT 2207.09583 / Uni-Mol 2302.02211 / MolCLR 2111.14088（剔除）
- 本轮类器官：MorphoOrgaAgent 正确号 arXiv:2609.08696（近似号 2609.08669/2509.08696/2609.08966 均错）；PMID 42687608 张冠李戴剔除
- 本轮 EEG：EEGPT 两个同名工作别互安 id（2410.19779 已改 BrainGPT ≠ NeurIPS2024 Wang 等）；BIOT 是 2023 非新工作；MIRepNet 才是首个 MI 基础模型；LaBraM/MOABB 摘要无 MI 不能称「MI SOTA」

**留白（核实不到就不填数）**：2024–2026 BBBP/PneumoniaMNIST 无权威持续更新 SOTA 榜单；类器官无官方 SOTA 榜单（里程碑标签不写分数）；PhysioNet 官方不提供 acc 基线；NeurIPS EEGPT arXiv id、MIRepNet 逐数据集 acc、Neuralink 同行评审论文未核实到。

同步到开源仓库 github.com/aidless/bme-zero-hardware，present 报告与脚本。

---

## 七、第四轮：外部验证、跨被试与概率校准（2026-11-24）

> 前三轮的重新评估都还在**同一个数据集内部**（BBBP 内换 split、S001 内换 seed、PBMC3k 内换 res）。审稿人真正的下一问是："**换个数据集、换个被试，还行吗？概率可信吗？**" 本轮用三个新方向回答：BACE（第二分子数据集）、DermaMNIST（第二影像数据集）、EEGBCI S2-5（跨被试），外加一轮概率校准。结论是判别力、泛化力、概率可信三件事里，前面的单数字只证明了第一件的一部分。

### 7.1 制药 · 换数据集 BACE：融合增量不再显著

第二个 MoleculeNet 数据集 BACE（BACE1 抑制剂二分类，1513 分子，URL 用小写 `bace.csv`，大写返回 403；sha256 `f3fb9ce9…737`）。canonical SMILES 去重后仍 1513（无效 0、重复 0、标签冲突 0），671 个 Bemis-Murcko scaffold，DeepChem 式 scaffold split 1210/151/152，三集合 scaffold 重叠 0。

| 表征 | random AUC [95% CI] | scaffold AUC [95% CI] |
|---|---|---|
| ECFP4-2048 | 0.8805 [0.843, 0.914] | 0.8909 [0.834, 0.940] |
| MACCS-167 | 0.8674 [0.824, 0.907] | 0.8654 [0.797, 0.921] |
| ECFP4+desc | 0.8862 [0.850, 0.919] | **0.9027 [0.846, 0.950]** |
| ECFP4+MACCS+desc（2223 维融合） | **0.8912 [0.856, 0.923]** | 0.8998 [0.843, 0.947] |

- **配对 bootstrap（融合 − 单 ECFP4）**：random +0.0108 [−0.000, +0.022]、scaffold +0.0088 [−0.005, +0.024]，**两处 CI 都跨 0**。融合是两个数据集上都为正的小增量，但在 BACE 上统计不显著；scaffold 上 `ECFP4+desc`（0.9027）还略高于完整融合（0.8998）。所以**"融合永远最优"不成立**，诚实写法是"融合稳定为正、幅度小且不显著"。
- **5 seed 分解**：融合 scaffold 同时重切分+换 RF seed → 0.8882±0.0197（range 0.862–0.913）；固定切分只换 RF seed → 0.8980±0.0012。**split 扰动比模型 seed 扰动大一个数量级**——真正的方差来源是数据怎么切，不是随机种子。
- **口径**：MoleculeNet 原文（arXiv:1703.00564v3）BACE 在 **Table 5**（不是 Table 6）：RF 0.867±0.008 / KernelSVM 0.862±0.000 / GC 0.783±0.014 / Weave 0.806±0.002（已 web_fetch 证伪任务给的错映射——0.783 是 GC 不是 RF）。我们的 0.8998 高于官方 RF +0.033，但官方是 GP 调参三次平均、默认 ECFP4 位数未钉死，我们是固定 RF500、2223 维融合、canonical 去重、自实现 80/10/10 scaffold 填充，CI 只反映 test 抽样方差——**只能作定性外部验证，不写同口径 SOTA 超越**。描述子 StandardScaler 在全量去重数据上拟合（忠实 BBBP v2，但非严格 train-only），已记入 limitations。

### 7.2 影像 · 换数据集 DermaMNIST：协议可迁移，口径不可比

第二个 MedMNIST 数据集 DermaMNIST（7 类皮肤病，官方划分 7007/1003/2005，类别 5 占 test 66.9% 长尾明显）。同一套 ResNet18 ImageNet 迁移协议，条件 Lambda 处理通道（灰度 repeat 成 3 通道、RGB 保留——无条件 repeat 会得到 9 通道触发 conv1 shape 错，已实测修正），val macro-AUC 选模选中 finetune epoch 2。

| 指标 | 实测 | 95% CI（1000×分层 bootstrap） |
|---|---|---|
| test accuracy | 0.7751 | [0.7601, 0.7900] |
| macro-AUC | 0.9343 | [0.9254, 0.9429] |
| ECE(15 bins) | 0.0539 | — |
| multiclass Brier | 0.3200 | — |

- 最弱类 class4 melanoma AUC 0.8797、次弱 class2 0.8913，混淆矩阵里 class2/4 大量误判成占多数的 class5。
- **口径**：官方 medmnist.com benchmark 列 DermaMNIST ResNet-18(224) AUC 0.920 / ACC 0.754，本轮数值略高但预训练与上采样路径未对齐，**只作参考不宣称 SOTA**。与 PneumoniaMNIST 二分类 AUC 0.9721 在指标定义、类别数、难度上都不同，**区间不可直接相减**。可写的结论是：协议在更难的多类任务上仍达到较高判别力（macro-AUC 0.934）和中等准确度（0.775），少数类与概率过度自信是后续点。

### 7.3 EEG · 换被试 S2-5：0.76 塌到近随机（本轮最关键）

前三轮的 S001 acc 0.7778 全来自**单个被试**。本轮换 EEGBCI subject 2–5（每人 runs 4/8/12、45 trials），做 subject-specific 5-fold 与 leave-one-subject-out（LOSO）。

- **先修一个确定性 bug**：首版 `bench_eeg_cross_subject.py:93` 用 `(prob>=0.5).astype(int)` 得 0/1，但真实标签是 {1,2}，导致所有 y=2 判错、subject-specific acc 被系统性压到 0.2444（AUC 不受影响）——**AUC 正常不能证明 acc 正确**。改成 `np.where(prob>=0.5,2,1)` 整脚本重跑（144s），并用原 `mini_project_eeg.py` 同款路径独立复核 S002=0.4889 一致。
- **subject-specific**（5-fold OOF）：S2 0.4889 / S3 0.5778 / S4 0.5556 / S5 0.4889，macro acc **0.5278**、macro AUC 0.5741——连单被试内的换人基线都只有 0.53。
- **LOSO**（留出被试完全不进 fit/缩放/阈值）：逐被试 acc 0.4889/0.4444/0.4222/0.6222；**pooled acc 0.4944、AUC 0.4851**，bootstrap CI acc [0.4222, 0.5667]、AUC [0.3979, 0.5701]，相对 macro 掉 3.33pp / AUC 掉 0.0891。
- **判读**：CI 覆盖 0.5，**本小样本 CSP+LR 协议下无跨被试优势的任何证据**。S001 的 0.76 是被试特异的幸运抽样，不能外推。文献 CTNet subject-specific 82–88% vs cross-subject 58–76% 的量级，在本复现里表现为 0.53→0.49，而非 0.76→0.58–0.76（本协议整体低于 CTNet 的深度模型）。**这是全部四轮里对原始作品集最不利、也最该写进简历口径的一条**。

### 7.4 概率校准：BBBP 可信，EEG 严重失校准

新增 `bench_calibration.py`：对 BBBP（ECFP4+RF）和 EEG（CSP+LR）的 OOF 概率算 ECE(15 bins)/MCE/Brier/LogLoss/sharpness，并做 temperature scaling（对半切，避免同批 fit+eval）。踩坑：RF `predict_proba` 给精确 0/1，取 logit 会除零，变换前 clip 到 [1e-6, 1−1e-6]。

| 数据集 | n | ECE | MCE | Brier | LogLoss | temperature scaling |
|---|---:|---:|---:|---:|---:|---|
| BBBP | 2039 | **0.0409** | 0.2855 | 0.0853 | 0.2956 | T=0.877，test-half ECE 0.0365→0.0375（本就校准，无需） |
| EEG S001 | 45 | **0.1521** | 0.6758 | 0.1425 | 0.4294 | T=0.891，ECE 0.2187→0.1694（仍超标） |

BBBP 概率基本可信（ECE<0.05）；EEG 严重过度自信（ECE 0.152，MCE 0.676）+ 45 试次小样本，temperature scaling 也救不回来。叠加第三轮的多 seed 不稳（0.76±0.05）和本轮的跨被试塌缩（0.49），EEG 这个数字有两层独立警告。

### 7.5 第四轮交付与新增脚本

| 方向 | 脚本 | 新增检验 |
|---|---|---|
| 制药 | `bench_mol_bace.py` | 第二数据集 BACE + 4 表征对比 + 配对 bootstrap + 5-seed 分解（含 `--selftest`） |
| 影像 | `bench_imaging_derma.py` | 第二数据集 DermaMNIST 7 类 + macro-AUC + 校准 + 混淆矩阵（GPU） |
| EEG | `bench_eeg_cross_subject.py` | S2-5 跨被试 LOSO + 标签阈值 bug 修复 + 12 EDF sha256 |
| 校准 | `bench_calibration.py` | ECE/MCE/Brier/LogLoss + temperature scaling（对半切） |

新增实测档案：`notes/2026-11-25-{bace-external-validation,derma-external-validation,eeg-cross-subject,calibration-analysis}.md`，指标 JSON 在 `data/{bace_rigorous_metrics.json,derma_rigorous_metrics.json,calibration/calibration_metrics.json,eeg/eeg_cross_subject_metrics.json}`，图在 `figures/{derma_reliability,derma_confusion,eeg_cross_subject,calibration_bbbp,calibration_eeg}.png`。文献核实覆盖全部新方向（BACE 的 MoleculeNet Table 5 映射已 web_fetch 证伪错值）。

**第四轮净结论**：判别力（单数据集 AUC）、泛化力（跨数据集/跨被试）、概率可信（ECE）是三件独立的事；本作品集在判别力上有扎实数字并配齐 CI，但跨被试泛化（EEG 0.49）和概率校准（EEG ECE 0.152）暴露了明显短板——这些是比"超过 SOTA"更值得写进简历讨论部分的诚实发现。

---

## 八、第五轮：把 21 项动作清单逐条做完（2026-11-25/26）

> 第四轮结束后列了 21 项"有价值动作"，按 P0 硬伤 / P1 验证 / P1 文献 / P1 产出 / P2 工程分五档。本轮把其中可在本机做的全部做完，派四路子代理并行，主会话做文献留白与工程收口。核心增量是：把第四轮暴露的三个短板（EEG 跨被试、scaler 泄漏疑虑、0.23 gap）逐个查清，并在第三、第四个数据集上验证"融合永远最优"这句话到底成不成立。

### 8.1 P0 硬伤逐条修复

| 编号 | 动作 | 实测结论 |
|---|---|---|
| P0-1 | EEG 跨被试扩到 9 人 + 强模型对照 | CSP+LR 的 LOSO pooled acc 0.5185、CI [0.4691, 0.5704] 覆盖 0.5（泛化仍不成立）；换 Riemannian TangentSpace+LR 后 pooled acc 0.5802、CI [0.5333, 0.6272] 不含 0.5，AUC 配对提升 +0.0965、CI [0.0259, 0.1691] 排除 0，逐被试 7 胜 2 平 0 负，McNemar p=0.052 为边界值 |
| P0-2 | EEG 跨被试概率校准 | CSP raw ECE 0.0696 → temperature 0.0158；Tangent raw ECE 0.2510 → isotonic 0.0509（ECE 最优），但 proper score（Brier/LogLoss）仍是 temperature 最优 |
| P0-3 | 制药 scaler 改严格 train-only | BBBP 三表征 ΔAUC = 0 / +0.0004 / −0.0002，BACE = 0 / −0.0011 / −0.0009：全量拟合 scaler 无实质泄漏（<0.002） |
| P0-4 | 0.23 gap 根因归因 | 我们 BBBP scaffold 0.9384 vs MoleculeNet 官方 RF 0.714 差 0.2244；单因素切到 DeepChem 精确 split 一项就消掉 0.2188（占 gap 97.5%），其余 ECFP4 位数/RF 树数/去重/特征组合各贡献 ≤0.031；全部切到官方同款后 0.7018，残差 0.0122。**gap 的绝大部分是切分实现差异，不是模型超越** |
| P0-5 | 收紧融合表述 | 见 8.2 ClinTox——第三个数据集上融合显著劣于 MACCS，原"融合最优"表述已全面收紧 |

**8.1 的关键判读**：P0-1 是好事也是坏事——换对模型（黎曼切空间）后 EEG 跨被试确实有弱—中等证据（AUC 提升 CI 排除 0），但 McNemar p=0.052 恰在边界，只能写"弱证据、待更大样本复现"，不能写成"已解决跨被试问题"。P0-3 与 P0-4 则把前两轮两条自我怀疑（scaler 泄漏、0.23 gap）都证伪为"协议差异、非缺陷/非超越"。

### 8.2 P1 验证扩展：三个新数据集再度改写结论

**ClinTox（第三分子数据集，P1-9）**：endpoint 用 CT_TOX（下载文件为 FDA_APPROVED + CT_TOX 两列，不是 HIV；gzip sha256 `7324a84a…9697`）。1484 行 → canonical 去重 1461（无效 4 / 重复 19 / 冲突 19），正例仅 7.05%，scaffold split 1168/146/147。

| 表征 | scaffold AUC（5 seed 均值±std） | random AUC | 配对融合−MACCS（scaffold） |
|---|---|---|---|
| ECFP4-2048 | 0.6760±0.0160 | 0.8623 | — |
| **MACCS-167** | **0.8949±0.0047** | **0.9444** | 融合 −0.0901，CI [−0.1508, −0.0349] 显著为负 |
| ECFP4+desc | 0.6802±0.0150 | 0.8708 | — |
| 融合 2223 维 | 0.8036±0.0165 | 0.9188 | 见左 |

融合相对 ECFP4、ECFP4+desc 显著为正，但**相对 MACCS 显著为负**——这是第三个数据集上"融合永远最优"被推翻（BBBP 融合微升不显著、BACE 融合不显著、ClinTox 融合显著更差）。三个数据集一致的诚实写法是：**融合相对单 ECFP4 稳定为正、幅度小，但不是普适最优，MACCS 在稀疏高不平衡端点上更强**。

**类器官金标准（P1-6）**：用 DSB2018 / BBBC038 stage1_train（32 图，CC0）把 COCO polygon 栅格化为实例 mask，跑同一条 Otsu+watershed 流水线算逐图 Dice/IoU，并加一个更强方法对照。基线（sigma2 / min_dist10 / peak3）Dice **0.7765** [0.7452, 0.8055]、IoU 0.6428；256 尺寸自适应版 Dice 0.8041 [0.7777, 0.8302]，配对 adapt−baseline +0.0276 [0.0145, 0.0419]（26/32 更好）；**Cellpose3 nuclei 对照 Dice 0.8547** [0.8160, 0.8853]，配对 Cellpose−adapt +0.0506 [0.0154, 0.0765]（29/32 更好）——传统 Otsu+watershed 与专用模型之间有可量化的差距。oracle 逐图取最优 0.8068 是调参上界、不是泛化。96 个 mask 从 npz 复算 Dice mismatch=0。踩坑：Cellpose 初版用 uint8 + 固定 diameter=60 得错误值 0.0876，改 float32 + 自动直径后才得 0.8547，失败与修正都记进 run.log。边界：2D HeLa/DSB 核 Dice ≠ 3D 类器官质量，体积 Dice、囊腔形态、跨设备泛化仍需 3D 人工体积标注。

**单细胞分辨率与跨组织（P1-7 / P1-10）**：PBMC3k 在 res=0.5/0.7/1.0 下 Leiden 得 7/8/10 群，ARI 0.5878/0.6312/1.0；通讯 BH-FDR 显著对数 96/125/194（res=1.0 从 .h5ad kNN 重跑为 194，与更早 83/126 的口径不同，属脚本重跑值，按本报告以重跑值为准）。官方常用 res=0.7 下 B/单核/血小板/T 主要谱系仍可辨，但 NK 与 cycling 细分不再保持。跨组织用 10x heart_1k_v3，实测是 **mm10 小鼠心脏**（跨组织+跨物种），QC 1011→879→146 细胞、8 群；8/8 人类 PBMC canonical marker 符号均不匹配，心脏主群由 Tnnt2/Myh6/Actc1 与 Col1a1/Dcn 定义，ComBat 仅 smoke test。**组织边界一句话**：marker 与通讯结论只在组织、物种、批次、聚类分辨率内成立，不可直接外推。

**影像跨数据集（P1-8）**：新增 `bench_imaging_extra.py`，在 PathMNIST（9 类）与 RetinaMNIST（5 类）上跑同一套 ResNet18 ImageNet 迁移协议 + val macro-AUC 选模 + 1000× 分层 bootstrap + ECE。踩坑：medmnist 3.x 用 `INFO[name]["label"]` 而非 `cls.info`；PathMNIST 无 `.samples` 属性；多类 Brier 用 `np.eye` one-hot。

| 数据集 | 规模（train/val/test） | test macro-AUC [95% CI] | test acc | ECE(15) |
|---|---|---|---|---|
| PathMNIST（9 类组织病理） | 89996/10004/7180 | **0.9919** [0.9908, 0.9930] | 0.9104 | 0.0476 |
| RetinaMNIST（5 类视网膜眼底） | 1080/120/400 | **0.7422** [0.7036, 0.7788] | 0.5475 | 0.0846 |

这组结果是"协议可迁移性有边界"的直接证据：同一套迁移学习在组织病理上达到 0.99 高判别力，但在小样本、类别不均衡、需要细粒度眼底判读的 RetinaMNIST 上掉到 0.74 / acc 0.55。加上第四轮的 DermaMNIST 0.9343，影像在三个不同数据集上是 0.99 / 0.93 / 0.74 的梯度——**难度由数据集决定，不是单一数字能代表的**；且三者的类别数、指标口径、预训练路径都不同，不可横向相减或宣称 SOTA。（结果落 `data/imaging_extra_metrics.json`。）

### 8.3 P1 文献留白收口

- **EEGPT id 坐实**：官方 NeurIPS 2024 页（hash `4540d267…`）确认 Guagnyu Wang 等、NeurIPS 37(2024) Main、DOI 10.52202/079017-1239、10M 参数、code github.com/BINE022/EEGPT，**官方页无 arXiv id**——此前"arXiv 号未核实到"结论坐实，统一改用 DOI 引用。
- **MIRepNet 逐数据集 acc 留白**：deepwiki 页显示结果在 asset/*.jpg 图片里、正文无逐数据集 acc 表格；本机 modlens 视觉桥未配置（实测报 "No vision provider is set up"），两条通道都取不到，**如实留白不编数字**。
- Neuralink PRIME 为 ClinicalTrials.gov NCT06429735（临床试验注册，非同行评审论文）；本轮未核实到 Neuralink 同行评审论文，继续留白。

### 8.4 P1 产出 / P2 工程

| 项 | 交付 |
|---|---|
| requirements 锁版本 | numpy 2.5.2 / pandas 2.3.3 / sklearn 1.9.1 / torch 2.14.1+cu130 / torchvision 0.29.2+cu132 / medmnist 3.0.2 / rdkit 2026.09.1 / scanpy 1.12.6 / scikit-image 0.26 / mne 1.13.2 / pyriemann ≥0.6 |
| FETCH_DATA 补 5 节 | BACE（小写 bace.csv）/ DermaMNIST / ClinTox / heart_1k_v3 / BCI IV-2a |
| README 短板表 | 明列六个数字无一 SOTA + 修 BME-DEEP-DIVE.md 链接（在仓库根不在 docs）+ 补验证 notes |
| 回归测试 | `smoke_test.py` 静态检查（语法编译 / README 链接 / metrics JSON schema / figures 非空），实测 5/5 PASS |
| claim ledger | `bme-claim-ledger.json` 9 个核心数字绑脚本，`research_ledger verify` 9/9 PASS |
| 简历口径审计 | `notes/2026-11-25-BME-简历口径审计.md`：6 条声称逐条给"实测 / 判定 / 替代说法"，不改简历正文（共享写点 + 事实闸） |

**第五轮净结论**：21 项里 P0 五项全部修复或证伪，P1 验证项把"融合最优""跨被试不可用"两条都改写成更精确的口径——融合只在相对单 ECFP4 时稳定为正、在 ClinTox 上显著不如 MACCS；EEG 换黎曼模型后有弱—中等跨被试证据但仍处 McNemar 边界。真正沉淀下来的不是更高的分数，而是把每个数字的可信边界钉死：切分实现能解释 97.5% 的 gap、scaler 泄漏 <0.002、跨被试 CI 是否覆盖 0.5、概率 ECE 能不能被校准救回。
