# 生物医学工程深挖报告 · 从入门基线到审稿级 benchmark

> 目的：把已开源作品集（github.com/aidless/bme-zero-hardware）的五个方向从"单数字 demo"升级到"经得起审稿的 benchmark"。核心动作有两层：① 联网逐条核实 2024–2026 真实前沿（号-题一致 + 数字溯源，杜绝编造 DOI）；② 用更严格的实验协议重估我们的基线，并诚实标注口径差异。
>
> 日期：2026-11-24（第三轮补齐类器官 & EEG） · 证据分级：🟢 已打开正文核对标题/年份/数字 ｜ 🟡 仅搜索摘要未验证（不得当事实）｜ ⚪ 自报未独立复现

---

## 零、一句话结论

深挖之后，**我们作品集的六个数字没有一个能直接写成"超过 SOTA"**——不是因为结果差，而是因为 SOTA 对比必须口径一致（切分方式、预训练与否、特征维度、是否去重、被试内还是跨被试）。本报告的真正产出是：把每个数字放回它真实的口径里，并给出审稿人会问的 bootstrap 置信区间、scaffold split、seed 稳定性、多重检验校正。结果反而更可信——融合特征在严格 scaffold split 下没有掉，反而略升；而 EEG 的 0.7778 经多 seed 重估实为 0.76±0.05，单 seed 数字偏乐观。文献核实覆盖全部五个方向，累计证伪 / 纠正 9 个易记错的文献 id，并对不存在官方榜单的方向如实留白。

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
