# 生物医学工程深挖报告 · 从入门基线到审稿级 benchmark

> 目的：把已开源作品集（github.com/aidless/bme-zero-hardware）的五个方向从"单数字 demo"升级到"经得起审稿的 benchmark"。核心动作有两层：① 联网逐条核实 2024–2026 真实前沿（号-题一致 + 数字溯源，杜绝编造 DOI）；② 用更严格的实验协议重估我们的基线，并诚实标注口径差异。
>
> 日期：2026-11-24 · 证据分级：🟢 已打开正文核对标题/年份/数字 ｜ 🟡 仅搜索摘要未验证（不得当事实）｜ ⚪ 自报未独立复现

---

## 零、一句话结论

深挖之后，**我们作品集的四个数字没有一个能直接写成"超过 SOTA"**——不是因为结果差，而是因为 SOTA 对比必须口径一致（切分方式、预训练与否、特征维度、是否去重）。本报告的真正产出是：把每个数字放回它真实的口径里，并给出审稿人会问的 bootstrap 置信区间、scaffold split、seed 稳定性。结果反而更可信——融合特征在严格 scaffold split 下没有掉，反而略升。

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

## 四、类器官 & EEG（本次未新增前沿核实，沿用之前结论）

- 类器官：脑类器官存活近 6 年（2026，Arlotta 团队）、Nature Neuroscience 2025 "Preconfigured neuronal firing sequences in human brain organoids"、类器官智能 shell MEA（PubMed 40027665）。我们能做的是已有显微图像的分割定量（230 核），湿实验与活体培养不可。
- EEG：EEGBCI 真实 EDF 解码（CSP test AUC 0.8775）。前沿为端到端可学习滤波器（EEGNet 等），需更大校准数据集；当前 6GB GPU + 3 run 体量下，CSP+LR 是稳健基线。

## 五、跨方向总表：数字的真实口径（深挖核心产出）

| 原作品集表述 | 深挖后的诚实表述 | 口径问题 |
|---|---|---|
| 影像 ResNet18 AUC 0.9741 超过 CNN | ✅ 保留，但写区间 **0.9721 [0.957, 0.983]**，注明迁移学习 vs 从头训练不可对标 SOTA | seed 抖动 + SOTA 口径 |
| 制药 ECFP4+MACCS+desc AUC 0.9405 | ✅ 保留，补 **去重+scaffold split 0.9413 [0.904, 0.972]**；注明 random/scaffold 不可对标 MoleculeNet 0.714 | 64 个重复 SMILES + split 实现 |
| 单细胞 126 对显著通讯 | ⚠️ 降级为 **83 对（BH-FDR q<0.05）**；核心 CD74-MIF survived | 630 次检验未校正 |
| 单细胞 res=1.0 得 10 群 | ⚠️ 补注：**res=1.0 是社区值，官方教程 0.5/0.7**；跨 res ARI 0.63–0.93 | 分辨率敏感 |

## 六、交付与复现

三个新增审稿级脚本（均本机实测、stdout 数字落报告，非编造）：
- `bench_imaging_rigorous.py` — ResNet18 迁移学习 + 1000× bootstrap CI（GPU）
- `bench_mol_rigorous.py` — v1 scaffold split（有协议缺陷，保留作对照）
- `bench_mol_rigorous_v2.py` — canonical 去重 + DeepChem scaffold split + bootstrap CI（**推荐**）
- `bench_scrna_rigorous.py` — Leiden 聚类 ARI 稳定性 + BH-FDR 多重检验校正

前沿引用全部经期刊 abs 页或 arXiv 号-题一致验号；证伪的 4 个 arXiv id（scGPT 2304.05360 / scBERT 2207.09583 / Uni-Mol 2302.02211 / MolCLR 2111.14088）明确剔除。**"2024–2026 BBBP/PneumoniaMNIST 官方 SOTA AUC"未核实到权威持续更新榜单，不写具体数字。**

同步到开源仓库 github.com/aidless/bme-zero-hardware，present 报告与脚本。
