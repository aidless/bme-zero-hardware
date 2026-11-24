# 类器官（organoid）2024–2026 前沿文献核实清单

> 核实日期：2026-11-24｜核实人：organoid-lit 子代理（供 lead 审稿级深挖报告引用）
> 方法：所有新增条目均通过 `web_fetch` 实取出版社页面 / arXiv abs 页 / Crossref API 做「号-题一致」验号；数字只从实取正文摘取，不凭记忆写 id 或数字。
> 证据等级：**已验号** = 出版社/abs 页与 Crossref 双源标题、卷期、DOI 一致；**已确认（前轮）** = lead 指定前轮已核实，本轮未重复核；**未核实到原文** = 只拿到引用它的官方二手页，一手 PDF 未取到。

## 一、本轮新增核实（6 条 + 2 条补充）

### 1. 人早期脑类器官发育的形态动力学（脑类器官 SOTA / 基础机制）— 已验号
- **准确标题**：Morphodynamics of human early brain organoid development
- **作者/团队**：Akanksha Jain（一作）；Barbara Treutlein（通讯）；Matthias P. Lutolf、J. Gray Camp 等；ETH Zürich / Basel
- **期刊+卷期页码+年份**：Nature 644: 1010–1019（2025），2025-06-18 在线
- **DOI**：10.1038/s41586-025-09151-3
- **核心贡献（一句话）**：用长时程活体光片成像 + 双通道多嵌合标记与计算解复用， Weeks 级追踪非诱导脑类器官的肌动蛋白/微管/膜/核/核膜动态，揭示 Matrigel 介导的 WNT–YAP1–WLS 机械感知调控管腔扩张与区域化。
- **关键数字**（均取自正文）：day0 约 500 个 iPSC；并行成像 16 个类器官、1–3 周；188 h、30 min 分辨率成像；day4→day8 体积约 4 倍；平均管腔数 day5→day6 由 3.7±2.5 增至 13.4±2.5、随后因融合降至 5.4；无基质时管腔峰值 day7.8 vs Matrigel day5.3；PEG 硬度约 100 Pa。
- **核实状态**：已验号（Nature 出版社页 + Crossref 一致）
- **来源 URL**：https://www.nature.com/articles/s41586-025-09151-3

### 2. 类器官 + 器官芯片整合综述（organoid-on-chip）— 已验号
- **准确标题**：Integrating organoids and organ-on-a-chip devices
- **作者/团队**：Yimu Zhao（一作）；Milica Radisic（通讯）；University of Toronto
- **期刊+卷期页码+年份**：Nature Reviews Bioengineering 2: 588–608（2024），2024-07-02 在线
- **DOI**：10.1038/s44222-024-00207-z
- **核心贡献（一句话）**：系统综述类器官与器官芯片在多器官（肠/肾/肺/肝/胰/脑/视网膜/心/肿瘤）的整合策略，并点明血管化、批次差异与验证标准三大瓶颈。
- **关键数字**：综述类，无单一实验数字；覆盖 9 类器官系统。
- **核实状态**：已验号（出版社页 + Crossref 一致）
- **来源 URL**：https://www.nature.com/articles/s44222-024-00207-z

### 3. 自组装腔室样心脏类器官（心脏类器官 SOTA / 心脏毒性）— 已验号
- **准确标题**：Self-assembled chamber-like cardiac organoids for modeling cardiac chamber formation and cardiotoxicity assessment
- **作者/团队**：Xinle Zou（一作）；Duanqing Pei（通讯）等
- **期刊+卷期页码+年份**：Nature Communications 17: Article 7096（2026），2026-06-02 在线
- **DOI**：10.1038/s41467-026-73822-6
- **核心贡献（一句话）**：由 hPSC 自组装出外层心肌 + 内层内皮、可自发收缩的腔室样心脏类器官（CCO），重现心腔形成（FGF/WNT/BMP/RA/AA 信号、HAND1/TFAP2B/NR2F2）并用于心脏毒性评估。
- **关键数字**（取自正文）：day11.5 形成单一腔室、day90.5 后仍稳定，成功率 95%（N=8, n>600）；scRNA：87.4% 心肌细胞（TNNT2+）、76.2% NR2F2+ 心房样、11.5% MYL3+ 心室样；day21.5–68 约 60 BPM；RMP −60.99±4.91 mV、APA 119.05±8.69 mV、APD50 88.57±25.65 ms；SB431542 使腔室面积约 2×（P=7.25e−5）；ETS1 敲除不能形成腔室。
- **核实状态**：已验号（出版社页 + Crossref 一致）
- **来源 URL**：https://www.nature.com/articles/s41467-026-73822-6

### 4. 个体化胶质母细胞瘤类器官自动药筛（肿瘤类器官 SOTA / 药筛）— 已验号
- **准确标题**：Automated and personalized glioblastoma tumor organoid drug screening platform exposes sensitivity to proteasome and HDAC inhibitors
- **作者/团队**：Gerhard Jungwirth（一作）；Christel Herold-Mende（通讯）；Heidelberg
- **期刊+卷期页码+年份**：npj Precision Oncology 10: Article 228（2026），2026-06-16 在线
- **DOI**：10.1038/s41698-026-01552-5
- **核心贡献（一句话）**：机器人自动化高通量药筛（aHTS）用于患者来源 GBM 肿瘤类器官，术后 8 天内即给出药敏谱，揭示对蛋白酶体与 HDAC 抑制剂的敏感性，并复现了 marizomib 临床三期失败。
- **关键数字**（取自正文）：11 例 GBM 患者（9 新诊断 + 2 复发）；166 种 FDA 肿瘤药（NCI AODX）、2.5 µM；共 924 个肿瘤类器官；仅 0.048%（8/166）药使活性 ≤50%；最敏感：bortezomib 21.8%、ixazomib 24.1%、carfilzomib 24.2%、romidepsin 24.9%、panobinostat 31.5%；IC50：carfilzomib 91 nM、bortezomib 64 nM、ixazomib 38 nM、romidepsin <3 nM；Cmax/IC50 有利：carfilzomib 64.62×、romidepsin 232.33×；marizomib 效价 >100× 更低（IC50>10 µM），与 EORTC-1709 三期失败一致。
- **核实状态**：已验号（出版社页 + Crossref 一致）
- **来源 URL**：https://www.nature.com/articles/s41698-026-01552-5

### 5. MorphoOrgaAgent：基础模型驱动的类器官分析多智能体（AI + 类器官）— 已验号
- **准确标题**：MorphoOrgaAgent: A Foundation-Model-Based Multi-Agent System for Autonomous Organoid Analysis
- **作者/团队**：Hanyi Zhang、Maximilian Hoermann、Lion J. Gleiter、Yiling Xu、Bettina Katalin Budai、Hans-Ulrich Kauczor、Carsten Marr、Tingying Peng（ peng-lab）
- **出处+年份**：arXiv:2609.08696 [cs.MA]，2026-09-08 提交；15 页；被 MICCAI 2026 第二届 "Agentic AI for Medicine" Workshop 接收
- **DOI**：10.48550/arXiv.2609.08696（DOI 待注册，以 arXiv id 为准）
- **核心贡献（一句话）**：用 Cellpose 几何提示 + SAM3 文本提示实现零样本类器官分割，由 TaskUnderstandingAgent + 混合分割 + ReportAgent 组成多智能体，并提出 MorphoOrgaVQA 基准，代码开源于 github.com/peng-lab/MorphoOrgaAgent。
- **关键数字**：15 页；零样本（zero-shot）分割；基准名 MorphoOrgaVQA。
- **核实状态**：已验号（arXiv abs 页标题与 id 一致）
- **来源 URL**：https://arxiv.org/abs/2609.08696
- **易记错 id 提示**：⚠️ 该工作正确号为 **arXiv:2609.08696**。凭记忆容易写错的近似号（如 2609.08669 / 2509.08696 / 2609.08966）均为错误；引用时务必回查 abs 页。

### 6. FDA 药物研发中逐步淘汰动物实验的监管路线图（监管进展）— 官方页已验，一手 PDF 未取到
- **准确标题**：(a) FDA Announces Plan to Phase Out Animal Testing Requirement for Monoclonal Antibodies and Other Drugs（新闻稿，2025-04-10）；(b) FDA Achieves Year 1 Goals in Reducing Animal Testing in Drug Development（新闻稿，2026-04-20）
- **作者/机构**：U.S. Food and Drug Administration（ commissioner Marty Makary）
- **出处+年份**：FDA 官网新闻稿 2025 / 2026
- **DOI**：无（政府新闻稿）；一手路线图 PDF：/media/186092/download（"Roadmap to Reducing Animal Testing in Preclinical Safety Studies"）
- **核心贡献（一句话）**：FDA 以 NAMs（AI 计算毒理 + 细胞系 + 类器官 + 器官芯片）减少/优化/替代单抗等药物的动物实验，IND 阶段立即实施；一年后发布进展，认定首个 AI 药物研发工具并扩展证据权重法。
- **关键数字**（取自官方新闻稿）：>90% 通过动物实验的药物最终未获 FDA 批准；鲎源内毒素检测转型每年可免 >100 万只动物；明确点名肝/心/免疫类器官与器官芯片用于毒性测试。
- **核实状态**：新闻稿 **已验**（web_fetch 实取官网 HTML）；路线图 PDF 一手原文 **未核实到原文**（firecrawl 抓取 PDF 返回 HTTP 402），故一手路线图措辞不写具体数字，仅引官方新闻稿。
- **来源 URL**：
  - https://www.fda.gov/news-events/press-announcements/fda-announces-plan-phase-out-animal-testing-requirement-monoclonal-antibodies-and-other-drugs
  - https://www.fda.gov/news-events/press-announcements/fda-achieves-year-1-goals-reducing-animal-testing-drug-development

### 补充 A（可选用）：类器官革命的现代化与全球公平（政策/views）
- **准确标题**：Modernizing drug development and ensuring global equity in the organoid revolution
- **出处**：Communications Biology，2026-04-17；PMID 41998277；DOI 10.1038/s42003-026-10076-4
- **核实状态**：已验号（NCBI eutils esummary 实取题名/期刊/日期/DOI 一致；PMC 正文页遇 reCAPTCHA 未直取全文）
- **来源 URL**：https://pubmed.ncbi.nlm.nih.gov/41998277/

### 补充 B（未采用，记录以免重复）：肠类器官整合进 FDA NAMs
- NCBI 检索 "Integrating Human Intestinal Organoids into FDA New Approach Methodologies for Drug Discovery" 命中 PMID 42687608 / 41766415，但 esummary 显示题名为 "Advances in New Approach Methodologies at FDA's Division of Applied Regulatory Science," Clin Pharmacol Ther 2026 Nov, DOI 10.1002/cpt.70472 —— 与检索词不符，**未作为本清单条目**，避免张冠李戴。

## 二、前轮已确认、本轮未重复核（lead 指定，可直接引用为已确认）

| 文献 | 关键数字/事实 | 状态 |
|---|---|---|
| 脑类器官存活近 6 年（Paola Arlotta 团队，2026） | 存活近 6 年 | 已确认（前轮） |
| van der Molen et al. 2025 Nature Neuroscience, "Preconfigured neuronal firing sequences in human brain organoids" | 人脑类器官中预置的神经元放电序列 | 已确认（前轮） |
| 类器官智能 shell MEA 神经调控，PubMed 40027665 | shell MEA 用于类器官智能神经调控 | 已确认（前轮） |
| Frontiers organoid intelligence 综述 | https://www.frontiersin.org/journals/science/articles/10.3389/fsci.2023.1017235/full | 已确认（前轮） |

## 三、易记错 arXiv id / 留白声明

- **易记错的 id（已纠正）**：MorphoOrgaAgent 正确 id = **arXiv:2609.08696**。本轮所有其他 arXiv 引用一律以 abs 页实取为准，未凭记忆写任何新 id。
- **SOTA 榜单留白**：未核实到官方 2024–2026 类器官 SOTA/排行榜（不存在这样的官方榜单），因此**不写具体"排名/第一/SOTA 分数"数字**；上表"心脏类器官 SOTA""肿瘤类器官 SOTA"仅为代表性里程碑工作的标签，不代表官方排名。
- **一手 PDF 留白**：FDA 路线图 PDF 与 PMC 全文页未直取（HTTP 402 / reCAPTCHA），相关结论只引官方新闻稿与 NCBI eutils 元数据，不写未取到的原文数字。

## 四、工具/路径经验（供后续复用）

- PMC 网页 `web_fetch` 触发 reCAPTCHA（"Checking your browser"）；改用 NCBI eutils `esearch`/`esummary`（python urllib）可稳定取题名/期刊/日期/DOI。
- FDA `/media/*.pdf` 走 web_fetch 会失败；引用 FDA 立场时用其 HTML 新闻稿页。
- Crossref `https://api.crossref.org/works/<DOI>` 用于二次确认出版社卷期/页码/文章号/发布日期/一作，论文条目 1–4 均双源一致。
- 本会话 bing 搜索返回 0 结果，web_search 用 keenable 兜底。
