# 生物医学工程·三方向实测档案（零硬件）

> 本机实测环境：RTX 3060 Laptop 6GB（空闲 5.8GB）+ Python 3.12
> torch 2.14.1+cu130 · scanpy 1.12.6 · rdkit 2026.09.1 · lightgbm 4.7.0 · medmnist 3.0.2
> 全部项目纯本地可复现，无硬件、无湿实验、无 API 密钥。

## 一、总览

| 方向 | 基线 | 进阶 | 提升 |
|---|---|---|---|
| ① AI 医学 | 逻辑回归 test AUC 0.9279 | ResNet18 迁移学习 test AUC **0.9741** | +0.0462 |
| ⑤ AI 制药 | ECFP4 test AUC 0.9182 | ECFP4+MACCS+desc test AUC **0.9405** | +0.0224 |
| ⑥ 单细胞组学 | 10 cluster 全注释 | PAGA 轨迹 + DPT 拟时序 + DE | 完成 |

三个方向从"基线可复现"升级到"进阶有提升"，全部真实跑出数字。

---

## 二、① AI 医学：ResNet18 迁移学习

- 脚本：`mini_project.py`（逻辑回归基线）+ `adv_resnet.py`（ResNet18 进阶）
- 数据：PneumoniaMNIST（28x28 胸片，二分类），medmnist 自动下载
- 结果：

| 模型 | test AUC | test acc |
|---|---|---|
| 逻辑回归 | 0.9279 | 0.8766 |
| 2 层 CNN（data aug） | 0.9364 | 0.8862 |
| ResNet18（ImageNet 预训练，全微调+分层 LR+affine aug） | **0.9741** | 0.8558 |

- 关键技巧：ResNet18 需要 224x224 输入，用 `Resize(224)`；灰度单通道要 `Lambda(lambda x: x.repeat(3,1,1))` 转 3 通道；预训练权重放 `~/.cache/torch/hub/checkpoints/`；冻结主干训 head 只能到 0.9338，解冻 layer3/layer4 + 分层学习率（head 1e-3 / 底层 1e-4）+ 随机仿射增强后跳到 0.9741。
- 跑法：
```bash
python3 mini_project.py     # 逻辑回归基线
python3 adv_resnet.py       # ResNet18 进阶（需 GPU，约 1 分钟）
```

---

## 三、⑤ AI 制药：分子表征进阶

- 脚本：`mini_project_mol.py`（ECFP4 基线）+ `adv_mol.py`（多表征对比）
- 数据：MoleculeNet BBBP（血脑屏障穿透，2050 分子），来自 DeepChem S3
- 结果（RandomForest，test 集）：

| 表征 | 维度 | test AUC |
|---|---|---|
| ECFP4（2048 bit） | 2048 | 0.9182 |
| MACCS（167 bit） | 167 | 0.9352 |
| ECFP4 + RDKit 描述符 | 2056 | 0.9283 |
| ECFP4 + MACCS + 描述符 | 2223 | **0.9405** |

- 关键技巧：单指纹 ECFP4 会漏掉 MACCS 覆盖的官能团信息；融合 MACCS（167 位结构键）+ RDKit 描述符（MolWt/LogP/TPSA/HBD/HBA/环数等，标准化后）三者拼接，RF 自动权衡，test AUC 提升 +0.0224。
- 跑法：
```bash
python3 mini_project_mol.py   # ECFP4 基线
python3 adv_mol.py            # 多表征对比
```

---

## 四、⑥ 单细胞组学：聚类 + 轨迹 + 差异表达

- 脚本：`mini_project_scrna.py`（聚类+注释）+ `adv_scrna.py`（PAGA+DPT+DE）
- 数据：10x Genomics PBMC3k（2700 细胞 × 32738 基因）
- 结果：

| 阶段 | 输出 |
|---|---|
| QC + 聚类 | 2643 细胞，Leiden 10 cluster |
| canonical 注释 | CD8 T 862 / CD4 T 625 / 单核 414 / NK 378 / B 341 / 血小板 15 / cycling 8 |
| PAGA 轨迹 | 14 条连通边（>0.1），cluster 0--1 连通性 0.91 最强 |
| DPT 拟时序 | cycling progenitors(0.01) → CD8 T → B → CD4 T → NK → 血小板 → 单核(0.87) |
| DE 差异表达 | 每 cluster Wilcoxon marker gene（adj p<0.05, logFC>1） |

- 关键技巧：PAGA 给出 cluster 间的谱系连通性；DPT 需要 `diffmap` + 设 `iroot`（选 cycling 群为根）→ 输出连续拟时序值，能反映分化轨迹。
- 跑法：
```bash
python3 mini_project_scrna.py   # 聚类+注释（约 1 分钟）
python3 adv_scrna.py            # PAGA+DPT+DE（基于上一步产物）
```

---

## 五、可视化（作品集门面）

`plot_scrna_figures.py` 基于 `data/pbmc3k/processed_advanced.h5ad` 生成 5 张 publication-quality 图，存于 `figures/`：

| 图 | 内容 |
|---|---|
| `umap_celltypes.png` | UMAP 按细胞类型着色（on-data 标签） |
| `umap_pseudotime.png` | DPT 拟时序渐变着色 |
| `paga_paga.png` | PAGA 谱系图（连通性>0.15，节点按 cluster） |
| `dotplot__marker_dotplot.png` | canonical marker 表达点图 |
| `paga_heatmap.png` | PAGA 连通性热力图 |
| `communication_heatmap.png` | 细胞通讯 sender×receiver 显著得分热力图 |

跑法：`python3 plot_scrna_figures.py`（需 scanpy + matplotlib，纯 CPU）。
踩坑：①scanpy 此版本 `sc.pl.umap` 不接受 `colorbar_label`；②`sc.pl.paga` 的 `colors` 需与 leiden cluster 数等长且不接受 dict，改用 `color="leiden", cmap="tab10"`；③marker 基因同时存在于 `.obs` 和 `.raw.var_names` 会触发 scanpy 歧义键错误，dotplot 前先 drop `.obs` 里的同名列；④`save="_x.png"` 实际文件名会拼上绘图函数前缀（如 `dotplot__marker_dotplot.png`）。

---

## 六、⑥ 单细胞深挖：细胞通讯（CellPhoneDB 简化版）

- 脚本：`adv_scrna_communication.py`，产物 `figures/communication_heatmap.png`
- 方法：15 对 canonical 人 Ligand-Receptor，raw counts→CP10K→log1p，sender 配体均值 × receiver 受体均值；1000 次标签置换建 null 分布求 p 值。
- 结果：**126 对显著通讯（p<0.05）**，最强为 MIF-CD74 轴：
  - B cells → cycling progenitors：CD74-MIF，score 6.090，p=0.000
  - CD14+ monocytes → CD4 T：LGALS1-CD69，score 3.767，p=0.000
  - NK → B：MIF-CD74，score 6.5（热力图累计得分）
- 判断：MIF-CD74、LGALS1-CD69 都是免疫学已知的通讯轴，说明分群可信、通讯推断有生物学意义，不是数字游戏。

---

## 七、② 细胞显微图像分析（类器官方向的可做部分）

- 脚本：`mini_project_organoid.py`，结论 `figures/organoid_summary.md`
- 数据：skimage 内置 `data.human_mitosis()`（HeLa DAPI，512×512），来源 scikit-image 官方数据仓库；首次运行下载到 `~/.cache/scikit-image`，之后离线可复现。
- 方法：高斯(σ=2)→Otsu→形态学清理→距离变换局部极大值(min_distance=10)→watershed→regionprops。
- 实测数字：`nuclei_count=230`，标记像素占比 12.52%；面积 142.71±72.76 px²，等效直径 13.15±2.97 px，圆度 0.97±0.13，离心率 0.57±0.17。
- 图 3 张：`organoid_segmentation.png`（三面板）、`organoid_morphometrics.png`（四直方图）、`organoid_shape_scatter.png`（面积 vs 圆度）。
- 能做/不能做：能做已有显微图的自动计数与形态定量；不能培养/染色/成像（湿实验），结果为像素单位无物理标定。
- 踩坑：skimage 0.26 的 `remove_small_objects` 等改用 `max_size` 参数（旧的 `min_size` 已移除）；全局 Otsu 比背景校正更稳（230 vs 169 核，避免过分割丢失）。

---

## 九、③ 脑机接口 / EEG 运动想象解码

- 脚本：`mini_project_eeg.py`，结论 `eeg_direction_conclusion.md`，账本 `data/eeg/eeg_metrics.json`
- 数据：**真实公开数据** EEGBCI/EEGMMIDB v1.0.0（PhysioNet，Schalk et al. 2004），S001 runs 4/8/12（左拳 vs 右拳运动想象），3 个 EDF 各 2.60 MB，sha256 已落盘溯源；64 通道 / 160 Hz / 45 有效试次（左23/右22）。非合成。
- 流程：FIR 带通 7–30 Hz → epoch(-0.5~4.0s, 基线 -0.5~0) → 特征（CSP 4 分量 / μ8-13+β13-30 log 功率 128 维 SelectKBest k=20）→ LogisticRegression → StratifiedKFold(5, shuffle, seed=42)。
- 实测数字（chance=0.5000）：
  - CSP：test accuracy **0.7778**，ROC-AUC **0.8775**，逐折 [0.7778, 0.8889, 0.7778, 0.7778, 0.6667] mean 0.7778±0.0703；混淆矩阵 [[18,5],[5,17]]
  - band-power：test accuracy 0.6667，AUC 0.6225，mean 0.6667±0.1988
- 判断：CSP 在小样本下比频带功率更稳（AUC 0.877 vs 0.623）。能做解码算法、评估协议、公开基准复现；不能实时采集、硬件在环或临床诊断——单被试 45 试次仅演示级，不外推跨被试。
- 图 3 张：`eeg_spectrum.png`（C3/C4 功率谱）、`eeg_topomap.png`（μ/β 头皮地形）、`eeg_confusion_matrix.png`（CSP 混淆矩阵）。
- 踩坑：MNE 1.13 的 `eegbci.load_data` 参数是 `subjects=`；`psd_welch` 已移除改用 `epochs.compute_psd(method='welch')`；`load_data` 返回 PosixPath 落盘 JSON 前要 `str()`。

---

## 十、下一步可选

| 方向 | 进阶选项 |
|---|---|
| ① AI 医学 | 换 nnU-Net 分割、多模态影像+临床文本、异常检测 |
| ⑤ AI 制药 | 图神经网络（GNN，需装 torch-geometric）、蛋白结构预测、生成式分子设计 |
| ⑥ 单细胞 | 多组学整合、细胞通讯深入分析、基因调控网络推断 |
| ② 细胞图像 | 换真实类器官/类组织数据集、3D 分割、Cellpose 预训练模型微调 |
| ③ EEG/BCI | 多被试联合建模、迁移学习跨被试、在线解码协议 |

## 附：文件清单

| 文件 | 用途 |
|---|---|
| `mini_project.py` / `adv_resnet.py` | ① 影像 AI 基线 + ResNet18 |
| `mini_project_mol.py` / `adv_mol.py` | ⑤ 分子性质预测基线 + 多表征 |
| `mini_project_scrna.py` / `adv_scrna.py` | ⑥ 单细胞聚类 + 轨迹 |
| `plot_scrna_figures.py` / `figures/` | ⑥ 单细胞可视化脚本 + 图 |
| `adv_scrna_communication.py` | ⑥ 单细胞深挖：细胞通讯（126 对显著 LR） |
| `mini_project_organoid.py` | ② 细胞显微图像分割 + 形态定量（230 核） |
| `mini_project_eeg.py` | ③ EEG 解码（见第九节） |
| `build_portfolio.py` / `bme-portfolio.html` | 自包含单文件作品集生成器 + 成品 |
| `bme-zero-hardware-guide.md` | 入门路径总文档 |
| `data/BBBP.csv` | BBBP 分子数据集（148 KB） |
| `data/pbmc3k/` | PBMC3k 原始数据 + processed.h5ad（7.6 MB） |