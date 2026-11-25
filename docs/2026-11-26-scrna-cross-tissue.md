# P1-10 跨组织泛化：10x heart_1k_v3（实测，2026-11-26）

## 结论

PBMC 的 canonical marker 集 **不作修改直接迁移到心脏数据：8/8 人类符号均不匹配**；这不是简单组织差异，下载文件的 `matrix/features/genome` 实测全为 **mm10**，即样本是**小鼠心脏**，PBMC3k 是人类 hg19。即使做小鼠同源基因替换，淋巴细胞 marker 在该心脏细胞谱系中也不再能定义主要群；心脏主体由 Tnnt2/Myh6/Myh7/Actc1（心肌）与 Col1a1/Col3a1/Dcn（成纤维）定义。跨组织比较细胞类型之前，**先做可靠的人类—小鼠 ortholog 映射，再做批次整合**；只做 ComBat 不够。

## 数据来源与校验

- URL：https://cf.10xgenomics.com/samples/cell-exp/3.0.0/heart_1k_v3/heart_1k_v3_filtered_feature_bc_matrix.h5
- HTTP 200，5395471 bytes，sha256 `4e78c3bb814f6dcce0566f64c006c0b8e42270d82cf896eac658911eee59eea6`
- 下载后读取 h5 属性：31053/31053 features 的 genome 都是 `mm10`（原始文件 1011 细胞 × 31053 基因）。provenance 落在 `data/heart_1k_v3/provenance.json`。

## 同套流程与计数

QC（min_genes=200、min_cells=3、mt%<5，线粒体前缀 `mt-`）→ normalize_total 1e4 → log1p → 2000 HVG → scale(max=10) → PCA 50 → neighbors k=10,n_pcs=40 → Leiden res=1.0。

- 1011 → 879（min_genes）→ **146**（mt<5）细胞；31053 → 16619 基因。
- Leiden 得 **8 群**，cluster sizes：0=30, 1=22, 2=26, 3=16, 4=19, 5=20, 6=6, 7=7。
- 标记规则注释（细胞数）：cardiomyocytes 46，fibroblasts 32，ambiguous 29，immune(Ptprc+) 20，cycling 19。
- 心脏 marker 峰：Tnnt2 cluster0=3.222、Myh6 cluster3=2.430、Actc1 cluster3=3.225、Tnni3 cluster3=2.952、Nppa cluster3=2.743、Col1a1 cluster6=2.535、Dcn cluster6=2.426、Ptprc cluster5=1.520、Mki67 cluster4=1.507、Hbb-bs cluster1=4.646。
- 从产物 `.obs['leiden']` 可复算 8 群；完整数值在 `data/heart_1k_v3/results.json`。

## marker 组织边界

- 人类符号 CD3D/CD8A/CD79A/MS4A1/NKG7/LYZ/CD14/PPBP 在心脏矩阵中**存在数 0/8**（PBMC3k 中为 8/8）。
- 有可测小鼠同源基因的只有 6 个：Cd3d, Cd8a, Cd79a, Nkg7, Lyz2, Cd14；Ms4a1/Ppbp 在心脏产物中无表达记录。
- 小鼠同源基因峰群落在不同心脏群上，且和心脏细胞类型不完全对应：Cd3d/Cd8a 峰在 cluster4（Mki67+ cycling，1.821/1.336），Nkg7 峰在 cluster5（Ptprc+ immune，0.686），Lyz2 峰在 cluster1（ambiguous，0.380），Cd14 峰在 cluster5（0.106），Cd79a 峰在 cluster0（0.000）。这证明“有同源基因”不等于“marker 规则跨组织成立”。
- 通讯结论同样受边界限制：PBMC 的 15-LR/置换网络是在血液免疫细胞群上测得；心脏的 Tnnt2/Col1a1/Pecam1 群不存在同一受体-配体背景，不能把 PBMC 的 sender→receiver 显著边直接外推为心脏机制。

## 批次整合探测

- `bbknn` 未安装（`ModuleNotFoundError`）。
- `sc.pp.combat` 在按大写符号相交的 182 个基因、2789 细胞（PBMC 2643 + heart 146）上成功执行，但这是 smoke test；大小写拼合不等于经过验证的跨物种 ortholog 映射，且桡动脉/心脏组织与血液的细胞组成差异无法被 ComBat 消除。
- 结论：需要 ortholog 映射 + 批次整合 + 共同细胞类型锚点，三者齐备才可做跨组织细胞类型比对。

## 产物

- 脚本：`bench_scrna_cross_tissue.py`
- 数据：`data/heart_1k_v3/heart_1k_v3_filtered_feature_bc_matrix.h5`、`data/heart_1k_v3/provenance.json`、`data/heart_1k_v3/results.json`、`data/heart_1k_v3/processed_heart.h5ad`
- 图：`figures/scrna_heart_umap.png`、`figures/scrna_heart_markers.png`
