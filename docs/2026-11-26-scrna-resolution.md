# P1-7 单细胞分辨率复核（实测，2026-11-26）

## 结论

官方教程常用 res=0.7 在本数据上复现为 **8 群**，与 res=1.0 基线的 ARI=**0.631225**；主要免疫细胞类型（B、单核、血小板、T）仍可辨认，但 **NK 与 cycling 群不再由 canonical marker 规则独立分出**。res=0.5 为 **7 群**、ARI=**0.587820**，合并更明显。res=1.0 为 **10 群**、ARI=**1.0**。因此“官方 res 下我们的注释结论保持”只在**主要谱系层面**成立，细粒度结论（NK/cycling）不保持。

## 方法与证据链

- 输入：`data/pbmc3k/processed.h5ad`，2643 细胞 × 2000 HVG，raw=13714 基因；源文件 sha256 `495c3cce4957592c53512b80114569399bcad0320bb08ba4b6d06288fa984bb2`。
- 每个 res 在输入文件已有 kNN 图（n_neighbors=10, n_pcs=40）上重跑 `sc.tl.leiden(flavor='igraph', n_iterations=2, random_state=0)`；表达峰使用 raw counts 的 log1p(CP10K)。
- 三个产物 `data/pbmc3k/processed_res05.h5ad`、`data/pbmc3k/processed_res07.h5ad`、`data/pbmc3k/processed_res10.h5ad` 均在 `.uns['scrna_resolution_check']` 内含完整 JSON（群数、ARI、marker peaks、通讯结果）。脚本 `bench_scrna_resolution.py` 另打印 `===BEGIN_MEASURED_JSON===` 包裹的完整 JSON。
- 判据复算：从每个落盘 h5ad 的 `.obs['leiden_res_XX'].unique()` 重新计数，依次为 7/8/10，与内存结果一致（`verification.matches_in_memory=true`）。

## 实测汇总

| res | 群数 | ARI vs res=1.0 | 群大小（按 cluster id 排序） | marker 规则注释（细胞数） | 多数投票迁移基线注释（细胞数） | 通讯检验数 | p<0.05 | BH-FDR q<0.05 |
|---|---:|---:|---|---|---|---:|---:|---:|
| 0.5 | 7 | 0.587820 | 1179, 341, 640, 12, 427, 36, 8 | CD4T 1179, 单核 676, CD8T 435, B 341, 血小板 12 | CD8T 1606, 单核 640, B 341, CD4T 36, 血小板 12, cycling 8 | 630 | 129 | 96 |
| 0.7 | 8 | 0.631225 | 1151, 341, 483, 15, 154, 36, 8 | CD4T 1151, 单核 673, CD8T 463, B 341, 血小板 15 | CD4T 1187, 单核 483, CD8T 455, B 341, NK 154, 血小板 15, cycling 8 | 840 | 179 | 125 |
| 1.0 | 10 | 1.000000 | 542, 589, 320, 341, 223, 155, 414, 36, 15, 8 | CD8T 870, 单核 673, CD4T 589, B 341, NK 155, 血小板 15 | CD8T 862, CD4T 625, 单核 414, NK 378, B 341, 血小板 15, cycling 8 | 1350 | 255 | 194 |

通讯检验为 15 个 LR 对 × sender≠receiver，1000 次标签置换，BH-FDR；与基线脚本的 7 个 `cell_type` 群 630 检验/q<0.05=83 **统计单位不同**（本表按该分辨率的 Leiden 群），不可直接当作同一数值比较。

## canonical marker 峰（log1p CP10K）

| marker | res=0.5 峰群/均值 | res=0.7 峰群/均值 | res=1.0 峰群/均值 |
|---|---:|---:|---:|
| CD3D | 0 / 2.3796 | 0 / 2.3750 | 2 / 2.4205 |
| CD8A | 4 / 0.9086 | 3 / 0.8676 | 2 / 1.1781 |
| CD79A | 1 / 2.7917 | 1 / 2.7917 | 3 / 2.7917 |
| MS4A1 | 1 / 2.3956 | 1 / 2.3956 | 3 / 2.3956 |
| NKG7 | 4 / 3.0768 | 3 / 2.9757 | 5 / 3.2398 |
| LYZ | 2 / 3.2365 | 2 / 3.4055 | 6 / 3.4449 |
| CD14 | 2 / 1.3695 | 2 / 1.7097 | 6 / 1.8105 |
| PPBP | 3 / 4.0225 | 5 / 3.8660 | 8 / 3.8660 |

B（CD79A/MS4A1）、单核（LYZ/CD14）、血小板（PPBP）的峰群在全部三档都得到一致注释；T/NK 在较低分辨率出现混合：0.5/0.7 的 NKG7 峰群同时 CD8A 较高，固定规则将其判为 CD8T，而多数投票中 0.7 的 cluster 4 来自基线 NK。这解释了“主要类型稳定、细类型不稳定”。

## 产物

- 脚本：`bench_scrna_resolution.py`
- 数据：`data/pbmc3k/processed_res05.h5ad`、`data/pbmc3k/processed_res07.h5ad`、`data/pbmc3k/processed_res10.h5ad`
- 图：`figures/scrna_resolution_umap.png`、`figures/scrna_resolution_marker_peaks.png`
- 完整数值：各 h5ad 的 `.uns['scrna_resolution_check']` JSON 与脚本 stdout JSON。
