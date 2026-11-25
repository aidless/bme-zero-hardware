# P1-6：类器官方向补金标准 Dice/IoU（2026-11-26）

## 结论

补上了此前最大的缺口：现有 Otsu+watershed 流水线第一次有了**像素级 ground truth 指标**——在 DSB2018/BBBC038 的 32 张带 GT 核分割图上，传统流水线 Dice=0.7765（512 调参）/0.8041（256 适配），更强方法 Cellpose3 nuclei Dice=0.8547；96 组 Dice 全部可从落盘 GT/pred mask 逐图复算，0 mismatch。**2D 核分割 Dice 不等于 3D 类器官分割质量**，本轮只补"分割准确性可量化"，未补"3D 类器官/形态质量"。

## 数据集与授权（实测可达，两通道以上确认）

| 项目 | 值 |
| --- | --- |
| 选用数据集 | DSB2018 / BBBC038 stage1_train 子集，32 张 256×256，核数 20–60/图 |
| 官方来源 | https://bbbc.broadinstitute.org/BBBC038 |
| 下载镜像 | https://huggingface.co/datasets/DataTorchio/nuclei-segmentation-dsb2018 |
| 授权 | CC0 1.0（public domain），来源页与镜像页均标注 |
| annotations.json SHA-256 | `0a440ae7722ffab0adc29b0156179e0ff15d52b67f67f0e61acf4bc12734bc95` |
| 子集清单 | `data/organoid/gt/dsb2018/manifest.json`（每张图的 URL + SHA-256 + GT 核数） |
| 首图 SHA-256 | `e837ea0f851b62ea656b456be02bcb61e32e5a9e93b23a69514f915cac522578` |
| GT 形式 | COCO polygon → `skimage.draw.polygon` 栅格化为实例标签；逐核 mask，可直接算像素 Dice |

弃用的候选（均实测，不编造）：

- `cellpose.org/static/data/test.zip`：HTTP 404 / 返回 challenge HTML，curl 与 web_fetch 两通道一致。
- DeepCell S3 数据端点：HTTP 403。
- BBBC002：只有细胞计数 GT，无像素级 mask。
- BBBC007：已下载（images SHA-256 `b7009e2fce0a3152a5c9adda916eaa699d09696f4bd02a7d05d12d041e30c6d1`，outlines `6a5246f9a9d743d22eafdb409fae638a8461af97e9ff9c4a92f25eba236224d3`），但 GT 是手绘轮廓线（solidity≈0.44 的环，官方用 boundary metric），不适合直接算像素 Dice，弃用。

## 方法与配置

流水线复用 `bench_organoid_rigorous.py` 的 segment 逻辑：Gaussian σ=2 → Otsu → 去小目标/小孔 → 距离变换 → `peak_local_max` 种子 → watershed → 面积过滤。两套配置：

- `baseline_512tuned`：min_distance=10, peak_threshold=3, min_object=40, min_hole=200, area=[80,2000]（原 512 HeLa 调参直接搬用）。
- `adapt_256`：min_distance=6, peak_threshold=3, min_object=20, min_hole=100, area=[40,1200]（针对 256×256 / 核更小）。
- 更强方法对照：Cellpose 3.1.1.2，`model_type='nuclei'`，CPU，`channels=[0,0]`，自动直径（每张图 7.3–18.7 px），float32 输入；nuclei 模型权重下载 25.3 MB。初次误用 uint8 + 固定 diameter=60 得 Dice=0.0876（预测大量空），已废弃，不作为结果。

逐图计算：像素级 TP/FP/FN、Dice、IoU、Precision、Recall、F1；实例级 Hungarian 匹配（`linear_sum_assignment`，IoU≥0.5）得 match-F1、mean match IoU、计数误差。汇总为均值±std 与按图 bootstrap 95% CI（seed=42，B=2000）。

## 实测结果（n=32，均值±std [bootstrap 95% CI]）

| 方法 | Dice | IoU | Precision | Recall | match-F1@0.5 |
| --- | --- | --- | --- | --- | --- |
| baseline_512tuned | 0.7765±0.0890 [0.7452, 0.8055] | 0.6428±0.1160 [0.6028, 0.6811] | 0.8124±0.1063 | 0.7622±0.1380 | 0.6162±0.2197 |
| adapt_256 | 0.8041±0.0778 [0.7777, 0.8302] | 0.6789±0.1049 [0.6433, 0.7140] | 0.8165±0.0999 | 0.8166±0.1404 | 0.7148±0.1854 |
| Cellpose3 nuclei | 0.8547±0.1023 [0.8160, 0.8853] | 0.7577±0.1329 [0.7095, 0.7992] | 0.9034±0.0430 | 0.8303±0.1575 | 0.8696±0.1623 |

配对对照（逐图差值的 bootstrap）：

| 对比 | Dice 差均值±std [95% CI] | 方向 |
| --- | --- | --- |
| adapt_256 − baseline_512tuned | +0.0276±0.0393 [0.0145, 0.0419] | adapt 更好 26/32，打平 5，baseline 更好 1 |
| Cellpose3 − adapt_256 | +0.0506±0.0882 [0.0154, 0.0765] | Cellpose 更好 29/32 |
| Cellpose3 − baseline_512tuned | +0.0782±0.0901 [0.0440, 0.1065] | — |

传统流水线相对更强方法仍有约 0.05 Dice 的差距；oracle 逐图取两配置最好为 0.8068，这只是调参上界，不是泛化数字。

## 可复算验证（判据）

- 每张图每方法存 `data/organoid/gt/masks/{config}_id{id}.npz`，含 `gt_mask` / `pred_mask` / `gt_inst` / `pred_inst`；`config` ∈ `baseline_512tuned` / `adapt_256` / `cellpose3`（Cellpose 的 npz stem 为 `cellpose3_id{id}`）。
- `bench_organoid_gt.py` 末尾从 npz 重算全部 96 个 Dice，与 `metrics.json`/`per_image.csv` 对账：`n_checked=96`，`n_mismatch=0`；`verification` 字段已落盘。
- 逐图指标见 `per_image.csv`（96 行）；汇总与参数见 `metrics.json`；运行输出见 `run.log`。

## 边界：这个 GT 验证补上了什么、还缺什么

**补上了**：把"只有形状指标（核数、面积 CV、加噪保持率）"升级为"与人工金标准逐图对比的分割准确性"；能区分过分割/欠分割（Precision 0.82 vs Recall 0.76 等），并给出与更强方法的量化差距。数据是可公开复核的，不是自评。

**还缺**：**2D HeLa 核分割的 Dice 不等于 3D 类器官分割质量**。类器官是 3D 结构，涉及 Z 层、囊腔/边界、细胞层厚度、形变和批次差异；本轮数据是 2D 单层核、32 张小子集、单一染色通道、单一标注团队，结论只对"这类 2D 核图上的 Otsu+watershed 准确性"成立。未补 3D volumetric Dice、类器官边界/面积/圆度、单克隆生长轨迹、跨设备泛化。下一步若要做类器官金标准，需 3D 共聚焦/光片 Z-stack + 人工体积标注，再算 3D Dice/表面距离（如 ASSD），本轮数字不能外推。

## 产物路径

- 脚本：`bench_organoid_gt.py`
- 数据：`data/organoid/gt/dsb2018/`（images/ + annotations.json + manifest.json），子集图像 URL/SHA-256 在 manifest
- 指标与证据：`data/organoid/gt/metrics.json`、`data/organoid/gt/per_image.csv`、`data/organoid/gt/cellpose_metrics.json`、`data/organoid/gt/run.log`、`data/organoid/gt/masks/*.npz`（96 个）
- 图：`figures/organoid_gt_dice.png`、`figures/organoid_gt_paired.png`、`figures/organoid_gt_cellpose.png`、`figures/organoid_gt_examples.png`

复现命令（工作区根，Python 3.12 / skimage 0.26）：

```bash
python3 bench_organoid_gt.py
```

Cellpose 对照的一次性复现环境为 `/tmp/cellpose3-venv`（cellpose 3.1.1.2），nuclei 权重 25.3 MB，32 图 CPU 推理 643.3 s。重跑脚本会读取已落盘的 `cellpose_metrics.json` 与 npz，不再需要 Cellpose 运行环境。

## 声明

所有数字来自 `metrics.json` / `cellpose_metrics.json` / `run.log` 的实测输出；无编造。GT 的 license 为 CC0，来源 URL 与 SHA-256 已落盘。未改 README.md / BME-DEEP-DIVE.md / 既有 tracked 文件。
