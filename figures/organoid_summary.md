# 类器官/细胞显微图像分析 — 实测结论

## 数据来源（真实公开数据集，非编造）
- `skimage.data.human_mitosis()`：HeLa 细胞 DAPI 染色荧光显微图像，512x512 uint8。
- 来源：scikit-image 官方 data 模块，原始文件 `AS_09125_050116030001_D03f00d0.tif`
  （scikit-image GitLab 数据仓库，commit `2cdc5ce89b334d28f06a58c9f0ca21aa6992a5ba`）。
  首次访问下载到 `~/.cache/scikit-image`，之后完全离线。
- 无湿实验、无细胞培养；数据集已公开、可直接复算。

## 方法
高斯平滑(sigma=2.0) -> Otsu 阈值 -> 去小物体/填洞 -> 距离变换局部极大值种子
(min_distance=10, threshold_abs=3.0) -> watershed -> regionprops 形态学定量。

## 实测数字（mean +/- SD，样本标准差 ddof=1）
- 局部极大值种子数：247
- 分割细胞核数：**230**
- 标记像素占比：12.52%
- 面积 (px^2)：142.71 +/- 72.76，中位数 123.00，范围 [80, 584]
- 等效直径 (px)：13.15 +/- 2.97，中位数 12.51
- 圆度 4piA/P^2：0.97 +/- 0.13，中位数 1.010
- 离心率：0.57 +/- 0.17，中位数 0.553

## 结论（纯计算能做什么 / 不能做什么）
**能做**：对已有显微图像做自动细胞核计数、面积/直径/圆度/离心率分布定量、
聚集/接触核分离（watershed）、组间形态统计对比、分割结果可视化与批量流水线；
纯 CPU 可复现，适合作为类器官/细胞图像的自动化表型分析原型。

**不能做**：不产生任何湿实验数据——不能培养类器官、不能染色/成像、不能做实时活细胞成像、
不能验证生物学功能或因果（增殖/分化/药效需湿实验+对照）；无物理尺度标定（结果为像素单位），
也不替代经专业标注的分割金标准评估；单张公开示例图像不代表真实类器官 3D 结构的复杂度。

## 产物
- `mini_project_organoid.py`（本脚本，`python3 mini_project_organoid.py` 可重跑）
- `figures/organoid_segmentation.png`, `figures/organoid_morphometrics.png`, `figures/organoid_shape_scatter.png`
