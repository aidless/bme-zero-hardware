#!/usr/bin/env python3
"""mini_project_organoid.py -- 显微图像细胞核分割与形态学定量（纯计算，无湿实验）。

数据集：skimage.data.human_mitosis()
    真实公开荧光显微图像，HeLa 细胞 DAPI 染色细胞核，512x512 uint8。
    来源：scikit-image 官方 data 模块，原始文件
    AS_09125_050116030001_D03f00d0.tif（scikit-image GitLab 数据仓库，
    commit 2cdc5ce89b334d28f06a58c9f0ca21aa6992a5ba）。首次访问会下载到
    ~/.cache/scikit-image，之后完全离线可复现。

流程：
    高斯滤波 -> Otsu 阈值 -> 形态学清理 -> 距离变换 + 局部极大值种子
    -> watershed 分离接触核 -> regionprops 形态学定量。

所有输出数字均由本脚本运行时实测，写入 figures/organoid_summary.md。
使用 matplotlib Agg 后端，纯 CPU，无需 GPU。
"""
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from skimage import data, filters, morphology, segmentation, measure, feature
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", category=FutureWarning)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def segment_nuclei(image, sigma=2.0, min_distance=10, peak_threshold=3.0,
                   min_object=40, min_hole=200, area_low=80, area_high=2000):
    """Return (labels, binary, distance, seeds, regions) for one microscopy frame."""
    smooth = filters.gaussian(image, sigma=sigma)
    threshold = filters.threshold_otsu(smooth)
    binary = smooth > threshold
    # skimage >=0.26: `max_size` is the size threshold (objects <= max_size removed)
    binary = morphology.remove_small_objects(binary, max_size=min_object)
    binary = morphology.remove_small_holes(binary, max_size=min_hole)

    distance = ndi.distance_transform_edt(binary)
    coords = feature.peak_local_max(
        distance, min_distance=min_distance, labels=binary,
        threshold_abs=peak_threshold,
    )
    seeds = np.zeros(distance.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        seeds[r, c] = idx

    labels = segmentation.watershed(-distance, seeds, mask=binary)
    regions = [r for r in measure.regionprops(labels)
               if area_low <= r.area <= area_high]
    # keep only the region labels that passed the area filter
    keep = np.isin(labels, [r.label for r in regions])
    labels = np.where(keep, labels, 0)
    return labels, binary, distance, len(coords), threshold, regions


def main():
    image = data.human_mitosis()
    img = image.astype(np.float64) / image.max()

    labels, binary, distance, n_seeds, threshold, regions = segment_nuclei(img)

    areas = np.array([r.area for r in regions], dtype=float)
    diameters = np.array([r.equivalent_diameter_area for r in regions], dtype=float)
    circularity = np.array([4 * np.pi * r.area / r.perimeter ** 2 for r in regions], dtype=float)
    eccentricity = np.array([r.eccentricity for r in regions], dtype=float)

    def ms(x):
        return f"{x.mean():.2f} +/- {x.std(ddof=1):.2f}"

    print("=== DATASET ===")
    print(f"dataset: skimage.data.human_mitosis() [HeLa nuclei, DAPI]")
    print(f"shape={image.shape}, dtype={image.dtype}, intensity_range=({image.min()},{image.max()}), mean={image.mean():.2f}")
    print(f"threshold_otsu(normalized)={threshold:.4f}")
    print("=== SEGMENTATION ===")
    print(f"local_maxima_seeds={n_seeds}")
    print(f"nuclei_count={len(regions)}")
    print(f"labeled_pixel_fraction={np.count_nonzero(labels)/labels.size*100:.2f}%")
    print("=== MORPHOMETRICS (mean +/- std, sample std ddof=1) ===")
    print(f"area_px2:            {ms(areas)}   median={np.median(areas):.2f}  range=({areas.min():.0f},{areas.max():.0f})")
    print(f"equiv_diameter_px:   {ms(diameters)}   median={np.median(diameters):.2f}")
    print(f"circularity_4piA_P2: {ms(circularity)}   median={np.median(circularity):.3f}")
    print(f"eccentricity:        {ms(eccentricity)}   median={np.median(eccentricity):.3f}")

    # ---- Figure 1: original / binary mask / watershed overlay ----
    marked = segmentation.mark_boundaries(img, labels, color=(1.0, 0.35, 0.2))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("(a) Original - human_mitosis (HeLa nuclei)")
    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title("(b) Otsu mask + morphology cleanup")
    axes[2].imshow(marked)
    for r in regions:
        axes[2].plot(r.centroid[1], r.centroid[0], "b.", markersize=2, alpha=0.6)
    axes[2].set_title(f"(c) Watershed seeds: {len(regions)} nuclei")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Nuclei segmentation pipeline - skimage.data.human_mitosis()", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "organoid_segmentation.png"))
    plt.close(fig)

    # ---- Figure 2: morphometric distributions ----
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    specs = [
        (axes[0, 0], areas, "Nuclear area (px$^2$)", "count", "#2b6cb0", "area_px2"),
        (axes[0, 1], diameters, "Equivalent diameter (px)", "count", "#2f855a", "diameter_px"),
        (axes[1, 0], circularity, "Circularity  $4\\pi A/P^2$", "count", "#c05621", "circularity"),
        (axes[1, 1], eccentricity, "Eccentricity", "count", "#6b46c1", "eccentricity"),
    ]
    for ax, vals, xlabel, ylabel, color, _ in specs:
        ax.hist(vals, bins=28, color=color, alpha=0.85, edgecolor="white", linewidth=0.6)
        m, s = vals.mean(), vals.std(ddof=1)
        ax.axvline(m, color="black", linewidth=1.4, linestyle="-")
        ax.axvline(m - s, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axvline(m + s, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"mean={m:.2f}, SD={s:.2f}, N={len(vals)}")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Morphometric distributions - {len(regions)} segmented nuclei (mean +/- SD)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG_DIR, "organoid_morphometrics.png"))
    plt.close(fig)

    # ---- Figure 3: area vs circularity scatter with regression-free guidance ----
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(areas, circularity, s=18, c=diameters, cmap="viridis", alpha=0.8, edgecolor="none")
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Equivalent diameter (px)")
    ax.set_xlabel("Nuclear area (px$^2$)")
    ax.set_ylabel("Circularity  $4\\pi A/P^2$")
    ax.set_title(f"Area vs shape - {len(regions)} nuclei")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "organoid_shape_scatter.png"))
    plt.close(fig)

    # ---- written summary (numbers are runtime-measured, not hard-coded) ----
    summary = f"""# 类器官/细胞显微图像分析 — 实测结论

## 数据来源（真实公开数据集，非编造）
- `skimage.data.human_mitosis()`：HeLa 细胞 DAPI 染色荧光显微图像，{image.shape[0]}x{image.shape[1]} uint8。
- 来源：scikit-image 官方 data 模块，原始文件 `AS_09125_050116030001_D03f00d0.tif`
  （scikit-image GitLab 数据仓库，commit `2cdc5ce89b334d28f06a58c9f0ca21aa6992a5ba`）。
  首次访问下载到 `~/.cache/scikit-image`，之后完全离线。
- 无湿实验、无细胞培养；数据集已公开、可直接复算。

## 方法
高斯平滑(sigma=2.0) -> Otsu 阈值 -> 去小物体/填洞 -> 距离变换局部极大值种子
(min_distance=10, threshold_abs=3.0) -> watershed -> regionprops 形态学定量。

## 实测数字（mean +/- SD，样本标准差 ddof=1）
- 局部极大值种子数：{n_seeds}
- 分割细胞核数：**{len(regions)}**
- 标记像素占比：{np.count_nonzero(labels)/labels.size*100:.2f}%
- 面积 (px^2)：{ms(areas)}，中位数 {np.median(areas):.2f}，范围 [{areas.min():.0f}, {areas.max():.0f}]
- 等效直径 (px)：{ms(diameters)}，中位数 {np.median(diameters):.2f}
- 圆度 4piA/P^2：{ms(circularity)}，中位数 {np.median(circularity):.3f}
- 离心率：{ms(eccentricity)}，中位数 {np.median(eccentricity):.3f}

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
"""
    with open(os.path.join(FIG_DIR, "organoid_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    print("saved figures/ and figures/organoid_summary.md")


if __name__ == "__main__":
    main()
