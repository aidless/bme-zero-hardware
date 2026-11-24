#!/usr/bin/env python3
"""细胞图像分割审稿级重估（第三轮深挖）。

对照 mini_project_organoid.py 的单组参数结果，补上审稿人会问的三件事：
  1. 参数敏感性网格 —— sigma × min_distance × peak_threshold，看核计数与
     面积均值随参数的波动（单组参数得出的数字是否稳健）
  2. bootstrap CI —— 对面积/直径/圆度分布重采样，给 95% 区间
  3. 加噪鲁棒性 —— 对图像叠加不同强度高斯噪声，看核计数保持率

诚实边界：human_mitosis 是 HeLa 单层细胞核（2D），不是真正的 3D 类器官；
且无人工分割金标准，无法算 Dice/IoU —— 这些限制写进产物，不自吹。

所有数字运行时实测，无硬编码。纯 CPU。运行：python3 bench_organoid_rigorous.py
"""
import os
import warnings
import json
import itertools
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage import data, filters, morphology, segmentation, measure, feature
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.getcwd()
FIG_DIR = os.path.join(ROOT, "figures")
RESULT_DIR = os.path.join(ROOT, "data", "organoid")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = 42
N_BOOT = 2000


def segment(img, sigma=2.0, min_distance=10, peak_threshold=3.0,
            min_object=40, min_hole=200, area_low=80, area_high=2000):
    smooth = filters.gaussian(img, sigma=sigma)
    thr = filters.threshold_otsu(smooth)
    binary = smooth > thr
    binary = morphology.remove_small_objects(binary, max_size=min_object)
    binary = morphology.remove_small_holes(binary, max_size=min_hole)
    dist = ndi.distance_transform_edt(binary)
    coords = feature.peak_local_max(dist, min_distance=min_distance,
                                    labels=binary, threshold_abs=peak_threshold)
    seeds = np.zeros(dist.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        seeds[r, c] = i
    labels = segmentation.watershed(-dist, seeds, mask=binary)
    regions = [r for r in measure.regionprops(labels)
               if area_low <= r.area <= area_high]
    areas = np.array([r.area for r in regions], dtype=float)
    return len(regions), areas, len(coords), thr


def main():
    t0 = time.time()
    np.random.seed(SEED)
    print("=" * 72)
    print("CELL-IMAGE RIGOROUS REEVALUATION -- sensitivity / bootstrap / noise")
    print("=" * 72)

    image = data.human_mitosis()
    img = image.astype(np.float64) / image.max()
    print(f"[data] human_mitosis() shape={image.shape} dtype={image.dtype} "
          f"range=({image.min()},{image.max()})")
    print("[note] HeLa monolayer 2D nuclei, NOT a 3D organoid; "
          "no ground-truth mask -> no Dice/IoU possible")

    results = {}

    # ---------- 1. 参数敏感性网格 ----------
    sigmas = [1.5, 2.0, 2.5, 3.0]
    min_dists = [8, 10, 12]
    peak_thrs = [2.0, 3.0, 4.0]
    grid = []
    for sigma, md, pt in itertools.product(sigmas, min_dists, peak_thrs):
        n, areas, nseeds, thr = segment(img, sigma=sigma, min_distance=md,
                                        peak_threshold=pt)
        grid.append({"sigma": sigma, "min_distance": md, "peak_threshold": pt,
                     "n_nuclei": n, "area_mean": float(areas.mean()) if len(areas) else 0.0,
                     "n_seeds": nseeds, "otsu_thr": float(thr)})
    counts = np.array([g["n_nuclei"] for g in grid])
    area_means = np.array([g["area_mean"] for g in grid])
    print(f"[grid] n_configs={len(grid)} nuclei_count mean={counts.mean():.1f} "
          f"std={counts.std(ddof=1):.1f} range=[{counts.min()},{counts.max()}] cv={counts.std(ddof=1)/counts.mean():.3f}")
    print(f"[grid] area_mean across configs mean={area_means.mean():.1f} "
          f"std={area_means.std(ddof=1):.1f}")
    baseline = [g for g in grid if g["sigma"] == 2.0 and g["min_distance"] == 10
                and g["peak_threshold"] == 3.0][0]
    print(f"[grid] baseline(2.0,10,3.0) nuclei={baseline['n_nuclei']} "
          f"area_mean={baseline['area_mean']:.1f}")
    results["sensitivity_grid"] = {
        "configs": grid,
        "nuclei_count_mean": float(counts.mean()),
        "nuclei_count_std": float(counts.std(ddof=1)),
        "nuclei_count_cv": float(counts.std(ddof=1) / counts.mean()),
        "nuclei_count_range": [int(counts.min()), int(counts.max())],
        "area_mean_range": [float(area_means.min()), float(area_means.max())],
        "baseline": baseline,
    }

    # ---------- 2. bootstrap CI（baseline 参数的形态分布） ----------
    n, areas, nseeds, thr = segment(img)  # baseline params
    smooth = filters.gaussian(img, sigma=2.0)
    binary = morphology.remove_small_objects(smooth > thr, max_size=40)
    binary = morphology.remove_small_holes(binary, max_size=200)
    dist = ndi.distance_transform_edt(binary)
    coords = feature.peak_local_max(dist, min_distance=10, labels=binary,
                                    threshold_abs=3.0)
    seeds = np.zeros(dist.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        seeds[r, c] = i
    labels = segmentation.watershed(-dist, seeds, mask=binary)
    regs = [r for r in measure.regionprops(labels) if 80 <= r.area <= 2000]
    diameters = np.array([r.equivalent_diameter_area for r in regs], dtype=float)
    circ = np.array([4 * np.pi * r.area / r.perimeter ** 2 for r in regs], dtype=float)
    ecc = np.array([r.eccentricity for r in regs], dtype=float)

    boot = {"area": [], "diameter": [], "circularity": [], "eccentricity": []}
    arrs = {"area": areas, "diameter": diameters, "circularity": circ, "eccentricity": ecc}
    for _ in range(N_BOOT):
        idx = np.random.randint(0, len(areas), len(areas))
        for k, arr in arrs.items():
            boot[k].append(arr[idx].mean())
    ci = {}
    for k in boot:
        ci[k] = [float(v) for v in np.percentile(boot[k], [2.5, 97.5])]
        print(f"[bootstrap] {k:12s} mean={arrs[k].mean():.3f} "
              f"95%CI=[{ci[k][0]:.3f},{ci[k][1]:.3f}]")
    results["bootstrap_ci95"] = {
        k: {"mean": float(arrs[k].mean()), "ci95": ci[k]} for k in arrs
    }
    results["baseline_morphometrics"] = {
        "n_nuclei": len(regs), "n_seeds": nseeds,
        "labeled_pixel_fraction": float(np.count_nonzero(labels) / labels.size * 100),
    }

    # ---------- 3. 加噪鲁棒性 ----------
    noise_levels = [0.0, 0.02, 0.05, 0.10, 0.15]
    noise_res = []
    base_count = len(regs)
    for nl in noise_levels:
        counts_nl = []
        for rep in range(5):  # 每档 5 次重复取均值，避免单次噪声偶然
            noisy = img + np.random.normal(0, nl, img.shape)
            noisy = np.clip(noisy, 0, 1)
            nn, _, _, _ = segment(noisy)
            counts_nl.append(nn)
        ret = np.mean(counts_nl) / base_count
        noise_res.append({"noise_sigma": nl, "n_count_mean": float(np.mean(counts_nl)),
                          "retention": float(ret), "reps": 5})
        print(f"[noise] sigma={nl:.2f} nuclei_mean={np.mean(counts_nl):.1f} "
              f"retention={ret:.3f}")
    results["noise_robustness"] = noise_res

    # ---------- 4. 图：参数敏感性热图 ----------
    # 固定 peak_threshold=3.0，看 sigma × min_distance 的核计数
    mat = np.zeros((len(sigmas), len(min_dists)))
    for i, s in enumerate(sigmas):
        for j, md in enumerate(min_dists):
            rec = [g for g in grid if g["sigma"] == s and g["min_distance"] == md
                   and g["peak_threshold"] == 3.0][0]
            mat[i, j] = rec["n_nuclei"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(min_dists))); ax.set_xticklabels(min_dists)
    ax.set_yticks(range(len(sigmas))); ax.set_yticklabels(sigmas)
    ax.set_xlabel("min_distance (px)"); ax.set_ylabel("gaussian sigma")
    for i in range(len(sigmas)):
        for j in range(len(min_dists)):
            ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center",
                    color="white", fontsize=10)
    ax.set_title(f"Nuclei count sensitivity (peak_thr=3.0)\n"
                 f"range [{int(mat.min())},{int(mat.max())}], CV={counts.std(ddof=1)/counts.mean():.3f}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p_fig = os.path.join(FIG_DIR, "organoid_sensitivity_heatmap.png")
    fig.savefig(p_fig, dpi=150)
    plt.close(fig)
    print(f"[figure] {p_fig}")

    results["metadata"] = {
        "dataset": "skimage.data.human_mitosis() [HeLa nuclei, DAPI, 2D]",
        "ground_truth": "none -> Dice/IoU not computable; honest limitation",
        "limitations": [
            "2D monolayer nuclei, not 3D organoid structure",
            "no manual ground-truth mask, no segmentation accuracy metric",
            "single public example image, no biological functionality validation",
        ],
        "elapsed_s": float(time.time() - t0),
    }
    out = os.path.join(RESULT_DIR, "organoid_rigorous_metrics.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"[artifact] {out}")
    print(f"[done] elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
