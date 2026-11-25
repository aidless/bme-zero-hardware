#!/usr/bin/env python3
"""P1-6 补金标准：在带 GT 的公开核分割数据上算可复算的 Dice/IoU。

数据集: DSB2018 / BBBC038 stage1_train 小子集 (DataTorchio HF 镜像, CC0)。
流水线: 复用 bench_organoid_rigorous.py 的 Otsu+watershed 逻辑，逐图对 GT 计算
   - 像素级: Dice / IoU / Precision / Recall / F1
   - 实例级: Hungarian 匹配后的 matched@IoU0.5、match-F1、计数误差
   - 配置: baseline(512 调参) 与 adapt_256(针对 256px 缩放)
   - bootstrap CI (按图重采样, seed=42, B=2000)
   - 对照: 多配置 oracle 上界 (非泛化)
所有数字逐图落盘 metrics.json 与 per_image.csv；GT/pred mask 落 npz 供逐图重算。

边界: 2D 核分割 Dice != 3D 类器官分割质量。见 notes/2026-11-26-organoid-gt.md。
"""
import os, json, csv, time, hashlib, warnings

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import filters, morphology, segmentation, measure, draw, feature

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.getcwd()
GT_DIR = os.path.join(ROOT, "data", "organoid", "gt")
DSB = os.path.join(GT_DIR, "dsb2018")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
SEED = 42
N_BOOT = 2000

MASK_DIR = os.path.join(GT_DIR, "masks")
os.makedirs(MASK_DIR, exist_ok=True)
RESULT_DIR = GT_DIR  # metrics.json / per_image.csv 落 data/organoid/gt/


# ---------- 数据加载 ----------
def load_gt():
    ann = json.load(open(os.path.join(DSB, "annotations.json"), encoding="utf-8"))
    manifest = json.load(open(os.path.join(DSB, "manifest.json"), encoding="utf-8"))
    by_image = {}
    for a in ann["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    samples = []
    for m in manifest:
        iid = m["image_id"]
        img = np.array(Image.open(os.path.join(DSB, "images", m["file_name"])))
        gray = img[..., 0].astype(np.float64) / 255.0  # RGBA: 单通道(核染色)
        H, W = gray.shape[:2]
        inst = np.zeros((H, W), dtype=np.int32)
        for k, a in enumerate(sorted(by_image[iid], key=lambda x: x["id"]), start=1):
            for poly in a["segmentation"]:
                rs, cs = draw.polygon(np.asarray(poly[1::2]),
                                      np.asarray(poly[0::2]), shape=(H, W))
                inst[rs, cs] = k
        samples.append({"image_id": iid, "file_name": m["file_name"],
                        "img": gray, "gt_inst": inst, "gt_mask": inst > 0,
                        "sha256": m["sha256"], "gt_url": m["url"]})
    return samples


# ---------- 流水线 (复用 bench_organoid_rigorous.py segment) ----------
def segment_labels(img, sigma=2.0, min_distance=10, peak_threshold=3.0,
                   min_object=40, min_hole=200, area_low=80, area_high=2000):
    smooth = filters.gaussian(img, sigma=sigma)
    thr = float(filters.threshold_otsu(smooth))
    binary = smooth > thr
    binary = morphology.remove_small_objects(binary, min_size=min_object)
    binary = morphology.remove_small_holes(binary, area_threshold=min_hole)
    dist = ndi.distance_transform_edt(binary)
    coords = feature.peak_local_max(dist, min_distance=min_distance,
                                    labels=binary, threshold_abs=peak_threshold)
    seeds = np.zeros(dist.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        seeds[r, c] = i
    labels = segmentation.watershed(-dist, seeds, mask=binary)
    keep = np.zeros(labels.max() + 1, dtype=bool)
    for r in measure.regionprops(labels):
        if area_low <= r.area <= area_high:
            keep[r.label] = True
    out = np.where(keep[labels], labels, 0).astype(np.int32)
    return binary, out, thr, len(coords)


# ---------- 指标 ----------
def pixel_metrics(pred, gt):
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "dice": float(dice), "iou": float(iou),
            "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def instance_metrics(pred_inst, gt_inst, iou_thr=0.5):
    gids = [r.label for r in measure.regionprops(gt_inst)]
    pids = [r.label for r in measure.regionprops(pred_inst)]
    if not gids or not pids:
        return {"n_gt": len(gids), "n_pred": len(pids), "n_matched_05": 0,
                "match_f1_05": 0.0, "mean_match_iou": 0.0,
                "count_error": len(pids) - len(gids)}
    G = np.stack([(gt_inst == lab) for lab in gids]).astype(np.float64).reshape(len(gids), -1)
    P = np.stack([(pred_inst == lab) for lab in pids]).astype(np.float64).reshape(len(pids), -1)
    inter = G @ P.T
    union = G.sum(1)[:, None] + P.sum(1)[None, :] - inter
    iou = inter / np.maximum(union, 1)
    rows, cols = linear_sum_assignment(-iou)
    match_iou = iou[rows, cols]
    n05 = int((match_iou >= iou_thr).sum())
    prec = n05 / len(pids); rec = n05 / len(gids)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n_gt": len(gids), "n_pred": len(pids), "n_matched_05": n05,
            "match_f1_05": float(f1), "mean_match_iou": float(match_iou.mean()),
            "count_error": int(len(pids) - len(gids))}


def bootstrap_ci(values, seed=SEED, n=N_BOOT):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"mean": None, "std": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(n)])
    return {"mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "ci95": [float(x) for x in np.percentile(means, [2.5, 97.5])]}


def aggregate(rows):
    return {m: bootstrap_ci([r[m] for r in rows])
            for m in ["dice", "iou", "precision", "recall", "f1",
                      "match_f1_05", "mean_match_iou"]}


# ---------- 主流程 ----------
def main():
    t0 = time.time()
    np.random.seed(SEED)
    samples = load_gt()
    print("[data] DSB2018/BBBC038 subset n=%d 256x256 nuclei GT" % len(samples))
    print("[data] license=CC0; source=DataTorchio HF mirror of BBBC038 stage1_train")

    configs = {
        "baseline_512tuned": dict(sigma=2.0, min_distance=10, peak_threshold=3.0,
                                  min_object=40, min_hole=200, area_low=80, area_high=2000),
        "adapt_256": dict(sigma=2.0, min_distance=6, peak_threshold=3.0,
                          min_object=20, min_hole=100, area_low=40, area_high=1200),
    }

    results = {"dataset": {
        "name": "DSB2018 / BBBC038 stage1_train subset",
        "source_url": "https://huggingface.co/datasets/DataTorchio/nuclei-segmentation-dsb2018",
        "origin": "https://bbbc.broadinstitute.org/BBBC038",
        "license": "CC0 1.0 (public domain)",
        "n_images": len(samples), "image_size": [256, 256],
        "image_ids": [s["image_id"] for s in samples],
        "annotations_sha256": hashlib.sha256(
            open(os.path.join(DSB, "annotations.json"), "rb").read()).hexdigest(),
        "subset_manifest": "data/organoid/gt/dsb2018/manifest.json",
        "gt_rasterization": "COCO polygon -> skimage.draw.polygon -> instance label",
    }, "configs": {}, "per_image": [], "comparison": {}, "verification": {},
        "metadata": {}}

    per_image_rows = []
    for cname, params in configs.items():
        print("\n[config] %s  params=%s" % (cname, params))
        rows = []
        for s in samples:
            _, pred_inst, thr, nseed = segment_labels(s["img"], **params)
            pred_mask = pred_inst > 0
            pm = pixel_metrics(pred_mask, s["gt_mask"])
            im = instance_metrics(pred_inst, s["gt_inst"])
            row = {"image_id": s["image_id"], "file_name": s["file_name"],
                   "config": cname, "image_sha256": s["sha256"],
                   "otsu_thr": thr, "n_seeds": nseed, **pm, **im}
            rows.append(row)
            per_image_rows.append(row)
            # 逐图 mask 存档 (供 verifier 重算 Dice)
            np.savez_compressed(
                os.path.join(MASK_DIR, "%s_id%d.npz" % (cname, s["image_id"])),
                gt_mask=s["gt_mask"], pred_mask=pred_mask,
                gt_inst=s["gt_inst"], pred_inst=pred_inst)
        agg = aggregate(rows)
        results["configs"][cname] = {"params": params, "n_images": len(rows),
                                     "aggregate": agg}
        print("[agg] %s Dice=%.4f [%.4f,%.4f] IoU=%.4f matchF1@.5=%.4f" % (
            cname, agg["dice"]["mean"], agg["dice"]["ci95"][0], agg["dice"]["ci95"][1],
            agg["iou"]["mean"], agg["match_f1_05"]["mean"]))

    # ---------- 可选: Cellpose3 nuclei 对照 (mask/指标由独立 run 落盘) ----------
    cp_path = os.path.join(RESULT_DIR, "cellpose_metrics.json")
    if os.path.exists(cp_path):
        cp_raw = json.load(open(cp_path, encoding="utf-8"))
        cp_rows = []
        for r in cp_raw["per_image"]:
            d = np.load(os.path.join(MASK_DIR, "cellpose3_id%d.npz" % r["image_id"]))
            gt_ref = np.load(os.path.join(MASK_DIR, "baseline_512tuned_id%d.npz" % r["image_id"]))
            im = instance_metrics(d["pred_inst"], gt_ref["gt_inst"])
            row = {k: r[k] for k in ["image_id", "file_name", "image_sha256",
                                     "tp", "fp", "fn", "dice", "iou", "precision",
                                     "recall", "f1"]}
            row.update(im)
            row["config"] = "cellpose3_nuclei"
            cp_rows.append(row)
            per_image_rows.append(row)
        configs["cellpose3_nuclei"] = None  # 仅用于 by_cfg/图形枚举
        results["configs"]["cellpose3_nuclei"] = {
            "params": {k: cp_raw[k] for k in ["model", "model_type", "gpu", "diameter",
                                               "diameter_auto", "channels", "device",
                                               "input_dtype"]},
            "n_images": len(cp_rows), "aggregate": aggregate(cp_rows),
            "elapsed_s": cp_raw.get("elapsed_s"),
        }
        print("[cellpose3] Dice=%.4f [%.4f,%.4f] IoU=%.4f" % (
            results["configs"]["cellpose3_nuclei"]["aggregate"]["dice"]["mean"],
            results["configs"]["cellpose3_nuclei"]["aggregate"]["dice"]["ci95"][0],
            results["configs"]["cellpose3_nuclei"]["aggregate"]["dice"]["ci95"][1],
            results["configs"]["cellpose3_nuclei"]["aggregate"]["iou"]["mean"]))

    # ---------- 对照: 两套传统配置的配对差 + oracle 上界 ----------
    by_cfg = {}
    for cname in configs:
        by_cfg[cname] = {r["image_id"]: r for r in per_image_rows if r["config"] == cname}
    ids = [s["image_id"] for s in samples]
    diffs = [by_cfg["adapt_256"][i]["dice"] - by_cfg["baseline_512tuned"][i]["dice"] for i in ids]
    results["comparison"] = {
        "paired_dice_diff_adapt_minus_baseline": bootstrap_ci(diffs),
        "n_adapt_better": int(sum(d > 0 for d in diffs)),
        "n_baseline_better": int(sum(d < 0 for d in diffs)),
        "n_tie": int(sum(d == 0 for d in diffs)),
        "note": "oracle=max per-image over configs is a tuning upper bound, NOT generalization",
        "oracle_dice_upper_bound": float(np.mean([
            max(by_cfg[c][i]["dice"] for c in ["baseline_512tuned", "adapt_256"]) for i in ids])),
    }
    if "cellpose3_nuclei" in by_cfg:
        cp_da = [by_cfg["cellpose3_nuclei"][i]["dice"] - by_cfg["adapt_256"][i]["dice"] for i in ids]
        cp_db = [by_cfg["cellpose3_nuclei"][i]["dice"] - by_cfg["baseline_512tuned"][i]["dice"] for i in ids]
        results["comparison"]["cellpose_minus_adapt_dice_diff"] = bootstrap_ci(cp_da)
        results["comparison"]["cellpose_minus_baseline_dice_diff"] = bootstrap_ci(cp_db)
        results["comparison"]["n_cellpose_better_than_adapt"] = int(sum(d > 0 for d in cp_da))
        print("[compare] cellpose-adapt Dice diff mean=%.4f (better %d/%d)" % (
            results["comparison"]["cellpose_minus_adapt_dice_diff"]["mean"],
            results["comparison"]["n_cellpose_better_than_adapt"], len(ids)))
    print("[compare] adapt-baseline Dice diff mean=%.4f (adapt better %d/%d)" % (
        results["comparison"]["paired_dice_diff_adapt_minus_baseline"]["mean"],
        results["comparison"]["n_adapt_better"], len(ids)))

    # ---------- 验证: 从落盘 npz 重算 Dice, 与 per_image 对账 ----------
    mismatches = []
    for r in per_image_rows:
        stem = "cellpose3_id%d" % r["image_id"] if r["config"] == "cellpose3_nuclei" \
            else "%s_id%d" % (r["config"], r["image_id"])
        d = np.load(os.path.join(MASK_DIR, stem + ".npz"))
        recomputed = pixel_metrics(d["pred_mask"], d["gt_mask"])["dice"]
        if abs(recomputed - r["dice"]) > 1e-12:
            mismatches.append({"config": r["config"], "image_id": r["image_id"],
                               "json": r["dice"], "recomputed": recomputed})
    results["verification"] = {
        "rule": "each Dice recomputed from saved gt_mask/pred_mask npz",
        "n_checked": len(per_image_rows), "n_mismatch": len(mismatches),
        "mismatches": mismatches,
    }
    print("[verify] recomputed %d Dice from npz; mismatches=%d" % (
        len(per_image_rows), len(mismatches)))

    # ---------- 落盘 ----------
    results["per_image"] = per_image_rows
    results["metadata"] = {
        "seed": SEED, "n_boot": N_BOOT,
        "pipeline": "gaussian->Otsu->remove_small_obj/holes->distance->peak_local_max->watershed->area filter",
        "boundary": "2D nuclei Dice != 3D organoid segmentation quality",
        "elapsed_s": float(time.time() - t0),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = os.path.join(RESULT_DIR, "metrics.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("[artifact] %s" % out)

    with open(os.path.join(RESULT_DIR, "per_image.csv"), "w", newline="", encoding="utf-8") as fh:
        fields = ["image_id", "file_name", "config", "tp", "fp", "fn",
                  "dice", "iou", "precision", "recall", "f1",
                  "n_gt", "n_pred", "n_matched_05", "match_f1_05",
                  "mean_match_iou", "count_error", "otsu_thr", "n_seeds"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in per_image_rows:
            w.writerow(r)

    make_figures(results, samples, by_cfg)
    print("[done] elapsed=%.1fs" % (time.time() - t0))


def make_figures(results, samples, by_cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = [s["image_id"] for s in samples]

    # Fig1: 像素 Dice 分布 (baseline vs adapt)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = [by_cfg["baseline_512tuned"][i]["dice"] for i in ids]
    ap = [by_cfg["adapt_256"][i]["dice"] for i in ids]
    series = [bp, ap]
    names = ["baseline\n(512-tuned)", "adapt_256"]
    colors = ["#E08A3C", "#0F4D92"]
    if "cellpose3_nuclei" in by_cfg:
        series.append([by_cfg["cellpose3_nuclei"][i]["dice"] for i in ids])
        names.append("Cellpose3\nnuclei")
        colors.append("#4F8F6B")
    ax.boxplot(series, tick_labels=names, patch_artist=True, widths=0.55)
    for patch, color in zip(ax.patches, colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    for j, (v, color) in enumerate(zip(series, colors), start=1):
        ax.scatter(np.random.default_rng(SEED + j).normal(j, 0.05, len(v)), v,
                   s=14, color=color, alpha=0.7, zorder=3)
    ax.set_ylabel("Pixel-wise Dice")
    ax.set_title("DSB2018 subset (n=%d): Otsu+watershed vs Cellpose vs GT" % len(ids))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "organoid_gt_dice.png"), dpi=300)
    plt.close(fig)

    # Fig2: 逐图 Dice 对比散点 (paired)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(bp, ap, s=28, color="#0F4D92", alpha=0.75)
    lim = [0, 1]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Baseline (512-tuned) Dice")
    ax.set_ylabel("Adapt_256 Dice")
    ax.set_title("Paired per-image Dice (n=%d)" % len(ids))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "organoid_gt_paired.png"), dpi=300)
    plt.close(fig)

    # Fig2b: 传统流水线 vs 更强方法 (paired)
    if "cellpose3_nuclei" in by_cfg:
        cp = [by_cfg["cellpose3_nuclei"][i]["dice"] for i in ids]
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.scatter(ap, cp, s=28, color="#4F8F6B", alpha=0.8)
        ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Adapt_256 Dice (Otsu+watershed)")
        ax.set_ylabel("Cellpose3 nuclei Dice")
        ax.set_title("Paired per-image Dice (n=%d)" % len(ids))
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "organoid_gt_cellpose.png"), dpi=300)
        plt.close(fig)

    # Fig3: 样例 GT vs 预测叠加
    vis_ids = ids[:4]
    fig, axes = plt.subplots(len(vis_ids), 3, figsize=(9, 3 * len(vis_ids)))
    from skimage.color import label2rgb
    for i, iid in enumerate(vis_ids):
        s = next(s for s in samples if s["image_id"] == iid)
        axes[i, 0].imshow(s["img"], cmap="gray"); axes[i, 0].set_title("image id=%d" % iid)
        axes[i, 1].imshow(label2rgb(s["gt_inst"], bg_label=0), interpolation="nearest")
        axes[i, 1].set_title("GT (%d nuclei)" % by_cfg["adapt_256"][iid]["n_gt"])
        pred_inst = np.load(os.path.join(MASK_DIR, "adapt_256_id%d.npz" % iid))["pred_inst"]
        axes[i, 2].imshow(label2rgb(pred_inst, bg_label=0), interpolation="nearest")
        axes[i, 2].set_title("pred Dice=%.3f" % by_cfg["adapt_256"][iid]["dice"])
        for j in range(3):
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "organoid_gt_examples.png"), dpi=200)
    plt.close(fig)
    print("[figure] organoid_gt_dice.png / organoid_gt_paired.png / "
          "organoid_gt_cellpose.png / organoid_gt_examples.png")


if __name__ == "__main__":
    main()
