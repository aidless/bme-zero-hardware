#!/usr/bin/env python3
"""第八轮 · 同架构纯预训练消融：把第七轮 +0.049 拆成"域"与"架构"两部分。

第七轮同协议 head-to-head：Rad-DINO(ViT-B/14, 胸片自监督) 0.9735 vs ResNet18(ImageNet监督) 0.9242，Δ=+0.0493。
但两者**架构也不同**（ViT vs CNN），所以 +0.049 是"预训练数据域 + 架构"的合并效应。

本轮按 ablation-design 的模块替换法，一次只变一个因素：
  - 新增臂 dinov2-base：与 Rad-DINO **完全同架构（ViT-B/14, 768d CLS）+ 同预处理（rad-dino processor, 518px）
    + 同线性探针**，唯一差别是预训练数据——dinov2-base 在自然图像自监督，Rad-DINO 在胸片自监督。
  - 因此 rad-dino − dinov2-base = 纯"预训练域"贡献（架构、预处理、探针全部固定）。
  - dinov2-base − resnet18 ≈ 架构（ViT自监督 vs CNN监督）残余贡献。

分解：total(+0.0493) = domain(rad−dinov2) + arch_residual(dinov2−resnet18)

诚实边界：dinov2-base 的自然图像自监督 ≠ ResNet18 的 ImageNet 监督，"架构残余"里仍混有监督方式差异，
不可读成纯"架构"效应；本轮只做单因素（域）的干净消融，第二项标"架构+监督方式合并残余"。
"""
import json
import os
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import binomtest
from torch.utils.data import DataLoader

torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "frontier")
SEED = 42
N_BOOT = 2000
BATCH = 8


def extract_dinov2(model, processor, ds, name):
    feats, labels = [], []
    def collate(items):
        imgs = [pre(img.convert("RGB")) if not isinstance(img, Image.Image)
                else pre(img.convert("RGB")) for img, _ in items]
        # processor 接受图像列表，批量返回 pixel_values
        batch = processor(imgs, return_tensors="pt")
        y = torch.tensor([int(np.asarray(l).ravel()[0]) for _, l in items])
        return batch, y
    def pre(img):
        return img  # 占位，实际由 processor 处理
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ds), BATCH):
            items = [ds[j] for j in range(i, min(i + BATCH, len(ds)))]
            imgs = [im.convert("RGB") if isinstance(im, Image.Image) else Image.fromarray(im).convert("RGB")
                    for im, _ in items]
            batch = processor(imgs, return_tensors="pt")
            pv = batch["pixel_values"].cuda()
            out = model(pixel_values=pv)
            cls = out.last_hidden_state[:, 0].cpu().numpy()  # (B,768)
            feats.append(cls)
            labels.extend(int(np.asarray(l).ravel()[0]) for _, l in items)
            if (i // BATCH + 1) % 15 == 0:
                print(f"  {name} {i+len(items)}/{len(ds)}", flush=True)
    return np.concatenate(feats, axis=0), np.asarray(labels)  # 末批不足 BATCH，需 concatenate 而非 asarray


def paired_bootstrap(y, p_a, p_b, seed=SEED):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    d = np.asarray(d)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main():
    from transformers import AutoModel, AutoImageProcessor
    from medmnist import PneumoniaMNIST

    torch.manual_seed(SEED); np.random.seed(SEED)
    print("loading dinov2-base (natural-image self-supervised ViT-B/14)...", flush=True)
    model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
    # 用 rad-dino 的 processor 保证预处理与第七轮 Rad-DINO 臂完全一致（唯一变量=权重/预训练域）
    proc = AutoImageProcessor.from_pretrained("microsoft/rad-dino", use_fast=False)
    print(f"processor size: {getattr(proc,'size',None)}", flush=True)

    ds_tr = PneumoniaMNIST(split="train", download=True, transform=None)
    ds_te = PneumoniaMNIST(split="test", download=True, transform=None)
    Xtr, ytr = extract_dinov2(model, proc, ds_tr, "train")
    Xte, yte = extract_dinov2(model, proc, ds_te, "test")
    print(f"dinov2 feats {Xtr.shape} {Xte.shape}", flush=True)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED))
    clf.fit(Xtr, ytr)
    prob_dino = clf.predict_proba(Xte)[:, 1]
    auc_dino = float(roc_auc_score(yte, prob_dino))
    acc_dino = float(accuracy_score(yte, (prob_dino >= 0.5).astype(int)))

    prob_rd = np.load(os.path.join(OUT, "raddino_test_prob.npy"))
    prob_r18 = np.load(os.path.join(OUT, "resnet18_test_prob.npy"))
    assert len(prob_rd) == len(prob_r18) == len(yte)
    auc_rd = float(roc_auc_score(yte, prob_rd))
    auc_r18 = float(roc_auc_score(yte, prob_r18))

    # 三个配对差
    d_domain = paired_bootstrap(yte, prob_rd, prob_dino)       # rad-dino − dinov2：纯预训练域
    d_arch = paired_bootstrap(yte, prob_dino, prob_r18)        # dinov2 − resnet18：架构+监督方式残余
    d_total = paired_bootstrap(yte, prob_rd, prob_r18)         # rad-dino − resnet18：总差

    # McNemar rad-dino vs dinov2（同阈值 0.5）
    c_rd = (prob_rd >= 0.5).astype(int) == yte
    c_dn = (prob_dino >= 0.5).astype(int) == yte
    b = int(np.sum(c_rd & ~c_dn)); c = int(np.sum(~c_rd & c_dn))
    mcnemar_p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0

    out = {
        "protocol": "frozen ViT-B/14 CLS + StandardScaler + LR linear probe (no finetune), same split, same processor(518px)",
        "n_test": int(len(yte)), "feature_dim": int(Xtr.shape[1]),
        "arms": {
            "rad_dino_chestxray": {"arch": "ViT-B/14", "pretrain": "chest X-ray DINOv2", "auc": auc_rd},
            "dinov2_base_natural": {"arch": "ViT-B/14", "pretrain": "natural images DINOv2", "auc": auc_dino, "acc": acc_dino},
            "resnet18_imagenet": {"arch": "ResNet18", "pretrain": "ImageNet supervised", "auc": auc_r18},
        },
        "decomposition": {
            "total_raddino_minus_resnet18": {"point": auc_rd - auc_r18, "ci95": d_total},
            "domain_raddino_minus_dinov2": {"point": auc_rd - auc_dino, "ci95": d_domain,
                                            "interpret": "纯预训练数据域（架构/预处理/探针全固定）"},
            "arch_residual_dinov2_minus_resnet18": {"point": auc_dino - auc_r18, "ci95": d_arch,
                                                    "interpret": "架构(ViT vs CNN)+监督方式(自监督 vs 监督)合并残余，非纯架构"},
            "additivity_check": float((auc_rd - auc_dino) + (auc_dino - auc_r18) - (auc_rd - auc_r18)),
        },
        "mcnemar_raddino_vs_dinov2": {"b_rd_only": b, "c_dino_only": c, "p": mcnemar_p},
        "verdict": {
            "domain_ci_excludes_0": bool(d_domain[0] > 0 or d_domain[1] < 0),
            "statement": ("预训练域有可测效应" if (d_domain[0] > 0 or d_domain[1] < 0)
                          else "同架构下预训练域差异不显著"),
        },
        "seed": SEED, "n_boot": N_BOOT,
    }
    p = os.path.join(OUT, "ablation_metrics.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    np.save(os.path.join(OUT, "dinov2_test_prob.npy"), prob_dino)
    print(json.dumps(out, indent=2))
    print(f"saved {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
