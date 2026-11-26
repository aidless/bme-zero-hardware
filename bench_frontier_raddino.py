#!/usr/bin/env python3
"""第七轮 · 前沿基础模型实测：Rad-DINO 冻结 CLS 特征 vs 我们的经典模型。

问题：前六轮我们一直说"口径不同不能写 SOTA"，但从未真正把前沿基础模型
拉到我们的数据上做同口径 head-to-head。这一轮补上：
  - Rad-DINO（Nature Mach Intell 2025, arXiv:2401.10815）冻结 ViT-B/14 CLS 特征
  - 同一 PneumoniaMNIST 官方 test split（624 张）
  - linear probe（StandardScaler + LogisticRegression）在 train 上拟合
  - bootstrap CI（2000×分层）+ ECE，与历史数字同表对比：
      LR(像素) 0.9279 / CNN 0.9364 / ResNet18 ImageNet 迁移 0.9741

诚实边界：
  - Rad-DINO 预训练于胸片（CXR），PneumoniaMNIST 也是胸片，领域对口；
    但我们的 ResNet18 用 ImageNet 权重（自然图像），二者预训练数据不同，
    仍是"不同预训练口径"，不是同一 backbone 的公平对比——只做描述性 head-to-head。
  - 冻结特征 + linear probe 是微调-free 协议，不与端到端 finetune 混谈。
"""
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "frontier")
os.makedirs(OUT, exist_ok=True)
SEED = 42
N_BOOT = 2000


def extract(dataset, enc, split_name):
    from PIL import Image
    feats, labels = [], []
    n = len(dataset)
    for i in range(n):
        img, lab = dataset[i]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        import torch
        with torch.no_grad():
            cls, _ = enc.extract_features([img])
        feats.append(cls[0].cpu().numpy())
        labels.append(int(np.asarray(lab).ravel()[0]))
        if (i + 1) % 200 == 0:
            print(f"  {split_name} {i+1}/{n}", flush=True)
    return np.asarray(feats), np.asarray(labels)


def bootstrap_ci(y, prob, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs, accs = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], prob[idx]))
        accs.append(accuracy_score(y[idx], (prob[idx] >= 0.5).astype(int)))
    return {
        "auc_ci": [float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))],
        "acc_ci": [float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975))],
    }


def main():
    import torch
    from medmnist import PneumoniaMNIST
    from rad_dino import RadDino

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("loading Rad-DINO...", flush=True)
    enc = RadDino().cuda().eval()
    print("extracting features...", flush=True)
    ds_tr = PneumoniaMNIST(split="train", download=True, transform=None)
    ds_va = PneumoniaMNIST(split="val", download=True, transform=None)
    ds_te = PneumoniaMNIST(split="test", download=True, transform=None)
    Xtr, ytr = extract(ds_tr, enc, "train")
    Xva, yva = extract(ds_va, enc, "val")
    Xte, yte = extract(ds_te, enc, "test")
    print(f"feature shapes: {Xtr.shape} {Xva.shape} {Xte.shape}", flush=True)

    # linear probe：train 拟合（不用 val 调参，linear probe 无超参；val 仅报告）
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, random_state=SEED))
    clf.fit(Xtr, ytr)

    def evaluate(X, y):
        prob = clf.predict_proba(X)[:, 1]
        return {
            "auc": float(roc_auc_score(y, prob)),
            "acc": float(accuracy_score(y, (prob >= 0.5).astype(int))),
            "prob": prob,
        }

    tr = evaluate(Xtr, ytr)
    va = evaluate(Xva, yva)
    te = evaluate(Xte, yte)
    ci = bootstrap_ci(yte, te["prob"])
    # ECE(15)
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(yte, te["prob"], n_bins=15, strategy="uniform")
    bins = np.linspace(0, 1, 16)
    idx = np.clip(np.digitize(te["prob"], bins) - 1, 0, 14)
    ece = 0.0
    for b in range(15):
        m = idx == b
        if m.sum():
            ece += m.mean() * abs(yte[m].mean() - te["prob"][m].mean())

    out = {
        "model": "microsoft/rad-dino (ViT-B/14, CLS 768d, frozen)",
        "pretrain": "chest X-ray (DINOv2 self-supervised), Nature Mach Intell 2025 / arXiv:2401.10815",
        "protocol": "frozen feature extraction + StandardScaler + LogisticRegression linear probe; no finetune",
        "dataset": "PneumoniaMNIST official split",
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "feature_dim": int(Xtr.shape[1]),
        "seed": SEED, "n_boot": N_BOOT,
        "train": {"auc": tr["auc"], "acc": tr["acc"]},
        "val": {"auc": va["auc"], "acc": va["acc"]},
        "test": {"auc": te["auc"], "acc": te["acc"],
                 "auc_ci95": ci["auc_ci"], "acc_ci95": ci["acc_ci"],
                 "ece15": float(ece)},
        "historical_comparison": {
            "pixel_logreg_auc": 0.9279,
            "cnn_2layer_auc": 0.9364,
            "resnet18_imagenet_transfer_auc": 0.9741,
            "note": "不同预训练口径（CXR vs ImageNet）与不同协议（冻结linear probe vs 端到端finetune），只做描述性对比，不宣称 SOTA 超越",
        },
    }
    p = os.path.join(OUT, "raddino_metrics.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    np.save(os.path.join(OUT, "raddino_test_prob.npy"), te["prob"])
    print(json.dumps({k: out[k] for k in ("test", "historical_comparison")}, indent=2))
    print(f"saved {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
