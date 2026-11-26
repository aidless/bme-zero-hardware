#!/usr/bin/env python3
"""第七轮 head-to-head v2：批量 GPU 提取 ResNet18 冻结特征（v1 逐张 CPU 太慢）。

同协议对比：冻结 backbone + StandardScaler + LR linear probe，不 finetune。
Rad-DINO(768d, 胸片预训练) test prob 读 raddino_test_prob.npy；
ResNet18(ImageNet, 512d) 这里批量提取。同一 PneumoniaMNIST test split，
配对 bootstrap + McNemar。回答：领域对口预训练值多少 AUC。
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

torch.set_num_threads(4)  # 防 CPU 线程争抢
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "frontier")
SEED = 42
N_BOOT = 2000
BATCH = 128


def extract(model, pre, ds, name):
    feats, labels = [], []
    from torch.utils.data import DataLoader
    def collate(items):
        imgs = []
        for img, lab in items:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            imgs.append(pre(img.convert("RGB")))
        return torch.stack(imgs), torch.tensor([int(np.asarray(l).ravel()[0]) for _, l in items])
    dl = DataLoader(ds, batch_size=BATCH, collate_fn=collate, num_workers=0)
    model.eval()
    with torch.no_grad():
        for i, (xb, yb) in enumerate(dl):
            f = model(xb.cuda()).cpu().numpy()
            feats.append(f)
            labels.append(yb.numpy())
            if (i + 1) % 10 == 0:
                print(f"  {name} {(i+1)*BATCH}/{len(ds)}", flush=True)
    return np.concatenate(feats), np.concatenate(labels)


def main():
    import torchvision.models as models
    from torchvision import transforms
    from medmnist import PneumoniaMNIST

    torch.manual_seed(SEED); np.random.seed(SEED)
    wpath = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    m = models.resnet18(weights=None)
    m.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
    m.fc = torch.nn.Identity()
    m = m.cuda().eval()
    pre = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds_tr = PneumoniaMNIST(split="train", download=True, transform=None)
    ds_te = PneumoniaMNIST(split="test", download=True, transform=None)
    print("extracting ResNet18 (GPU batch)...", flush=True)
    Xtr, ytr = extract(m, pre, ds_tr, "train")
    Xte, yte = extract(m, pre, ds_te, "test")
    print(f"ResNet18 feats {Xtr.shape} {Xte.shape}", flush=True)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED))
    clf.fit(Xtr, ytr)
    prob_r18 = clf.predict_proba(Xte)[:, 1]
    auc_r18 = float(roc_auc_score(yte, prob_r18))
    acc_r18 = float(accuracy_score(yte, (prob_r18 >= 0.5).astype(int)))

    prob_rd = np.load(os.path.join(OUT, "raddino_test_prob.npy"))
    assert len(prob_rd) == len(yte), "test prob 长度不一致"
    auc_rd = float(roc_auc_score(yte, prob_rd))
    acc_rd = float(accuracy_score(yte, (prob_rd >= 0.5).astype(int)))

    rng = np.random.default_rng(SEED)
    d_auc, d_acc = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(yte), len(yte))
        if len(np.unique(yte[idx])) < 2:
            continue
        d_auc.append(roc_auc_score(yte[idx], prob_rd[idx]) - roc_auc_score(yte[idx], prob_r18[idx]))
        d_acc.append(accuracy_score(yte[idx], (prob_rd[idx] >= 0.5).astype(int)) -
                     accuracy_score(yte[idx], (prob_r18[idx] >= 0.5).astype(int)))
    d_auc, d_acc = np.asarray(d_auc), np.asarray(d_acc)
    ci_auc = [float(np.quantile(d_auc, 0.025)), float(np.quantile(d_auc, 0.975))]
    ci_acc = [float(np.quantile(d_acc, 0.025)), float(np.quantile(d_acc, 0.975))]

    c_rd = (prob_rd >= 0.5).astype(int) == yte
    c_r18 = (prob_r18 >= 0.5).astype(int) == yte
    b = int(np.sum(c_rd & ~c_r18)); c = int(np.sum(~c_rd & c_r18))
    mcnemar_p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0

    out = {
        "protocol": "frozen backbone + StandardScaler + LogisticRegression linear probe (no finetune), same split",
        "n_test": int(len(yte)),
        "arms": {
            "rad_dino_cxr": {"auc": auc_rd, "acc": acc_rd, "pretrain": "chest X-ray DINOv2", "feature_dim": 768},
            "resnet18_imagenet": {"auc": auc_r18, "acc": acc_r18, "pretrain": "ImageNet supervised", "feature_dim": 512},
        },
        "paired_diff_rad_dino_minus_resnet18": {
            "auc_diff": auc_rd - auc_r18, "auc_diff_ci95": ci_auc,
            "acc_diff": acc_rd - acc_r18, "acc_diff_ci95": ci_acc,
            "mcnemar_b_rd_only": b, "mcnemar_c_r18_only": c, "mcnemar_p": mcnemar_p,
        },
        "verdict": {
            "auc_ci_excludes_0": bool(ci_auc[0] > 0 or ci_auc[1] < 0),
            "statement": ("领域预训练有可测优势" if (ci_auc[0] > 0 or ci_auc[1] < 0)
                          else "同协议下两 backbone 无显著差异"),
        },
        "seed": SEED, "n_boot": N_BOOT,
        "note": "v2: GPU batch 提取 ResNet18；v1 逐张 CPU 太慢已弃",
    }
    with open(os.path.join(OUT, "headtohead_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.save(os.path.join(OUT, "resnet18_test_prob.npy"), prob_r18)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
