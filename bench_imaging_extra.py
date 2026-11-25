#!/usr/bin/env python3
"""P1-8: PathMNIST / RetinaMNIST 跨数据集外部验证（沿用 derma 的 ResNet18 迁移学习协议）。

对两个额外 medmnist 数据集各跑一遍，验证"ImageNet 预训练 ResNet18 + 分层解冻"
协议能否迁移到 ①影像 之外的更多子任务。口径：
  - 官方划分 train/val/test，val macro-AUC 选模，test 只评一次
  - bootstrap 95% CI（分层，1000 次）
  - ECE(15 bins) + Brier
  - 与 medmnist.com 官方 leaderboard 数字对比（预训练/增强不同，仅定性）
输出： data/imaging_extra_metrics.json
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from medmnist import INFO, PathMNIST, RetinaMNIST

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BINS = 15


def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def make_loaders(cls, batch):
    tf_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.RandomAffine(8, translate=(0.08, 0.08)),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    ])
    tr = cls(split="train", transform=tf_train, download=True)
    va = cls(split="val", transform=tf_eval, download=True)
    te = cls(split="test", transform=tf_eval, download=True)
    return (DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0),
            DataLoader(va, batch_size=batch, shuffle=False, num_workers=0),
            DataLoader(te, batch_size=batch, shuffle=False, num_workers=0),
            tr, va, te)


def build_model(n_classes):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m.to(DEVICE)


def predict(model, loader, n_classes):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            with torch.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                out = model(x)
            p = torch.softmax(out.float(), dim=1).cpu().numpy()
            probs.append(p)
            y = y.numpy()
            ys.append(y.ravel() if y.ndim > 1 else y)
    return np.concatenate(probs), np.concatenate(ys)


def macro_auc(y, p):
    n = p.shape[1]
    y_bin = np.eye(n)[y]
    return roc_auc_score(y_bin, p, average="macro", multi_class="ovr")


def ece(y, p, bins=BINS):
    conf = p.max(1)
    pred = p.argmax(1)
    correct = (pred == y).astype(float)
    e = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            e += m.sum() / len(y) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def bootstrap_ci(y, p, fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    classes = np.unique(y)
    for _ in range(n):
        idx = np.concatenate([rng.choice(np.where(y == c)[0], np.sum(y == c), replace=True) for c in classes])
        try:
            vals.append(fn(y[idx], p[idx]))
        except Exception:
            pass
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run(name, cls, epochs_head=4, epochs_ft=3):
    set_seed()
    train, val, test, tr, va, te = make_loaders(cls, 128)
    n_classes = len(INFO[name]["label"])
    model = build_model(n_classes)
    opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs_head):
        model.train()
        for x, y in train:
            x, y = x.to(DEVICE), y.to(DEVICE).ravel().long()
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                out = model(x)
            loss = crit(out.float(), y)
            loss.backward()
            opt.step()
    # 解冻 layer3/4
    for n, p in model.named_parameters():
        p.requires_grad = ("layer3" in n or "layer4" in n or "fc" in n)
    opt = torch.optim.AdamW([
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": [p for n, p in model.named_parameters() if ("layer3" in n or "layer4" in n) and p.requires_grad], "lr": 1e-4},
    ], weight_decay=1e-4)
    best_auc, best_state, best_ep = -1, None, -1
    for ep in range(epochs_ft):
        model.train()
        for x, y in train:
            x, y = x.to(DEVICE), y.to(DEVICE).ravel().long()
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                out = model(x)
            loss = crit(out.float(), y)
            loss.backward()
            opt.step()
        pv, yv = predict(model, val, n_classes)
        auc = macro_auc(yv, pv)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
    model.load_state_dict(best_state)
    pt, yt = predict(model, test, n_classes)
    test_auc = macro_auc(yt, pt)
    ci = bootstrap_ci(yt, pt, macro_auc)
    acc = accuracy_score(yt, pt.argmax(1))
    return {
        "dataset": name,
        "n_classes": n_classes,
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "best_val_macro_auc": float(best_auc), "best_finetune_epoch": best_ep,
        "test_macro_auc": float(test_auc), "test_macro_auc_ci95": list(ci),
        "test_acc": float(acc), "test_ece15": ece(yt, pt),
        "test_brier": float(np.mean(np.sum((pt - np.eye(n_classes)[yt]) ** 2, axis=1))),
        "confusion_matrix": confusion_matrix(yt, pt.argmax(1)).tolist(),
    }


if __name__ == "__main__":
    print(f"device={DEVICE}", flush=True)
    out = {"task": "P1-8 imaging cross-dataset: PathMNIST + RetinaMNIST",
           "protocol": "ResNet18 ImageNet, head 4ep -> unfreeze layer3/4 3ep, val macro-AUC model select, 1000x stratified bootstrap",
           "device": str(DEVICE), "seed": SEED, "results": {}}
    for name, cls, eh, ef in [("pathmnist", PathMNIST, 4, 3), ("retinamnist", RetinaMNIST, 5, 4)]:
        print(f"--- {name} ---", flush=True)
        r = run(name, cls, eh, ef)
        out["results"][name] = r
        print(json.dumps(r, ensure_ascii=False)[:500], flush=True)
    with open("data/imaging_extra_metrics.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("SAVED data/imaging_extra_metrics.json", flush=True)
