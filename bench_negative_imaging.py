#!/usr/bin/env python3
"""第九轮阴性对照（影像）：冻结 ResNet18 特征 + linear probe 标签置换检验。

前八轮影像阳性数字（PneumoniaMNIST AUC 0.97、DermaMNIST macro-AUC 0.93）
都在真实标签上得到。本脚本做阴性对照：特征 X 与 official split 完全不变，
只把标签 y 随机置换，重跑 StandardScaler + LR linear probe，看 test AUC
是否塌回 0.5。若置换后仍高，说明流水线有泄漏。

backbone 用 ResNet18(ImageNet) 而非 Rad-DINO：阴性对照检验的是
"冻结特征 + linear probe + official split + bootstrap" 这段流水线，
与 bench_frontier_headtohead_v2.py 的 R18 臂同构；backbone 选择不影响结论。
DermaMNIST 7 类用 macro one-vs-rest AUC。

Run: python3 bench_negative_imaging.py
     python3 bench_negative_imaging.py --selftest
"""
import json
import os
import sys
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "negative")
os.makedirs(OUT, exist_ok=True)
SEED = 42
N_PERM = 100
BATCH = 128


def extract(model, pre, ds, name):
    from torch.utils.data import DataLoader
    feats, labels = [], []
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
        for xb, yb in dl:
            feats.append(model(xb.cuda()).cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def probe_auc(Xtr, ytr, Xte, yte, n_classes):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)
    if n_classes == 2:
        return float(roc_auc_score(yte, prob[:, 1]))
    yhot = np.eye(n_classes)[yte]
    return float(roc_auc_score(yhot, prob, average="macro", multi_class="ovr"))


def perm_test(Xtr, ytr, Xte, yte, n_classes, real_auc, n=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    y_all = np.concatenate([ytr, yte])
    n_tr = len(ytr)
    null = []
    for i in range(n):
        yp = rng.permutation(y_all)
        a = probe_auc(Xtr, yp[:n_tr], Xte, yp[n_tr:], n_classes)
        null.append(a)
        if (i + 1) % 25 == 0:
            print(f"    perm {i+1}/{n}  auc={a:.4f}", flush=True)
    null = np.asarray(null)
    p_emp = float((np.sum(null >= real_auc) + 1) / (n + 1))
    return {
        "n_perm": n,
        "null_mean": float(null.mean()), "null_std": float(null.std(ddof=1)),
        "null_median": float(np.median(null)),
        "null_p95": float(np.quantile(null, 0.95)),
        "null_max": float(null.max()), "null_min": float(null.min()),
        "real_auc": real_auc, "empirical_p": p_emp,
        "null_mean_deviation_from_half": float(null.mean() - 0.5),
    }


def selftest():
    # 置换不改变标签多重集
    rng = np.random.default_rng(SEED)
    y = np.array([0]*7 + [1]*3)
    yp = rng.permutation(y)
    assert sorted(yp.tolist()) == sorted(y.tolist()), "置换改变了标签多重集"
    # 同 seed 可复现
    a = np.random.default_rng(SEED).permutation(y)
    b = np.random.default_rng(SEED).permutation(y)
    assert np.array_equal(a, b), "同 seed 置换不可复现"
    # probe 在随机标签上应接近 0.5（粗断言，容差放宽）
    Xtr = rng.normal(size=(200, 10)); Xte = rng.normal(size=(80, 10))
    ytr = rng.integers(0, 2, 200); yte = rng.integers(0, 2, 80)
    a = probe_auc(Xtr, ytr, Xte, yte, 2)
    assert 0.3 < a < 0.7, f"随机标签 probe AUC 异常 {a}"
    print("SELFTEST PASS: permutation multiset / reproducibility / null probe near 0.5", flush=True)


def main():
    import torchvision.models as models
    from torchvision import transforms
    from medmnist import PneumoniaMNIST, DermaMNIST

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

    result = {
        "experiment": "negative-control-imaging",
        "created_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "protocol": "frozen ResNet18(ImageNet) + StandardScaler + LR linear probe; official split fixed; label vector shuffled across train+test per permutation; N=100",
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "scikit_learn": __import__("sklearn").__version__},
        "datasets": {},
    }

    for name, Cls, nc in [("pneumoniamnist", PneumoniaMNIST, 2), ("dermamnist", DermaMNIST, 7)]:
        print(f"=== {name} ===", flush=True)
        ds_tr = Cls(split="train", download=True, transform=None)
        ds_te = Cls(split="test", download=True, transform=None)
        Xtr, ytr = extract(m, pre, ds_tr, "train")
        Xte, yte = extract(m, pre, ds_te, "test")
        print(f"  feats {Xtr.shape} {Xte.shape} classes={nc}", flush=True)
        real = probe_auc(Xtr, ytr, Xte, yte, nc)
        print(f"  real AUC={real:.4f}", flush=True)
        st = perm_test(Xtr, ytr, Xte, yte, nc, real)
        st["n_train"] = len(ytr); st["n_test"] = len(yte); st["n_classes"] = nc
        st["verdict_clean"] = (abs(st["null_mean_deviation_from_half"]) <= 0.05 and st["empirical_p"] <= 0.01)
        result["datasets"][name] = st
        print(f"  null mean={st['null_mean']:.4f}±{st['null_std']:.4f} p95={st['null_p95']:.4f} "
              f"p={st['empirical_p']:.4f} clean={st['verdict_clean']}", flush=True)

    with open(os.path.join(OUT, "negative_imaging_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("SAVED data/negative/negative_imaging_metrics.json", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
