"""DermaMNIST external validation of the PneumoniaMNIST ResNet18 transfer protocol.

Run from the workspace root:
    python3 -u bench_imaging_derma.py 2>&1 | tee data/derma_rigorous_run.log

Outputs:
    data/derma_rigorous_metrics.json
    figures/derma_reliability.png
    figures/derma_confusion.png

Protocol: official MedMNIST splits; ImageNet ResNet18; freeze backbone for head-only
training, then unfreeze layer3/layer4 with stratified learning rates and affine
augmentation; select the best epoch by validation macro-AUC; report test accuracy,
macro/per-class AUC, ECE(15 bins), multiclass Brier, confusion matrix, and 1000x
stratified-bootstrap 95% CIs for test accuracy and macro-AUC.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import Compose, Lambda, RandomAffine, Resize, ToTensor
from medmnist import DermaMNIST

SEED = 42
HEAD_EPOCHS = 4
FINETUNE_MAX_EPOCHS = 10
PATIENCE = 4
BATCH_SIZE = 64
N_BOOT = 1000
N_BINS = 15
NUM_CLASSES = 7
WEIGHTS_PATH = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_weights() -> None:
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        subprocess.check_call(["curl", "-L", "-f", "-o", WEIGHTS_PATH, WEIGHTS_URL])
    if os.path.getsize(WEIGHTS_PATH) < 1_000_000:
        raise RuntimeError(f"ResNet18 weights look truncated: {WEIGHTS_PATH}")


def split_stats(dataset) -> dict:
    y = np.asarray(dataset.labels).reshape(-1).astype(int)
    counts = np.bincount(y, minlength=NUM_CLASSES)
    return {
        "n": int(len(dataset)),
        "num_classes": int(len(counts)),
        "class_counts": {str(i): int(counts[i]) for i in range(len(counts))},
        "class_fraction": {
            str(i): (round(float(counts[i] / len(y)), 6) if len(y) else 0.0)
            for i in range(len(counts))
        },
    }


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    yb = label_binarize(y, classes=list(range(NUM_CLASSES)))
    return float(np.mean([roc_auc_score(yb[:, i], p[:, i]) for i in range(NUM_CLASSES)]))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> dict:
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_id == b
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin": b,
                    "lower": float(edges[b]),
                    "upper": float(edges[b + 1]),
                    "n": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                }
            )
            continue
        c = float(conf[mask].mean())
        a = float(correct[mask].mean())
        ece += (n / len(y)) * abs(a - c)
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "n": n,
                "mean_confidence": c,
                "accuracy": a,
            }
        )
    yb = label_binarize(y, classes=list(range(NUM_CLASSES)))
    brier = float(np.mean(np.sum((p - yb) ** 2, axis=1)))
    return {"ece": float(ece), "brier": brier, "bins": rows}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot: int = N_BOOT) -> dict:
    rng = np.random.default_rng(SEED)
    groups = [np.where(y == i)[0] for i in range(NUM_CLASSES)]
    if any(len(g) == 0 for g in groups):
        raise RuntimeError("A class has no test samples; stratified bootstrap is invalid")
    accs, aucs = [], []
    for _ in range(n_boot):
        idx = np.concatenate(
            [g[rng.integers(0, len(g), len(g))] for g in groups]
        )
        rng.shuffle(idx)
        yb = label_binarize(y[idx], classes=list(range(NUM_CLASSES)))
        accs.append(float(accuracy_score(y[idx], p[idx].argmax(axis=1))))
        aucs.append(
            float(
                np.mean(
                    [roc_auc_score(yb[:, i], p[idx, i]) for i in range(NUM_CLASSES)]
                )
            )
        )
    acc_lo, acc_hi = np.percentile(accs, [2.5, 97.5])
    auc_lo, auc_hi = np.percentile(aucs, [2.5, 97.5])
    return {
        "n_boot": n_boot,
        "method": "stratified bootstrap with replacement within each class, seed=42",
        "accuracy_ci": [float(acc_lo), float(acc_hi)],
        "macro_auc_ci": [float(auc_lo), float(auc_hi)],
        "accuracy_bootstrap_mean": float(np.mean(accs)),
        "macro_auc_bootstrap_mean": float(np.mean(aucs)),
    }


def save_figures(y: np.ndarray, p: np.ndarray, yhat: np.ndarray, calib: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="perfect calibration")
    nonempty = [b for b in calib["bins"] if b["n"] > 0]
    centers = [0.5 * (b["lower"] + b["upper"]) for b in nonempty]
    accs = [b["accuracy"] for b in nonempty]
    confs = [b["mean_confidence"] for b in nonempty]
    ns = np.array([b["n"] for b in nonempty])
    ax.bar(centers, accs, width=0.06, color="#3b6fb6", alpha=0.75, label="observed accuracy")
    ax.plot(confs, accs, "o", color="#b22222", ms=4.5, label="bin centers")
    for c, a, n in zip(confs, accs, ns):
        ax.plot([c, c], [a, c], color="gray", lw=0.6, alpha=0.7)
        ax.text(c, c + 0.015, str(n), ha="center", va="bottom", fontsize=6, color="gray")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Mean predicted confidence",
           ylabel="Empirical accuracy",
           title=f"DermaMNIST reliability (15 bins, N={len(y)})\n"
                 f"ECE={calib['ece']:.4f}, Brier={calib['brier']:.4f}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/derma_reliability.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y, yhat, labels=list(range(NUM_CLASSES))),
        display_labels=[str(i) for i in range(NUM_CLASSES)],
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d", xticks_rotation=0)
    ax.set(xlabel="Predicted class", ylabel="True class",
           title="DermaMNIST test confusion matrix (best val epoch)")
    fig.tight_layout()
    fig.savefig("figures/derma_confusion.png", dpi=200)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    set_seed()
    ensure_weights()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; torch={torch.__version__}; seed={SEED}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    tf = Compose([Resize(224), ToTensor(),
                  Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)])
    tf_aug = Compose([
        Resize(224),
        RandomAffine(degrees=8, translate=(0.08, 0.08)),
        ToTensor(),
        Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    ])
    print("downloading/loading official DermaMNIST splits into ./data", flush=True)
    train_ds = DermaMNIST(split="train", download=True, transform=tf_aug, root="./data")
    val_ds = DermaMNIST(split="val", download=True, transform=tf, root="./data")
    test_ds = DermaMNIST(split="test", download=True, transform=tf, root="./data")
    stats = {
        "train": split_stats(train_ds),
        "val": split_stats(val_ds),
        "test": split_stats(test_ds),
    }
    label_map = {
        str(k): str(v) for k, v in (train_ds.info.get("label") or {}).items()
    }
    print("split_stats=" + json.dumps(stats, ensure_ascii=False), flush=True)
    print("label_map=" + json.dumps(label_map, ensure_ascii=False), flush=True)

    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              generator=gen, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = resnet18(weights=None)
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    @torch.no_grad()
    def predict(loader):
        model.eval()
        ys, ps = [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            ps.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            ys.append(np.asarray(y).reshape(-1).astype(int))
        return np.concatenate(ys), np.concatenate(ps)

    @torch.no_grad()
    def evaluate(loader):
        y, p = predict(loader)
        yhat = p.argmax(axis=1)
        return {
            "accuracy": float(accuracy_score(y, yhat)),
            "macro_auc": macro_auc(y, p),
        }

    def train_epoch(loader, optimizer, tag):
        model.train()
        total_loss, seen = 0.0, 0
        for batch, (x, y) in enumerate(loader, start=1):
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y).reshape(-1).long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(y)
            seen += len(y)
            if batch % 100 == 0:
                print(f"  {tag} batch={batch} loss={total_loss / seen:.4f}", flush=True)
        return total_loss / seen

    history = []
    best = {"macro_auc": -1.0, "accuracy": -1.0, "stage": None, "epoch": None,
            "state": None}

    def checkpoint(stage, epoch, metrics):
        if (metrics["macro_auc"] > best["macro_auc"] or
                (metrics["macro_auc"] == best["macro_auc"] and
                 metrics["accuracy"] > best["accuracy"])):
            best.update(stage=stage, epoch=epoch, macro_auc=metrics["macro_auc"],
                        accuracy=metrics["accuracy"],
                        state={k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()})
            return True
        return False

    print(f"[stage1] frozen backbone, head-only training for {HEAD_EPOCHS} epochs",
          flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.fc.requires_grad_(True)
    opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    for ep in range(1, HEAD_EPOCHS + 1):
        loss = train_epoch(train_loader, opt, f"head ep{ep}")
        m = evaluate(val_loader)
        flag = " *best" if checkpoint("head", ep, m) else ""
        history.append({"stage": "head", "epoch": ep, "train_loss": loss, **m})
        print(f"[head ep{ep}] train_loss={loss:.4f} val_acc={m['accuracy']:.4f} "
              f"val_macro_auc={m['macro_auc']:.4f}{flag}", flush=True)

    model.load_state_dict(best["state"])
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("fc") or name.startswith("layer3") or
                         name.startswith("layer4"))
    opt = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.layer3.parameters(), "lr": 1e-4},
    ])
    print(f"[stage2] unfreeze layer3/layer4, affine augmentation, max "
          f"{FINETUNE_MAX_EPOCHS} epochs, patience={PATIENCE}", flush=True)
    last_improve = 0
    for ep in range(1, FINETUNE_MAX_EPOCHS + 1):
        loss = train_epoch(train_loader, opt, f"ft ep{ep}")
        m = evaluate(val_loader)
        improved = checkpoint("finetune", ep, m)
        if improved:
            last_improve = ep
        history.append({"stage": "finetune", "epoch": ep, "train_loss": loss, **m})
        print(f"[ft ep{ep}] train_loss={loss:.4f} val_acc={m['accuracy']:.4f} "
              f"val_macro_auc={m['macro_auc']:.4f}"
              f"{' *best' if improved else ''}", flush=True)
        if ep - last_improve >= PATIENCE:
            print(f"[stage2] early stop at epoch {ep}", flush=True)
            break

    model.load_state_dict(best["state"])
    print(f"[selected] stage={best['stage']} epoch={best['epoch']} "
          f"val_acc={best['accuracy']:.4f} val_macro_auc={best['macro_auc']:.4f}",
          flush=True)

    y_test, p_test = predict(test_loader)
    yhat = p_test.argmax(axis=1)
    yb = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
    per_class_auc = {
        label_map.get(str(i), str(i)): float(roc_auc_score(yb[:, i], p_test[:, i]))
        for i in range(NUM_CLASSES)
    }
    per_class_auc_by_id = {
        str(i): float(roc_auc_score(yb[:, i], p_test[:, i]))
        for i in range(NUM_CLASSES)
    }
    test_acc = float(accuracy_score(y_test, yhat))
    test_auc = float(np.mean(list(per_class_auc_by_id.values())))
    calib = reliability(y_test, p_test)
    boot = bootstrap_ci(y_test, p_test)
    print(f"[test] acc={test_acc:.4f} macro_auc={test_auc:.4f}", flush=True)
    print("[test] per_class_auc=" + json.dumps(per_class_auc_by_id), flush=True)
    print(f"[test] ECE15={calib['ece']:.4f} Brier={calib['brier']:.4f}", flush=True)
    print("[test] confusion_rows_true_columns_pred=" +
          json.dumps(confusion_matrix(y_test, yhat, labels=list(range(NUM_CLASSES))).tolist()),
          flush=True)
    print("[test] bootstrap=" + json.dumps(boot), flush=True)

    save_figures(y_test, p_test, yhat, calib)

    with open(WEIGHTS_PATH, "rb") as f:
        weights_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    result = {
        "experiment": "DermaMNIST external validation of ResNet18 transfer protocol",
        "run_timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "environment": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "medmnist": __import__("medmnist").__version__,
            "sklearn": __import__("sklearn").__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "weights_path": WEIGHTS_PATH,
            "weights_sha256": weights_sha256,
        },
        "dataset": {
            "name": "DermaMNIST",
            "splits": stats,
            "label_map": label_map,
            "root": "./data",
            "official_split": True,
        },
        "protocol": {
            "architecture": "torchvision ResNet18, ImageNet1K_V1 state_dict",
            "input": "Resize(224), ToTensor, repeat grayscale to 3 channels (native RGB retained)",
            "augmentation": "RandomAffine(degrees=8, translate=(0.08,0.08)) in stage2",
            "loss": "CrossEntropyLoss, 7 classes",
            "head_epochs": HEAD_EPOCHS,
            "head_lr": 1e-3,
            "finetune_layers": ["layer3", "layer4", "fc"],
            "finetune_lr": {"fc": 1e-3, "layer3": 1e-4, "layer4": 1e-4},
            "finetune_max_epochs": FINETUNE_MAX_EPOCHS,
            "early_stop_patience": PATIENCE,
            "selection_metric": "validation macro-AUC (one-vs-rest)",
            "batch_size": BATCH_SIZE,
            "amp": use_amp,
            "cudnn_deterministic": True,
            "selected": {"stage": best["stage"], "epoch": best["epoch"],
                         "val_accuracy": best["accuracy"],
                         "val_macro_auc": best["macro_auc"]},
            "history": history,
        },
        "test": {
            "n": int(len(y_test)),
            "accuracy": test_acc,
            "macro_auc": test_auc,
            "per_class_auc_by_id": per_class_auc_by_id,
            "per_class_auc_by_name": per_class_auc,
            "weakest_class_by_id": min(per_class_auc_by_id, key=per_class_auc_by_id.get),
            "weakest_class_auc": min(per_class_auc_by_id.values()),
            "confusion_matrix": confusion_matrix(
                y_test, yhat, labels=list(range(NUM_CLASSES))).tolist(),
            "calibration": calib,
            "bootstrap": boot,
        },
        "comparability_note": (
            "PneumoniaMNIST is binary (test AUC 0.9721 [0.9569,0.9833]); "
            "DermaMNIST is 7-class. Macro-AUC and accuracy are not the same "
            "metric/caliber; this is a transferability check, not a SOTA claim."
        ),
        "artifacts": {
            "metrics": "data/derma_rigorous_metrics.json",
            "stdout_log": "data/derma_rigorous_run.log",
            "reliability": "figures/derma_reliability.png",
            "confusion": "figures/derma_confusion.png",
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    os.makedirs("data", exist_ok=True)
    with open("data/derma_rigorous_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[done] artifacts written; runtime={result['runtime_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
