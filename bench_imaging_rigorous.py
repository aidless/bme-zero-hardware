"""Rigorous re-evaluation of the ResNet18 PneumoniaMNIST result: transfer-learning
protocol + 1000x bootstrap 95% CI on test AUC.

Protocol note (important for honest reporting):
- Our number uses ImageNet-pretrained ResNet18 with fine-tuning (head + layer3/4).
- Public sota2.com leaderboard ResNet-18 ERM AUC=0.9556 is TRAINED FROM SCRATCH.
These are different protocols; a transfer-learning test AUC should NOT be written
as "beats SOTA" without aligning the protocol.

Run: python3 bench_imaging_rigorous.py   (GPU or CPU)
"""
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Lambda, RandomAffine
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from medmnist import PneumoniaMNIST

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev)

tf = Compose([Resize(224), ToTensor(), Lambda(lambda x: x.repeat(3, 1, 1))])
tf_aug = Compose([Resize(224), RandomAffine(degrees=8, translate=(0.08, 0.08)), ToTensor(),
                  Lambda(lambda x: x.repeat(3, 1, 1))])
tr = PneumoniaMNIST(split="train", download=True, transform=tf_aug)
va = PneumoniaMNIST(split="val", download=True, transform=tf)
te = PneumoniaMNIST(split="test", download=True, transform=tf)
mk = lambda ds, shuf: DataLoader(ds, batch_size=64, shuffle=shuf, num_workers=0)
trL, vaL, teL = mk(tr, True), mk(va, False), mk(te, False)

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(dev)
opt = torch.optim.Adam([
    {"params": model.fc.parameters(), "lr": 1e-3},
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.layer3.parameters(), "lr": 1e-4},
], lr=1e-4)
crit = nn.BCEWithLogitsLoss()


@torch.no_grad()
def ev(loader):
    model.eval(); ys, ps = [], []
    for x, y in loader:
        p = torch.sigmoid(model(x.to(dev))).cpu().numpy().ravel()
        ps.append(p); ys.append(y.numpy().ravel())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return accuracy_score(y, p >= 0.5), roc_auc_score(y, p), f1_score(y, p >= 0.5), y, p


best = 0.0; state = None
for ep in range(1, 9):
    model.train()
    for x, y in trL:
        opt.zero_grad()
        loss = crit(model(x.to(dev)).ravel(), y.float().to(dev).ravel())
        loss.backward(); opt.step()
    a, u, f, _, _ = ev(vaL)
    flag = ""
    if u > best:
        best = u; state = {k: v.clone() for k, v in model.state_dict().items()}; flag = " *"
    print(f"ep{ep} val acc={a:.4f} auc={u:.4f} f1={f:.4f}{flag}")
model.load_state_dict(state)
a, u, f, y_te, p_te = ev(teL)
print(f"\n[test] acc={a:.4f} auc={u:.4f} f1={f:.4f}")

# 1000x bootstrap 95% CI on test AUC (stratified resample)
rng = np.random.default_rng(SEED)
n_pos = int(y_te.sum()); n_neg = len(y_te) - n_pos
pos_pool = np.where(y_te == 1)[0]; neg_pool = np.where(y_te == 0)[0]
aucs = []
for _ in range(1000):
    pos = pos_pool[rng.integers(0, n_pos, n_pos)]
    neg = neg_pool[rng.integers(0, n_neg, n_neg)]
    aucs.append((p_te[pos][:, None] > p_te[neg][None, :]).mean() +
                0.5 * (p_te[pos][:, None] == p_te[neg][None, :]).mean())
lo, hi = np.percentile(aucs, [2.5, 97.5])
print(f"\n[test AUC 95% CI] {u:.4f} [{lo:.4f}, {hi:.4f}]  (bootstrap n=1000)")
print(f"protocol: ImageNet-pretrained ResNet18, fine-tune head+layer3/4, "
      f"val-selected epoch (best val AUC={best:.4f})")
print("NOTE: sota2.com 0.9556 is ResNet-18 FROM-SCRATCH (different protocol; not a direct comparison)")