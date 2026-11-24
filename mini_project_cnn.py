"""Mini project v2: PyTorch CNN on PneumoniaMNIST.
Goal (verifier): beat the logistic-regression baseline test AUC = 0.9279.
Pure CPU. Install: pip install medmnist torch torchvision
Run:     python3 mini_project_cnn.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomAffine
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import medmnist
from medmnist import PneumoniaMNIST

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)   # 28x28 -> 28x28
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.do = nn.Dropout(0.3)
        self.fc = nn.Linear(32 * 7 * 7, 1)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.max_pool2d(x, 2)                      # 14x14
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)                      # 7x7
        x = x.view(x.size(0), -1)
        x = self.do(x)
        return self.fc(x).squeeze(1)                # logits

def loaders():
    va = PneumoniaMNIST(split="val", download=True, transform=Compose([ToTensor()]))
    te = PneumoniaMNIST(split="test", download=True, transform=Compose([ToTensor()]))
    tr = PneumoniaMNIST(split="train", download=True,
                        transform=Compose([RandomAffine(degrees=8, translate=(0.08, 0.08)), ToTensor()]))
    mk = lambda ds, shuf: DataLoader(ds, batch_size=64, shuffle=shuf)
    return mk(tr, True), mk(va, False), mk(te, False)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.float()
        p = torch.sigmoid(model(x)).numpy()
        ys.append(y.numpy().ravel()); ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    pred = (p >= 0.5).astype(int)
    return accuracy_score(y, pred), roc_auc_score(y, p), f1_score(y, pred)

def main():
    tr, va, te = loaders()
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = PneumoniaMNIST(split="train").labels.ravel().astype(int)
    n_pos, n_neg = pos.sum(), len(pos) - pos.sum()
    wpos = torch.tensor(n_neg / n_pos, dtype=torch.float32).sqrt()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=wpos)

    best_auc, best_state = 0.0, None
    for ep in range(1, 11):
        model.train()
        for x, y in tr:
            x = x.float()
            y = y.float().ravel()
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
        va_acc, va_auc, va_f1 = evaluate(model, va)
        flag = ""
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            flag = " *"
        print(f"ep{ep:02d} val acc={va_acc:.4f} auc={va_auc:.4f} f1={va_f1:.4f}{flag}")

    model.load_state_dict(best_state)
    te_acc, te_auc, te_f1 = evaluate(model, te)
    print(f"\n[test] acc={te_acc:.4f} auc={te_auc:.4f} f1={te_f1:.4f}")
    print(f"baseline(logreg) test AUC=0.9279 -> CNN "
          f"{'BEAT' if te_auc > 0.9279 else 'NOT BEAT'} "
          f"(delta={te_auc - 0.9279:+.4f})")

if __name__ == "__main__":
    main()
