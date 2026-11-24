"""Advanced v1: ResNet18 transfer learning on PneumoniaMNIST.
Goal: beat the 2-layer CNN baseline test AUC = 0.9364.
GPU (RTX 3060 6GB) or CPU. Pretrained ImageNet weights (local /tmp).
Run: python3 adv_resnet.py
"""
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Lambda, RandomAffine
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from medmnist import PneumoniaMNIST
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev)

tf = Compose([Resize(224), ToTensor(), Lambda(lambda x: x.repeat(3,1,1))])
tf_aug = Compose([Resize(224), RandomAffine(degrees=8, translate=(0.08,0.08)), ToTensor(), Lambda(lambda x: x.repeat(3,1,1))])
tr = PneumoniaMNIST(split="train", download=True, transform=tf_aug)
va = PneumoniaMNIST(split="val", download=True, transform=tf)
te = PneumoniaMNIST(split="test", download=True, transform=tf)
# standard loader collates PIL fine after ToTensor
mk = lambda ds,shuf: DataLoader(ds, batch_size=64, shuffle=shuf, num_workers=0)
trL, vaL, teL = mk(tr,True), mk(va,False), mk(te,False)

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(dev)
opt = torch.optim.Adam([
    {"params": model.fc.parameters(), "lr": 1e-3},
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.layer3.parameters(), "lr": 1e-4},
], lr=1e-4)  # full-ish finetune: head + last 2 blocks
crit = nn.BCEWithLogitsLoss()

@torch.no_grad()
def ev(loader):
    model.eval(); ys,ps=[],[]
    for x,y in loader:
        p=torch.sigmoid(model(x.to(dev))).cpu().numpy().ravel()
        ps.append(p); ys.append(y.numpy().ravel())
    y,p=np.concatenate(ys),np.concatenate(ps)
    return accuracy_score(y,p>=0.5), roc_auc_score(y,p), f1_score(y,p>=0.5)

best=0.0; state=None
for ep in range(1,9):
    model.train()
    for x,y in trL:
        opt.zero_grad()
        loss=crit(model(x.to(dev)).ravel(), y.float().to(dev).ravel())
        loss.backward(); opt.step()
    a,u,f=ev(vaL); flag=""
    if u>best: best=u; state={k:v.clone() for k,v in model.state_dict().items()}; flag=" *"
    print(f"ep{ep} val acc={a:.4f} auc={u:.4f} f1={f:.4f}{flag}")
model.load_state_dict(state)
a,u,f=ev(teL)
print(f"\n[test] acc={a:.4f} auc={u:.4f} f1={f:.4f}")
print(f"CNN baseline auc=0.9364 -> ResNet18 {'BEAT' if u>0.9364 else 'NOT BEAT'} (delta={u-0.9364:+.4f})")
