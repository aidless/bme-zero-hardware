"""First reproducible mini-project: PneumoniaMNIST binary classification.
Direction (13) computational medicine / medical AI. Pure CPU, no hardware.
Install: pip install medmnist scikit-learn
Run:     python3 mini_project.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import medmnist
from medmnist import PneumoniaMNIST

np.random.seed(42)

def to_numpy(ds):
    x = ds.imgs.astype(np.float32) / 255.0
    x = x.reshape(len(ds), -1)              # 28x28 -> 784, no CNN needed
    y = ds.labels.ravel().astype(int)
    return x, y

train = PneumoniaMNIST(split="train", download=True)
val   = PneumoniaMNIST(split="val",   download=True)
test  = PneumoniaMNIST(split="test",  download=True)

Xtr, ytr = to_numpy(train)
Xva, yva = to_numpy(val)
Xte, yte = to_numpy(test)
print(f"train={Xtr.shape} val={Xva.shape} test={Xte.shape}  "
      f"positive_rate={yte.mean():.3f}")

sc = StandardScaler().fit(Xtr)
Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

clf = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")
clf.fit(Xtr, ytr)

for name, X, y in [("val", Xva, yva), ("test", Xte, yte)]:
    pred = clf.predict(X)
    prob = clf.predict_proba(X)[:, 1]
    print(f"[{name}] acc={accuracy_score(y,pred):.4f} "
          f"auc={roc_auc_score(y,prob):.4f} f1={f1_score(y,pred):.4f}")

print("\n[test] classification report:")
print(classification_report(yte, clf.predict(Xte), target_names=["normal","pneumonia"]))
