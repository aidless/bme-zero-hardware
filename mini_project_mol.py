"""Mini project v3: AI drug discovery — BBBP (blood-brain barrier penetration).
Direction (16) computer-aided drug design. Pure CPU. No wet lab.
Dataset: MoleculeNet BBBP (2053 molecules, binary labels).
Tools: rdkit (ECFP/Morgan fingerprint) + sklearn/lightgbm.
Install: pip install rdkit scikit-learn lightgbm pandas
Run:     python3 mini_project_mol.py
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
RDLogger.DisableLog("rdApp.*")
SEED = 42

df = pd.read_csv("data/BBBP.csv")
print(f"loaded {len(df)} molecules, label balance={df.p_np.mean():.3f}")

# SMILES -> Morgan fingerprint (ECFP4, 2048 bits)
def featurize(smiles):
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            fps.append(np.zeros(2048, dtype=np.int8)); continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        fps.append(np.frombuffer(fp.ToBitString().encode(), dtype="u1") - ord("0"))
    return np.array(fps)

X = featurize(df.smiles)
y = df.p_np.values.astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"train={Xtr.shape} test={Xte.shape}  pos_rate={ytr.mean():.3f}")

results = {}
for name, clf in [
    ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
]:
    try:
        from lightgbm import LGBMClassifier
        clf2 = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1)
    except ImportError:
        clf2 = None
    for nm, model in [(name, clf), ("LightGBM", clf2)]:
        if model is None: continue
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(yte, prob)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred)
        results[nm] = auc
        print(f"[{nm}] test acc={acc:.4f} auc={auc:.4f} f1={f1:.4f}")

best = max(results, key=results.get)
print(f"\nbest={best} auc={results[best]:.4f}")
