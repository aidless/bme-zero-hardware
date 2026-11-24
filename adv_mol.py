"""Advanced v2: richer molecular representation for BBBP.
Goal: beat ECFP4-only baseline test AUC = 0.9182.
Compare: ECFP4 vs MACCS vs ECFP4+RDKit-descriptors fusion, with RandomForest.
Run: python3 adv_mol.py
"""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*")
SEED=42

df = pd.read_csv("data/BBBP.csv")
y = df.p_np.values.astype(int)

def ecfp(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return np.zeros(2048,dtype=np.int8)
    return np.frombuffer(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048).ToBitString().encode(),dtype="u1")-48

def maccs(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return np.zeros(167,dtype=np.int8)
    return np.frombuffer(MACCSkeys.GenMACCSKeys(m).ToBitString().encode(),dtype="u1")-48

desc_names = ["MolWt","MolLogP","NumHDonors","NumHAcceptors","TPSA","NumRotatableBonds","RingCount","HeavyAtomCount"]
def desc(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return np.zeros(len(desc_names))
    f=getattr(Descriptors,"MolLogP",None)
    vals=[]
    for n in desc_names:
        try: vals.append(float(getattr(Descriptors,n)(m)))
        except Exception: vals.append(0.0)
    return vals

print("featurizing...")
E=np.array([ecfp(s) for s in df.smiles])
M=np.array([maccs(s) for s in df.smiles])
D=np.array([desc(s) for s in df.smiles])
Dsc=StandardScaler().fit_transform(D)

reps={
    "ECFP4 (2048b)": E,
    "MACCS (167b)": M,
    "ECFP4+desc": np.hstack([E,Dsc]),
    "ECFP4+MACCS+desc": np.hstack([E,M,Dsc]),
}
results={}
for name,X in reps.items():
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=SEED)
    clf=RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=-1)
    clf.fit(Xtr,ytr)
    auc=roc_auc_score(yte,clf.predict_proba(Xte)[:,1])
    results[name]=auc
    print(f"[{name:20s}] dim={X.shape[1]:5d} test AUC={auc:.4f}")
base=results["ECFP4 (2048b)"]
best=max(results,key=results.get)
print(f"\nbaseline ECFP4={base:.4f} -> best={best} ({results[best]:.4f}, delta={results[best]-base:+.4f})")
