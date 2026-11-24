"""Rigorous re-evaluation of BBBP representations: random split vs scaffold split.

Why: a random stratified split lets near-identical Bemis-Murcko scaffolds leak
across train/test, inflating AUC. Scaffold split holds out whole scaffold
families, which better reflects prospective (new chemotype) generalization.

Adds 1000x bootstrap 95% CI on test AUC for every representation/split pair.

Run: python3 bench_mol_rigorous.py
Data: data/BBBP.csv (DeepChem MoleculeNet BBBP, 2050 molecules)
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
SEED = 42
N_BOOT = 1000


def scaffold_of(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return "NONE"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)


def scaffold_split(smiles_list, frac_test=0.2):
    """DeepChem-style: group by BM scaffold, sort groups by size ascending
    (smallest scaffold families -> test), so no scaffold appears in both sides."""
    groups = defaultdict(list)
    for i, s in enumerate(smiles_list):
        groups[scaffold_of(s)].append(i)
    order = sorted(groups.keys(), key=lambda k: (len(groups[k]), k))
    n_test_target = int(len(smiles_list) * frac_test)
    test_idx = []
    for k in order:
        if len(test_idx) >= n_test_target:
            break
        test_idx.extend(groups[k])
    test_idx = np.array(sorted(test_idx))
    train_idx = np.array(sorted(set(range(len(smiles_list))) - set(test_idx)))
    return train_idx, test_idx, len(groups), max(len(v) for v in groups.values())


def ecfp(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return np.zeros(2048, dtype=np.int8)
    return np.frombuffer(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048).ToBitString().encode(), dtype="u1") - 48


def maccs(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return np.zeros(167, dtype=np.int8)
    return np.frombuffer(MACCSkeys.GenMACCSKeys(m).ToBitString().encode(), dtype="u1") - 48


desc_names = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds", "RingCount", "HeavyAtomCount"]


def desc(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return np.zeros(len(desc_names))
    return [float(getattr(Descriptors, n)(m)) for n in desc_names]


def bootstrap_auc(y_true, y_score, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    pos_pool = np.where(y_true == 1)[0]
    neg_pool = np.where(y_true == 0)[0]
    aucs = []
    for _ in range(n):
        pos = pos_pool[rng.integers(0, n_pos, n_pos)]
        neg = neg_pool[rng.integers(0, n_neg, n_neg)]
        aucs.append((y_score[pos][:, None] > y_score[neg][None, :]).mean() +
                    0.5 * (y_score[pos][:, None] == y_score[neg][None, :]).mean())
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


df = pd.read_csv("data/BBBP.csv")
y = df.p_np.values.astype(int)
smiles = df.smiles.tolist()

print("featurizing...")
E = np.array([ecfp(s) for s in smiles])
M = np.array([maccs(s) for s in smiles])
D = np.array([desc(s) for s in smiles])
Dsc = StandardScaler().fit_transform(D)
reps = {
    "ECFP4 (2048b)": E,
    "MACCS (167b)": M,
    "ECFP4+desc": np.hstack([E, Dsc]),
    "ECFP4+MACCS+desc": np.hstack([E, M, Dsc]),
}

tr_s, te_s, n_scaf, max_grp = scaffold_split(smiles)
print(f"scaffold stats: {n_scaf} unique BM scaffolds, largest family={max_grp} molecules")
print(f"scaffold split -> train={len(tr_s)} test={len(te_s)} "
      f"(test positive rate={y[te_s].mean():.3f}, train={y[tr_s].mean():.3f})")

print("\n{:<22s} {:^27s} {:^27s}".format("", "RANDOM split", "SCAFFOLD split"))
print("-" * 80)
summary = []
rand_tr, rand_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
for name, X in reps.items():
    row = {"rep": name, "dim": X.shape[1]}
    for split_name, (tr, te) in [("random", (rand_tr, rand_te)), ("scaffold", (tr_s, te_s))]:
        clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], proba)
        lo, hi = bootstrap_auc(y[te], proba)
        row[split_name] = {"auc": auc, "lo": lo, "hi": hi}
    r = row["random"]
    s = row["scaffold"]
    print(f"{name:22s}  random  AUC={r['auc']:.4f} [{r['lo']:.3f},{r['hi']:.3f}]   "
          f"scaffold AUC={s['auc']:.4f} [{s['lo']:.3f},{s['hi']:.3f}]  "
          f"gap={s['auc']-r['auc']:+.4f}")
    summary.append(row)

print("\n=== key comparison (best representation on each split) ===")
for split_name in ["random", "scaffold"]:
    best = max(summary, key=lambda d: d[split_name]["auc"])
    print(f"{split_name:9s}: best={best['rep']}  AUC={best[split_name]['auc']:.4f} "
          f"95%CI=[{best[split_name]['lo']:.3f},{best[split_name]['hi']:.3f}]")

r_best = max(summary, key=lambda d: d["random"]["auc"])["random"]["auc"]
s_best = max(summary, key=lambda d: d["scaffold"]["auc"])["scaffold"]["auc"]
print(f"\ninflation of random vs scaffold (best-of-each): {r_best - s_best:+.4f} AUC")
