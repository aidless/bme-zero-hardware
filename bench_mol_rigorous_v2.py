"""Audit-grade BBBP benchmark, v2 (corrects v1's two protocol errors).

v1 (bench_mol_rigorous.py) had two flaws surfaced by the literature check:
  1. No canonical-SMILES dedup: 2050 raw rows -> only 1986 unique canonical
     structures (64 duplicate representations) -> identity leakage.
  2. Custom scaffold split (smallest-first, unstratified) instead of the
     DeepChem ScaffoldSplitter algorithm -> test label drift, unreproducible.

v2 fixes both:
  - dedup by RDKit canonical SMILES (keep first), drop invalid molecules
  - DeepChem-style scaffold split: group by Bemis-Murcko scaffold, sort groups
    by size DESC, fill train(80%)/valid(10%)/test(10%) in that order
  - 1000x bootstrap 95% CI on test AUC

Reference protocol: Wu et al. MoleculeNet, arXiv:1703.00564v3, Chem. Sci.
2018, 9, 513-530. Note MoleculeNet reports scaffold-split RF ~0.714 on BBBP
with 1024-bit features; comparisons must use the same split & features.

Run: python3 bench_mol_rigorous_v2.py
Data: data/BBBP.csv (DeepChem MoleculeNet BBBP)
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


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None else None


def scaffold_of_mol(m):
    if m is None:
        return "NONE"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)


def deepchem_scaffold_split(mols, frac_train=0.8, frac_valid=0.1, seed=SEED):
    """Match DeepChem ScaffoldSplitter.gen_scaffold_sets:
    group by BM scaffold, sort groups by size DESC, fill train then valid then test."""
    groups = defaultdict(list)
    for i, m in enumerate(mols):
        groups[scaffold_of_mol(m)].append(i)
    rng = np.random.default_rng(seed)
    scaffold_items = list(groups.items())
    rng.shuffle(scaffold_items)                       # break size-tie dependence
    scaffold_items.sort(key=lambda kv: -len(kv[1]))   # largest scaffold family first
    n = len(mols)
    train_cutoff = frac_train * n
    valid_cutoff = (frac_train + frac_valid) * n
    train_idx, valid_idx, test_idx = [], [], []
    for _, idxs in scaffold_items:
        if len(train_idx) + len(idxs) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(idxs) > valid_cutoff:
                test_idx.extend(idxs)
            else:
                valid_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    return np.array(train_idx), np.array(valid_idx), np.array(test_idx), len(groups)


def ecfp(m, n_bits):
    if m is None:
        return np.zeros(n_bits, dtype=np.int8)
    return np.frombuffer(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits).ToBitString().encode(), dtype="u1") - 48


def maccs(m):
    if m is None:
        return np.zeros(167, dtype=np.int8)
    return np.frombuffer(MACCSkeys.GenMACCSKeys(m).ToBitString().encode(), dtype="u1") - 48


desc_names = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds", "RingCount", "HeavyAtomCount"]


def desc(m):
    if m is None:
        return np.zeros(len(desc_names))
    return [float(getattr(Descriptors, n)(m)) for n in desc_names]


def bootstrap_auc(y_true, y_score, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n_pos = int(y_true.sum()); n_neg = len(y_true) - n_pos
    pp = np.where(y_true == 1)[0]; np_ = np.where(y_true == 0)[0]
    aucs = []
    for _ in range(n):
        pos = pp[rng.integers(0, n_pos, n_pos)]
        neg = np_[rng.integers(0, n_neg, n_neg)]
        aucs.append((y_score[pos][:, None] > y_score[neg][None, :]).mean() +
                    0.5 * (y_score[pos][:, None] == y_score[neg][None, :]).mean())
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ---- load + canonicalize + dedup ----
df = pd.read_csv("data/BBBP.csv")
seen, mols, ys = set(), [], []
for s, lab in zip(df.smiles, df.p_np):
    m = Chem.MolFromSmiles(str(s))
    c = Chem.MolToSmiles(m) if m is not None else None
    if c is None or c in seen:      # drop invalid & duplicate representations
        continue
    seen.add(c); mols.append(m); ys.append(int(lab))
y = np.array(ys)
print(f"loaded {len(df)} rows -> {len(mols)} unique valid canonical molecules "
      f"({len(df)-len(mols)} dropped)")

# featurize
print("featurizing...")
E2048 = np.array([ecfp(m, 2048) for m in mols])
E1024 = np.array([ecfp(m, 1024) for m in mols])
M = np.array([maccs(m) for m in mols])
D = StandardScaler().fit_transform(np.array([desc(m) for m in mols]))
reps = {
    "ECFP4-1024": E1024,
    "ECFP4-2048": E2048,
    "ECFP4-2048+desc": np.hstack([E2048, D]),
    "ECFP4+MACCS+desc": np.hstack([E2048, M, D]),
}

tr_s, va_s, te_s, n_scaf = deepchem_scaffold_split(mols)
print(f"scaffold split: {n_scaf} unique BM scaffolds -> "
      f"train={len(tr_s)} valid={len(va_s)} test={len(te_s)} "
      f"(test pos-rate={y[te_s].mean():.3f}, train={y[tr_s].mean():.3f})")

# verify no scaffold overlap between train and test
def scafset(idxs):
    return {scaffold_of_mol(mols[i]) for i in idxs}
overlap = scafset(tr_s) & scafset(te_s)
print(f"train/test scaffold overlap = {len(overlap)} (must be 0)")

print("\n{:<20s} {:^27s} {:^27s}".format("", "RANDOM split", "SCAFFOLD split (DeepChem)"))
print("-" * 82)
summary = []
rand_tr, rand_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
for name, X in reps.items():
    row = {"rep": name, "dim": X.shape[1]}
    for split_name, (tr, te) in [("random", (rand_tr, rand_te)), ("scaffold", (tr_s, te_s))]:
        clf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], proba)
        lo, hi = bootstrap_auc(y[te], proba)
        row[split_name] = {"auc": auc, "lo": lo, "hi": hi}
    r, s = row["random"], row["scaffold"]
    print(f"{name:20s}  AUC={r['auc']:.4f} [{r['lo']:.3f},{r['hi']:.3f}]   "
          f"AUC={s['auc']:.4f} [{s['lo']:.3f},{s['hi']:.3f}]   gap={s['auc']-r['auc']:+.4f}")
    summary.append(row)

print("\n=== summary ===")
for split_name in ["random", "scaffold"]:
    best = max(summary, key=lambda d: d[split_name]["auc"])
    print(f"{split_name:9s} best: {best['rep']}  AUC={best[split_name]['auc']:.4f} "
          f"95%CI=[{best[split_name]['lo']:.3f},{best[split_name]['hi']:.3f}]")
r_best = max(d["random"]["auc"] for d in summary)
s_best = max(d["scaffold"]["auc"] for d in summary)
print(f"random-vs-scaffold inflation (best each): {r_best - s_best:+.4f} AUC")
print("\nreference (MoleculeNet 2018, Table 6, scaffold split, different feature/split impl):")
print("  RF 0.714 / KernelSVM 0.729 / GraphConv 0.690  -- compare only qualitatively")
