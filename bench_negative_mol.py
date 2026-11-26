"""Negative control: label-permutation test of the molecular benchmark pipeline.

Only new files are written: data/negative/negative_mol_metrics.json (and this script).

The feature matrix X, Bemis-Murcko scaffold groups and the train/test indices are
computed ONCE per dataset, copied verbatim from the three existing benchmarks:
  - BBBP:   bench_mol_rigorous_v2.py (top-level script; functions copied verbatim)
  - BACE:   bench_mol_bace.py        (helpers imported; identical protocol)
  - ClinTox:bench_mol_clintox.py + bench_mol_trainonly.py (helpers imported)
Each permutation shuffles the FULL label vector (train+valid+test together),
rejecting draws that leave either train or test without both classes, then
refits the same RF500(seed 42) and scores the same held-out test set.

Run:       python3 -u bench_negative_mol.py
Selftest:  python3 -u bench_negative_mol.py --selftest
Verify:    python3 -u bench_negative_mol.py --verify
Optional smoke: python3 -u bench_negative_mol.py --n-perm 2 --out <tmp.json>
"""
import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from bench_mol_bace import (
    canonicalize as bace_canonicalize,
    deepchem_scaffold_split as bace_split,
    fingerprints as bace_fingerprints,
    assert_split_ok as bace_assert_split_ok,
)
from bench_mol_trainonly import (
    canonicalize as generic_canonicalize,
    raw_features,
    build_reps,
    deepchem_scaffold_split as generic_split,
    split_ok as generic_split_ok,
    stratified_boot_indices,
    bootstrap_ci,
)

RDLogger.DisableLog("rdApp.*")

OUT_PATH = Path("data/negative/negative_mol_metrics.json")
MAIN_SEED = 42
PERM_SEED = 20261126
N_PERM = 100
N_BOOT_BBBP = 1000   # bench_mol_rigorous_v2.py N_BOOT
N_BOOT_OTHER = 2000  # bench_mol_bace.py / bench_mol_clintox.py N_BOOT
FUSION = "ECFP4+MACCS+desc"
DESC_NAMES = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "RingCount", "HeavyAtomCount",
]
REFERENCE_AUC = {
    "BBBP": {"auc": 0.9413, "source": "bench_mol_rigorous_v2.py fusion/scaffold (also BME-DEEP-DIVE.md:73-76 via bench_mol_bace.py BBBP_REFERENCE)"},
    "BACE": {"auc": 0.899848828420257, "source": "data/bace_rigorous_metrics.json results/scaffold/ECFP4+MACCS+desc.auc"},
    "ClinTox": {"auc": 0.7962962962962963, "source": "data/clintox_metrics.json results/scaffold_80_10_10/representations/ECFP4+MACCS+desc.auc_seed42"},
}


# ---- BBBP helpers copied verbatim from bench_mol_rigorous_v2.py:38-105 ----
def bb_scaffold_of_mol(m):
    if m is None:
        return "NONE"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)


def bb_deepchem_scaffold_split(mols, frac_train=0.8, frac_valid=0.1, seed=MAIN_SEED):
    groups = defaultdict(list)
    for i, m in enumerate(mols):
        groups[bb_scaffold_of_mol(m)].append(i)
    rng = np.random.default_rng(seed)
    scaffold_items = list(groups.items())
    rng.shuffle(scaffold_items)
    scaffold_items.sort(key=lambda kv: -len(kv[1]))
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


def bb_ecfp(m, n_bits):
    if m is None:
        return np.zeros(n_bits, dtype=np.int8)
    return np.frombuffer(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits).ToBitString().encode(), dtype="u1") - 48


def bb_maccs(m):
    if m is None:
        return np.zeros(167, dtype=np.int8)
    return np.frombuffer(MACCSkeys.GenMACCSKeys(m).ToBitString().encode(), dtype="u1") - 48


def bb_desc(m):
    if m is None:
        return np.zeros(len(DESC_NAMES))
    return [float(getattr(Descriptors, n)(m)) for n in DESC_NAMES]


def bb_bootstrap_auc(y_true, y_score, n=N_BOOT_BBBP, seed=MAIN_SEED):
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


# ---- loaders: X / split computed exactly once per dataset ----
def load_bbbp():
    df = pd.read_csv("data/BBBP.csv")
    seen, mols, ys = set(), [], []
    for s, lab in zip(df.smiles, df.p_np):
        m = Chem.MolFromSmiles(str(s))
        c = Chem.MolToSmiles(m) if m is not None else None
        if c is None or c in seen:
            continue
        seen.add(c); mols.append(m); ys.append(int(lab))
    y = np.asarray(ys, dtype=int)
    E = np.asarray([bb_ecfp(m, 2048) for m in mols])
    M = np.asarray([bb_maccs(m) for m in mols])
    D = StandardScaler().fit_transform(np.asarray([bb_desc(m) for m in mols]))
    X = np.hstack([E, M, D])
    tr, va, te, n_scaf = bb_deepchem_scaffold_split(mols)
    split = {"train": tr, "test": te}
    assert not ({bb_scaffold_of_mol(mols[i]) for i in tr} & {bb_scaffold_of_mol(mols[i]) for i in te})
    stats = {"raw_rows": int(len(df)), "unique_valid": int(len(mols)),
             "invalid_or_duplicate_dropped": int(len(df) - len(mols)), "unique_scaffolds": int(n_scaf)}
    return X, y, split, stats


def load_bace():
    df = pd.read_csv("data/BACE.csv")
    mols, y, _, stats = bace_canonicalize(df)
    X = bace_fingerprints(mols)[FUSION]
    s = bace_split(mols, MAIN_SEED)
    bace_assert_split_ok(mols, s)
    return X, y, {"train": s["train"], "test": s["test"]}, stats


def load_clintox():
    df = pd.read_csv("data/ClinTox.csv.gz")
    mols, y, stats = generic_canonicalize(df, "smiles", "CT_TOX")
    raw = raw_features(mols)
    s = generic_split(mols, MAIN_SEED)
    generic_split_ok(mols, s)
    reps, _ = build_reps(raw, s["train"], "train_only")
    return reps[FUSION], y, {"train": s["train"], "test": s["test"]}, stats


# ---- permutation machinery ----
def permuted_labels(y, train, test, rng):
    """Permute the full label vector; reject draws with a degenerate train or test set."""
    while True:
        yp = rng.permutation(y)
        if 0 < yp[train].sum() < len(train) and 0 < yp[test].sum() < len(test):
            return yp


def fit_predict(X, y, train, test, seed=MAIN_SEED):
    clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    clf.fit(X[train], y[train])
    score = clf.predict_proba(X[test])[:, 1]
    assert np.isfinite(score).all()
    return score


def ci_lower(name, y_test, score):
    if name == "BBBP":
        return bb_bootstrap_auc(y_test, score)[0]
    boot = stratified_boot_indices(y_test, N_BOOT_OTHER, MAIN_SEED)
    return float(bootstrap_ci(score, boot)["lo"])


def run_null(X, y, split, n_perm, seed, name=None):
    train, test = split["train"], split["test"]
    rng = np.random.default_rng(seed)
    aucs, lows = [], []
    for _ in range(n_perm):
        yp = permuted_labels(y, train, test, rng)
        score = fit_predict(X, yp, train, test, MAIN_SEED)
        aucs.append(float(roc_auc_score(yp[test], score)))
        if name is not None:
            lows.append(ci_lower(name, yp[test], score))
    return np.asarray(aucs), np.asarray(lows)


def null_stats(aucs, real_auc, lows):
    return {
        "n": int(len(aucs)),
        "mean": float(aucs.mean()),
        "std_sample": float(aucs.std(ddof=1)),
        "median": float(np.median(aucs)),
        "p95": float(np.percentile(aucs, 95)),
        "max": float(aucs.max()),
        "min": float(aucs.min()),
        "aucs": [float(a) for a in aucs],
        "empirical_p": float((np.sum(aucs >= real_auc) + 1) / (len(aucs) + 1)),
        "bootstrap_ci_lower_gt_0_5_count": int(np.sum(lows > 0.5)) if len(lows) else None,
    }


def selftest():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(75, dtype=int), np.zeros(75, dtype=int)]
    X = rng.normal(size=(150, 30))
    split = {"train": np.arange(120), "test": np.arange(120, 150)}
    r = np.random.default_rng(123)
    for _ in range(50):
        assert np.array_equal(np.sort(r.permutation(y)), np.sort(y))
    for _ in range(20):
        yp = permuted_labels(y, split["train"], split["test"], r)
        assert np.array_equal(np.sort(yp), np.sort(y))
        assert 0 < yp[split["train"]].sum() < 120 and 0 < yp[split["test"]].sum() < 30
    a1, _ = run_null(X, y, split, 6, 7)
    a2, _ = run_null(X, y, split, 6, 7)
    assert np.array_equal(a1, a2), "same seed must be reproducible"
    a3, _ = run_null(X, y, split, 6, 8)
    assert not np.array_equal(a1, a3), "different seed must change the null"
    a4, _ = run_null(X, y, split, 20, 9)
    assert abs(a4.mean() - 0.5) < 0.15, f"null mean too far from 0.5: {a4.mean()}"
    print(f"SELFTEST PASS: multiset preserved, class-preserving rejection, same-seed reproducible, "
          f"null mean={a4.mean():.4f} (|mean-0.5|<0.15)", flush=True)


def verify():
    out = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    for name, d in out["datasets"].items():
        n = d["null"]
        a = np.asarray(n["aucs"], dtype=float)
        assert len(a) == n["n"]
        assert abs(a.mean() - n["mean"]) < 1e-9
        assert abs(a.std(ddof=1) - n["std_sample"]) < 1e-9
        assert abs(np.median(a) - n["median"]) < 1e-9
        assert abs(np.percentile(a, 95) - n["p95"]) < 1e-9
        assert abs(a.max() - n["max"]) < 1e-9 and abs(a.min() - n["min"]) < 1e-9
        p = (np.sum(a >= d["real_auc"]) + 1) / (len(a) + 1)
        assert abs(p - n["empirical_p"]) < 1e-12
        assert n["n"] >= 100
        print(f"VERIFY PASS {name}: n={n['n']} mean={n['mean']:.4f} std={n['std_sample']:.4f} "
              f"p95={n['p95']:.4f} p={p:.4f}", flush=True)
    print("VERIFY PASS: all null statistics recomputed from JSON", flush=True)


def main(n_perm, out_path):
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = {
        "task": "Ninth-round negative control: label permutation of the molecular benchmark pipeline",
        "protocol": {
            "permutation": "full label vector (train+valid+test) permuted with one fixed RNG seed per dataset; draws leaving train or test without both classes are rejected",
            "invariants": "X, BM scaffold groups and train/test indices computed once; only labels permuted and RF refit",
            "model": "RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight=None), identical to real experiments",
            "features": "ECFP4(2048)+MACCS(167)+8 descriptors (2223d); BBBP/BACE scaler fit on full set, ClinTox train-only, exactly as the real experiments",
            "split": "DeepChem-style Bemis-Murcko scaffold 80/10/10, seed 42; train+test used (valid labels permuted but not scored)",
            "ci": "BBBP 1000 stratified bootstrap; BACE/ClinTox 2000 stratified bootstrap; 95th percentile CI lower bound > 0.5 counted as a false positive",
            "empirical_p": "(count(null AUC >= real AUC) + 1) / (N + 1)",
            "verdict_rule": "null mean in 0.5±0.05 AND empirical p <= 0.01 => pipeline clean",
        },
        "environment": {
            "python": sys.version.split()[0], "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "platform": platform.platform(),
        },
        "script": {"path": "bench_negative_mol.py", "sha256": script_sha,
                   "n_perm": int(n_perm), "perm_seed": PERM_SEED, "model_seed": MAIN_SEED},
        "datasets": {},
    }
    loaders = {"BBBP": load_bbbp, "BACE": load_bace, "ClinTox": load_clintox}
    for name, loader in loaders.items():
        X, y, split, dedup = loader()
        train, test = split["train"], split["test"]
        assert 0 < y[train].sum() < len(train) and 0 < y[test].sum() < len(test)
        real_score = fit_predict(X, y, train, test, MAIN_SEED)
        real_auc = float(roc_auc_score(y[test], real_score))
        aucs, lows = run_null(X, y, split, n_perm, PERM_SEED, name=name)
        stats = null_stats(aucs, real_auc, lows)
        clean = bool(abs(stats["mean"] - 0.5) <= 0.05 and stats["empirical_p"] <= 0.01)
        data_sha = hashlib.sha256(Path({
            "BBBP": "data/BBBP.csv", "BACE": "data/BACE.csv",
            "ClinTox": "data/ClinTox.csv.gz",
        }[name]).read_bytes()).hexdigest()
        out["datasets"][name] = {
            "data": {"dedup": dedup, "sha256": data_sha, "n": int(len(y)),
                     "pos": int(y.sum()), "neg": int(len(y) - y.sum())},
            "split": {"feature_dim": int(X.shape[1]), "train_n": int(len(train)),
                      "test_n": int(len(test)), "train_pos": int(y[train].sum()),
                      "test_pos": int(y[test].sum()),
                      "test_neg": int(len(test) - y[test].sum())},
            "real_auc": real_auc,
            "reference_auc": REFERENCE_AUC[name],
            "null": stats,
            "verdict": {"null_mean_within_0.05_of_0.5": bool(abs(stats["mean"] - 0.5) <= 0.05),
                        "empirical_p_le_0.01": bool(stats["empirical_p"] <= 0.01),
                        "pipeline_clean": clean},
        }
        print(f"{name}: real_auc={real_auc:.4f} ref={REFERENCE_AUC[name]['auc']:.4f} "
              f"null mean={stats['mean']:.4f} std={stats['std_sample']:.4f} "
              f"p95={stats['p95']:.4f} p={stats['empirical_p']:.4f} "
              f"FP_CI={stats['bootstrap_ci_lower_gt_0_5_count']} clean={clean}", flush=True)
    out["overall_verdict"] = {
        "pipeline_clean": all(d["verdict"]["pipeline_clean"] for d in out["datasets"].values()),
        "n_datasets": len(out["datasets"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE {out_path} ({out_path.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.verify:
        verify()
    else:
        main(args.n_perm, args.out)
