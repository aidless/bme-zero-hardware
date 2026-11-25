"""Train-only scaler audit for BBBP and BACE scaffold benchmarks.

Only new files are written: data/trainonly_metrics.json.
Run: python3 -u bench_mol_trainonly.py
Self-check: python3 -u bench_mol_trainonly.py --selftest
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
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

DATASETS = {
    "BBBP": {"path": Path("data/BBBP.csv"), "smiles": "smiles", "label": "p_np"},
    "BACE": {"path": Path("data/BACE.csv"), "smiles": "mol", "label": "Class"},
}
OUT_PATH = Path("data/trainonly_metrics.json")
DESC_NAMES = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "RingCount", "HeavyAtomCount",
]
REPS = ["ECFP4-2048", "ECFP4+desc", "ECFP4+MACCS+desc"]
FUSION = "ECFP4+MACCS+desc"
MAIN_SEED = 42
SEEDS = [42, 0, 7, 2024, 123]
N_BOOT = 2000

LOG = []


def emit(msg):
    print(msg, flush=True)
    LOG.append(str(msg))


def canonicalize(df, smiles_col, label_col):
    seen, mols, ys, invalid, dup, conflicts = set(), [], [], 0, 0, 0
    by_canonical = {}
    for raw, label in zip(df[smiles_col], df[label_col]):
        mol = Chem.MolFromSmiles(str(raw))
        if mol is None:
            invalid += 1
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            dup += 1
            if by_canonical[can] != int(label):
                conflicts += 1
            continue
        seen.add(can)
        by_canonical[can] = int(label)
        mols.append(mol)
        ys.append(int(label))
    stats = {
        "raw_rows": int(len(df)), "unique_valid": int(len(mols)),
        "invalid_dropped": invalid, "duplicate_dropped": dup,
        "conflicting_label_duplicates": conflicts,
    }
    return mols, np.asarray(ys, dtype=int), stats


def scaffold_of(mol):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def deepchem_scaffold_split(mols, seed=MAIN_SEED):
    groups = defaultdict(list)
    for i, mol in enumerate(mols):
        groups[scaffold_of(mol)].append(i)
    rng = np.random.default_rng(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: -len(kv[1]))
    n, train, valid, test = len(mols), [], [], []
    train_cutoff, valid_cutoff = 0.8 * n, 0.9 * n
    for _, idxs in items:
        if len(train) + len(idxs) > train_cutoff:
            if len(train) + len(valid) + len(idxs) > valid_cutoff:
                test.extend(idxs)
            else:
                valid.extend(idxs)
        else:
            train.extend(idxs)
    return {
        "train": np.asarray(train, dtype=int),
        "valid": np.asarray(valid, dtype=int),
        "test": np.asarray(test, dtype=int),
    }


def split_ok(mols, split, require_nonempty=True):
    parts = ("train", "valid", "test")
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            a = {scaffold_of(mols[k]) for k in split[parts[i]]}
            b = {scaffold_of(mols[k]) for k in split[parts[j]]}
            assert not (a & b), f"scaffold overlap {parts[i]}/{parts[j]}"
    assert len(set(split["train"]) | set(split["valid"]) | set(split["test"])) == len(mols)
    if require_nonempty:
        assert min(len(split[p]) for p in parts) > 0


def raw_features(mols):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ecfp = np.asarray([gen.GetFingerprint(m) for m in mols], dtype=np.int8)
    maccs = np.asarray([MACCSkeys.GenMACCSKeys(m) for m in mols], dtype=np.int8)
    desc = np.asarray([[float(getattr(Descriptors, n)(m)) for n in DESC_NAMES] for m in mols])
    assert ecfp.shape[1] == 2048 and maccs.shape[1] == 167
    assert np.isfinite(desc).all()
    return {"ecfp": ecfp, "maccs": maccs, "desc": desc}


def scaled_desc(raw_desc, train_idx, mode):
    """Fit only on train_idx in train_only mode; full_fit is the historical leakage."""
    fit_idx = train_idx if mode == "train_only" else np.arange(len(raw_desc))
    scaler = StandardScaler().fit(raw_desc[fit_idx])
    if mode == "train_only":
        train_std = raw_desc[train_idx].std(axis=0)
        expected_scale = np.where(train_std > 0, train_std, 1.0)
        assert np.allclose(scaler.mean_, raw_desc[train_idx].mean(axis=0), atol=1e-12)
        assert np.allclose(scaler.scale_, expected_scale, atol=1e-12)
    return scaler.transform(raw_desc), scaler


def build_reps(raw, train_idx, mode):
    desc, scaler = scaled_desc(raw["desc"], train_idx, mode)
    return {
        "ECFP4-2048": raw["ecfp"],
        "ECFP4+desc": np.hstack([raw["ecfp"], desc]),
        FUSION: np.hstack([raw["ecfp"], raw["maccs"], desc]),
    }, scaler


def stratified_boot_indices(y, n_boot=N_BOOT, seed=MAIN_SEED):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    assert len(pos) > 0 and len(neg) > 0
    return pos[rng.integers(0, len(pos), (n_boot, len(pos)))], neg[rng.integers(0, len(neg), (n_boot, len(neg)))]


def auc_from_indices(score, pos, neg):
    s = np.asarray(score)
    sp, sn = s[pos][:, None], s[neg][None, :]
    return (sp > sn).mean() + 0.5 * (sp == sn).mean()


def bootstrap_ci(score, boot):
    vals = np.asarray([auc_from_indices(score, boot[0][b], boot[1][b]) for b in range(len(boot[0]))])
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)), "mean": float(vals.mean())}


def paired_delta_ci(score_a, score_b, boot):
    vals = np.asarray([
        auc_from_indices(score_a, boot[0][b], boot[1][b])
        - auc_from_indices(score_b, boot[0][b], boot[1][b])
        for b in range(len(boot[0]))
    ])
    return {"delta": float(vals.mean()), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}


def fit_predict(X, y, train, test, seed=MAIN_SEED):
    clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    clf.fit(X[train], y[train])
    score = clf.predict_proba(X[test])[:, 1]
    assert np.isfinite(score).all()
    return score


def split_stats(y, split):
    return {name: {"n": int(len(idx)), "pos": int(y[idx].sum()),
                   "neg": int(len(idx) - y[idx].sum()),
                   "positive_rate": float(y[idx].mean())}
            for name, idx in split.items()}


def leakage_proof(raw_desc, split):
    train, test = split["train"], split["test"]
    scaler = StandardScaler().fit(raw_desc[train])
    assert np.allclose(scaler.mean_, raw_desc[train].mean(axis=0), atol=1e-12)
    train_std = raw_desc[train].std(axis=0)
    expected_scale = np.where(train_std > 0, train_std, 1.0)
    assert np.allclose(scaler.scale_, expected_scale, atol=1e-12), (scaler.scale_, expected_scale)
    shifted = raw_desc.copy()
    shifted[test] += 1e3
    train_shifted_scaler = StandardScaler().fit(shifted[train])
    heldout_shift_can_change_train_params = bool(
        not np.allclose(scaler.mean_, train_shifted_scaler.mean_, atol=1e-12)
        or not np.allclose(scaler.scale_, train_shifted_scaler.scale_, atol=1e-12)
    )
    train_only_params_invariant = not heldout_shift_can_change_train_params
    full_scaler = StandardScaler().fit(raw_desc)
    full_shifted_scaler = StandardScaler().fit(shifted)
    full_fit_detector_is_red = bool(
        not np.allclose(full_scaler.mean_, full_shifted_scaler.mean_, atol=1e-12)
        or not np.allclose(full_scaler.scale_, full_shifted_scaler.scale_, atol=1e-12)
    )
    mean_differs = bool(not np.allclose(scaler.mean_, full_scaler.mean_, atol=1e-12))
    assert train_only_params_invariant and full_fit_detector_is_red
    return {
        "method": (
            "Strict train-only fit is proved three ways: scaler.mean_/scale_ equal train-fold moments; "
            "adding 1000 to held-out descriptor rows leaves the train-only fitted parameters unchanged; "
            "the same perturbation changes full-fit parameters (negative control turns RED)."
        ),
        "train_rows_fit": int(len(train)),
        "heldout_rows_not_fit": int(len(test)),
        "scaler_mean_matches_train_moments": True,
        "scaler_scale_matches_train_moments": True,
        "heldout_perturbation_leaves_train_params_unchanged": train_only_params_invariant,
        "full_fit_negative_control_detected": full_fit_detector_is_red,
        "train_mean_differs_from_full_mean": mean_differs,
        "train_mean_l2_vs_full_mean": float(np.linalg.norm(scaler.mean_ - full_scaler.mean_)),
    }


def selftest():
    base = [f"C1{'C' * (n - 2)}C1" for n in range(3, 33)]
    smiles = base + ["not-a-smiles", base[0]]
    df = pd.DataFrame({"mol": smiles, "Class": [i % 2 for i in range(30)] + [1, 1]})
    mols, y, stats = canonicalize(df, "mol", "Class")
    assert stats == {"raw_rows": 32, "unique_valid": 30, "invalid_dropped": 1,
                     "duplicate_dropped": 1, "conflicting_label_duplicates": 1}, stats
    split = deepchem_scaffold_split(mols)
    split_ok(mols, split)
    raw = raw_features(mols)
    proof = leakage_proof(raw["desc"], split)
    assert proof["heldout_perturbation_leaves_train_params_unchanged"] and proof["full_fit_negative_control_detected"]
    boot = stratified_boot_indices(y[split["test"]], n_boot=20, seed=0)
    score = np.linspace(0, 1, len(split["test"]))
    ci = bootstrap_ci(score, boot)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    delta = paired_delta_ci(score, score, boot)
    assert abs(delta["delta"]) < 1e-12
    print("SELFTEST PASS: canonical dedup, zero-overlap split, train-only scaler proof, negative control, bootstrap CI", flush=True)


def main():
    out = {
        "task": "Train-only StandardScaler audit of BBBP and BACE scaffold benchmarks",
        "protocol": {
            "canonicalization": "RDKit MolFromSmiles -> MolToSmiles, keep first; invalid dropped; label conflicts counted",
            "split": "DeepChem-style Bemis-Murcko scaffold split, shuffle groups seed 42, descending group size, fill 80/10/10; pairwise overlap asserted zero",
            "descriptors": DESC_NAMES,
            "full_fit": "Historical protocol: StandardScaler fit on all deduplicated rows before splitting",
            "train_only": "Strict protocol: StandardScaler fit on scaffold train fold only, then transform valid/test without refitting",
            "model": "RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight=None)",
            "bootstrap": "2000 stratified resamples of the scaffold test set; percentile 95% CI; seed 42",
            "seed_stability": "Five RF seeds 42/0/7/2024/123 on the fixed scaffold split, fusion representation only",
        },
        "environment": {
            "python": sys.version.split()[0], "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "platform": platform.platform(),
        },
        "datasets": {},
    }
    for name, cfg in DATASETS.items():
        path = cfg["path"]
        assert path.exists(), f"missing {path}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        df = pd.read_csv(path)
        mols, y, dedup = canonicalize(df, cfg["smiles"], cfg["label"])
        assert set(np.unique(y)).issubset({0, 1})
        raw = raw_features(mols)
        split = deepchem_scaffold_split(mols)
        split_ok(mols, split)
        groups = defaultdict(list)
        for i, mol in enumerate(mols):
            groups[scaffold_of(mol)].append(i)
        boot = stratified_boot_indices(y[split["test"]])
        emit(f"{name}: {dedup}; split={split_stats(y, split)}; test labels pos={int(y[split['test']].sum())} neg={len(split['test'])-int(y[split['test']].sum())}")
        ds = {
            "data": {"path": str(path), "sha256": digest, "dedup": dedup,
                     "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))}},
            "split": {"unique_scaffolds": len(groups), "sizes": split_stats(y, split),
                      "train_indices_first20": split["train"][:20].tolist(),
                      "test_indices": split["test"].tolist()},
            "scaler_proof": leakage_proof(raw["desc"], split),
            "results": {}, "paired_delta_trainonly_minus_fullfit": {},
            "seed_stability": {"seeds": SEEDS},
        }
        scaled = {}
        scores = {}
        for mode in ("full_fit", "train_only"):
            reps, _ = build_reps(raw, split["train"], mode)
            scaled[mode] = reps
            scores[mode] = {}
            ds["results"][mode] = {}
            for rep in REPS:
                proba = fit_predict(reps[rep], y, split["train"], split["test"], MAIN_SEED)
                scores[mode][rep] = proba
                auc = float(roc_auc_score(y[split["test"]], proba))
                ci = bootstrap_ci(proba, boot)
                ds["results"][mode][rep] = {"auc": auc, "ci95": ci, "dim": int(reps[rep].shape[1])}
                emit(f"{name:4s} {mode:10s} {rep:18s} AUC={auc:.4f} [{ci['lo']:.3f},{ci['hi']:.3f}]")
        for rep in REPS:
            ds["paired_delta_trainonly_minus_fullfit"][rep] = paired_delta_ci(
                scores["train_only"][rep], scores["full_fit"][rep], boot,
            )
        for mode in ("full_fit", "train_only"):
            vals = []
            for seed in SEEDS:
                proba = fit_predict(scaled[mode][FUSION], y, split["train"], split["test"], seed)
                vals.append({"seed": seed, "auc": float(roc_auc_score(y[split["test"]], proba))})
            a = np.asarray([v["auc"] for v in vals])
            ds["seed_stability"][mode] = vals
            ds["seed_stability"][mode + "_summary"] = {
                "mean": float(a.mean()), "std_sample": float(a.std(ddof=1)),
                "min": float(a.min()), "max": float(a.max()),
            }
        out["datasets"][name] = ds
    out["answer"] = {
        name: {
            rep: {
                "full_fit_auc": out["datasets"][name]["results"]["full_fit"][rep]["auc"],
                "train_only_auc": out["datasets"][name]["results"]["train_only"][rep]["auc"],
                "auc_change": out["datasets"][name]["results"]["train_only"][rep]["auc"]
                              - out["datasets"][name]["results"]["full_fit"][rep]["auc"],
            }
            for rep in REPS
        }
        for name in DATASETS
    }
    out["run_log"] = list(LOG)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    emit(f"WROTE {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        main()
