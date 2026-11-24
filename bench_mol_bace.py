"""BACE external validation, matched to the BBBP v2 protocol.

Data: DeepChem MoleculeNet BACE (classification, canonical SMILES column `mol`).
Protocol: canonical-SMILES dedup -> Bemis-Murcko scaffold split (group size
descending, DeepChem-style) -> four fingerprint/descriptor representations ->
RandomForest(500) -> 2000x stratified bootstrap AUC CI.

Only new files are written: data/bace_rigorous_metrics.json.
Run: python3 -u bench_mol_bace.py
Self-check: python3 -u bench_mol_bace.py --selftest
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

DATA_PATH = Path("data/BACE.csv")
OUT_PATH = Path("data/bace_rigorous_metrics.json")
DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
MAIN_SEED = 42
N_BOOT = 2000
SEEDS = [42, 0, 7, 2024, 123]
DESC_NAMES = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "RingCount", "HeavyAtomCount",
]
FUSION = "ECFP4+MACCS+desc"
ECFP = "ECFP4-2048"

OFFICIAL_REFERENCE = {
    "paper": "Wu et al., MoleculeNet: A Benchmark for Molecular Machine Learning, arXiv:1703.00564v3",
    "verified_url": "https://arxiv.org/html/1703.00564v3#S6.T5",
    "abs_url": "https://arxiv.org/abs/1703.00564",
    "verification_date": "2026-11-24",
    "correct_table": "Table 5 (PCBA, MUV, HIV and BACE Performances), NOT Table 6; Table 6 is BBBP/Tox21/ToxCast/SIDER/ClinTox",
    "bace_scaffold_test_auc_mean_std": {
        "RF": {"mean": 0.867, "std": 0.008},
        "KernelSVM": {"mean": 0.862, "std": 0.000},
        "GC": {"mean": 0.783, "std": 0.014},
        "Weave": {"mean": 0.806, "std": 0.002},
    },
    "requested_mapping_correction": (
        "The requested mapping RF=0.783 / KernelSVM=0.867 / GraphConv=0.806 is wrong. "
        "Verified Table 5: RF=0.867±0.008, KernelSVM=0.862±0.000, GC=0.783±0.014, Weave=0.806±0.002. "
        "0.783 belongs to GC, 0.867 to RF, 0.806 to Weave; KernelSVM is 0.862."
    ),
    "caliber": (
        "Three runs after Gaussian-process hyperparameter optimization; scaffold split; "
        "AUC-ROC; non-graph models use RDKit ECFP4 by default. Our RF is fixed at 500 trees, "
        "no hyperparameter search, 80/10/10 DeepChem-style scaffold fill, canonical dedup, "
        "and 2223-dimensional fusion features; comparison is qualitative, not SOTA-equivalent."
    ),
}

BBBP_REFERENCE = {
    "source": "BME-DEEP-DIVE.md:73-76 and bench_mol_rigorous_v2.py",
    "ECFP4-2048": {"random": 0.9122, "scaffold": 0.9100},
    FUSION: {"random": 0.9288, "scaffold": 0.9413},
}


def canonicalize(df):
    """Return first canonical molecule per SMILES, with exact dedup accounting."""
    seen, mols, ys, rows, invalid, dup, conflicts = set(), [], [], [], 0, 0, 0
    label_by_canonical = {}
    for i, (raw, label) in enumerate(zip(df["mol"], df["Class"])):
        mol = Chem.MolFromSmiles(str(raw))
        if mol is None:
            invalid += 1
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            dup += 1
            if label_by_canonical[can] != int(label):
                conflicts += 1
            continue
        seen.add(can)
        label_by_canonical[can] = int(label)
        mols.append(mol)
        ys.append(int(label))
        rows.append(i)
    return mols, np.asarray(ys, dtype=int), rows, {
        "raw_rows": int(len(df)),
        "unique_valid": int(len(mols)),
        "invalid_dropped": int(invalid),
        "duplicate_dropped": int(dup),
        "conflicting_label_duplicates": int(conflicts),
    }


def scaffold_of(mol):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_groups(mols):
    groups = defaultdict(list)
    for i, mol in enumerate(mols):
        groups[scaffold_of(mol)].append(i)
    return groups


def deepchem_scaffold_split(mols, seed=MAIN_SEED):
    """Mirror bench_mol_rigorous_v2.py: shuffle items, stable sort by descending size."""
    rng = np.random.default_rng(seed)
    items = list(scaffold_groups(mols).items())
    rng.shuffle(items)
    items.sort(key=lambda kv: -len(kv[1]))
    n = len(mols)
    train_cutoff, valid_cutoff = 0.8 * n, 0.9 * n
    train, valid, test = [], [], []
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


def group_overlap(mols, a, b):
    sa = {scaffold_of(mols[i]) for i in a}
    sb = {scaffold_of(mols[i]) for i in b}
    return len(sa & sb)


def assert_split_ok(mols, split, require_nonempty=True):
    parts = ("train", "valid", "test")
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            assert group_overlap(mols, split[parts[i]], split[parts[j]]) == 0
    assert len(set(split["train"]) | set(split["valid"]) | set(split["test"])) == len(mols)
    if require_nonempty:
        assert min(len(split[p]) for p in parts) > 0


def fingerprints(mols):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ecfp = np.asarray([gen.GetFingerprint(m) for m in mols], dtype=np.int8)
    maccs = np.asarray([MACCSkeys.GenMACCSKeys(m) for m in mols], dtype=np.int8)
    desc = np.asarray([[float(getattr(Descriptors, name)(m)) for name in DESC_NAMES] for m in mols])
    assert np.isfinite(desc).all()
    # Same protocol as BBBP v2: fit the 8-descriptor StandardScaler on the full
    # dataset before splitting. This is recorded as a limitation, not hidden.
    desc = StandardScaler().fit_transform(desc)
    return {
        ECFP: ecfp,
        "MACCS-167": maccs,
        "ECFP4+desc": np.hstack([ecfp, desc]),
        FUSION: np.hstack([ecfp, maccs, desc]),
    }


def stratified_boot_indices(y, n_boot=N_BOOT, seed=MAIN_SEED):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return pos[rng.integers(0, len(pos), (n_boot, len(pos)))], neg[rng.integers(0, len(neg), (n_boot, len(neg)))]


def auc_from_indices(score, pos, neg):
    s = np.asarray(score)
    sp, sn = s[pos][:, None], s[neg][None, :]
    return (sp > sn).mean() + 0.5 * (sp == sn).mean()


def bootstrap_ci(score, boot):
    vals = np.asarray([auc_from_indices(score, boot[0][b], boot[1][b]) for b in range(len(boot[0]))])
    return {
        "lo": float(np.percentile(vals, 2.5)),
        "hi": float(np.percentile(vals, 97.5)),
        "mean": float(vals.mean()),
    }


def paired_delta_ci(score_a, score_b, boot):
    vals = np.asarray([
        auc_from_indices(score_a, boot[0][b], boot[1][b])
        - auc_from_indices(score_b, boot[0][b], boot[1][b])
        for b in range(len(boot[0]))
    ])
    return {"delta": float(np.mean(vals)), "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5))}


def fit_predict(X, y, train, test, seed=MAIN_SEED):
    clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    clf.fit(X[train], y[train])
    score = clf.predict_proba(X[test])[:, 1]
    assert np.isfinite(score).all()
    return score


def split_stats(y, split):
    return {
        name: {
            "n": int(len(idx)),
            "pos": int(y[idx].sum()),
            "neg": int(len(idx) - y[idx].sum()),
            "positive_rate": float(y[idx].mean()),
        }
        for name, idx in split.items()
    }


def selftest():
    smiles = [
        "c1ccccc1", "c1ccccc1C", "c1ccccc1CC", "C1CCCCC1", "CC1CCCCC1",
        "c1ccncc1", "CCO", "CCN", "CCCCO", "CCCCN", "not-a-smiles", "c1ccccc1",
    ]
    df = pd.DataFrame({"mol": smiles, "Class": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]})
    mols, ys, rows, stats = canonicalize(df)
    assert stats == {"raw_rows": 12, "unique_valid": 10, "invalid_dropped": 1,
                     "duplicate_dropped": 1, "conflicting_label_duplicates": 1}, stats
    split = deepchem_scaffold_split(mols)
    assert_split_ok(mols, split, require_nonempty=False)
    assert group_overlap(mols, [0], [0]) == 1  # deliberately overlapping input must be detected
    assert y_split_assertion()
    rng = np.random.default_rng(0)
    ysm = np.r_[np.ones(12, dtype=int), np.zeros(12, dtype=int)]
    sc = rng.random(24)
    boot = stratified_boot_indices(ysm, n_boot=50, seed=0)
    ci = bootstrap_ci(sc, boot)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    d = paired_delta_ci(sc, sc, boot)
    assert abs(d["delta"]) < 1e-12
    print("SELFTEST PASS: canonical dedup, invalid drop, label conflict, zero-overlap split, overlap detector, bootstrap CI", flush=True)


def y_split_assertion():
    mols = [Chem.MolFromSmiles(s) for s in ["c1ccccc1", "c1ccccc1C", "CCO", "CCN", "CCCCO", "CCCCN"]]
    split = deepchem_scaffold_split(mols)
    assert_split_ok(mols, split, require_nonempty=False)
    return True


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"missing {DATA_PATH}; download from {DATA_URL}")
    raw = DATA_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    df = pd.read_csv(DATA_PATH)
    assert {"mol", "Class"}.issubset(df.columns)

    mols, y, source_rows, dedup = canonicalize(df)
    assert dedup["unique_valid"] == len(mols)
    assert set(np.unique(y)).issubset({0, 1})
    print(f"dedup: {dedup}; labels={dict(zip(*np.unique(y, return_counts=True)))}", flush=True)

    reps = fingerprints(mols)
    print(f"features: {[(k, int(v.shape[1])) for k, v in reps.items()]}", flush=True)

    scaffold = deepchem_scaffold_split(mols, MAIN_SEED)
    assert_split_ok(mols, scaffold)
    groups = scaffold_groups(mols)
    print(f"scaffold split: {len(groups)} BM scaffolds; sizes={split_stats(y, scaffold)}", flush=True)

    random_split = {
        "train": np.array(train_test_split(np.arange(len(y)), test_size=0.2,
                                           stratify=y, random_state=MAIN_SEED)[0]),
        "test": np.array(train_test_split(np.arange(len(y)), test_size=0.2,
                                          stratify=y, random_state=MAIN_SEED)[1]),
    }
    assert set(random_split["train"]).isdisjoint(random_split["test"])

    scores = {}
    results = {}
    boots = {
        "random": stratified_boot_indices(y[random_split["test"]]),
        "scaffold": stratified_boot_indices(y[scaffold["test"]]),
    }
    for split_name, split in (("random", random_split), ("scaffold", scaffold)):
        scores[split_name] = {}
        results[split_name] = {}
        for name, X in reps.items():
            proba = fit_predict(X, y, split["train"], split["test"], MAIN_SEED)
            scores[split_name][name] = proba
            auc = float(roc_auc_score(y[split["test"]], proba))
            ci = bootstrap_ci(proba, boots[split_name])
            results[split_name][name] = {"auc": auc, "ci95": ci, "dim": int(X.shape[1])}
            print(f"{split_name:8s} {name:18s} AUC={auc:.4f} [{ci['lo']:.3f},{ci['hi']:.3f}]", flush=True)

    delta = {}
    for split_name in ("random", "scaffold"):
        delta[split_name] = paired_delta_ci(
            scores[split_name][FUSION], scores[split_name][ECFP], boots[split_name],
        )
        print(f"{split_name:8s} paired fusion-minus-ECFP4 delta="
              f"{delta[split_name]['delta']:+.4f} [{delta[split_name]['lo']:+.3f},{delta[split_name]['hi']:+.3f}]", flush=True)

    stability = {"seeds": SEEDS, "split_and_model_seed": [], "fixed_split_model_seed_only": []}
    fixed_scores = {}
    for seed in SEEDS:
        s = deepchem_scaffold_split(mols, seed)
        assert_split_ok(mols, s)
        proba = fit_predict(reps[FUSION], y, s["train"], s["test"], seed)
        auc = float(roc_auc_score(y[s["test"]], proba))
        stability["split_and_model_seed"].append({
            "seed": seed, "auc": auc, "sizes": split_stats(y, s),
        })
        p2 = fit_predict(reps[FUSION], y, scaffold["train"], scaffold["test"], seed)
        a2 = float(roc_auc_score(y[scaffold["test"]], p2))
        fixed_scores[seed] = p2
        stability["fixed_split_model_seed_only"].append({"seed": seed, "auc": a2})
    for key in ("split_and_model_seed", "fixed_split_model_seed_only"):
        vals = np.asarray([r["auc"] for r in stability[key]])
        stability[key + "_summary"] = {
            "mean": float(vals.mean()), "std_sample": float(vals.std(ddof=1)),
            "min": float(vals.min()), "max": float(vals.max()),
        }
        print(f"stability {key}: mean={vals.mean():.4f} std={vals.std(ddof=1):.4f} "
              f"range=[{vals.min():.4f},{vals.max():.4f}]", flush=True)

    fusion_gain = {
        split_name: {
            "bace": results[split_name][FUSION]["auc"] - results[split_name][ECFP]["auc"],
            "bbbp": BBBP_REFERENCE[FUSION][split_name] - BBBP_REFERENCE[ECFP][split_name],
        }
        for split_name in ("random", "scaffold")
    }
    cross_dataset_fusion_better = all(
        fusion_gain[s]["bace"] > 0 and fusion_gain[s]["bbbp"] > 0 for s in fusion_gain
    )
    official = OFFICIAL_REFERENCE["bace_scaffold_test_auc_mean_std"]
    official_comparison = {
        model: {
            "official_mean": v["mean"],
            "ours_fusion_auc": results["scaffold"][FUSION]["auc"],
            "ours_minus_official": results["scaffold"][FUSION]["auc"] - v["mean"],
        }
        for model, v in official.items()
    }

    out = {
        "task": "BACE external validation of BBBP v2 molecular benchmark protocol",
        "protocol": {
            "canonicalization": "RDKit MolFromSmiles -> MolToSmiles, keep first occurrence; invalid dropped",
            "split": "DeepChem-style Bemis-Murcko scaffold split: shuffle scaffold groups with seed, stable sort by descending group size, fill 80/10/10 train/valid/test; verify all pairwise scaffold overlaps are zero",
            "random_split": "train_test_split(test_size=0.2, stratify=y, random_state=42)",
            "descriptors": DESC_NAMES,
            "descriptor_scaler": "StandardScaler fit on full deduplicated dataset before splitting, exactly as BBBP v2; strict train-only scaling was not used and is a limitation",
            "model": "RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight=None)",
            "class_weight_reason": "BACE labels 822 negative / 691 positive (54.5%/45.8%); mild imbalance, so default None retained to match BBBP v2",
            "bootstrap": "2000 stratified resamples of the held-out test set (positives and negatives separately), percentile 95% CI; fixed bootstrap seed 42",
            "seed_stability": "Five seeds 42/0/7/2024/123: (a) rerun scaffold tie-shuffle and RF random_state; (b) fixed scaffold split with RF random_state only",
        },
        "data": {
            "url": DATA_URL,
            "requested_uppercase_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BACE.csv",
            "requested_uppercase_http_status": 403,
            "working_url_status": 200,
            "path": str(DATA_PATH),
            "bytes": len(raw),
            "sha256": digest,
            "raw_columns": int(df.shape[1]),
            "key_columns": ["mol", "Class"],
            "dedup": dedup,
            "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        },
        "environment": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "scaffold_split": {
            "unique_scaffolds": int(len(groups)),
            "largest_scaffold_size": int(max(len(v) for v in groups.values())),
            "sizes": split_stats(y, scaffold),
            "pairwise_overlaps": {
                "train_valid": group_overlap(mols, scaffold["train"], scaffold["valid"]),
                "train_test": group_overlap(mols, scaffold["train"], scaffold["test"]),
                "valid_test": group_overlap(mols, scaffold["valid"], scaffold["test"]),
            },
        },
        "random_split_sizes": split_stats(y, random_split),
        "results": results,
        "paired_bootstrap_fusion_minus_ecfp4": delta,
        "stability": stability,
        "bbbp_reference": BBBP_REFERENCE,
        "fusion_gain_vs_ecfp4": fusion_gain,
        "cross_dataset_fusion_better_than_ecfp4_all_splits": bool(cross_dataset_fusion_better),
        "official_reference": OFFICIAL_REFERENCE,
        "official_comparison_fusion_scaffold": official_comparison,
        "honest_caliber_conclusion": (
            "Our scaffold AUC is compared with MoleculeNet Table 5 (not Table 6), where RF=0.867±0.008, "
            "KernelSVM=0.862±0.000, GC=0.783±0.014, Weave=0.806±0.002. Different feature dimension, "
            "canonical dedup, fixed 500-tree RF without GP hyperparameter search, and split implementation "
            "mean this is a qualitative external-validation reference, not an apples-to-apples SOTA claim. "
            "The requested model-to-number mapping was falsified by the fetched source."
        ),
        "limitations": [
            "StandardScaler was fitted on the full dataset before splitting, matching BBBP v2 but not strict train-only preprocessing.",
            "MoleculeNet's exact scaffold-split implementation and fold membership were not re-run; DeepChem-style group filling is protocol-equivalent to BBBP v2, not byte-identical to DeepChem's historical loader.",
            "The 95% CIs quantify test-set sampling variance only, not split variability or model-selection uncertainty.",
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        main()
