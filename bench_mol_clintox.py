"""ClinTox third-dataset check for molecular fusion gain.

Primary endpoint: CT_TOX (clinical toxicity; 1 = positive). FDA_APPROVED is a
separate regulatory-status endpoint and is not the toxicity target.

Only new files are written: data/clintox_metrics.json.
Run: python3 -u bench_mol_clintox.py
Self-check: python3 -u bench_mol_clintox.py --selftest
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import RDLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from bench_mol_trainonly import (
    MAIN_SEED, SEEDS, N_BOOT, FUSION, canonicalize, deepchem_scaffold_split,
    raw_features, build_reps, fit_predict, split_stats, split_ok,
    stratified_boot_indices, bootstrap_ci, auc_from_indices,
)

RDLogger.DisableLog("rdApp.*")

DATA_PATH = Path("data/ClinTox.csv.gz")
OUT_PATH = Path("data/clintox_metrics.json")
SOURCE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
REPS = ["ECFP4-2048", "MACCS", "ECFP4+desc", FUSION]
LOG = []


def emit(msg):
    print(msg, flush=True)
    LOG.append(str(msg))


def stratified_random_split(y):
    """Conventional stratified 80/10/10 sensitivity split."""
    idx = np.arange(len(y))
    train_tmp, test = train_test_split(idx, test_size=0.1, stratify=y,
                                       random_state=MAIN_SEED)
    # Split 80/10/10: test is 10% of all rows; valid is 10% of all rows,
    # i.e. (n_test / n_remaining) of the post-test remainder.
    valid_frac = len(test) / len(train_tmp)
    train, valid = train_test_split(train_tmp, test_size=valid_frac, stratify=y[train_tmp],
                                    random_state=MAIN_SEED)
    split = {"train": np.asarray(train), "valid": np.asarray(valid),
             "test": np.asarray(test)}
    assert len(set(split["train"]) | set(split["valid"]) | set(split["test"])) == len(y)
    assert not (set(split["train"]) & set(split["valid"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["valid"]) & set(split["test"]))
    return split


def make_reps(raw, train_idx):
    reps, scaler = build_reps(raw, train_idx, "train_only")
    # Insert MACCS-only alongside the three baseline representations.
    return {
        "ECFP4-2048": reps["ECFP4-2048"],
        "MACCS": raw["maccs"],
        "ECFP4+desc": reps["ECFP4+desc"],
        FUSION: reps[FUSION],
    }, scaler


def paired_delta_ci_multi(scores_a, scores_b, boot):
    assert scores_a.shape == scores_b.shape
    vals = np.asarray([
        np.mean([
            auc_from_indices(scores_a[s], boot[0][b], boot[1][b])
            - auc_from_indices(scores_b[s], boot[0][b], boot[1][b])
            for s in range(len(scores_a))
        ])
        for b in range(len(boot[0]))
    ])
    return {
        "delta_mean": float(vals.mean()),
        "ci95": {"lo": float(np.percentile(vals, 2.5)),
                 "hi": float(np.percentile(vals, 97.5))},
        "significant_at_0_05": bool(np.percentile(vals, 97.5) < 0 or np.percentile(vals, 2.5) > 0),
    }


def selftest():
    y = np.asarray([0] * 90 + [1] * 10)
    split = stratified_random_split(y)
    for name, idx in split.items():
        assert len(idx) > 0 and 0 < y[idx].sum() < len(idx), (name, y[idx].sum(), len(idx))
    assert len(set(split["train"]) | set(split["valid"]) | set(split["test"])) == len(y)
    print("SELFTEST PASS: stratified split is disjoint, exhaustive and class-preserving", flush=True)


def main():
    df = pd.read_csv(DATA_PATH)
    assert {"smiles", "FDA_APPROVED", "CT_TOX"} <= set(df.columns)
    # CT_TOX is the only requested toxicity-like endpoint; FDA_APPROVED is regulatory status.
    mols, y, stats = canonicalize(df, "smiles", "CT_TOX")
    assert set(np.unique(y)) == {0, 1}
    raw = raw_features(mols)
    scaffold = deepchem_scaffold_split(mols, MAIN_SEED)
    split_ok(mols, scaffold)
    stratified = stratified_random_split(y)
    splits = {
        "scaffold_80_10_10": scaffold,
        "stratified_random_80_10_10_sensitivity": stratified,
    }

    overall = {
        "n": int(len(y)), "pos": int(y.sum()), "neg": int(len(y) - y.sum()),
        "positive_rate": float(y.mean()),
    }
    emit(f"ClinTox CT_TOX unique valid={len(y)} pos={int(y.sum())} neg={int(len(y)-y.sum())} pos_rate={y.mean():.6f}")
    results, paired = {}, {}
    for split_name, split in splits.items():
        emit(f"--- {split_name} ---")
        emit("split_stats=" + json.dumps(split_stats(y, split), sort_keys=True))
        y_test = y[split["test"]]
        boot = stratified_boot_indices(y_test, N_BOOT, MAIN_SEED)
        reps, _ = make_reps(raw, split["train"])
        split_results = {}
        all_scores = {}
        for rep_name in REPS:
            scores = np.asarray([
                fit_predict(reps[rep_name], y, split["train"], split["test"], seed)
                for seed in SEEDS
            ])
            all_scores[rep_name] = scores
            aucs = [float(roc_auc_score(y_test, s)) for s in scores]
            ci = bootstrap_ci(scores[0], boot)
            a = np.asarray(aucs)
            split_results[rep_name] = {
                "feature_dim": int(reps[rep_name].shape[1]),
                "aucs_by_seed": [{"seed": int(s), "auc": float(v)} for s, v in zip(SEEDS, aucs)],
                "auc_mean": float(a.mean()), "auc_std_sample": float(a.std(ddof=1)),
                "auc_seed42": float(aucs[0]), "ci95_seed42": ci,
            }
            emit(f"{rep_name:18s} mean={a.mean():.4f} std={a.std(ddof=1):.4f} seed42={aucs[0]:.4f} [{ci['lo']:.3f},{ci['hi']:.3f}]")
        fusion = all_scores[FUSION]
        paired[split_name] = {}
        for other in REPS:
            if other == FUSION:
                continue
            delta = paired_delta_ci_multi(fusion, all_scores[other], boot)
            delta["comparison"] = f"{FUSION} - {other}"
            paired[split_name][other] = delta
            emit(f"fusion - {other:18s} delta={delta['delta_mean']:+.4f} CI=[{delta['ci95']['lo']:+.4f},{delta['ci95']['hi']:+.4f}] significant={delta['significant_at_0_05']}")
        results[split_name] = {
            "split_stats": split_stats(y, split),
            "test_bootstrap": {"n_boot": N_BOOT, "seed": MAIN_SEED, "stratified_by_label": True},
            "representations": split_results,
        }

    primary = paired["scaffold_80_10_10"]
    fusion_positive_all = all(v["delta_mean"] > 0 for v in primary.values())
    fusion_significant_any = any(v["significant_at_0_05"] for v in primary.values())
    conclusion = {
        "fusion_positive_vs_all_comparators": bool(fusion_positive_all),
        "fusion_significant_vs_any_comparator": bool(fusion_significant_any),
        "primary_vs_ecfp4": primary["ECFP4-2048"],
        "primary_vs_ecfp4_plus_desc": primary["ECFP4+desc"],
        "text": (
            "On the primary ClinTox scaffold split, fusion is assessed against all three other representations. "
            "Significance is a paired bootstrap CI over 2000 stratified test-set resamples, averaged across the five RF seeds."
        ),
    }
    out = {
        "task": "Third-dataset fusion-gain check",
        "endpoint": {
            "name": "CT_TOX", "meaning": "clinical toxicity", "positive_label": 1,
            "reason": "The downloaded ClinTox file has FDA_APPROVED and CT_TOX, not HIV; CT_TOX is the toxicity endpoint.",
        },
        "source": {"url": SOURCE_URL, "path": str(DATA_PATH),
                   "gzip_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()},
        "data": {
            "columns": list(df.columns), "raw_rows": int(len(df)),
            "canonicalization": stats, "overall": overall,
            "FDA_APPROVED_counts_not_used": {str(k): int(v) for k, v in df["FDA_APPROVED"].value_counts().sort_index().items()},
        },
        "protocol": {
            "primary_split": "DeepChem-style scaffold 80/10/10 with shuffled equal-size scaffold groups, seed 42",
            "sensitivity_split": "Stratified random 80/10/10, seed 42",
            "scaler": "Strict train-only StandardScaler fit on train fold for descriptor-containing representations",
            "model": "RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)",
            "seeds": SEEDS, "representations": REPS,
            "bootstrap": "2000 stratified bootstrap resamples of held-out test labels; percentile 95% CI",
            "paired_delta": "Fusion minus comparator, paired by bootstrap resample and averaged across five RF seeds",
        },
        "results": results,
        "paired_fusion_gains": paired,
        "conclusion": conclusion,
        "environment": {
            "python": sys.version.split()[0], "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "platform": platform.platform(),
        },
        "run_log": list(LOG),
    }
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
