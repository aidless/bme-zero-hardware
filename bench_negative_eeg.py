#!/usr/bin/env python3
"""EEGBCI 标签置换阴性对照（第九轮，A 臂 CSP+LR）。

复用 bench_eeg_extended.py 的 load_subjects/build_model/metrics，不复制数据加载
与预处理逻辑。epochs、subject-specific 5-fold 划分只按真实标签计算一次；
每次置换只打乱单个被试的 45 个标签，重跑 CSP fit + StandardScaler + LR。

Run:
  python3 bench_negative_eeg.py --selftest
  python3 bench_negative_eeg.py
  python3 bench_negative_eeg.py --verify-json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from bench_eeg_extended import (
    BASELINE, H_FREQ, L_FREQ, MNE_DIR, N_BOOT, N_SPLITS, RUNS, SEED,
    SUBJECTS, T_MAX, T_MIN, bootstrap_ci_single, build_model, load_subjects, metrics, sha256_file,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path.cwd()
OUT_JSON = ROOT / "data" / "negative" / "negative_eeg_metrics.json"
N_PERM = 100
PERM_SEED_STRIDE = 1000
BOOT_SEED = 42


def permute_labels(y: np.ndarray, seed: int) -> np.ndarray:
    """Full-vector permutation; label multiset is invariant."""
    return np.random.default_rng(seed).permutation(np.asarray(y))


def fixed_folds(ys: dict) -> dict:
    """Fold indices are derived from REAL labels exactly once."""
    folds = {}
    for s in SUBJECTS:
        y = ys[s]
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds[str(s)] = [
            {"train": [int(v) for v in tr], "valid": [int(v) for v in va]}
            for tr, va in cv.split(np.zeros((len(y), 1)), y)
        ]
    return folds


def fit_oof(X: np.ndarray, y: np.ndarray, folds: list) -> tuple[np.ndarray, list]:
    """Same CSP+LR arm as bench_eeg_extended.py subject_specific(); CSP refit per fold."""
    prob = np.zeros(len(y), dtype=float)
    fold_rows = []
    for fold_no, split in enumerate(folds, start=1):
        tr = np.asarray(split["train"])
        va = np.asarray(split["valid"])
        if len(np.unique(y[tr])) < 2:
            raise RuntimeError(f"permutation produced single-class fold train set: fold={fold_no}")
        pipe = build_model("csp_lr", SEED + fold_no)
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[va])[:, 1]
        prob[va] = p
        row = metrics(y[va], p)
        row.update({"fold": fold_no, "n_train": int(len(tr)), "n_valid": int(len(va))})
        fold_rows.append(row)
    return prob, fold_rows


def macro_from_subjects(rows: dict, key: str) -> float:
    return float(np.mean([rows[str(s)][key] for s in SUBJECTS]))


def macro_bootstrap_ci(values_by_subject: list, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Subject-level percentile bootstrap for the macro metric (9 subjects, resampled with replacement)."""
    v = np.asarray(values_by_subject, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boot = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return [float(lo), float(hi)]


def trial_bootstrap_ci(y: np.ndarray, prob: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Identical to bench_eeg_extended.bootstrap_ci_single (trial-level percentile bootstrap)."""
    return bootstrap_ci_single(y, prob, n_boot=n_boot, seed=seed)


def distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std_ddof1": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "n": int(len(values)),
    }


def selftest() -> None:
    y = np.asarray([1] * 13 + [2] * 32)
    yp = permute_labels(y, 1234)
    assert sorted(yp.tolist()) == sorted(y.tolist()), "permutation changed label multiset"
    assert np.array_equal(permute_labels(y, 1234), permute_labels(y, 1234)), "same seed not reproducible"
    assert not np.array_equal(permute_labels(y, 1234), permute_labels(y, 1235)), "different seeds collide"
    yy = np.asarray([1, 2, 1, 2])
    prob = np.asarray([0.9, 0.1, 0.4, 0.6])
    m = metrics(yy, prob)
    assert np.array_equal(np.where(prob >= 0.5, 2, 1), [2, 1, 1, 2])
    assert m["acc"] == 0.5 and m["auc"] == 0.25, m
    # Historical failure mode: truncating class-2 probabilities to int yields zeros.
    assert not np.array_equal(np.where(prob >= 0.5, 2, 1), prob.astype(int))
    print("SELFTEST PASS: multiset / reproducibility / {1,2} threshold mapping", flush=True)


def verify_json(path: Path) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    errs = []
    script = ROOT / d["artifacts"]["script"]
    actual_script_sha = sha256_file(script)
    if actual_script_sha != d["artifacts"]["script_sha256"]:
        errs.append(f"script sha256 mismatch: {actual_script_sha}")
    for rec in d["source_files"]:
        if sha256_file(ROOT / rec["path"]) != rec["sha256"]:
            errs.append(f"EDF sha256 mismatch: {rec['path']}")
    real = d["real"]["per_subject"]
    for key in ("acc", "auc"):
        recomputed = float(np.mean([real[str(s)][key] for s in SUBJECTS]))
        if abs(recomputed - d["real"]["macro"][key]) > 1e-12:
            errs.append(f"real macro {key}: {recomputed} != {d['real']['macro'][key]}")
    null_auc, null_acc = [], []
    fp_macro_auc = fp_macro_acc = fp_subject_auc = 0
    for row in d["permutations"]:
        for key, arr in (("acc", null_acc), ("auc", null_auc)):
            recomputed = float(np.mean([row["per_subject"][str(s)][key] for s in SUBJECTS]))
            arr.append(recomputed)
            if abs(recomputed - row["macro"][key]) > 1e-12:
                errs.append(f"perm {row['index']} macro {key} mismatch")
        fp_macro_auc += int(row["macro"]["bootstrap"]["auc"][0] > 0.5)
        fp_macro_acc += int(row["macro"]["bootstrap"]["acc"][0] > 0.5)
        for s in SUBJECTS:
            fp_subject_auc += int(row["per_subject"][str(s)]["bootstrap"]["auc_ci95"][0] > 0.5)
    n = len(null_auc)
    if n < 100:
        errs.append(f"N={n} below protocol minimum 100")
    for key, arr in (("auc", null_auc), ("acc", null_acc)):
        want = distribution(np.asarray(arr))
        got = d["null_distribution"][key]
        for stat in ("mean", "std_ddof1", "median", "p95", "max", "min", "n"):
            if abs(want[stat] - got[stat]) > 1e-12:
                errs.append(f"null {key}.{stat}: {want[stat]} != {got[stat]}")
        want_p = (float(np.sum(np.asarray(arr) >= d["real"]["macro"][key])) + 1) / (n + 1)
        if abs(want_p - d["empirical_tests"][key]["p"]) > 1e-12:
            errs.append(f"empirical p {key}: {want_p} != {d['empirical_tests'][key]['p']}")
    fp = d["false_positive_counts"]
    for key, val in (("macro_auc_permutations", fp_macro_auc),
                     ("macro_acc_permutations", fp_macro_acc),
                     ("subject_auc_permutation_pairs", fp_subject_auc)):
        if fp[key] != val:
            errs.append(f"false positive {key}: {fp[key]} != {val}")
    if errs:
        print("VERIFY FAIL:\n" + "\n".join(errs), flush=True)
        raise SystemExit(2)
    print(f"VERIFY PASS: {path} | N={n} | script+27 EDF sha256 | "
          "null stats / empirical p / false-positive counts recomputed", flush=True)


def main(n_perm: int, output: Path) -> None:
    if n_perm < 100:
        raise SystemExit(f"protocol requires N>=100, got {n_perm}")
    t0 = time.time()
    print(f"[setup] N={n_perm} seeds={SEED}+perm*{PERM_SEED_STRIDE}+subject", flush=True)
    Xs, ys, files, subject_meta = load_subjects()
    folds = fixed_folds(ys)
    assert len(files) == 27

    real_rows = {}
    real_folds = {}
    print("[real] refit CSP+LR on unshuffled labels", flush=True)
    for s in SUBJECTS:
        prob, fold_rows = fit_oof(Xs[s], ys[s], folds[str(s)])
        row = metrics(ys[s], prob)
        real_rows[str(s)] = row
        real_folds[str(s)] = fold_rows
        print(f"[real] S{s:03d} acc={row['acc']:.6f} auc={row['auc']:.6f}", flush=True)
    real_macro = {key: macro_from_subjects(real_rows, key) for key in ("acc", "auc")}
    real_boot = {
        key: macro_bootstrap_ci([real_rows[str(s)][key] for s in SUBJECTS], seed=BOOT_SEED + i)
        for i, key in enumerate(("acc", "auc"))
    }
    print(f"[real] macro acc={real_macro['acc']:.6f} auc={real_macro['auc']:.6f}", flush=True)

    permutations = []
    for i in range(n_perm):
        perm_rows = {}
        for s in SUBJECTS:
            seed = SEED + i * PERM_SEED_STRIDE + s
            yp = permute_labels(ys[s], seed)
            assert sorted(yp.tolist()) == sorted(ys[s].tolist())
            prob, _ = fit_oof(Xs[s], yp, folds[str(s)])
            row = metrics(yp, prob)
            row["bootstrap"] = trial_bootstrap_ci(yp, prob)
            perm_rows[str(s)] = row
        macro = {key: macro_from_subjects(perm_rows, key) for key in ("acc", "auc")}
        macro["bootstrap"] = {
            key: macro_bootstrap_ci(
                [perm_rows[str(s)][key] for s in SUBJECTS], seed=BOOT_SEED + 10000 * i + k
            )
            for k, key in enumerate(("acc", "auc"))
        }
        permutations.append({
            "index": i + 1,
            "permutation_seeds": {str(s): SEED + i * PERM_SEED_STRIDE + s for s in SUBJECTS},
            "per_subject": perm_rows,
            "macro": macro,
        })
        if (i + 1) % 5 == 0 or i + 1 == n_perm:
            auc_so_far = np.asarray([p["macro"]["auc"] for p in permutations])
            print(f"[perm] {i+1}/{n_perm} macro_auc={macro['auc']:.6f} "
                  f"running_mean={auc_so_far.mean():.6f} elapsed={time.time()-t0:.1f}s", flush=True)

    null = {
        "auc": distribution(np.asarray([p["macro"]["auc"] for p in permutations])),
        "acc": distribution(np.asarray([p["macro"]["acc"] for p in permutations])),
    }
    empirical_tests = {}
    for key in ("acc", "auc"):
        arr = np.asarray([p["macro"][key] for p in permutations])
        ge = int(np.sum(arr >= real_macro[key]))
        empirical_tests[key] = {
            "definition": f"(null >= real + 1)/(N+1); null_ge_real={ge}",
            "null_ge_real": ge,
            "p": float((ge + 1) / (n_perm + 1)),
            "significant_at_0.01": bool((ge + 1) / (n_perm + 1) <= 0.01),
        }
    fp_counts = {
        "definition": "percentile bootstrap 95% CI lower bound strictly > 0.5 under shuffled labels",
        "macro_auc_permutations": int(sum(
            p["macro"]["bootstrap"]["auc"][0] > 0.5 for p in permutations)),
        "macro_acc_permutations": int(sum(
            p["macro"]["bootstrap"]["acc"][0] > 0.5 for p in permutations)),
        "subject_auc_permutation_pairs": int(sum(
            r["bootstrap"]["auc_ci95"][0] > 0.5            for p in permutations for r in p["per_subject"].values())),
        "n_macro_tests": n_perm,
        "n_subject_tests": n_perm * len(SUBJECTS),
        "nominal_alpha": 0.05,
        "nominal_alpha_times_n_macro": 0.05 * n_perm,
    }
    auc_mean_in_band = abs(null["auc"]["mean"] - 0.5) <= 0.05
    auc_p_ok = empirical_tests["auc"]["significant_at_0.01"]
    verdict = {
        "null_auc_mean_in_0.5_pm_0.05": bool(auc_mean_in_band),
        "real_auc_empirical_p_le_0.01": bool(auc_p_ok),
        "pipeline_clean": bool(auc_mean_in_band and auc_p_ok),
        "note": "verdict follows protocol criterion on AUC; acc is reported as secondary evidence",
    }
    reference_json = ROOT / "data" / "eeg" / "eeg_extended_metrics.json"
    reference = json.loads(reference_json.read_text(encoding="utf-8"))
    ref_macro = reference["pooled_loso"]["csp_lr"]
    ref_acc = ref_macro["macro_subject_specific_acc"]
    ref_auc = ref_macro["macro_subject_specific_auc"]

    payload = {
        "experiment": "EEGBCI negative control: label-permutation test of subject-specific CSP+LR",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "protocol": {
            "source_protocol": "notes/2026-11-26-negative-control-protocol.md",
            "reused_arm": "bench_eeg_extended.py A arm / csp_lr (imported build_model, load_subjects, metrics)",
            "subjects": SUBJECTS,
            "runs": RUNS,
            "n_edf": len(files),
            "labels": "T1=1(left), T2=2(right)",
            "filter": f"FIR {L_FREQ}-{H_FREQ} Hz (firwin)",
            "epoch_window_s": [T_MIN, T_MAX],
            "baseline_s": list(BASELINE),
            "n_channels": 64,
            "sfreq_hz": 160.0,
            "model": "CSP(4, ledoit_wolf, log, average_power) + StandardScaler + LogisticRegression(C=1, max_iter=2000)",
            "cv": f"StratifiedKFold({N_SPLITS}, shuffle=True, random_state={SEED}) computed ONCE on real labels",
            "permutation_scope": "each subject's full 45-label vector shuffled; X and fold indices untouched",
            "permutation_seed_formula": f"{SEED} + permutation_index(0-based)*{PERM_SEED_STRIDE} + subject",
            "n_permutations": n_perm,
            "label_threshold_rule": "np.where(prob >= 0.5, 2, 1) with y in {1,2}",
            "macro_definition": "unweighted mean of 9 subject-specific OOF metrics",
            "bootstrap": {
                "macro": f"subject-level percentile bootstrap, {N_BOOT} resamples of 9 subject values, 2.5/97.5",
                "subject": f"trial-level percentile bootstrap reused from bench_eeg_extended.bootstrap_ci_single, {N_BOOT} resamples, seed {BOOT_SEED}",
            },
            "device": "CPU only (OMP/OPENBLAS/MKL/NUMEXPR threads=1)",
            "small_sample_note": "45 trials/subject; subject-level null variance is expected to be large and is reported, not hidden",
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "mne": __import__("mne").__version__,
            "numpy": np.__version__, "scipy": __import__("scipy").__version__,
            "sklearn": __import__("sklearn").__version__,
            "pyriemann": __import__("pyriemann").__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifacts": {
            "script": "bench_negative_eeg.py",
            "script_sha256": sha256_file(Path(__file__)),
            "metrics": str(output.relative_to(ROOT)),
            "reused_script": "bench_eeg_extended.py",
            "reference_metrics": "data/eeg/eeg_extended_metrics.json",
        },
        "reference_metrics": {
            "prior_csp_lr_macro_acc": ref_acc,
            "prior_csp_lr_macro_auc": ref_auc,
            "remeasured_acc": real_macro["acc"],
            "remeasured_auc": real_macro["auc"],
            "abs_diff_acc": abs(real_macro["acc"] - ref_acc),
            "abs_diff_auc": abs(real_macro["auc"] - ref_auc),
        },
        "source_files": files,
        "subjects": subject_meta,
        "fold_indices_from_real_labels": folds,
        "real": {
            "per_subject": real_rows,
            "per_subject_folds": real_folds,
            "macro": real_macro,
            "macro_bootstrap_ci95": real_boot,
        },
        "null_distribution": null,
        "empirical_tests": empirical_tests,
        "false_positive_counts": fp_counts,
        "verdict": verdict,
        "permutations": permutations,
        "elapsed_s": float(time.time() - t0),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"[artifact] {output.relative_to(ROOT)}", flush=True)
    print(f"[done] elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--verify-json", action="store_true")
    p.add_argument("--n-permutations", type=int, default=N_PERM)
    p.add_argument("--output", type=Path, default=OUT_JSON)
    args = p.parse_args()
    if args.selftest:
        selftest()
    elif args.verify_json:
        verify_json(args.output)
    else:
        # 相对路径按 cwd 解析为绝对路径后再做 relative_to（否则 ValueError）
        out_path = args.output if args.output.is_absolute() else (ROOT / args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        main(args.n_permutations, out_path)
