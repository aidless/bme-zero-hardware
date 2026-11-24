#!/usr/bin/env python3
"""EEGBCI external validation: subject-specific and leakage-free LOSO CSP+LR.

Protocol (fixed before execution):
  subjects 2/3/4/5, runs 4/8/12; 64 channels, 160 Hz;
  FIR 7--30 Hz; epochs -0.5--4.0 s; baseline -0.5--0 s;
  T1=left (1), T2=right (2); CSP(4, ledoit_wolf, log) + scaler + LR.
  Within-subject: stratified shuffled 5-fold, seed 42.
  LOSO: for each held-out subject, stratified shuffled 5-fold ONLY on the
  other four subjects. Each of the five fold models predicts the held-out
  subject; primary test metrics use the mean of those five probabilities.
  A separate model refit on all four training subjects is also evaluated.
  Pooled LOSO: concatenate the primary OOF predictions for all held-out
  subjects and bootstrap trials with replacement, 2000 resamples.

CSP, StandardScaler, and LogisticRegression live inside sklearn Pipeline,
so every fit operation is restricted to the corresponding training indices.
No test-subject trial participates in fitting, threshold selection, or CV.
All numerical results are computed at runtime and written to JSON.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy
import sklearn
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mne.set_log_level("error")

ROOT = Path.cwd()
DATA_DIR = ROOT / "data" / "eeg"
MNE_DIR = DATA_DIR / "mne_data"
OUT_JSON = DATA_DIR / "eeg_cross_subject_metrics.json"
OUT_FIG = ROOT / "figures" / "eeg_cross_subject.png"

SUBJECTS = [2, 3, 4, 5]
RUNS = [4, 8, 12]
L_FREQ, H_FREQ = 7.0, 30.0
T_MIN, T_MAX = -0.5, 4.0
BASELINE = (-0.5, 0.0)
N_SPLITS = 5
SEED = 42
N_BOOT = 2000
CSP_COMPONENTS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pipeline(seed: int = SEED):
    return make_pipeline(
        CSP(n_components=CSP_COMPONENTS, reg="ledoit_wolf", log=True,
            transform_into="average_power"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
    )


def metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "n_left": int(np.sum(y == 1)),
        "n_right": int(np.sum(y == 2)),
        "acc": float(accuracy_score(y, np.where(prob >= 0.5, 2, 1))),
        "auc": float(roc_auc_score(y, prob)),
    }


def bootstrap_ci(y: np.ndarray, prob: np.ndarray, n_boot: int = N_BOOT,
                 seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    accs, aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        pi = prob[idx]
        accs.append(accuracy_score(yi, np.where(pi >= 0.5, 2, 1)))
        aucs.append(roc_auc_score(yi, pi))
    return {
        "acc_ci95": [float(v) for v in np.percentile(accs, [2.5, 97.5])],
        "auc_ci95": [float(v) for v in np.percentile(aucs, [2.5, 97.5])],
        "n_boot_valid": len(accs),
        "method": "nonparametric trial-level percentile bootstrap; replacement; stratified by chance only",
    }


def load_subjects() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[dict], dict]:
    print("[download] EEGMMIDB v1.0.0 subjects 2/3/4/5, runs 4/8/12", flush=True)
    paths = eegbci.load_data(subjects=SUBJECTS, runs=RUNS, path=str(MNE_DIR),
                             update_path=False, verbose="error")
    paths = [Path(p) for p in paths]
    assert len(paths) == len(SUBJECTS) * len(RUNS), len(paths)
    files = []
    for i, path in enumerate(paths):
        subject = SUBJECTS[i // len(RUNS)]
        run = RUNS[i % len(RUNS)]
        expected = f"S{subject:03d}R{run:02d}.edf"
        assert path.name == expected, (path, expected)
        rec = {
            "subject": subject, "run": run, "path": str(path.relative_to(ROOT)),
            "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path),
        }
        files.append(rec)
        print(f"[sha256] S{subject:03d}R{run:02d} {rec['sha256']} "
              f"size={rec['size_bytes']}", flush=True)

    montage = make_standard_montage("standard_1020")
    Xs, ys = {}, {}
    subject_meta = {}
    for subject in SUBJECTS:
        xparts, yparts, run_counts = [], [], {}
        for run in RUNS:
            path = next(p for p in paths if p.name == f"S{subject:03d}R{run:02d}.edf")
            raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
            eegbci.standardize(raw)
            raw.set_montage(montage)
            assert len(raw.ch_names) == 64, raw.ch_names
            assert raw.info["sfreq"] == 160.0, raw.info["sfreq"]
            raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose="error")
            events, _ = mne.events_from_annotations(
                raw, event_id={"T1": 1, "T2": 2}, verbose="error")
            epochs = mne.Epochs(
                raw, events, event_id={"T1_left": 1, "T2_right": 2},
                tmin=T_MIN, tmax=T_MAX, baseline=BASELINE, preload=True,
                verbose="error")
            x, y = epochs.get_data(), epochs.events[:, -1]
            xparts.append(x)
            yparts.append(y)
            run_counts[str(run)] = {
                "n": int(len(y)), "n_left": int(np.sum(y == 1)),
                "n_right": int(np.sum(y == 2)),
            }
            print(f"[epoch] S{subject:03d} run{run}: n={len(y)} "
                  f"left={int(np.sum(y==1))} right={int(np.sum(y==2))} "
                  f"shape={x.shape}", flush=True)
        Xs[subject], ys[subject] = np.concatenate(xparts), np.concatenate(yparts)
        assert Xs[subject].shape == (len(ys[subject]), 64, 721), Xs[subject].shape
        assert set(np.unique(ys[subject])) == {1, 2}
        subject_meta[str(subject)] = {
            "n_epochs": int(len(ys[subject])), "n_left": int(np.sum(ys[subject] == 1)),
            "n_right": int(np.sum(ys[subject] == 2)), "runs": run_counts,
        }
    return Xs, ys, files, subject_meta


def subject_specific(Xs: dict[int, np.ndarray], ys: dict[int, np.ndarray]) -> dict:
    results, predictions = {}, {}
    for subject in SUBJECTS:
        X, y = Xs[subject], ys[subject]
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        prob = np.zeros(len(y), dtype=float)
        fold_rows = []
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            pipe = build_pipeline(SEED + fold)
            pipe.fit(X[tr], y[tr])
            p = pipe.predict_proba(X[va])[:, 1]
            prob[va] = p
            row = metrics(y[va], p)
            row.update({"fold": fold, "n_train": int(len(tr)), "n_valid": int(len(va))})
            fold_rows.append(row)
            print(f"[specific] S{subject:03d} fold{fold} "
                  f"acc={row['acc']:.4f} auc={row['auc']:.4f}", flush=True)
        result = metrics(y, prob)
        result.update({
            "folds": fold_rows,
            "fold_acc_mean": float(np.mean([r["acc"] for r in fold_rows])),
            "fold_acc_std": float(np.std([r["acc"] for r in fold_rows])),
            "fold_auc_mean": float(np.mean([r["auc"] for r in fold_rows])),
            "fold_auc_std": float(np.std([r["auc"] for r in fold_rows])),
            "bootstrap": bootstrap_ci(y, prob),
            "caliber": "within-subject stratified shuffled 5-fold OOF",
        })
        results[str(subject)] = result
        predictions[str(subject)] = {"y": y, "prob": prob}
        print(f"[specific] S{subject:03d} OOF acc={result['acc']:.4f} "
              f"auc={result['auc']:.4f} CI_acc={result['bootstrap']['acc_ci95']}", flush=True)
    return results, predictions


def loso(Xs: dict[int, np.ndarray], ys: dict[int, np.ndarray]) -> tuple[dict, dict]:
    results, predictions = {}, {}
    for heldout in SUBJECTS:
        train_subjects = [s for s in SUBJECTS if s != heldout]
        Xtrain = np.concatenate([Xs[s] for s in train_subjects])
        ytrain = np.concatenate([ys[s] for s in train_subjects])
        Xtest, ytest = Xs[heldout], ys[heldout]
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_prob = np.zeros((N_SPLITS, len(ytest)), dtype=float)
        fold_rows = []
        print(f"[LOSO] heldout=S{heldout:03d}, train={train_subjects}, "
              f"train_n={len(ytrain)}, test_n={len(ytest)}", flush=True)
        for fold, (tr, va) in enumerate(cv.split(Xtrain, ytrain)):
            pipe = build_pipeline(SEED + fold)
            # Fitting scope is only training-subject indices. The held-out
            # subject is not present in Xtrain/ytrain at all.
            pipe.fit(Xtrain[tr], ytrain[tr])
            pva = pipe.predict_proba(Xtrain[va])[:, 1]
            fold_prob[fold] = pipe.predict_proba(Xtest)[:, 1]
            row = metrics(ytrain[va], pva)
            row.update({
                "fold": fold, "n_train": int(len(tr)), "n_valid": int(len(va)),
                "test_metrics": metrics(ytest, fold_prob[fold]),
            })
            fold_rows.append(row)
            print(f"[LOSO] S{heldout:03d} fold{fold} "
                  f"train_valid_acc={row['acc']:.4f} "
                  f"test_acc={row['test_metrics']['acc']:.4f} "
                  f"test_auc={row['test_metrics']['auc']:.4f}", flush=True)
        ensemble_prob = fold_prob.mean(axis=0)
        primary = metrics(ytest, ensemble_prob)
        primary["bootstrap"] = bootstrap_ci(ytest, ensemble_prob)
        primary.update({
            "heldout_subject": heldout, "train_subjects": train_subjects,
            "n_train_subject_epochs": int(len(ytrain)),
            "n_test_epochs": int(len(ytest)), "train_cv": "stratified shuffled 5-fold on training subjects only",
            "prediction_rule": "arithmetic mean of predict_proba[:,1] from five training-only fold models",
            "fold_train_valid_acc_mean": float(np.mean([r["acc"] for r in fold_rows])),
            "fold_train_valid_acc_std": float(np.std([r["acc"] for r in fold_rows])),
            "fold_train_valid_auc_mean": float(np.mean([r["auc"] for r in fold_rows])),
            "fold_train_valid_auc_std": float(np.std([r["auc"] for r in fold_rows])),
            "fold_test_acc_mean": float(np.mean([r["test_metrics"]["acc"] for r in fold_rows])),
            "fold_test_acc_std": float(np.std([r["test_metrics"]["acc"] for r in fold_rows])),
            "fold_test_auc_mean": float(np.mean([r["test_metrics"]["auc"] for r in fold_rows])),
            "fold_test_auc_std": float(np.std([r["test_metrics"]["auc"] for r in fold_rows])),
            "folds": fold_rows,
        })
        # Secondary, independently fit model on all four training subjects.
        full_pipe = build_pipeline(SEED)
        full_pipe.fit(Xtrain, ytrain)
        full_prob = full_pipe.predict_proba(Xtest)[:, 1]
        primary["full_training_refit"] = metrics(ytest, full_prob)
        print(f"[LOSO] S{heldout:03d} ENSEMBLE acc={primary['acc']:.4f} "
              f"auc={primary['auc']:.4f}; FULL-REFIT acc={primary['full_training_refit']['acc']:.4f} "
              f"auc={primary['full_training_refit']['auc']:.4f}", flush=True)
        results[str(heldout)] = primary
        predictions[str(heldout)] = {
            "y": ytest, "prob": ensemble_prob, "fold_prob": fold_prob,
            "full_refit_prob": full_prob,
        }
    return results, predictions


def make_figure(specific: dict, loso_results: dict, pooled: dict) -> None:
    labels = [f"S{s:03d}" for s in SUBJECTS] + ["Pooled"]
    ss_acc = [specific[str(s)]["acc"] for s in SUBJECTS] + [
        np.mean([specific[str(s)]["acc"] for s in SUBJECTS])]
    lo_acc = [loso_results[str(s)]["acc"] for s in SUBJECTS] + [pooled["acc"]]
    ss_auc = [specific[str(s)]["auc"] for s in SUBJECTS] + [
        np.mean([specific[str(s)]["auc"] for s in SUBJECTS])]
    lo_auc = [loso_results[str(s)]["auc"] for s in SUBJECTS] + [pooled["auc"]]
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    for ax, a, b, title, ylabel in [
        (axes[0], ss_acc, lo_acc, "A. Accuracy", "Accuracy"),
        (axes[1], ss_auc, lo_auc, "B. ROC AUC", "ROC AUC"),
    ]:
        bars1 = ax.bar(x - width / 2, a, width, label="Subject-specific 5-fold",
                       color="#3976b8")
        bars2 = ax.bar(x + width / 2, b, width, label="Held-out subject LOSO",
                       color="#df7846")
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for bars in (bars1, bars2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("EEGBCI motor imagery: within-subject vs. cross-subject generalization\n"
                 "Subjects 2-5, runs 4/8/12 | CSP(4, Ledoit-Wolf, log) + StandardScaler + LR",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[artifact] {OUT_FIG.relative_to(ROOT)}", flush=True)


def main() -> None:
    t0 = time.time()
    print("=" * 80, flush=True)
    print("EEG CROSS-SUBJECT EXTERNAL VALIDATION (seed=42)", flush=True)
    print("=" * 80, flush=True)
    Xs, ys, files, subject_meta = load_subjects()
    specific, _ = subject_specific(Xs, ys)
    loso_results, loso_pred = loso(Xs, ys)
    pooled_y = np.concatenate([loso_pred[str(s)]["y"] for s in SUBJECTS])
    pooled_prob = np.concatenate([loso_pred[str(s)]["prob"] for s in SUBJECTS])
    pooled_full_prob = np.concatenate([loso_pred[str(s)]["full_refit_prob"] for s in SUBJECTS])
    pooled = metrics(pooled_y, pooled_prob)
    pooled["bootstrap"] = bootstrap_ci(pooled_y, pooled_prob)
    pooled["full_training_refit"] = metrics(pooled_y, pooled_full_prob)
    specific_acc_mean = float(np.mean([specific[str(s)]["acc"] for s in SUBJECTS]))
    specific_auc_mean = float(np.mean([specific[str(s)]["auc"] for s in SUBJECTS]))
    pooled["macro_subject_specific_acc"] = specific_acc_mean
    pooled["macro_subject_specific_auc"] = specific_auc_mean
    pooled["acc_drop_pp_vs_subject_specific"] = float(100 * (specific_acc_mean - pooled["acc"]))
    pooled["auc_drop_vs_subject_specific"] = float(specific_auc_mean - pooled["auc"])
    pooled["per_subject_acc_drop_pp"] = {
        str(s): float(100 * (specific[str(s)]["acc"] - loso_results[str(s)]["acc"]))
        for s in SUBJECTS
    }
    print(f"[pooled LOSO] acc={pooled['acc']:.6f} auc={pooled['auc']:.6f}", flush=True)
    print(f"[pooled LOSO] acc CI={pooled['bootstrap']['acc_ci95']} "
          f"auc CI={pooled['bootstrap']['auc_ci95']}", flush=True)
    print(f"[drop] acc={pooled['acc_drop_pp_vs_subject_specific']:.4f} pp; "
          f"auc={pooled['auc_drop_vs_subject_specific']:.4f}", flush=True)

    payload = {
        "experiment": "EEGBCI external cross-subject validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "protocol": {
            "dataset": "EEGBCI/EEGMMIDB v1.0.0", "subjects": SUBJECTS, "runs": RUNS,
            "n_channels": 64, "sfreq_hz": 160.0, "filter": "FIR bandpass 7-30 Hz (firwin)",
            "epoch_window_s": [T_MIN, T_MAX], "baseline_s": list(BASELINE),
            "labels": {"T1": "left", "T2": "right"}, "n_samples_per_epoch": 721,
            "model": "CSP(4, reg=ledoit_wolf, log=True, average_power) + StandardScaler + LogisticRegression(C=1.0, max_iter=2000)",
            "subject_specific_cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
            "loso_cv": "stratified shuffled 5-fold fitted exclusively on the four training subjects; five fold models evaluated on the held-out subject",
            "primary_loso_prediction": "mean of five training-only fold-model probabilities",
            "secondary_loso_prediction": "model refit on all epochs of the four training subjects",
            "leakage_control": "CSP, scaler, and classifier are inside Pipeline; test subject is absent from training arrays and all fitting calls",
            "decision_threshold": 0.5,
        },
        "environment": {
            "python": platform.python_version(), "mne": mne.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "source_files": files,
        "subjects": subject_meta,
        "subject_specific": specific,
        "loso": loso_results,
        "pooled_loso": pooled,
        "caliber": {
            "pooled_primary": "subject-blocked leave-one-subject-out; one prediction per trial; macro trial-weighted across subjects",
            "pooled_full_refit": "secondary LOSO with each classifier refit on all four training subjects",
            "subject_specific_pooled_comparator": "unweighted macro mean of per-subject within-subject OOF accuracy/AUC",
            "not_estimated": "run-blocked cross-subject validation; hyperparameter optimization; CTNet retraining",
        },
        "artifacts": {
            "metrics": str(OUT_JSON.relative_to(ROOT)),
            "figure": str(OUT_FIG.relative_to(ROOT)),
            "script": str(Path(__file__).relative_to(ROOT)),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "elapsed_s": float(time.time() - t0),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    make_figure(specific, loso_results, pooled)
    print(f"[artifact] {OUT_JSON.relative_to(ROOT)}", flush=True)
    print(f"[done] elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
