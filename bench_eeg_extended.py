#!/usr/bin/env python3
"""EEGBCI round-4 extension: 9-subject cross-subject validation, Riemann vs CSP,
and subject-blocked probability calibration. Every number is computed at runtime.

Protocol (fixed before execution):
  subjects 1..9 (override via EEG_EXT_SUBJECTS), runs 4/8/12, 64 ch, 160 Hz;
  FIR 7--30 Hz, epochs -0.5--4.0 s, baseline -0.5--0 s, T1=left(1)/T2=right(2).
  Models: CSP(4, ledoit_wolf, log) + StandardScaler + LR  [baseline]
          Covariances(lwf) + TangentSpace(riemann) + StandardScaler + LR [strong]
  Subject-specific: stratified shuffled 5-fold OOF, seed 42.
  LOSO: for each held-out subject, stratified shuffled 5-fold ONLY on the other
  subjects; the five fold models predict the held-out subject, probabilities are
  averaged. A separate refit on all training subjects is also scored.
  Pooled metrics concatenate per-subject held-out OOF predictions; trial-level
  percentile bootstrap with 2000 resamples (same indices for both models).
  Calibration: leave-one-subject-out -- calibrator fit on OTHER subjects' pooled
  held-out OOF probabilities, applied to the held-out subject. Temperature and
  isotonic regression; ECE(15 equal-width bins)/MCE/Brier/LogLoss.
  CPU only. Run: python3 -u bench_eeg_extended.py 2>&1 | tee data/eeg/eeg_extended_run.log
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pyriemann
import scipy
import sklearn
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mne.set_log_level("error")
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path.cwd()
DATA_DIR = ROOT / "data" / "eeg"
MNE_DIR = DATA_DIR / "mne_data"
OUT_JSON = DATA_DIR / "eeg_extended_metrics.json"
FIG_DIR = ROOT / "figures"

SUBJECTS = [int(x) for x in os.environ.get("EEG_EXT_SUBJECTS", "1,2,3,4,5,6,7,8,9").split(",")]
RUNS = [4, 8, 12]
L_FREQ, H_FREQ = 7.0, 30.0
T_MIN, T_MAX = -0.5, 4.0
BASELINE = (-0.5, 0.0)
N_SPLITS = 5
SEED = 42
N_BOOT = 2000
CSP_COMPONENTS = 4
EPS = 1e-6
MODELS = ["csp_lr", "tangent_lr"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(name: str, seed: int):
    if name == "csp_lr":
        return make_pipeline(
            CSP(n_components=CSP_COMPONENTS, reg="ledoit_wolf", log=True,
                transform_into="average_power"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        )
    if name == "tangent_lr":
        return make_pipeline(
            Covariances(estimator="lwf"),
            TangentSpace(metric="riemann"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, random_state=seed),
        )
    raise ValueError(name)


def metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    """CRITICAL: y in {1,2}; threshold rule must be np.where(prob>=0.5,2,1)."""
    return {
        "n": int(len(y)),
        "n_left": int(np.sum(y == 1)),
        "n_right": int(np.sum(y == 2)),
        "acc": float(accuracy_score(y, np.where(prob >= 0.5, 2, 1))),
        "auc": float(roc_auc_score(y, prob)),
    }


def prob01(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(prob, float), EPS, 1 - EPS)


def ece_mce(y01: np.ndarray, p: np.ndarray, n_bins: int = 15):
    """Equal-width bin ECE/MCE (Guo et al. 2017), identical to bench_calibration.py."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece, mce = 0.0, 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        gap = abs(p[mask].mean() - y01[mask].mean())
        ece += mask.sum() / len(y01) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def calibration_metrics(y01: np.ndarray, p: np.ndarray) -> dict:
    p = prob01(p)
    ece, mce = ece_mce(y01, p, 15)
    return {
        "n": int(len(y01)),
        "ece": ece,
        "mce": mce,
        "brier": float(np.mean((p - y01) ** 2)),
        "logloss": float(-np.mean(y01 * np.log(p) + (1 - y01) * np.log(1 - p))),
        "sharpness_std": float(np.std(p)),
        "mean_conf": float(np.mean(p)),
        "acc_at_0.5": float(accuracy_score(y01, (p >= 0.5).astype(int))),
    }


def fit_temperature(y01: np.ndarray, p: np.ndarray) -> float:
    pc = prob01(p)

    def obj(t: float) -> float:
        q = prob01(1 / (1 + np.exp(-(np.log(pc / (1 - pc)) / t))))
        return float(-np.mean(y01 * np.log(q) + (1 - y01) * np.log(1 - q)))

    res = minimize_scalar(obj, bounds=(0.2, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return prob01(1 / (1 + np.exp(-(np.log(prob01(p) / (1 - prob01(p))) / t))))


def fit_isotonic(y01: np.ndarray, p: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y01)


def bootstrap_all(y: np.ndarray, probs: dict, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired trial-level percentile bootstrap; identical resampled indices per model."""
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = list(probs)
    accs = {k: [] for k in keys}
    aucs = {k: [] for k in keys}
    diffs = []
    auc_diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        cur = {}
        for k in keys:
            pi = probs[k][idx]
            cur[k] = np.where(pi >= 0.5, 2, 1)
            accs[k].append(accuracy_score(yi, cur[k]))
            aucs[k].append(roc_auc_score(yi, pi))
        diffs.append(accuracy_score(yi, cur["tangent_lr"]) - accuracy_score(yi, cur["csp_lr"]))
        auc_diffs.append(aucs["tangent_lr"][-1] - aucs["csp_lr"][-1])
    out = {
        "n_boot_valid": len(diffs),
        "method": "paired nonparametric trial-level percentile bootstrap; replacement; 2000 resamples; seed 42",
        "per_model": {
            k: {
                "acc_ci95": [float(v) for v in np.percentile(accs[k], [2.5, 97.5])],
                "auc_ci95": [float(v) for v in np.percentile(aucs[k], [2.5, 97.5])],
            }
            for k in keys
        },
        "paired_diff": {
            "definition": "tangent_lr - csp_lr",
            "acc_diff_ci95": [float(v) for v in np.percentile(diffs, [2.5, 97.5])],
            "auc_diff_ci95": [float(v) for v in np.percentile(auc_diffs, [2.5, 97.5])],
        },
    }
    return out


def load_subjects():
    print(f"[download] EEGMMIDB v1.0.0 subjects={SUBJECTS} runs={RUNS}", flush=True)
    t0 = time.time()
    paths = [Path(p) for p in eegbci.load_data(
        subjects=SUBJECTS, runs=RUNS, path=str(MNE_DIR), update_path=False, verbose="error")]
    lookup = {(int(p.name[1:4]), int(p.name[5:7])): p for p in paths}
    assert len(lookup) == len(SUBJECTS) * len(RUNS), (len(lookup), paths)
    files = []
    for s in SUBJECTS:
        for r in RUNS:
            p = lookup[(s, r)]
            rec = {
                "subject": s, "run": r, "path": str(p.relative_to(ROOT)),
                "size_bytes": int(p.stat().st_size), "sha256": sha256_file(p),
            }
            files.append(rec)
            print(f"[sha256] S{s:03d}R{r:02d} {rec['sha256']} size={rec['size_bytes']}", flush=True)
    print(f"[download] resolved {len(files)} EDFs in {time.time()-t0:.1f}s", flush=True)

    montage = make_standard_montage("standard_1020")
    Xs, ys, subject_meta = {}, {}, {}
    for s in SUBJECTS:
        xparts, yparts, run_counts = [], [], {}
        for r in RUNS:
            raw = mne.io.read_raw_edf(lookup[(s, r)], preload=True, verbose="error")
            eegbci.standardize(raw)
            raw.set_montage(montage)
            assert len(raw.ch_names) == 64, raw.ch_names
            assert raw.info["sfreq"] == 160.0, raw.info["sfreq"]
            raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose="error")
            events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose="error")
            epochs = mne.Epochs(raw, events, event_id={"T1_left": 1, "T2_right": 2},
                                tmin=T_MIN, tmax=T_MAX, baseline=BASELINE, preload=True,
                                verbose="error")
            x, y = epochs.get_data(), epochs.events[:, -1]
            xparts.append(x)
            yparts.append(y)
            run_counts[str(r)] = {
                "n": int(len(y)), "n_left": int(np.sum(y == 1)), "n_right": int(np.sum(y == 2))}
            print(f"[epoch] S{s:03d}R{r:02d} n={len(y)} left={int(np.sum(y==1))} "
                  f"right={int(np.sum(y==2))} shape={x.shape}", flush=True)
        Xs[s] = np.concatenate(xparts)
        ys[s] = np.concatenate(yparts)
        assert Xs[s].shape == (len(ys[s]), 64, 721), Xs[s].shape
        assert set(np.unique(ys[s])) == {1, 2}
        subject_meta[str(s)] = {
            "n_epochs": int(len(ys[s])), "n_left": int(np.sum(ys[s] == 1)),
            "n_right": int(np.sum(ys[s] == 2)), "runs": run_counts,
        }
        print(f"[subject] S{s:03d} total_n={len(ys[s])} left={int(np.sum(ys[s]==1))} "
              f"right={int(np.sum(ys[s]==2))}", flush=True)
    return Xs, ys, files, subject_meta


def subject_specific(Xs, ys):
    results = {m: {} for m in MODELS}
    predictions = {m: {} for m in MODELS}
    for s in SUBJECTS:
        X, y = Xs[s], ys[s]
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        probs = {m: np.zeros(len(y), dtype=float) for m in MODELS}
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            for m in MODELS:
                pipe = build_model(m, SEED + fold)
                pipe.fit(X[tr], y[tr])
                p = pipe.predict_proba(X[va])[:, 1]
                probs[m][va] = p
                row = metrics(y[va], p)
                row.update({"fold": fold, "n_train": int(len(tr)), "n_valid": int(len(va))})
                results[m].setdefault(str(s), {"folds": []})["folds"].append(row)
                print(f"[specific] {m} S{s:03d} fold{fold} acc={row['acc']:.4f} auc={row['auc']:.4f}", flush=True)
        for m in MODELS:
            r = metrics(y, probs[m])
            folds = results[m][str(s)]["folds"]
            r.update({
                "folds": folds,
                "fold_acc_mean": float(np.mean([f["acc"] for f in folds])),
                "fold_acc_std": float(np.std([f["acc"] for f in folds])),
                "fold_auc_mean": float(np.mean([f["auc"] for f in folds])),
                "fold_auc_std": float(np.std([f["auc"] for f in folds])),
                "bootstrap": bootstrap_ci_single(y, probs[m]),
                "caliber": "within-subject stratified shuffled 5-fold OOF",
            })
            results[m][str(s)] = r
            predictions[m][str(s)] = {"y": y.tolist(), "prob": probs[m].tolist()}
            print(f"[specific] {m} S{s:03d} OOF acc={r['acc']:.6f} auc={r['auc']:.6f} "
                  f"acc_CI={r['bootstrap']['acc_ci95']}", flush=True)
    return results, predictions


def bootstrap_ci_single(y, prob, n_boot=N_BOOT, seed=SEED):
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
    }


def loso(Xs, ys):
    results = {m: {} for m in MODELS}
    predictions = {m: {} for m in MODELS}
    for heldout in SUBJECTS:
        train_subjects = [s for s in SUBJECTS if s != heldout]
        Xtrain = np.concatenate([Xs[s] for s in train_subjects])
        ytrain = np.concatenate([ys[s] for s in train_subjects])
        Xtest, ytest = Xs[heldout], ys[heldout]
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_prob = {m: np.zeros((N_SPLITS, len(ytest)), dtype=float) for m in MODELS}
        print(f"[LOSO] heldout=S{heldout:03d} train={train_subjects} "
              f"train_n={len(ytrain)} test_n={len(ytest)}", flush=True)
        for fold, (tr, va) in enumerate(cv.split(Xtrain, ytrain)):
            for m in MODELS:
                pipe = build_model(m, SEED + fold)
                pipe.fit(Xtrain[tr], ytrain[tr])
                pva = pipe.predict_proba(Xtrain[va])[:, 1]
                fold_prob[m][fold] = pipe.predict_proba(Xtest)[:, 1]
                row = metrics(ytrain[va], pva)
                row.update({
                    "fold": fold, "n_train": int(len(tr)), "n_valid": int(len(va)),
                    "test_metrics": metrics(ytest, fold_prob[m][fold]),
                })
                results[m].setdefault(str(heldout), {"folds": []})["folds"].append(row)
                print(f"[LOSO] {m} S{heldout:03d} fold{fold} train_valid_acc={row['acc']:.4f} "
                      f"test_acc={row['test_metrics']['acc']:.4f} "
                      f"test_auc={row['test_metrics']['auc']:.4f}", flush=True)
        for m in MODELS:
            ensemble = fold_prob[m].mean(axis=0)
            primary = metrics(ytest, ensemble)
            primary.update({
                "heldout_subject": heldout, "train_subjects": train_subjects,
                "n_train_subject_epochs": int(len(ytrain)), "n_test_epochs": int(len(ytest)),
                "prediction_rule": "mean of five training-only fold-model predict_proba[:,1]",
                "folds": results[m][str(heldout)]["folds"],
            })
            full = build_model(m, SEED)
            full.fit(Xtrain, ytrain)
            full_prob = full.predict_proba(Xtest)[:, 1]
            primary["full_training_refit"] = metrics(ytest, full_prob)
            results[m][str(heldout)] = primary
            predictions[m][str(heldout)] = {
                "y": ytest.tolist(), "prob": ensemble.tolist(),
                "full_refit_prob": full_prob.tolist(),
            }
            print(f"[LOSO] {m} S{heldout:03d} ENSEMBLE acc={primary['acc']:.6f} "
                  f"auc={primary['auc']:.6f}; FULL-REFIT acc={primary['full_training_refit']['acc']:.6f} "
                  f"auc={primary['full_training_refit']['auc']:.6f}", flush=True)
    return results, predictions


def pooled_loso(loso_predictions, specific_results):
    out = {}
    for m in MODELS:
        y = np.concatenate([np.asarray(loso_predictions[m][str(s)]["y"]) for s in SUBJECTS])
        p = np.concatenate([np.asarray(loso_predictions[m][str(s)]["prob"]) for s in SUBJECTS])
        pf = np.concatenate([np.asarray(loso_predictions[m][str(s)]["full_refit_prob"]) for s in SUBJECTS])
        r = metrics(y, p)
        r["bootstrap"] = bootstrap_ci_single(y, p)
        r["full_training_refit"] = metrics(y, pf)
        r["macro_subject_specific_acc"] = float(np.mean([specific_results[m][str(s)]["acc"] for s in SUBJECTS]))
        r["macro_subject_specific_auc"] = float(np.mean([specific_results[m][str(s)]["auc"] for s in SUBJECTS]))
        r["acc_drop_pp_vs_subject_specific"] = float(
            100 * (r["macro_subject_specific_acc"] - r["acc"]))
        out[m] = r
        print(f"[pooled] {m} acc={r['acc']:.6f} auc={r['auc']:.6f} "
              f"acc_CI={r['bootstrap']['acc_ci95']} auc_CI={r['bootstrap']['auc_ci95']}", flush=True)
    return out


def compare_models(loso_predictions, pooled):
    y = np.concatenate([np.asarray(loso_predictions["csp_lr"][str(s)]["y"]) for s in SUBJECTS])
    probs = {
        m: np.concatenate([np.asarray(loso_predictions[m][str(s)]["prob"]) for s in SUBJECTS])
        for m in MODELS
    }
    boot = bootstrap_all(y, probs)
    pred_a = np.where(probs["csp_lr"] >= 0.5, 2, 1)
    pred_b = np.where(probs["tangent_lr"] >= 0.5, 2, 1)
    b = int(np.sum((pred_b == y) & (pred_a != y)))
    c = int(np.sum((pred_b != y) & (pred_a == y)))
    mcnemar = binomtest(b, b + c, 0.5, alternative="two-sided").pvalue if b + c > 0 else 1.0
    per_subject = {}
    for s in SUBJECTS:
        ys = np.asarray(loso_predictions["csp_lr"][str(s)]["y"])
        pa = np.asarray(loso_predictions["csp_lr"][str(s)]["prob"])
        pb = np.asarray(loso_predictions["tangent_lr"][str(s)]["prob"])
        per_subject[str(s)] = {
            "csp_acc": float(accuracy_score(ys, np.where(pa >= 0.5, 2, 1))),
            "tangent_acc": float(accuracy_score(ys, np.where(pb >= 0.5, 2, 1))),
        }
    wins = sum(1 for v in per_subject.values() if v["tangent_acc"] > v["csp_acc"])
    losses = sum(1 for v in per_subject.values() if v["tangent_acc"] < v["csp_acc"])
    ties = len(per_subject) - wins - losses
    out = {
        "same_subjects_same_split": True,
        "n_pooled_trials": int(len(y)),
        "csp_acc": pooled["csp_lr"]["acc"], "tangent_acc": pooled["tangent_lr"]["acc"],
        "acc_diff_pp": float(100 * (pooled["tangent_lr"]["acc"] - pooled["csp_lr"]["acc"])),
        "csp_auc": pooled["csp_lr"]["auc"], "tangent_auc": pooled["tangent_lr"]["auc"],
        "auc_diff": float(pooled["tangent_lr"]["auc"] - pooled["csp_lr"]["auc"]),
        "bootstrap": boot,
        "mcnemar": {
            "b_tangent_right_csp_wrong": b, "c_tangent_wrong_csp_right": c,
            "p_two_sided": float(mcnemar),
        },
        "per_subject": per_subject,
        "subject_level_win_tie_loss": {"win": wins, "tie": ties, "loss": losses},
    }
    print(f"[compare] csp_acc={out['csp_acc']:.6f} tangent_acc={out['tangent_acc']:.6f} "
          f"diff={out['acc_diff_pp']:+.4f} pp; csp_auc={out['csp_auc']:.6f} "
          f"tangent_auc={out['tangent_auc']:.6f} auc_diff={out['auc_diff']:+.6f}", flush=True)
    print(f"[compare] paired acc_diff_CI95={boot['paired_diff']['acc_diff_ci95']} "
          f"auc_diff_CI95={boot['paired_diff']['auc_diff_ci95']} "
          f"McNemar b={b} c={c} p={mcnemar:.6f}", flush=True)
    print(f"[compare] subject win/tie/loss={wins}/{ties}/{losses}", flush=True)
    return out


def calibrate_cross_subject(loso_predictions):
    out = {}
    for m in MODELS:
        y_all = np.concatenate([np.asarray(loso_predictions[m][str(s)]["y"]) for s in SUBJECTS])
        p_all = np.concatenate([np.asarray(loso_predictions[m][str(s)]["prob"]) for s in SUBJECTS])
        y01 = (y_all == 2).astype(int)
        sid = np.concatenate([np.full(len(loso_predictions[m][str(s)]["y"]), s) for s in SUBJECTS])
        p_temp = np.zeros_like(p_all)
        p_iso = np.zeros_like(p_all)
        t_values = {}
        for s in SUBJECTS:
            te = sid == s
            cal = ~te
            t = fit_temperature(y01[cal], p_all[cal])
            t_values[str(s)] = t
            p_temp[te] = apply_temperature(p_all[te], t)
            iso = fit_isotonic(y01[cal], p_all[cal])
            p_iso[te] = np.clip(iso.predict(p_all[te]), EPS, 1 - EPS)
        raw_m = calibration_metrics(y01, p_all)
        temp_m = calibration_metrics(y01, p_temp)
        iso_m = calibration_metrics(y01, p_iso)
        per_subject = {}
        for s in SUBJECTS:
            mask = sid == s
            per_subject[str(s)] = {
                "raw": calibration_metrics(y01[mask], p_all[mask]),
                "temperature": calibration_metrics(y01[mask], p_temp[mask]),
                "isotonic": calibration_metrics(y01[mask], p_iso[mask]),
            }
        out[m] = {
            "protocol": ("leave-one-subject-out calibration: calibrator fitted on the other "
                         "subjects' pooled LOSO held-out OOF probabilities, applied to the "
                         "held-out subject; predictions pooled over subjects"),
            "pooled_raw": raw_m,
            "pooled_temperature": temp_m,
            "pooled_isotonic": iso_m,
            "temperature_by_heldout_subject": t_values,
            "per_subject": per_subject,
        }
        out[m]["best_by_ece"] = min(["temperature", "isotonic"],
                                    key=lambda k: out[m]["pooled_" + k]["ece"])
        print(f"[calib] {m} raw ECE={raw_m['ece']:.6f} MCE={raw_m['mce']:.6f} "
              f"Brier={raw_m['brier']:.6f} LogLoss={raw_m['logloss']:.6f} "
              f"acc={raw_m['acc_at_0.5']:.6f}", flush=True)
        print(f"[calib] {m} temperature ECE={temp_m['ece']:.6f} MCE={temp_m['mce']:.6f} "
              f"Brier={temp_m['brier']:.6f} LogLoss={temp_m['logloss']:.6f} "
              f"acc={temp_m['acc_at_0.5']:.6f}; T_by_subject={t_values}", flush=True)
        print(f"[calib] {m} isotonic ECE={iso_m['ece']:.6f} MCE={iso_m['mce']:.6f} "
              f"Brier={iso_m['brier']:.6f} LogLoss={iso_m['logloss']:.6f} "
              f"acc={iso_m['acc_at_0.5']:.6f}", flush=True)
    return out


def make_figures(loso_predictions, pooled, calibration):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [f"S{s:03d}" for s in SUBJECTS] + ["Pooled"]
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(labels) * 1.35), 5.2),
                             constrained_layout=True)
    for ax, key, title in [(axes[0], "acc", "A. Held-out-subject accuracy"),
                           (axes[1], "auc", "B. Held-out-subject ROC AUC")]:
        def series(model):
            vals = []
            for s in SUBJECTS:
                yy = np.asarray(loso_predictions[model][str(s)]["y"])
                pp = np.asarray(loso_predictions[model][str(s)]["prob"])
                vals.append(accuracy_score(yy, np.where(pp >= 0.5, 2, 1)) if key == "acc"
                            else roc_auc_score(yy, pp))
            return vals + [pooled[model][key]]
        a, b = series("csp_lr"), series("tangent_lr")
        ax.bar(x - width / 2, a, width, label="CSP + LR", color="#3976b8")
        ax.bar(x + width / 2, b, width, label="TangentSpace + LR", color="#df7846")
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("EEGBCI motor imagery: cross-subject CSP+LR vs Riemannian TangentSpace+LR\n"
                 f"Subjects {SUBJECTS[0]:03d}-{SUBJECTS[-1]:03d} (n={len(SUBJECTS)}), runs 4/8/12, LOSO OOF",
                 fontsize=12, fontweight="bold")
    p = FIG_DIR / "eeg_extended_model_compare.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[artifact] {p.relative_to(ROOT)}", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    for row, m in enumerate(MODELS):
        y = np.concatenate([np.asarray(loso_predictions[m][str(s)]["y"]) for s in SUBJECTS])
        p_raw = np.concatenate([np.asarray(loso_predictions[m][str(s)]["prob"]) for s in SUBJECTS])
        y01 = (y == 2).astype(int)
        sid = np.concatenate([np.full(len(loso_predictions[m][str(s)]["y"]), s) for s in SUBJECTS])
        p_temp = np.zeros_like(p_raw)
        p_iso = np.zeros_like(p_raw)
        for s in SUBJECTS:
            te = sid == s
            cal = ~te
            t = calibration[m]["temperature_by_heldout_subject"][str(s)]
            p_temp[te] = apply_temperature(p_raw[te], t)
            iso = fit_isotonic(y01[cal], p_raw[cal])
            p_iso[te] = np.clip(iso.predict(p_raw[te]), EPS, 1 - EPS)
        for col, (pp, name) in enumerate([(p_raw, "raw"), (p_temp, "temperature"),
                                          (p_iso, "isotonic")]):
            ax = axes[row, col]
            bins = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(pp, bins) - 1, 0, 9)
            xp, yp = [], []
            for b in range(10):
                mask = idx == b
                if mask.sum() >= 3:
                    xp.append(float(pp[mask].mean()))
                    yp.append(float(y01[mask].mean()))
            ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="perfect")
            if xp:
                ax.plot(xp, yp, "o-", color="#2b6cb0", label="observed")
            ece = calibration[m]["pooled_" + name]["ece"]
            ax.set_title(f"{m} | {name} (ECE={ece:.4f})", loc="left", fontsize=10)
            ax.set_xlabel("mean predicted probability (class=2)")
            ax.set_ylabel("empirical frequency")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Cross-subject reliability diagrams (LOSO OOF, LOSO-fitted calibrators)",
                 fontsize=13, fontweight="bold")
    p = FIG_DIR / "eeg_extended_reliability.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[artifact] {p.relative_to(ROOT)}", flush=True)


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print(f"EEG EXTENDED EXPERIMENT | subjects={SUBJECTS} | seed={SEED} | CPU", flush=True)
    print("=" * 80, flush=True)
    print(f"[env] python={platform.python_version()} mne={mne.__version__} "
          f"sklearn={sklearn.__version__} numpy={np.__version__} scipy={scipy.__version__} "
          f"pyriemann={pyriemann.__version__}", flush=True)
    Xs, ys, files, subject_meta = load_subjects()
    specific, specific_pred = subject_specific(Xs, ys)
    loso_results, loso_pred = loso(Xs, ys)
    pooled = pooled_loso(loso_pred, specific)
    comparison = compare_models(loso_pred, pooled)
    calibration = calibrate_cross_subject(loso_pred)
    make_figures(loso_pred, pooled, calibration)

    payload = {
        "experiment": "EEGBCI round-4 extension: 9-subject cross-subject validation, "
                      "Riemannian vs CSP comparison, cross-subject calibration",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "protocol": {
            "dataset": "EEGBCI/EEGMMIDB v1.0.0", "subjects": SUBJECTS, "runs": RUNS,
            "n_channels": 64, "sfreq_hz": 160.0, "filter": "FIR bandpass 7-30 Hz (firwin)",
            "epoch_window_s": [T_MIN, T_MAX], "baseline_s": list(BASELINE),
            "labels": {"T1": "left", "T2": "right"}, "n_samples_per_epoch": 721,
            "models": {
                "csp_lr": "CSP(4, reg=ledoit_wolf, log=True, average_power) + StandardScaler + LogisticRegression(C=1.0, max_iter=2000)",
                "tangent_lr": "Covariances(estimator=lwf) + TangentSpace(metric=riemann) + StandardScaler + LogisticRegression(C=1.0, max_iter=1000)",
            },
            "subject_specific_cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
            "loso_cv": "stratified shuffled 5-fold fitted exclusively on all other subjects; five fold models predict held-out subject; probabilities averaged",
            "secondary_loso": "single model refit on all training subjects",
            "decision_threshold": 0.5,
            "label_threshold_rule": "np.where(prob >= 0.5, 2, 1) with y in {1,2}",
            "bootstrap": "paired trial-level percentile bootstrap, 2000 resamples, seed 42",
            "calibration": "leave-one-subject-out: fit temperature/isotonic on OTHER subjects' pooled LOSO OOF probs; apply to held-out subject",
            "device": "CPU only (OMP/OPENBLAS/MKL threads=1)",
        },
        "environment": {
            "python": platform.python_version(), "mne": mne.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "pyriemann": pyriemann.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        },
        "source_files": files,
        "subjects": subject_meta,
        "predictions": {
            "subject_specific": specific_pred,
            "loso": loso_pred,
        },
        "subject_specific": specific,
        "loso": loso_results,
        "pooled_loso": pooled,
        "model_comparison": comparison,
        "calibration": calibration,
        "caliber": {
            "pooled_primary": "subject-blocked leave-one-subject-out; one OOF prediction per trial; trial-weighted across held-out subjects",
            "comparison": "identical subjects, identical CV folds, identical trial indices for both models",
            "not_estimated": "run-blocked cross-subject validation; hyperparameter search; deep EEGNet",
        },
        "artifacts": {
            "metrics": str(OUT_JSON.relative_to(ROOT)),
            "log": "data/eeg/eeg_extended_run.log",
            "figures": [
                "figures/eeg_extended_model_compare.png",
                "figures/eeg_extended_reliability.png",
            ],
            "script": str(Path(__file__).relative_to(ROOT)),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "elapsed_s": float(time.time() - t0),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"[artifact] {OUT_JSON.relative_to(ROOT)}", flush=True)
    print(f"[done] elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
