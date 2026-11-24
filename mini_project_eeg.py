#!/usr/bin/env python3
"""EEG motor-imagery decoding mini project -- pure computation, real public data.

Data source (REAL, no hardware): EEGBCI / EEG Motor Movement/Imagery Dataset
v1.0.0, Schalk et al. 2004, subject 1, runs 4/8/12 = motor imagery of the
LEFT vs RIGHT fist. Files are fetched from https://physionet.org/files/eegmmidb/1.0.0/
into data/eeg/mne_data/ via MNE's downloader. Event codes: T0=rest, T1=left
fist imagery, T2=right fist imagery (verified against the dataset page).

Pipeline: band-pass filter -> epoch cut -> two feature families
(CSP log-variance and mu/beta band power) -> sklearn logistic regression ->
stratified 5-fold CV accuracy / ROC-AUC.

Every number printed by this script is computed at runtime. Nothing is
hard-coded. Run from the workspace root:  python3 mini_project_eeg.py
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import hashlib
import json
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.viz import plot_topomap
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mne.set_log_level("error")  # keep stdout to the numbers we actually report

# ---------------------------------------------------------------- parameters
SUBJECT = 1
RUNS = [4, 8, 12]  # motor imagery: left vs right fist (MNE run table)
L_FREQ, H_FREQ = 7.0, 30.0  # mu + beta passband
T_MIN, T_MAX = -0.5, 4.0
BASELINE = (-0.5, 0.0)
MU = (8.0, 13.0)
BETA = (13.0, 30.0)
N_SPLITS = 5
SEED = 42
LABEL_NAMES = {1: "T1 left-fist imagery", 2: "T2 right-fist imagery"}

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data", "eeg")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    print("=" * 72)
    print("EEG MOTOR-IMAGERY DECODING  (REAL DATA, NO HARDWARE)")
    print("=" * 72)
    print(f"mne={mne.__version__}  numpy={np.__version__}  seed={SEED}")

    # ---------------------------------------------------------- 1. data load
    paths = eegbci.load_data(
        subjects=SUBJECT,
        runs=RUNS,
        path=os.path.join(DATA_DIR, "mne_data"),
        update_path=False,
        verbose="error",
    )
    print(f"[data] source=EEGBCI v1.0.0 subject={SUBJECT} runs={RUNS}")
    for p in paths:
        print(f"[data] file={p} bytes={os.path.getsize(p)} sha256={sha256(p)}")

    montage = make_standard_montage("standard_1020")
    raws = []
    for p in paths:
        r = mne.io.read_raw_edf(p, preload=True, verbose="error")
        eegbci.standardize(r)
        r.set_montage(montage)
        raws.append(r)
    raw = mne.concatenate_raws(raws)

    sfreq = raw.info["sfreq"]
    n_channels = len(raw.ch_names)
    print(f"[data] concatenated duration={raw.times[-1]:.2f}s sfreq={sfreq:g}Hz channels={n_channels}")
    print(f"[data] annotations(all runs)={dict(__import__('collections').Counter(raw.annotations.description))}")

    # ------------------------------------------------------ 2. band-pass filter
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose="error")
    print(f"[filter] bandpass={L_FREQ}-{H_FREQ}Hz (FIR)")

    # ----------------------------------------------------------- 3. epoching
    event_id = {"T1_left": 1, "T2_right": 2}
    events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose="error")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=T_MIN,
        tmax=T_MAX,
        baseline=BASELINE,
        preload=True,
        verbose="error",
    )
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]
    n_left = int(np.sum(y == 1))
    n_right = int(np.sum(y == 2))
    print(f"[epochs] tmin={T_MIN} tmax={T_MAX} baseline={BASELINE} n_epochs={len(y)} "
          f"left={n_left} right={n_right} shape={X.shape}")
    if min(n_left, n_right) < N_SPLITS:
        raise RuntimeError(f"too few trials per class for {N_SPLITS}-fold CV")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    clf = lambda: LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)

    # ------------------------------------------------- 4a. CSP feature decode
    csp = CSP(n_components=4, reg="ledoit_wolf", log=True,
              transform_into="average_power")
    pipe_csp = make_pipeline(csp, StandardScaler(), clf())
    csp_fold_acc = cross_val_score(pipe_csp, X, y, cv=cv, scoring="accuracy")
    y_pred_csp = cross_val_predict(pipe_csp, X, y, cv=cv, method="predict")
    y_prob_csp = cross_val_predict(pipe_csp, X, y, cv=cv, method="predict_proba")[:, 1]
    csp_acc = accuracy_score(y, y_pred_csp)
    csp_auc = roc_auc_score(y, y_prob_csp)
    cm_csp = confusion_matrix(y, y_pred_csp, labels=[1, 2])
    print(f"[CSP] fold_accuracy={np.round(csp_fold_acc, 4).tolist()} "
          f"mean={csp_fold_acc.mean():.4f} std={csp_fold_acc.std():.4f}")
    print(f"[CSP] test_accuracy={csp_acc:.4f} test_auc={csp_auc:.4f} "
          f"confusion_matrix={cm_csp.tolist()}")

    # ------------------------------------------- 4b. mu/beta band-power decode
    psd = epochs.compute_psd(method="welch", fmin=1.0, fmax=40.0,
                             n_fft=256, verbose="error")
    freqs = np.asarray(psd.freqs)
    P = psd.get_data()  # (n_epochs, n_channels, n_freqs)
    mu_mask = (freqs >= MU[0]) & (freqs <= MU[1])
    beta_mask = (freqs > BETA[0]) & (freqs <= BETA[1])
    pow_mu = np.log(P[:, :, mu_mask].mean(axis=2))
    pow_beta = np.log(P[:, :, beta_mask].mean(axis=2))
    X_bp = np.concatenate([pow_mu, pow_beta], axis=1)
    print(f"[bandpower] mu_bins={int(mu_mask.sum())} beta_bins={int(beta_mask.sum())} "
          f"feature_shape={X_bp.shape}")

    pipe_bp = make_pipeline(SelectKBest(f_classif, k=20), StandardScaler(), clf())
    bp_fold_acc = cross_val_score(pipe_bp, X_bp, y, cv=cv, scoring="accuracy")
    y_prob_bp = cross_val_predict(pipe_bp, X_bp, y, cv=cv, method="predict_proba")[:, 1]
    y_pred_bp = cross_val_predict(pipe_bp, X_bp, y, cv=cv, method="predict")
    bp_acc = accuracy_score(y, y_pred_bp)
    bp_auc = roc_auc_score(y, y_prob_bp)
    print(f"[bandpower] fold_accuracy={np.round(bp_fold_acc, 4).tolist()} "
          f"mean={bp_fold_acc.mean():.4f} std={bp_fold_acc.std():.4f}")
    print(f"[bandpower] test_accuracy={bp_acc:.4f} test_auc={bp_auc:.4f}")
    print(f"[baseline] chance_accuracy={1.0 / 2:.4f}")

    # ----------------------------------------------------------- 5. figures
    # 5a. PSD at sensorimotor electrodes C3 / C4, per class
    sub = epochs.copy().pick(["C3", "C4"])
    psd_motor = sub.compute_psd(method="welch", fmin=1.0, fmax=40.0,
                                n_fft=256, verbose="error")
    pm = psd_motor.get_data()
    fm = np.asarray(psd_motor.freqs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = {1: "#1f77b4", 2: "#d62728"}
    for ax_idx, ch in enumerate(["C3", "C4"]):
        ci = psd_motor.ch_names.index(ch)
        for lab in (1, 2):
            curve = 10 * np.log10(pm[y == lab, ci, :].mean(axis=0))
            axes[ax_idx].plot(fm, curve, color=colors[lab],
                              label=LABEL_NAMES[lab], linewidth=1.6)
        axes[ax_idx].axvspan(MU[0], MU[1], color="grey", alpha=0.15, label="mu 8-13Hz")
        axes[ax_idx].set_title(f"Welch PSD @ {ch}")
        axes[ax_idx].set_xlabel("Frequency (Hz)")
        axes[ax_idx].grid(alpha=0.3)
    axes[0].set_ylabel("Power spectral density (dB)")
    axes[0].legend(fontsize=8)
    fig.suptitle("EEGBCI S001 runs 4/8/12 - grand-average PSD by imagery class", fontsize=12)
    fig.tight_layout()
    p_spectrum = os.path.join(FIG_DIR, "eeg_spectrum.png")
    fig.savefig(p_spectrum, dpi=150)
    plt.close(fig)
    print(f"[figure] {p_spectrum}")

    # 5b. topomap of right-minus-left log band power (mu and beta)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (title, power) in zip(
        axes,
        [("mu 8-13 Hz", pow_mu), ("beta 13-30 Hz", pow_beta)],
    ):
        diff = power[y == 2].mean(axis=0) - power[y == 1].mean(axis=0)
        ret = plot_topomap(diff, epochs.info, axes=ax, show=False,
                           cmap="RdBu_r", contours=0)
        fig.colorbar(ret[0], ax=ax, shrink=0.8, label="log-power right - left")
        ax.set_title(title)
    fig.suptitle("Right-minus-left imagery: sensorimotor band-power topography", fontsize=12)
    fig.tight_layout()
    p_topo = os.path.join(FIG_DIR, "eeg_topomap.png")
    fig.savefig(p_topo, dpi=150)
    plt.close(fig)
    print(f"[figure] {p_topo}")

    # 5c. confusion matrix of the CSP decoder
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm_csp,
        display_labels=["left", "right"],
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"CSP + logistic regression, {N_SPLITS}-fold CV\n"
                 f"accuracy={csp_acc:.3f}  AUC={csp_auc:.3f}")
    fig.tight_layout()
    p_cm = os.path.join(FIG_DIR, "eeg_confusion_matrix.png")
    fig.savefig(p_cm, dpi=150)
    plt.close(fig)
    print(f"[figure] {p_cm}")

    # ---------------------------------------------------------- 6. artifacts
    metrics = {
        "data_source": "EEGBCI / EEGMMIDB v1.0.0 (PhysioNet), REAL data, no hardware",
        "dataset_url": "https://physionet.org/content/eegmmidb/1.0.0/",
        "subject": SUBJECT,
        "runs": RUNS,
        "event_mapping": {"T1": "left-fist imagery", "T2": "right-fist imagery", "T0": "rest (excluded)"},
        "files": [
            {"path": str(p), "bytes": os.path.getsize(p), "sha256": sha256(p)}
            for p in paths
        ],
        "sfreq_hz": float(sfreq),
        "n_channels": int(n_channels),
        "bandpass_hz": [L_FREQ, H_FREQ],
        "epoch_window_s": [T_MIN, T_MAX],
        "baseline_s": list(BASELINE),
        "n_epochs": int(len(y)),
        "n_left": n_left,
        "n_right": n_right,
        "cv": {"scheme": f"StratifiedKFold({N_SPLITS}, shuffle=True)", "seed": SEED},
        "chance_accuracy": 0.5,
        "csp": {
            "n_components": 4,
            "fold_accuracy": [float(v) for v in csp_fold_acc],
            "fold_accuracy_mean": float(csp_fold_acc.mean()),
            "fold_accuracy_std": float(csp_fold_acc.std()),
            "test_accuracy": float(csp_acc),
            "test_auc": float(csp_auc),
            "confusion_matrix": cm_csp.tolist(),
        },
        "band_power": {
            "mu_hz": list(MU),
            "beta_hz": list(BETA),
            "n_features": int(X_bp.shape[1]),
            "k_selected": 20,
            "fold_accuracy": [float(v) for v in bp_fold_acc],
            "fold_accuracy_mean": float(bp_fold_acc.mean()),
            "fold_accuracy_std": float(bp_fold_acc.std()),
            "test_accuracy": float(bp_acc),
            "test_auc": float(bp_auc),
        },
        "figures": [p_spectrum, p_topo, p_cm],
    }
    p_metrics = os.path.join(DATA_DIR, "eeg_metrics.json")
    with open(p_metrics, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(f"[artifact] {p_metrics}")
    print(f"[done] elapsed={time.time() - t0:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
