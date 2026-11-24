#!/usr/bin/env python3
"""概率校准分析（第四轮深挖）。

前几轮只报 AUC/acc，但审稿标配还要问：预测概率可信吗？
本脚本对两个二分类方向重算 OOF 概率，按 uncertainty-quantification 门禁：
  - ECE(15 bins) / MCE / Brier / LogLoss，附 sharpness（预测概率 std）
  - ECE>0.05 时用 temperature scaling 校准，报告校准前后
  - reliability diagram + sharpness 分布图

方向：
  1) 制药 BBBP：ECFP4(2048) + RandomForest（与 bench_mol 同协议）
  2) EEG：EEGBCI S001 runs4/8/12 CSP(4)+LR（与 bench_eeg 同协议）

所有数字运行时实测，无硬编码。纯 CPU。python3 -u bench_calibration.py
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

RDLogger.DisableLog("rdApp.*")

ROOT = os.getcwd()
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "data", "calibration")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
SEED = 42


def ece_mce(y, p, n_bins=15):
    """Equal-width bin ECE/MCE (Guo et al. 2017)."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece, mce = 0.0, 0.0
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            rows.append((b, 0, float("nan"), float("nan")))
            continue
        conf, acc = p[m].mean(), y[m].mean()
        gap = abs(conf - acc)
        ece += m.sum() / len(y) * gap
        mce = max(mce, gap)
        rows.append((b, int(m.sum()), float(conf), float(acc)))
    return float(ece), float(mce), rows


def metrics(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ece, mce, rows = ece_mce(y, p, 15)
    return {"ece": ece, "mce": mce, "brier": brier, "logloss": logloss,
            "sharpness_std": float(np.std(p)), "mean_conf": float(np.mean(p)), "bins": rows}


def temperature_scale(y_cal, p_cal, y_te, p_te):
    """Fit scalar T on calibration split (minimize logloss), apply to test split."""
    # RF predict_proba can hit exact 0/1 -> logit denominator is zero. Clip first.
    eps = 1e-6
    pc = np.clip(np.asarray(p_cal, float), eps, 1 - eps)
    pt = np.clip(np.asarray(p_te, float), eps, 1 - eps)
    def obj(t):
        q = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) / t)))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return -np.mean(y_cal * np.log(q) + (1 - y_cal) * np.log(1 - q))
    res = minimize_scalar(obj, bounds=(0.2, 10.0), method="bounded")
    T = float(res.x)
    q_te = 1 / (1 + np.exp(-(np.log(pt / (1 - pt)) / T)))
    return T, np.clip(q_te, 1e-9, 1 - 1e-9)


def reliability_plot(y, p_raw, p_cal, title, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, pp, lab in [(ax1, p_raw, "raw"), (ax2, p_cal, "temperature-scaled")]:
        prob_true, prob_pred = [], []
        for lo in np.linspace(0, 1, 11)[:-1]:
            m = (pp >= lo) & (pp < lo + 0.1)
            if m.sum() >= 3:
                prob_pred.append(pp[m].mean()); prob_true.append(y[m].mean())
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
        ax.plot(prob_pred, prob_true, "o-", color="#2b6cb0")
        ax.set_xlabel("mean predicted probability"); ax.set_ylabel("empirical frequency")
        ax.set_title(f"{lab} (ECE={ece_mce(y, pp, 15)[0]:.4f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cal_bbbp():
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "BBBP.csv"), encoding="utf-8")))
    X, y, valid = [], [], []
    for r in rows:
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        X.append(np.asarray(fp, dtype=np.float32))
        y.append(int(r["p_np"])); valid.append(Chem.MolToSmiles(m))
    X, y = np.asarray(X), np.asarray(y)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    m_raw = metrics(y, p)
    # temperature scaling: 用一半 OOF 当校准集、一半当测试（避免在同一批上 fit+评）
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    half = len(perm) // 2
    ic, it = perm[:half], perm[half:]
    T, p_cal = temperature_scale(y[ic], p[ic], y[it], p[it])
    m_cal = metrics(y[it], p_cal)
    m_raw_te = metrics(y[it], p[it])
    print(f"[BBBP] n={len(y)} raw ECE={m_raw['ece']:.4f} MCE={m_raw['mce']:.4f} "
          f"Brier={m_raw['brier']:.4f} LogLoss={m_raw['logloss']:.4f} "
          f"sharpness_std={m_raw['sharpness_std']:.4f} mean_conf={m_raw['mean_conf']:.4f}")
    print(f"[BBBP] T={T:.3f} test-half ECE {m_raw_te['ece']:.4f}->{m_cal['ece']:.4f} "
          f"Brier {m_raw_te['brier']:.4f}->{m_cal['brier']:.4f} "
          f"LogLoss {m_raw_te['logloss']:.4f}->{m_cal['logloss']:.4f}")
    reliability_plot(y[it], p[it], p_cal, "BBBP ECFP4+RF calibration",
                     os.path.join(FIG_DIR, "calibration_bbbp.png"))
    return {"n": int(len(y)), "raw": m_raw, "test_half_raw": m_raw_te,
            "test_half_calibrated": m_cal, "temperature": T}


def cal_eeg():
    import mne
    from mne.channels import make_standard_montage
    from mne.datasets import eegbci
    from mne.decoding import CSP
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mne.set_log_level("error")
    paths = eegbci.load_data(subjects=1, runs=[4, 8, 12],
                             path=os.path.join(ROOT, "data", "eeg", "mne_data"),
                             update_path=False, verbose="error")
    montage = make_standard_montage("standard_1020")
    raws = []
    for pth in paths:
        r = mne.io.read_raw_edf(pth, preload=True, verbose="error")
        eegbci.standardize(r); r.set_montage(montage); raws.append(r)
    raw = mne.concatenate_raws(raws)
    raw.filter(7, 30, fir_design="firwin", verbose="error")
    events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose="error")
    ep = mne.Epochs(raw, events, event_id={"T1": 1, "T2": 2}, tmin=-0.5, tmax=4.0,
                    baseline=(-0.5, 0), preload=True, verbose="error")
    X, y = ep.get_data(), ep.events[:, -1]
    pipe = make_pipeline(CSP(n_components=4, reg="ledoit_wolf", log=True,
                             transform_into="average_power"),
                         StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    y01 = (y == 2).astype(int)
    m_raw = metrics(y01, p)
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(y)); half = len(perm) // 2
    ic, it = perm[:half], perm[half:]
    T, p_cal = temperature_scale(y01[ic], p[ic], y01[it], p[it])
    m_cal = metrics(y01[it], p_cal); m_raw_te = metrics(y01[it], p[it])
    print(f"[EEG] n={len(y)} raw ECE={m_raw['ece']:.4f} MCE={m_raw['mce']:.4f} "
          f"Brier={m_raw['brier']:.4f} LogLoss={m_raw['logloss']:.4f} "
          f"sharpness_std={m_raw['sharpness_std']:.4f} mean_conf={m_raw['mean_conf']:.4f}")
    print(f"[EEG] T={T:.3f} test-half ECE {m_raw_te['ece']:.4f}->{m_cal['ece']:.4f} "
          f"Brier {m_raw_te['brier']:.4f}->{m_cal['brier']:.4f} "
          f"LogLoss {m_raw_te['logloss']:.4f}->{m_cal['logloss']:.4f}")
    reliability_plot(y01[it], p[it], p_cal, "EEG CSP+LR calibration",
                     os.path.join(FIG_DIR, "calibration_eeg.png"))
    return {"n": int(len(y)), "raw": m_raw, "test_half_raw": m_raw_te,
            "test_half_calibrated": m_cal, "temperature": T}


def main():
    t0 = time.time()
    print("=" * 72)
    print("PROBABILITY CALIBRATION ANALYSIS (round 4 deep dive)")
    print("=" * 72)
    results = {"protocol": "OOF probabilities, ECE 15 bins + MCE + Brier + LogLoss "
                           "+ sharpness; temperature scaling fit on half of OOF, "
                           "evaluated on the other half to avoid same-data fit+eval"}
    print("--- BBBP ---")
    results["bbbp"] = cal_bbbp()
    print("--- EEG ---")
    results["eeg"] = cal_eeg()
    results["elapsed_s"] = float(time.time() - t0)
    out = os.path.join(RES_DIR, "calibration_metrics.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"[artifact] {out}")
    print(f"[done] elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
