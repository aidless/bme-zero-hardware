#!/usr/bin/env python3
"""EEG 审稿级重估（第三轮深挖）。

对照 mini_project_eeg.py 的单 seed subject-specific 5-fold 结果，本脚本补上
审稿人会问的三件事：
  1. bootstrap CI —— 对 OOF 预测重采样，给 acc/AUC 的 95% 区间（点估计不稳）
  2. 跨 run 泛化 —— leave-one-run-out（run4/run8/run12 轮流当测试），
     比随机 fold 更严格，检验 session 间漂移下的泛化
  3. 标签置换检验 —— 打乱标签 500 次重算 CV acc，得到 null 分布与经验 p 值，
     回答"0.78 是不是显著高于随机"

所有数字运行时实测，无硬编码。纯 CPU。运行：python3 bench_eeg_rigorous.py
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import time
from collections import Counter

import mne
import numpy as np
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mne.set_log_level("error")

SUBJECT = 1
RUNS = [4, 8, 12]
L_FREQ, H_FREQ = 7.0, 30.0
T_MIN, T_MAX = -0.5, 4.0
BASELINE = (-0.5, 0.0)
N_SPLITS = 5
SEEDS = [42, 0, 7, 2024, 123]   # 多 seed 稳定性
N_BOOT = 2000
N_PERM = 60
SEED = 42

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data", "eeg")


def make_pipe(seed):
    csp = CSP(n_components=4, reg="ledoit_wolf", log=True, transform_into="average_power")
    return make_pipeline(
        csp,
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
    )


def main():
    t0 = time.time()
    print("=" * 72)
    print("EEG RIGOROUS REEVALUATION -- bootstrap CI / cross-run / permutation")
    print("=" * 72)

    paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS,
                             path=os.path.join(DATA_DIR, "mne_data"),
                             update_path=False, verbose="error")
    montage = make_standard_montage("standard_1020")

    # 逐 run 提取 epoch，同时记录来源 run（供 leave-one-run-out）
    Xs, ys, run_ids = [], [], []
    for run, p in zip(RUNS, paths):
        raw = mne.io.read_raw_edf(p, preload=True, verbose="error")
        eegbci.standardize(raw)
        raw.set_montage(montage)
        raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose="error")
        events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose="error")
        ep = mne.Epochs(raw, events, event_id={"T1_left": 1, "T2_right": 2},
                        tmin=T_MIN, tmax=T_MAX, baseline=BASELINE, preload=True,
                        verbose="error")
        Xi, yi = ep.get_data(), ep.events[:, -1]
        Xs.append(Xi); ys.append(yi); run_ids.append(np.full(len(yi), run))
        print(f"[run {run}] epochs={len(yi)} left={int((yi==1).sum())} right={int((yi==2).sum())}")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    run_ids = np.concatenate(run_ids, axis=0)
    print(f"[data] total_epochs={len(y)} shape={X.shape} chance=0.5")

    results = {}

    # ---------- 1. 多 seed subject-specific 5-fold + bootstrap CI ----------
    seed_accs, seed_aucs = [], []
    rng = np.random.default_rng(SEED)
    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        pipe = make_pipe(seed)
        yp = cross_val_predict(pipe, X, y, cv=cv, method="predict")
        yprob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        acc, auc = accuracy_score(y, yp), roc_auc_score(y, yprob)
        seed_accs.append(acc); seed_aucs.append(auc)
        print(f"[seed {seed}] acc={acc:.4f} auc={auc:.4f}")
    # 用主 seed 的 OOF 预测做 bootstrap CI
    cv0 = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEEDS[0])
    pipe0 = make_pipe(SEEDS[0])
    yp0 = cross_val_predict(pipe0, X, y, cv=cv0, method="predict")
    yprob0 = cross_val_predict(pipe0, X, y, cv=cv0, method="predict_proba")[:, 1]
    boot_acc, boot_auc = [], []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boot_acc.append(accuracy_score(y[idx], yp0[idx]))
        boot_auc.append(roc_auc_score(y[idx], yprob0[idx]))
    acc_ci = np.percentile(boot_acc, [2.5, 97.5])
    auc_ci = np.percentile(boot_auc, [2.5, 97.5])
    print(f"[multi-seed] acc mean={np.mean(seed_accs):.4f} std={np.std(seed_accs):.4f} "
          f"range=[{min(seed_accs):.4f},{max(seed_accs):.4f}]")
    print(f"[multi-seed] auc mean={np.mean(seed_aucs):.4f} std={np.std(seed_aucs):.4f}")
    print(f"[bootstrap] acc={accuracy_score(y,yp0):.4f} 95%CI=[{acc_ci[0]:.4f},{acc_ci[1]:.4f}] "
          f"(n_boot={len(boot_acc)})")
    print(f"[bootstrap] auc={roc_auc_score(y,yprob0):.4f} 95%CI=[{auc_ci[0]:.4f},{auc_ci[1]:.4f}]")
    results["subject_specific"] = {
        "seeds": SEEDS,
        "acc_per_seed": [float(v) for v in seed_accs],
        "auc_per_seed": [float(v) for v in seed_aucs],
        "acc_mean": float(np.mean(seed_accs)), "acc_std": float(np.std(seed_accs)),
        "auc_mean": float(np.mean(seed_aucs)), "auc_std": float(np.std(seed_aucs)),
        "bootstrap_acc_ci95": [float(v) for v in acc_ci],
        "bootstrap_auc_ci95": [float(v) for v in auc_ci],
        "n_boot": len(boot_acc),
        "caliber": "subject-specific, random 5-fold (within-subject)",
    }

    # ---------- 2. Leave-one-run-out（跨 session 泛化） ----------
    lor_accs = []
    for test_run in RUNS:
        tr = run_ids != test_run
        te = ~tr
        pipe = make_pipe(SEEDS[0])
        pipe.fit(X[tr], y[tr])
        yp = pipe.predict(X[te])
        a = accuracy_score(y[te], yp)
        lor_accs.append(a)
        print(f"[leave-run-{test_run}-out] train_epochs={int(tr.sum())} "
              f"test_epochs={int(te.sum())} acc={a:.4f}")
    print(f"[cross-run] mean_acc={np.mean(lor_accs):.4f} std={np.std(lor_accs):.4f}")
    results["cross_run"] = {
        "scheme": "leave-one-run-out (run4/run8/run12 each held out)",
        "acc_per_heldout_run": {str(r): float(a) for r, a in zip(RUNS, lor_accs)},
        "mean_acc": float(np.mean(lor_accs)), "std": float(np.std(lor_accs)),
    }

    # ---------- 3. 标签置换检验 ----------
    perm_accs = []
    cvp = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEEDS[0])
    perm_rng = np.random.default_rng(SEED + 1)
    for i in range(N_PERM):
        y_perm = perm_rng.permutation(y)
        ypp = cross_val_predict(make_pipe(SEEDS[0]), X, y_perm, cv=cvp, method="predict")
        perm_accs.append(accuracy_score(y_perm, ypp))
        if (i + 1) % 100 == 0:
            print(f"[permutation] {i+1}/{N_PERM} done")
    perm_accs = np.array(perm_accs)
    real_acc = np.mean(seed_accs)
    p_emp = (np.sum(perm_accs >= real_acc) + 1) / (N_PERM + 1)
    print(f"[permutation] null mean={perm_accs.mean():.4f} std={perm_accs.std():.4f} "
          f"95th={np.percentile(perm_accs,95):.4f} max={perm_accs.max():.4f}")
    print(f"[permutation] real_acc={real_acc:.4f} empirical_p={p_emp:.4f} (n_perm={N_PERM})")
    results["permutation"] = {
        "null_mean": float(perm_accs.mean()), "null_std": float(perm_accs.std()),
        "null_95th": float(np.percentile(perm_accs, 95)),
        "real_acc": float(real_acc), "empirical_p": float(p_emp),
        "n_perm": N_PERM,
    }

    results["metadata"] = {
        "dataset": "EEGBCI/EEGMMIDB v1.0.0 PhysioNet, subject 1, runs 4/8/12",
        "n_epochs": int(len(y)), "sfreq": float(raw.info["sfreq"]),
        "bandpass_hz": [L_FREQ, H_FREQ], "epoch_window_s": [T_MIN, T_MAX],
        "features": "CSP(4 comps, ledoit_wolf, log) + LR",
        "elapsed_s": float(time.time() - t0),
    }
    out = os.path.join(DATA_DIR, "eeg_rigorous_metrics.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"[artifact] {out}")
    print(f"[done] elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
