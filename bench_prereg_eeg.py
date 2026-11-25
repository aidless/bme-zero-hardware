#!/usr/bin/env python3
"""预注册 1 · EEG 跨被试 · 协方差对齐（Tangent Space）vs CSP。

先冻预注册（PREREGISTRATION-1-EEG-crosssubject.md, sha256 见 data/prereg/prereg_eeg.sha256），
本脚本只按预注册执行，跑后不改判定。

主指标：pooled LOSO accuracy（9 折，405 trial 级 OOF）
次指标：pooled LOSO AUC、per-subject macro acc、校准 ECE
判定：B 臂 pooled LOSO acc 的 95% bootstrap CI 下界 > 0.5 → 存在跨被试优势
      B−A 配对 bootstrap 95% CI 排除 0 → 优于 CSP
"""
import hashlib
import json
import os
import sys

import numpy as np

SEED = 42
N_BOOT = 2000
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "eeg", "mne_data")
OUT = os.path.join(ROOT, "data", "prereg")
FIG = os.path.join(ROOT, "figures")
PREREG = os.path.join(ROOT, "PREREGISTRATION-1-EEG-crosssubject.md")
SHA_FILE = os.path.join(OUT, "prereg_eeg.sha256")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_prereg():
    """预注册完整性：文件 sha256 必须与冻结记录一致。"""
    if not os.path.exists(SHA_FILE):
        raise SystemExit("FATAL: 冻结 sha256 记录缺失，拒绝运行")
    frozen = open(SHA_FILE).read().split()[0]
    now = sha256(PREREG)
    if frozen != now:
        raise SystemExit(f"FATAL: 预注册已被改动（frozen={frozen[:12]} now={now[:12]}），拒绝运行")
    return now


def load_subject(mne, subject, runs=(4, 8, 12)):
    """返回 (X[trial, ch, t], y[trial] in {1,2}, sfreq)。"""
    xs, ys = [], []
    for run in runs:
        raw = mne.io.read_raw_edf(
            os.path.join(DATA, "MNE-eegbci-data", "files", "eegmmidb", "1.0.0",
                         subject, f"{subject}R{run:02d}.edf"),
            preload=True, verbose="ERROR")
        raw.rename_channels(lambda c: c.strip())
        sfreq = raw.info["sfreq"]
        events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose="ERROR")
        for ev in events:
            a = int(ev[0] - 0.5 * sfreq)
            b = int(ev[0] + 4.0 * sfreq)
            if a < 0 or b > len(raw):
                continue
            xs.append(raw.get_data(start=a, stop=b))
            ys.append(int(ev[2]))
    return np.asarray(xs), np.asarray(ys), sfreq


def make_epochs(raws):
    """FIR 7-30 Hz 带通（与既有 bench_eeg_extended 同口径）。"""
    from mne.filter import filter_data
    return filter_data(raws, 160.0, 7.0, 30.0, method="fir", verbose="ERROR")


def arm_csp(X, y, train_idx, test_idx):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from mne.decoding import CSP
    csp = CSP(n_components=4, reg=0.01, log=True, norm_trace=False)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    xt = csp.fit_transform(X[train_idx], y[train_idx])
    clf.fit(xt, y[train_idx])
    xv = csp.transform(X[test_idx])
    return clf.predict(xv), clf.predict_proba(xv)


def arm_tangent(X, y, train_idx, test_idx):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    cov = Covariances(estimator="lwf")
    ts = TangentSpace(metric="riemann", tsupdate=False)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    xt = ts.fit_transform(cov.fit_transform(X[train_idx]), y[train_idx])
    clf.fit(xt, y[train_idx])
    xv = ts.transform(cov.transform(X[test_idx]))
    return clf.predict(xv), clf.predict_proba(xv)


def metrics(y, pred, prob):
    from sklearn.metrics import accuracy_score, roc_auc_score
    return {
        "acc": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, prob[:, 1])),
        "n": int(len(y)),
    }


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    stats = v[idx].mean(axis=1)
    return [float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, (1 - alpha / 2)))]


def ece(y, prob, n_bins=15):
    conf = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def main():
    prereg_sha = check_prereg()
    import mne
    mne.set_log_level("ERROR")

    subjects = [f"S{i:03d}" for i in range(1, 10)]
    per_subject = {}
    for s in subjects:
        X, y, sfreq = load_subject(mne, s)
        X = make_epochs(X)
        per_subject[s] = (X, y)
        print(f"loaded {s}: X={X.shape} y={np.bincount(y)}", flush=True)

    # ---- LOSO：两臂同折 ----
    res = {"A_csp": {}, "B_tangent": {}}
    for held in subjects:
        train_subj = [s for s in subjects if s != held]
        Xtr = np.concatenate([per_subject[s][0] for s in train_subj])
        ytr = np.concatenate([per_subject[s][1] for s in train_subj])
        Xte, yte = per_subject[held]
        for name, fn in (("A_csp", arm_csp), ("B_tangent", arm_tangent)):
            pred, prob = fn(Xtr, ytr, np.arange(len(ytr)), np.arange(len(yte)))
            res[name][held] = {"y": yte.tolist(), "pred": pred.tolist(),
                               "prob": prob.tolist()}
        print(f"LOSO held {held} done", flush=True)

    # ---- pooled trial 级 ----
    pooled = {}
    for name in ("A_csp", "B_tangent"):
        y = np.concatenate([np.asarray(res[name][s]["y"]) for s in subjects])
        pred = np.concatenate([np.asarray(res[name][s]["pred"]) for s in subjects])
        prob = np.concatenate([np.asarray(res[name][s]["prob"]) for s in subjects])
        pooled[name] = metrics(y, pred, prob)
        pooled[name]["acc_ci"] = bootstrap_ci((pred == y).astype(float))
        pooled[name]["ece"] = ece(y, prob)
        # per-subject macro
        accs = [float(np.mean(np.asarray(res[name][s]["pred"]) == np.asarray(res[name][s]["y"])))
                for s in subjects]
        pooled[name]["macro_acc"] = float(np.mean(accs))
        pooled[name]["per_subject_acc"] = {s: float(a) for s, a in zip(subjects, accs)}

    # ---- 主判定：B 的 CI 下界 > 0.5 ----
    b = pooled["B_tangent"]
    primary = bool(b["acc_ci"][0] > 0.5)
    # ---- 次判定：B−A 配对 bootstrap ----
    rng = np.random.default_rng(SEED)
    diffs = []
    n = len(subjects)
    # trial 级配对：按 subject 拼接后顺序一致，直接对 pooled 数组配对重采样
    yA = np.concatenate([np.asarray(res["A_csp"][s]["y"]) for s in subjects])
    pA = np.concatenate([np.asarray(res["A_csp"][s]["pred"]) for s in subjects])
    pB = np.concatenate([np.asarray(res["B_tangent"][s]["pred"]) for s in subjects])
    idx = rng.integers(0, len(yA), size=(N_BOOT, len(yA)))
    diffs = (pB[idx] == yA[idx]).mean(axis=1) - (pA[idx] == yA[idx]).mean(axis=1)
    diff_ci = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
    secondary = bool(diff_ci[0] > 0 or diff_ci[1] < 0)

    out = {
        "prereg": "PREREGISTRATION-1-EEG-crosssubject.md",
        "prereg_sha256": prereg_sha,
        "frozen_sha256": open(SHA_FILE).read().split()[0],
        "subjects": subjects,
        "n_trials_total": int(sum(len(per_subject[s][1]) for s in subjects)),
        "primary_metric": "pooled LOSO accuracy",
        "arms": pooled,
        "paired_diff_B_minus_A": {"point": float((pB == yA).mean() - (pA == yA).mean()),
                                  "ci95": diff_ci},
        "verdict": {
            "primary_CI_lower_gt_0.5": primary,
            "secondary_ci_excludes_0": secondary,
            "statement": (
                "B 臂 pooled LOSO acc 95% CI 下界 > 0.5 → 存在跨被试优势"
                if primary else
                "B 臂 pooled LOSO acc 95% CI 下界 ≤ 0.5 → 无跨被试优势证据"),
        },
        "env": {"numpy": np.__version__, "seed": SEED, "n_boot": N_BOOT},
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "prereg_eeg_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"A": pooled["A_csp"], "B": pooled["B_tangent"],
                      "diff": out["paired_diff_B_minus_A"],
                      "verdict": out["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())