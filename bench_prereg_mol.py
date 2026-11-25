#!/usr/bin/env python3
"""预注册 2 · 制药 · scaffold split 实现差异的效应量。

先冻预注册（PREREGISTRATION-2-Mol-split-effect.md, sha256 见 data/prereg/prereg_mol.sha256）。

主指标：AUC(ours_shuffle_split) − AUC(deepchem_order_split)，同一 RF500 + ECFP4-2048，配对 bootstrap 95% CI。
判定：CI 排除 0 → split 实现差异有可测效应；跨 0 → 无证据。
"""
import hashlib
import json
import os
import sys
from urllib.request import urlopen

import numpy as np

SEED = 42
N_BOOT = 2000
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "prereg")
PREREG = os.path.join(ROOT, "PREREGISTRATION-2-Mol-split-effect.md")
SHA_FILE = os.path.join(OUT, "prereg_mol.sha256")
URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_prereg():
    if not os.path.exists(SHA_FILE):
        raise SystemExit("FATAL: 冻结 sha256 记录缺失，拒绝运行")
    frozen = open(SHA_FILE).read().split()[0]
    now = sha256_file(PREREG)
    if frozen != now:
        raise SystemExit(f"FATAL: 预注册已被改动（frozen={frozen[:12]} now={now[:12]}）")
    return now


def load_bbbp(path):
    import csv
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["smiles"], int(r["p_np"])))
    return rows


def canonicalize(smiles):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def scaffold_of(smiles):
    """照 bench_mol_gap.py 口径：传入 Mol 对象，keyword=mol。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem.Scaffolds import MurckoScaffold
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)


def _fill_groups(keyed_groups, n, frac):
    """按排好序的组填充，累加不越界（逐字照搬 bench_mol_gap.py:83-105 验证逻辑）。"""
    train_cutoff, valid_cutoff = frac[0] * n, (frac[0] + frac[1]) * n
    tr, va, te = [], [], []
    for group in keyed_groups:
        group = list(map(int, group))
        if len(tr) + len(group) > train_cutoff:
            if len(tr) + len(va) + len(group) > valid_cutoff:
                te.extend(group)
            else:
                va.extend(group)
        else:
            tr.extend(group)
    return np.asarray(tr, int), np.asarray(va, int), np.asarray(te, int)


def split_ours(groups, y, frac=(0.8, 0.1, 0.1), seed=SEED):
    """我们的实现：按簇大小升序、平局随机 shuffle（gap 基线口径）。"""
    rng = np.random.default_rng(seed)
    keyed = sorted(groups, key=lambda g: (len(g), rng.random()))
    return _fill_groups(keyed, len(y), frac)


def split_deepchem(groups, y, frac=(0.8, 0.1, 0.1)):
    """DeepChem 规则：骨架组按 (len, first_index) 降序（gap deepchem_exact 口径）。"""
    keyed = [sorted(map(int, g)) for g in groups]
    keyed = sorted(keyed, key=lambda g: (len(g), g[0]), reverse=True)
    return _fill_groups(keyed, len(y), frac)


def featurize(smiles_list):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import AllChem
    X, y = [], []
    for s, lab in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        X.append(np.asarray(fp, dtype=np.uint8))
        y.append(lab)
    return np.asarray(X), np.asarray(y)


def fit_auc(Xtr, ytr, Xte, yte, seed=SEED):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def main():
    prereg_sha = check_prereg()
    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(ROOT, "data", "BBBP.csv")
    if not os.path.exists(csv_path):
        with urlopen(URL, timeout=120) as r, open(csv_path, "wb") as f:
            f.write(r.read())
    rows = load_bbbp(csv_path)

    # canonical 去重（与 bench_mol_rigorous_v2 同口径）
    seen, uniq = {}, []
    invalid = dup = 0
    for smi, lab in rows:
        c = canonicalize(smi)
        if c is None:
            invalid += 1
            continue
        if c in seen:
            if seen[c] != lab:
                dup += 1
            continue
        seen[c] = lab
        uniq.append((c, lab))
    print(f"rows={len(rows)} unique={len(uniq)} invalid={invalid} dup={dup}", flush=True)

    smis = [u[0] for u in uniq]
    ys = np.asarray([u[1] for u in uniq])
    X, y = featurize(uniq)
    print(f"X={X.shape} pos={int(y.sum())} neg={int((1-y).sum())}", flush=True)

    # scaffold 分组
    scaf = [scaffold_of(s) for s in smis]
    groups = {}
    for i, g in enumerate(scaf):
        groups.setdefault(g, []).append(i)
    groups = [np.asarray(v, dtype=int) for v in groups.values()]
    print(f"n_scaffold_groups={len(groups)}", flush=True)

    out = {"prereg": "PREREGISTRATION-2-Mol-split-effect.md", "prereg_sha256": prereg_sha,
           "n_rows": len(rows), "n_unique": len(uniq), "n_scaffold": len(groups)}

    aucs = {}
    for name, fn in (("ours_shuffle", lambda: split_ours(groups, y)),
                     ("deepchem_order", lambda: split_deepchem(groups, y))):
        tr, va, te = fn()
        # 合并 train+valid 作训练（与 v2 口径一致：test 只做最终评估）
        tr_full = np.concatenate([np.asarray(tr, int), np.asarray(va, int)]).astype(int)
        te = np.asarray(te, int)
        a = fit_auc(X[tr_full], y[tr_full], X[te], y[te])
        aucs[name] = a
        out[name] = {"n_train": int(len(tr_full)), "n_test": int(len(te)),
                     "auc": a, "test_pos": int(y[te].sum()), "test_neg": int((1-y[te]).sum())}
        print(f"{name}: n_tr={len(tr_full)} n_te={len(te)} auc={a:.4f}", flush=True)

    # 配对 bootstrap：对 test 集重采样（同一 test 集，配对差）
    # 两切分 test 集不同 → 用各自 test 的 OOF 概率做配对不可行；改为
    # 分别 bootstrap 各自 AUC，报告差值 CI（独立重采样，保守）
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(N_BOOT):
        a1 = []
        a2 = []
        # 用正态近似从 AUC 的 bootstrap 分布抽样（DeLong 太远，用重采样 test 标签）
        diffs.append(aucs["ours_shuffle"] - aucs["deepchem_order"])
    # 由于 test 集不同，配对 bootstrap 不成立 → 改报「两套切分 AUC 差的 bootstrap 分布」
    # 用 test 集内的 bootstrap 重采样各自 AUC，再作差
    from sklearn.metrics import roc_auc_score
    tr1, va1, te1 = split_ours(groups, y)
    tr2, va2, te2 = split_deepchem(groups, y)
    tr1, va1, te1 = np.asarray(tr1, int), np.asarray(va1, int), np.asarray(te1, int)
    tr2, va2, te2 = np.asarray(tr2, int), np.asarray(va2, int), np.asarray(te2, int)
    from sklearn.ensemble import RandomForestClassifier
    clf1 = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1).fit(
        X[np.concatenate([tr1, va1])], y[np.concatenate([tr1, va1])])
    clf2 = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1).fit(
        X[np.concatenate([tr2, va2])], y[np.concatenate([tr2, va2])])
    p1 = clf1.predict_proba(X[te1])[:, 1]
    p2 = clf2.predict_proba(X[te2])[:, 1]
    d = []
    for _ in range(N_BOOT):
        i1 = rng.integers(0, len(te1), len(te1))
        i2 = rng.integers(0, len(te2), len(te2))
        d.append(roc_auc_score(y[te1][i1], p1[i1]) - roc_auc_score(y[te2][i2], p2[i2]))
    d = np.asarray(d)
    ci = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
    out["paired_diff"] = {"point": float(aucs["ours_shuffle"] - aucs["deepchem_order"]),
                          "ci95": ci,
                          "note": "两切分 test 集不同，CI 用各自 test 独立 bootstrap 重采样后作差（保守）"}
    out["verdict"] = {"ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0),
                      "statement": ("split 实现差异有可测效应" if (ci[0] > 0 or ci[1] < 0)
                                    else "无证据：CI 跨 0")}
    with open(os.path.join(OUT, "prereg_mol_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"aucs": aucs, "diff": out["paired_diff"], "verdict": out["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())