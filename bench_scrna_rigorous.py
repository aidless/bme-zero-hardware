"""Audit-grade re-check of the PBMC3k single-cell results.

Two reviewer-grade concerns with the original mini_project_scrna.py /
adv_scrna_communication.py:
  1. Clustering stability: is the Leiden partition an artifact of one
     resolution / seed? -> consensus ARI across seeds and resolutions.
  2. Multiple testing: 126 "significant" LR interactions at p<0.05 came from
     ~735 tests with NO correction -> apply Benjamini-Hochberg FDR.

Run: python3 bench_scrna_rigorous.py
Data: data/pbmc3k/processed.h5ad (from mini_project_scrna.py)
"""
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
from scipy.stats import rankdata

sc.settings.verbosity = 0
a = sc.read_h5ad("data/pbmc3k/processed.h5ad")
print(f"[load] {a.n_obs} cells x {a.n_vars} HVG, leiden labels present: {'leiden' in a.obs}")

# ---- 1. clustering stability: ARI of each new run vs the baseline leiden ----
base = a.obs["leiden"].astype(str).values
print("\n[clustering stability] ARI vs baseline leiden partition:")
for res in [0.6, 0.8, 1.0, 1.2]:
    row = []
    for seed in [0, 42, 123]:
        b = a.copy()
        sc.pp.neighbors(b, n_neighbors=10, n_pcs=40, random_state=seed)
        sc.tl.leiden(b, resolution=res, flavor="igraph", key_added="tmp", random_state=seed)
        row.append(adjusted_rand_score(base, b.obs["tmp"].astype(str).values))
    print(f"  res={res}: ARI seeds(0,42,123) = " +
          ", ".join(f"{v:.3f}" for v in row) + f"  mean={np.mean(row):.3f}")

# ---- 2. multiple-testing correction on the communication p-values ----
# Recompute p-values exactly as adv_scrna_communication.py does, then BH-FDR.
import pandas as pd
adv = sc.read_h5ad("data/pbmc3k/processed_advanced.h5ad")
labels = adv.obs["cell_type"].astype(str).values
ctypes = sorted(set(labels))
LR = [("CCL5","CCR5"),("CCL3","CCR5"),("CCL4","CCR5"),("IL7","IL7R"),
      ("IFNG","IFNGR1"),("TNF","TNFRSF1A"),("TGFB1","TGFBR1"),
      ("CD40LG","CD40"),("CTLA4","CD86"),("PDCD1","CD274"),
      ("LGALS9","HAVCR2"),("LAG3","HLA-DRA"),("MIF","CD74"),
      ("CD74","MIF"),("LGALS1","CD69")]
X = adv.raw.X.toarray() if hasattr(adv.raw.X, "toarray") else np.asarray(adv.raw.X)
genes = list(adv.raw.var_names); gidx = {g: i for i, g in enumerate(genes)}
Xn = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)
L = np.column_stack([Xn[:, gidx[l]] for l, r in LR])
R = np.column_stack([Xn[:, gidx[r]] for l, r in LR])
K = len(LR)
def cm(M, lab):
    out = np.zeros((len(ctypes), M.shape[1]))
    for i, c in enumerate(ctypes):
        out[i] = M[lab == c].mean(axis=0)
    return out
obs = cm(L, labels)[:, None, :] * cm(R, labels)[None, :, :]
rng = np.random.default_rng(42)
NPERM = 1000
null = np.empty((NPERM, len(ctypes), len(ctypes), K))
for p in range(NPERM):
    sh = rng.permutation(labels)
    null[p] = cm(L, sh)[:, None, :] * cm(R, sh)[None, :, :]
pval = (null >= obs).mean(axis=0)

# flatten all sender!=receiver tests
recs = []
for si, cs in enumerate(ctypes):
    for ri, cr in enumerate(ctypes):
        if si == ri:
            continue
        for k, (l, rec) in enumerate(LR):
            recs.append((pval[si, ri, k], obs[si, ri, k], cs, cr, l, rec))
n_tests = len(recs)
p_arr = np.array([r[0] for r in recs])
order = np.argsort(p_arr)
m = n_tests
q_sorted = np.empty(m)
prev = 1.0
for rank in range(m, 0, -1):
    idx = order[rank - 1]
    val = p_arr[idx] * m / rank
    prev = min(prev, val)
    q_sorted[idx] = prev          # store at ORIGINAL index, not sorted rank
n_raw = int((p_arr < 0.05).sum())
n_fdr = int((q_sorted < 0.05).sum())
print(f"\n[multiple testing] {n_tests} sender-receiver-LR tests:")
print(f"  p<0.05 (uncorrected):     {n_raw} significant")
print(f"  BH-FDR q<0.05:            {n_fdr} significant")
print(f"  expected false positives at p<0.05 under global null: {0.05*n_tests:.0f}")
# top survivors after FDR
surv = [(recs[i][1], recs[i][2], recs[i][3], recs[i][4], recs[i][5], p_arr[i], q_sorted[i])
        for i in range(m) if q_sorted[i] < 0.05]
surv.sort(key=lambda x: -x[0])
print("  top FDR-surviving interactions:")
for s, cs, cr, l, rec, pv, qv in surv[:8]:
    print(f"    {cs:18s}->{cr:18s} {l:6s}-{rec:7s} score={s:6.3f} p={pv:.3f} q={qv:.3f}")
