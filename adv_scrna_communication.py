"""Cell-cell communication (simplified CellPhoneDB) on PBMC3k.
Ligand-receptor mean-expression product + label-permutation null.
Consumes data/pbmc3k/processed_advanced.h5ad; writes numbers to stdout and
a sender x receiver communication heatmap to figures/.
Run: python3 adv_scrna_communication.py
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
NPERM = 1000
rng = np.random.default_rng(SEED)
sc.settings.verbosity = 0

a = sc.read_h5ad("data/pbmc3k/processed_advanced.h5ad")
labels = a.obs["cell_type"].astype(str).values
ctypes = sorted(set(labels))
print(f"[load] {a.n_obs} cells, {len(ctypes)} cell types")

# canonical human ligand -> receptor pairs known present in this dataset
LR = [("CCL5","CCR5"),("CCL3","CCR5"),("CCL4","CCR5"),("IL7","IL7R"),
      ("IFNG","IFNGR1"),("TNF","TNFRSF1A"),("TGFB1","TGFBR1"),
      ("CD40LG","CD40"),("CTLA4","CD86"),("PDCD1","CD274"),
      ("LGALS9","HAVCR2"),("LAG3","HLA-DRA"),("MIF","CD74"),
      ("CD74","MIF"),("LGALS1","CD69")]

# work from raw counts -> CP10K -> log1p (proper normalization)
X = a.raw.X.toarray() if hasattr(a.raw.X, "toarray") else np.asarray(a.raw.X)
genes = list(a.raw.var_names)
gidx = {g: i for i, g in enumerate(genes)}
lib = X.sum(axis=1, keepdims=True)
Xn = np.log1p(X / lib * 1e4)

L = np.column_stack([Xn[:, gidx[l]] for l, r in LR])   # ligand
R = np.column_stack([Xn[:, gidx[r]] for l, r in LR])   # receptor
K = len(LR)

def cluster_mean(M, lab):
    out = np.zeros((len(ctypes), M.shape[1]))
    for i, c in enumerate(ctypes):
        out[i] = M[lab == c].mean(axis=0)
    return out

Lm = cluster_mean(L, labels)   # sender ligand mean
Rm = cluster_mean(R, labels)   # receiver receptor mean
obs = Lm[:, None, :] * Rm[None, :, :]   # [sender, receiver, pair]

# permutation null: shuffle labels, recompute both means
null = np.empty((NPERM, len(ctypes), len(ctypes), K))
for p in range(NPERM):
    sh = rng.permutation(labels)
    lm = cluster_mean(L, sh); rm = cluster_mean(R, sh)
    null[p] = lm[:, None, :] * rm[None, :, :]
pval = (null >= obs).mean(axis=0)
print(f"[perm] {NPERM} permutations, null built")

# significant interactions (p<0.05), sorted by effect size
sig = []
for s, cs in enumerate(ctypes):
    for r, cr in enumerate(ctypes):
        if s == r: continue
        for k, (l, rec) in enumerate(LR):
            if pval[s, r, k] < 0.05:
                sig.append((obs[s, r, k], cs, cr, l, rec, pval[s, r, k]))
sig.sort(key=lambda x: -x[0])
print(f"\n[result] {len(sig)} significant sender->receiver LR interactions (p<0.05):")
for sc_, cs, cr, l, rec, pv in sig[:12]:
    print(f"  {cs:22s} -> {cr:22s}  {l:7s}-{rec:7s}  score={sc_:7.3f}  p={pv:.3f}")

# total communication strength heatmap (sum of significant scores only)
tot = np.zeros((len(ctypes), len(ctypes)))
for sc_, cs, cr, l, rec, pv in sig:
    tot[ctypes.index(cs), ctypes.index(cr)] += sc_

fig, ax = plt.subplots(figsize=(7, 5.6))
im = ax.imshow(tot, cmap="magma")
ax.set_xticks(range(len(ctypes))); ax.set_xticklabels(ctypes, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(ctypes))); ax.set_yticklabels(ctypes, fontsize=8)
ax.set_xlabel("receiver"); ax.set_ylabel("sender")
ax.set_title("Significant cell-cell communication (sum LR score, p<0.05)")
for i in range(len(ctypes)):
    for j in range(len(ctypes)):
        if tot[i, j] > 0:
            ax.text(j, i, f"{tot[i,j]:.1f}", ha="center", va="center",
                    color="cyan", fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig("figures/communication_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n[fig] figures/communication_heatmap.png")
