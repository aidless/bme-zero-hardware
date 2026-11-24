"""Advanced v3: trajectory inference + differential expression on PBMC3k.
Goal: add PAGA lineage + pseudotime + DE marker genes on top of clustering.
Run: python3 adv_scrna.py (uses data/pbmc3k/processed.h5ad from mini_project_scrna.py)
"""
import scanpy as sc
import numpy as np
import pandas as pd
sc.settings.verbosity=1

adata = sc.read_h5ad("data/pbmc3k/processed.h5ad")
print(f"[load] {adata.n_obs} cells, {adata.n_vars} HVGs, {adata.obs['leiden'].nunique()} clusters")

# --- differential expression: Wilcoxon test per cluster (one-vs-rest) ---
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", use_raw=True)
de = sc.get.rank_genes_groups_df(adata, group=None)
de_sig = de[(de["pvals_adj"] < 0.05) & (de["logfoldchanges"] > 1)]
n_de = de_sig.groupby("group").size()
print(f"[DE] significant marker genes (adj p<0.05, logFC>1) per cluster:")
print(n_de.to_string())

# --- PAGA trajectory (lineage among clusters) ---
sc.tl.paga(adata, groups="leiden")
# report connectivity above threshold
conn = adata.uns["paga"]["connectivities"].toarray()
clusters = sorted(adata.obs["leiden"].unique(), key=int)
edges=[]
for i in range(len(clusters)):
    for j in range(i+1,len(clusters)):
        if conn[i,j] > 0.1:
            edges.append((clusters[i],clusters[j],round(float(conn[i,j]),2)))
print(f"\n[PAGA] {len(edges)} edges with connectivity>0.1 (top 10):")
for e in sorted(edges,key=lambda x:-x[2])[:10]:
    print(f"  cluster {e[0]} -- {e[1]}: {e[2]}")

# --- diffusion pseudotime (DPT) ---
sc.tl.diffmap(adata)
# choose root = a progenitor-like cluster (cycling) if present, else cluster 0
root_mask = adata.obs["leiden"].astype(int) == 9
if root_mask.sum() > 0:
    adata.uns["iroot"] = np.where(root_mask.values)[0][0]
else:
    adata.uns["iroot"] = 0
sc.tl.dpt(adata)
pt = adata.obs["dpt_pseudotime"]
print(f"\n[DPT] pseudotime range={pt.min():.3f}-{pt.max():.3f}, mean={pt.mean():.3f}")
pt_by_type = adata.obs.groupby("cell_type")["dpt_pseudotime"].mean().sort_values()
print("[DPT] mean pseudotime by cell type:")
print(pt_by_type.to_string())

adata.write("data/pbmc3k/processed_advanced.h5ad")
print("\n[saved] data/pbmc3k/processed_advanced.h5ad")
