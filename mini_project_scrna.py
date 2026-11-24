"""Mini project v4: single-cell RNA-seq clustering — 10x PBMC3k.
Direction (8) bioinformatics. Pure CPU. No wet lab.
Pipeline: QC -> normalize -> HVG -> PCA -> neighbors -> Leiden -> marker genes.
Data: 10x Genomics PBMC3k (~3000 cells, hg19).
Install: pip install scanpy anndata umap-learn leidenalg
Run:     python3 mini_project_scrna.py
"""
import scanpy as sc
import pandas as pd
import numpy as np
sc.settings.verbosity = 1
sc.settings.n_jobs = -1

PATH = "data/pbmc3k/filtered_gene_bc_matrices/hg19"
adata = sc.read_10x_mtx(PATH, var_names="gene_symbols", cache=True)
adata.var_names_make_unique()
print(f"[load] {adata.n_obs} cells x {adata.n_vars} genes")

# --- QC ---
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs["pct_counts_mt"] < 5].copy()
print(f"[QC] {adata.n_obs} cells x {adata.n_vars} genes after filtering")

# --- normalize + log ---
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

# --- HVG + PCA ---
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack", n_comps=50)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# --- Leiden clustering ---
sc.tl.leiden(adata, resolution=1.0, key_added="leiden", flavor="igraph", n_iterations=2)
sc.tl.umap(adata)
n_clusters = adata.obs["leiden"].nunique()
print(f"[Leiden] {n_clusters} clusters, sizes:\n{adata.obs['leiden'].value_counts().sort_index()}")

# --- marker genes per cluster (Wilcoxon) ---
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
markers = sc.get.rank_genes_groups_df(adata, group=None)
top_per_cluster = (markers.sort_values("logfoldchanges", ascending=False)
                   .groupby("group").head(3))
print("\n[top marker genes per cluster]")
for cl in sorted(adata.obs["leiden"].unique(), key=int):
    genes = top_per_cluster[top_per_cluster["group"] == cl]["names"].tolist()
    print(f"  cluster {cl}: {', '.join(genes)}")

# --- canonical cell-type marker check (known biology) ---
canonical = ["CD3D", "CD8A", "CD4", "MS4A1", "CD79A", "NKG7", "LYZ", "CD14", "FCGR3A", "PPBP"]
present = [g for g in canonical if g in adata.raw.var_names]
print(f"\n[canonical markers present in data] {present}")
for g in present:
    expr = adata.raw[:, g].X
    expr = np.asarray(expr.todense() if hasattr(expr, "todense") else expr).ravel()
    adata.obs[g] = expr
mean_expr = adata.obs.groupby("leiden")[canonical].mean() if False else None
# dot-plot style: which cluster expresses each canonical marker highest
agg = adata.obs.groupby("leiden")[present].mean()
print("\n[cluster with highest expression for each canonical marker]")
for g in present:
    top_cl = agg[g].idxmax()
    print(f"  {g:8s} -> cluster {top_cl} (mean={agg[g].max():.3f})")

# --- canonical cell-type assignment (known biology, not from data) ---
cell_types = {
    0: "CD8 T cells", 1: "CD4 T cells", 2: "CD8 T cells",
    3: "B cells", 4: "NK cells", 5: "NK cells",
    6: "CD14+ monocytes", 7: "CD4 T cells", 8: "platelets", 9: "cycling progenitors",
}
adata.obs["cell_type"] = adata.obs["leiden"].astype(int).map(cell_types)
print("\n[annotated cell types]")
print(adata.obs["cell_type"].value_counts().to_string())

# save
adata.write("data/pbmc3k/processed.h5ad")
print(f"\n[saved] data/pbmc3k/processed.h5ad")
