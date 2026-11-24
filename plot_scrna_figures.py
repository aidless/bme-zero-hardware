"""Publication-quality figures for the PBMC3k single-cell project.
Consumes data/pbmc3k/processed_advanced.h5ad; writes PNGs into figures/.
Run: python3 plot_scrna_figures.py
"""
import os
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sc.settings.verbosity = 0
sc.settings.figdir = "figures"
os.makedirs("figures", exist_ok=True)

# publication-ish defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "svg.fonttype": "none",
})
sc.set_figure_params(dpi=150, dpi_save=150, frameon=False, fontsize=9, figsize=(6, 5))

adata = sc.read_h5ad("data/pbmc3k/processed_advanced.h5ad")
print(f"[load] {adata.n_obs} cells, cell types: {adata.obs['cell_type'].unique().tolist()}")

palette = {
    "CD8 T cells": "#1f77b4", "CD4 T cells": "#17becf", "B cells": "#2ca02c",
    "NK cells": "#9467bd", "CD14+ monocytes": "#d62728", "platelets": "#8c564b",
    "cycling progenitors": "#ff7f0e",
}

# 1. UMAP by cell type (labels on top of clusters)
sc.pl.umap(
    adata, color="cell_type", palette=palette, legend_loc="on data",
    legend_fontsize=8, legend_fontoutline=2, title="PBMC3k — cell types (Leiden)",
    show=False, save="_celltypes.png",
)
print("[fig] figures/umap_celltypes.png")

# 2. UMAP colored by DPT pseudotime
sc.pl.umap(
    adata, color="dpt_pseudotime", cmap="viridis", title="DPT pseudotime trajectory",
    show=False, save="_pseudotime.png",
)
print("[fig] figures/umap_pseudotime.png")

# 3. PAGA lineage overlaid on UMAP
sc.pl.paga(
    adata, color="leiden", cmap="tab10", layout="fr",
    threshold=0.15, node_size_scale=1.4, edge_width_scale=1.4,
    title="PAGA lineage (connectivity > 0.15)", show=False, save="_paga.png",
)
print("[fig] figures/paga.png")

# 4. Canonical marker dotplot
markers = ["CD3D", "CD8A", "CD4", "MS4A1", "CD79A", "NKG7", "GNLY",
           "LYZ", "CD14", "FCGR3A", "PPBP", "MKI67"]
markers = [g for g in markers if g in adata.raw.var_names]
# marker genes also live in .obs from the annotation step -> drop to avoid
# scanpy's ambiguous-key error when it checks both obs and raw.var_names
amb = adata.obs.columns.intersection(markers)
plot_adata = adata.copy()
plot_adata.obs = plot_adata.obs.drop(columns=list(amb))
sc.pl.dotplot(
    plot_adata, markers, groupby="cell_type", standard_scale="var",
    cmap="Blues", figsize=(9, 3.6), show=False, save="_marker_dotplot.png",
)
print("[fig] figures/marker_dotplot.png")

# 5. PAGA connectivity heatmap
conn = adata.uns["paga"]["connectivities"].toarray()
clusters = sorted(adata.obs["leiden"].unique(), key=int)
import numpy as np
fig, ax = plt.subplots(figsize=(5, 4.3))
im = ax.imshow(conn, cmap="magma", vmin=0, vmax=1)
ax.set_xticks(range(len(clusters))); ax.set_xticklabels(clusters)
ax.set_yticks(range(len(clusters))); ax.set_yticklabels(clusters)
ax.set_xlabel("cluster"); ax.set_ylabel("cluster"); ax.set_title("PAGA connectivity")
for i in range(len(clusters)):
    for j in range(len(clusters)):
        if conn[i, j] > 0.4:
            ax.text(j, i, f"{conn[i,j]:.2f}", ha="center", va="center",
                    color="cyan", fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig("figures/paga_heatmap.png", bbox_inches="tight")
plt.close(fig)
print("[fig] figures/paga_heatmap.png")

print("\nDone. 5 figures in figures/")
