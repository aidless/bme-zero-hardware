"""Cross-tissue extension (P1-10): run the PBMC pipeline on 10x heart_1k_v3.

The downloaded sample is mouse (mm10), not human: the matrix's features/genome
column is the ground truth. That makes this a tissue AND species extension.
All measurements are written to data/heart_1k_v3/results.json.

Run: python3 bench_scrna_cross_tissue.py
"""
import json
import os
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

SEED = 0
RES = 1.0
HERE = "data/heart_1k_v3"
MATRIX = f"{HERE}/heart_1k_v3_filtered_feature_bc_matrix.h5"
PBMC = "data/pbmc3k/processed.h5ad"

# Exact human PBMC3k canonical set and mouse orthologs present in mm10.
HUMAN_MARKERS = ["CD3D", "CD8A", "CD79A", "MS4A1", "NKG7", "LYZ", "CD14", "PPBP"]
ORTHOLOGS = {
    "CD3D": "Cd3d", "CD8A": "Cd8a", "CD79A": "Cd79a", "MS4A1": "Ms4a1",
    "NKG7": "Nkg7", "LYZ": "Lyz2", "CD14": "Cd14", "PPBP": "Ppbp",
}
CARDIAC_MARKERS = ["Tnnt2", "Myh6", "Myh7", "Actc1", "Tnni3", "Nppa", "Myl7",
                   "Col1a1", "Col3a1", "Dcn", "Lum", "Pecam1", "Vwf", "Cdh5",
                   "Acta2", "Tagln", "Myh11", "Pdgfrb", "Cspg4", "Ptprc",
                   "C1qa", "C1qb", "Adgre1", "Mki67", "Pf4", "Hbb-bs"]

sc.settings.verbosity = 0


def logcp10k(adata):
    X = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else np.asarray(adata.raw.X)
    return np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4), list(adata.raw.var_names)


def classify(profile):
    """Marker-driven annotation of heart clusters; fixed rules, no cluster IDs."""
    out = []
    for _, r in profile.iterrows():
        if r.get("Pf4", 0) > 2.0 and r.get("Ppbp", 0) > 1.0:
            out.append("platelets")
        elif r.get("Mki67", 0) > 1.0:
            out.append("cycling cells")
        elif r.get("Ptprc", 0) > 1.0 and r.get("Lyz2", 0) > 1.0:
            out.append("myeloid/immune")
        elif r.get("Ptprc", 0) > 1.0:
            out.append("immune (Ptprc+)")
        elif max(r.get("Tnnt2", 0), r.get("Myh6", 0), r.get("Myh7", 0), r.get("Actc1", 0)) > 1.0:
            out.append("cardiomyocytes")
        elif max(r.get("Col1a1", 0), r.get("Col3a1", 0), r.get("Dcn", 0), r.get("Lum", 0)) > 1.0:
            out.append("fibroblasts")
        elif max(r.get("Pecam1", 0), r.get("Vwf", 0), r.get("Cdh5", 0)) > 1.0:
            out.append("endothelial")
        elif max(r.get("Acta2", 0), r.get("Tagln", 0), r.get("Myh11", 0), r.get("Pdgfrb", 0)) > 1.0:
            out.append("mural/smooth muscle")
        else:
            out.append("ambiguous")
    return out


def main():
    import h5py
    with h5py.File(MATRIX) as f:
        genomes = [x.decode() for x in f["matrix/features/genome"][:]]
    provenance = json.load(open(f"{HERE}/provenance.json"))
    assert set(genomes) == {"mm10"}, set(genomes)
    results = {
        "provenance": provenance,
        "organism_measurement": {"genome_values": sorted(set(genomes)),
                                 "n_features": len(genomes)},
        "pipeline": {},
        "canonical_markers": {},
        "heart_markers": {},
        "annotation": {},
        "batch_integration_probe": {},
    }

    a = sc.read_10x_h5(MATRIX)
    a.var_names_make_unique()
    results["pipeline"]["raw_shape"] = [int(a.n_obs), int(a.n_vars)]

    a.var["mt"] = a.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True)
    n0 = a.n_obs
    sc.pp.filter_cells(a, min_genes=200)
    n1 = a.n_obs
    sc.pp.filter_genes(a, min_cells=3)
    v1 = a.n_vars
    a = a[a.obs["pct_counts_mt"] < 5].copy()
    results["pipeline"]["qc"] = {
        "cells_before": int(n0), "cells_after_min_genes_200": int(n1),
        "cells_after_mt_lt_5": int(a.n_obs),
        "genes_before": 31053, "genes_after_min_cells_3": int(v1),
        "mitochondrial_prefix": "mt- (mouse convention)",
    }

    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, svd_solver="arpack", n_comps=50, random_state=SEED)
    sc.pp.neighbors(a, n_neighbors=10, n_pcs=40, random_state=SEED)
    sc.tl.leiden(a, resolution=RES, flavor="igraph", key_added="leiden",
                 random_state=SEED, n_iterations=2)
    sc.tl.umap(a, random_state=SEED)
    n_clusters = int(a.obs["leiden"].nunique())
    results["pipeline"]["clustering"] = {
        "resolution": RES, "n_clusters": n_clusters,
        "cluster_sizes": a.obs["leiden"].value_counts().sort_index().astype(int).to_dict(),
        "n_hvg": int(a.n_vars), "n_pcs": 50, "n_neighbors": 10, "seed": SEED,
    }

    Xn, genes = logcp10k(a)
    gidx = {g: i for i, g in enumerate(genes)}
    present_human = [g for g in HUMAN_MARKERS if g in gidx]
    present_ortholog = {h: m for h, m in ORTHOLOGS.items() if m in gidx}
    results["canonical_markers"] = {
        "exact_human_symbols": HUMAN_MARKERS,
        "exact_human_symbols_present": present_human,
        "n_exact_human_symbols_present": len(present_human),
        "mouse_orthologs_measured": present_ortholog,
        "interpretation": "Exact human symbols do not transfer (case/species convention); "
                          "mouse orthologs are the biologically comparable measurement.",
    }

    clusters = sorted(a.obs["leiden"].unique())
    profile = {}
    for g in present_ortholog.values():
        profile[g] = [float(Xn[a.obs["leiden"].values == c, gidx[g]].mean()) for c in clusters]
    profile = pd.DataFrame(profile, index=clusters)
    heart_profile = {}
    for g in CARDIAC_MARKERS:
        if g in gidx:
            heart_profile[g] = [float(Xn[a.obs["leiden"].values == c, gidx[g]].mean()) for c in clusters]
    heart_profile = pd.DataFrame(heart_profile, index=clusters)
    labels = classify(heart_profile)
    ann = dict(zip(clusters, labels))
    a.obs["cell_type"] = a.obs["leiden"].astype(str).map(ann)

    results["heart_markers"] = {
        "mean_logcp10k_by_cluster": heart_profile.round(3).to_dict(orient="index"),
        "peak_cluster_by_marker": {
            g: {"cluster": str(heart_profile[g].idxmax()), "mean": round(float(heart_profile[g].max()), 3)}
            for g in heart_profile.columns},
    }
    results["annotation"] = {
        "rules": "fixed marker thresholds: Pf4/Ppbp platelets; Mki67 cycling; Ptprc+Lyz2 myeloid; "
                 "Ptprc immune; Tnnt2/Myh6/Myh7/Actc1 cardiomyocyte; Col1a1/Col3a1/Dcn/Lum fibroblast; "
                 "Pecam1/Vwf/Cdh5 endothelial; Acta2/Tagln/Myh11/Pdgfrb mural; else ambiguous",
        "cluster_to_cell_type": ann,
        "cell_type_counts": a.obs["cell_type"].value_counts().astype(int).to_dict(),
    }
    results["canonical_markers"]["mouse_ortholog_mean_logcp10k_by_cluster"] = (
        profile.round(3).to_dict(orient="index"))
    results["canonical_markers"]["mouse_ortholog_peak_cluster"] = {
        h: {"mouse_symbol": m, "cluster": str(profile[m].idxmax()),
            "mean": round(float(profile[m].max()), 3),
            "annotation": ann[str(profile[m].idxmax())]}
        for h, m in present_ortholog.items()}

    # PBMC exact marker availability, for the tissue-boundary table.
    pbmc = sc.read_h5ad(PBMC)
    results["canonical_markers"]["pbmc_exact_symbols_present"] = [
        g for g in HUMAN_MARKERS if g in pbmc.raw.var_names]

    # Batch integration capability probe (no deep-dive): BBKNN import + combat smoke.
    try:
        import bbknn  # noqa: F401
        results["batch_integration_probe"]["bbknn"] = "installed"
    except Exception as exc:
        results["batch_integration_probe"]["bbknn"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        # One-dataset ComBat is ill-posed (only one level of the batch factor);
        # we demonstrate it on the two-sample concatenated object with common
        # upper-cased symbols and record whatever scanpy actually reports.
        pv = pbmc.copy()
        pv.var_names = pv.var_names.str.upper()
        hv = a.copy()
        hv.var_names = hv.var_names.str.upper()
        common = sorted(set(pv.var_names) & set(hv.var_names))
        pv = pv[:, common].copy(); hv = hv[:, common].copy()
        pv.obs["batch"] = "PBMC"; hv.obs["batch"] = "heart"
        import anndata as ad
        combined = ad.concat([pv, hv], join="inner", label="batch",
                             keys=["PBMC", "heart"], index_unique="-")
        combined.obs["batch"] = combined.obs["batch"].astype("category")
        sc.pp.combat(combined, key="batch")
        results["batch_integration_probe"]["combat"] = {
            "status": "ran", "n_common_uppercase_symbols": len(common),
            "n_obs": int(combined.n_obs),
            "caveat": "upper-case symbol matching across species is only a smoke test; "
                      "it is not validated ortholog mapping",
        }
    except Exception as exc:
        results["batch_integration_probe"]["combat"] = {
            "status": "error", "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc().splitlines()[-3:],
        }
    results["batch_integration_probe"]["conclusion"] = (
        "Before cross-tissue cell-type comparison, both ortholog mapping (human->mouse) "
        "and batch integration are prerequisites; marker labels and communication "
        "networks are only valid within their tissue/species context."
    )

    a.write(f"{HERE}/processed_heart.h5ad")
    results["output_h5ad"] = f"{HERE}/processed_heart.h5ad"

    fig, ax = plt.subplots(figsize=(7, 5))
    sc.pl.umap(a, color="cell_type", ax=ax, show=False, title="heart_1k_v3 Leiden res=1.0")
    fig.tight_layout()
    fig.savefig("figures/scrna_heart_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_genes = ["Tnnt2", "Myh6", "Dcn", "Col1a1", "Pecam1", "Acta2", "Ptprc", "Lyz2", "Nkg7", "Cd3d"]
    plot_genes = [g for g in plot_genes if g in heart_profile.columns or g in profile.columns]
    mat = pd.DataFrame({g: heart_profile[g] if g in heart_profile else profile[g] for g in plot_genes})
    im = ax.imshow(mat.T.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(clusters))); ax.set_xticklabels(clusters)
    ax.set_yticks(range(len(plot_genes))); ax.set_yticklabels(plot_genes)
    ax.set_xlabel("Leiden cluster"); ax.set_title("Heart marker mean logCP10K")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig("figures/scrna_heart_markers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(f"{HERE}/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "n_cells": a.n_obs, "n_clusters": n_clusters,
        "cell_type_counts": results["annotation"]["cell_type_counts"],
        "exact_human_markers_present": present_human,
        "results_json": f"{HERE}/results.json",
        "combat_probe": results["batch_integration_probe"]["combat"],
        "bbknn_probe": results["batch_integration_probe"]["bbknn"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
