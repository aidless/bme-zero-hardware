"""Resolution re-check for PBMC3k Leiden clusters (P1-7).

For each resolution in (0.5, 0.7, 1.0):
  - recompute Leiden from the stored kNN graph of processed.h5ad,
  - compare with the baseline res=1.0 partition by ARI,
  - compute canonical-marker expression peaks and marker-driven annotations,
  - transfer baseline cell_type by majority vote for annotation stability,
  - rerun the 15-LR permutation communication test and apply BH-FDR.

Run: python3 bench_scrna_resolution.py
Outputs: data/pbmc3k/processed_res*.h5ad, figures/scrna_resolution_*.png,
         measured JSON on stdout (copied verbatim into the dated note).
"""
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

SEED = 0
NPERM = 1000
RESOLUTIONS = [0.5, 0.7, 1.0]
MARKERS = ["CD3D", "CD8A", "CD79A", "MS4A1", "NKG7", "LYZ", "CD14", "PPBP"]
LR = [("CCL5", "CCR5"), ("CCL3", "CCR5"), ("CCL4", "CCR5"), ("IL7", "IL7R"),
      ("IFNG", "IFNGR1"), ("TNF", "TNFRSF1A"), ("TGFB1", "TGFBR1"),
      ("CD40LG", "CD40"), ("CTLA4", "CD86"), ("PDCD1", "CD274"),
      ("LGALS9", "HAVCR2"), ("LAG3", "HLA-DRA"), ("MIF", "CD74"),
      ("CD74", "MIF"), ("LGALS1", "CD69")]

sc.settings.verbosity = 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalized_raw(adata):
    """Log-CP10K values from raw counts; the baseline used the same transform."""
    X = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else np.asarray(adata.raw.X)
    return np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)


def annotate_from_markers(profile):
    """Fixed, transcription-based rules; no cluster IDs or baseline labels used."""
    labels = []
    for _, r in profile.iterrows():
        if r["PPBP"] > 2.0:
            labels.append("platelets")
        elif r["CD79A"] > 1.0 and r["MS4A1"] > 0.8:
            labels.append("B cells")
        elif r["LYZ"] > 2.0 and r["CD14"] > 0.2:
            labels.append("CD14+ monocytes")
        elif r["NKG7"] > 2.0 and r["CD3D"] < 0.8:
            labels.append("NK cells")
        elif r["CD3D"] > 1.0 and r["CD8A"] > 0.4:
            labels.append("CD8 T cells")
        elif r["CD3D"] > 1.0:
            labels.append("CD4 T cells")
        else:
            labels.append("ambiguous")
    return labels


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m, dtype=float)
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        prev = min(prev, p[idx] * m / rank)
        q[idx] = prev
    return q


def communication(Xn, genes, labels):
    """CellPhoneDB-like mean product + permutation null + BH-FDR."""
    gidx = {g: i for i, g in enumerate(genes)}
    L = np.column_stack([Xn[:, gidx[l]] for l, _ in LR])
    R = np.column_stack([Xn[:, gidx[r]] for _, r in LR])
    groups = sorted(set(labels))
    k = len(LR)

    def means(lab):
        onehot = np.column_stack([(lab == c).astype(float) for c in groups])
        inv = 1.0 / np.maximum(onehot.sum(axis=0), 1)
        return (onehot.T @ L) * inv[:, None], (onehot.T @ R) * inv[:, None]

    lm, rm = means(labels)
    obs = lm[:, None, :] * rm[None, :, :]
    rng = np.random.default_rng(SEED)
    null = np.empty((NPERM, len(groups), len(groups), k), dtype=np.float32)
    for p in range(NPERM):
        sl, sr = means(rng.permutation(labels))
        null[p] = sl[:, None, :] * sr[None, :, :]
    pval = (null >= obs).mean(axis=0)

    recs = []
    for si, sender in enumerate(groups):
        for ri, receiver in enumerate(groups):
            if si == ri:
                continue
            for ki, (lig, rec) in enumerate(LR):
                recs.append((pval[si, ri, ki], obs[si, ri, ki], sender, receiver, lig, rec))
    p_arr = np.array([r[0] for r in recs])
    q_arr = bh_fdr(p_arr)
    survivors = sorted(
        [(float(r[1]), r[2], r[3], r[4], r[5], float(p_arr[i]), float(q_arr[i]))
         for i, r in enumerate(recs) if q_arr[i] < 0.05],
        key=lambda x: -x[0],
    )
    return {
        "n_groups": len(groups),
        "n_tests": len(recs),
        "p_lt_0.05": int((p_arr < 0.05).sum()),
        "q_lt_0.05": int((q_arr < 0.05).sum()),
        "top_fdr_survivors": [
            {"score": round(x[0], 4), "sender": x[1], "receiver": x[2],
             "LR": f"{x[3]}-{x[4]}", "p": round(x[5], 5), "q": round(x[6], 5)}
            for x in survivors[:5]
        ],
    }


def main():
    src = "data/pbmc3k/processed.h5ad"
    a = sc.read_h5ad(src)
    base_labels = a.obs["leiden"].astype(str).values
    Xn = normalized_raw(a)
    genes = list(a.raw.var_names)
    results = {
        "source_file": src,
        "source_sha256": sha256(src),
        "n_cells": int(a.n_obs),
        "n_hvg": int(a.n_vars),
        "n_raw_genes": int(a.raw.shape[1]),
        "method": {
            "leiden": "stored kNN graph n_neighbors=10 n_pcs=40; flavor=igraph, n_iterations=2, random_state=0",
            "marker_values": "log1p(CP10K) from .raw counts",
            "communication": f"{len(LR)} LR pairs, sender!=receiver, {NPERM} label permutations, BH-FDR",
        },
        "resolutions": {},
        "verification": {},
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, res in zip(axes, RESOLUTIONS):
        b = a.copy()
        key = f"leiden_res_{str(res).replace('.', '')}"
        sc.tl.leiden(b, resolution=res, key_added=key, flavor="igraph",
                     random_state=SEED, n_iterations=2)
        lab = b.obs[key].astype(str).values
        n_clusters = int(b.obs[key].nunique())

        profile = pd.DataFrame(
            {g: [float(Xn[b.obs[key].values == c, genes.index(g)].mean())
                 for c in sorted(b.obs[key].unique())] for g in MARKERS},
            index=sorted(b.obs[key].unique()),
        )
        marker_labels = dict(zip(profile.index, annotate_from_markers(profile)))
        b.obs["cell_type_marker"] = b.obs[key].astype(str).map(marker_labels)
        # transfer baseline annotation by majority vote for stability comparison
        transferred = {}
        for c in sorted(b.obs[key].unique()):
            vc = a.obs.loc[b.obs[key].values == c, "cell_type"].value_counts()
            transferred[c] = str(vc.index[0]) if len(vc) else "unassigned"
        b.obs["cell_type_majority"] = b.obs[key].astype(str).map(transferred)

        peaks = {}
        for g in MARKERS:
            top_id = str(profile[g].idxmax())
            peaks[g] = {
                "peak_cluster": top_id,
                "peak_mean": round(float(profile[g].max()), 4),
                "marker_annotation": marker_labels[top_id],
                "majority_baseline_annotation": transferred[top_id],
            }

        comm = communication(Xn, genes, lab)
        # UMAP colored by this resolution; explicit savefig avoids scanpy prefix quirks
        sc.pl.umap(b, color=key, ax=ax, show=False, title=f"Leiden res={res} ({n_clusters} clusters)",
                   legend_loc="on data")
        out_h5ad = f"data/pbmc3k/processed_res{str(res).replace('.', '')}.h5ad"
        b.uns["scrna_resolution_check"] = json.dumps(
            {"resolution": res, "n_clusters": n_clusters,
             "ARI_vs_baseline": round(float(adjusted_rand_score(base_labels, lab)), 6),
             "communication": comm, "marker_peaks": peaks}, ensure_ascii=False)
        b.write(out_h5ad)
        disk = sc.read_h5ad(out_h5ad, backed="r")
        disk_n = int(disk.obs[key].nunique())

        results["resolutions"][str(res)] = {
            "key": key,
            "n_clusters": n_clusters,
            "cluster_sizes": b.obs[key].value_counts().sort_index().astype(int).to_dict(),
            "ARI_vs_baseline_res1": round(float(adjusted_rand_score(base_labels, lab)), 6),
            "marker_peaks": peaks,
            "marker_annotation_counts": b.obs["cell_type_marker"].value_counts().astype(int).to_dict(),
            "majority_baseline_annotation_counts": b.obs["cell_type_majority"].value_counts().astype(int).to_dict(),
            "communication": comm,
            "output_h5ad": out_h5ad,
            "output_sha256": sha256(out_h5ad),
        }
        results["verification"][str(res)] = {
            "n_clusters_recomputed_from_disk_obs": disk_n,
            "matches_in_memory": disk_n == n_clusters,
        }

    fig.tight_layout()
    fig.savefig("figures/scrna_resolution_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # marker peak heatmap at all three resolutions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, res in zip(axes, RESOLUTIONS):
        r = results["resolutions"][str(res)]
        mat = pd.DataFrame(
            [{g: r["marker_peaks"][g]["peak_mean"] for g in MARKERS}], index=[str(res)]
        )
        im = ax.imshow(mat.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(MARKERS)))
        ax.set_xticklabels(MARKERS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([0]); ax.set_yticklabels([str(res)])
        ax.set_title(f"peak mean expression, res={res}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig("figures/scrna_resolution_marker_peaks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("===BEGIN_MEASURED_JSON===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("===END_MEASURED_JSON===")


if __name__ == "__main__":
    main()
