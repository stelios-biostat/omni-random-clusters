#!/usr/bin/env python3
"""Random-clustering baseline for the STA426 scRNA pipelines benchmark.

Assigns every cell to one of K clusters uniformly at random, where
K = (true number of clusters) + d_cluster, exactly as the real pipelines do.
The same outputs as the other `methods` modules are written, so the
`metrics` stage can score the baseline without any changes:

  {name}.clusters.tsv         cell_id, louvain, leiden  (random labels)
  {name}.pca.tsv              cell_id, PC1..PCn (log-normalised counts)
  {name}.timings.json         same keys as the pipelines
  {name}.clustering_info.json same keys as the pipelines

The random number generator is seeded with --seed, so results are reproducible.
"""

import argparse
import json
import os
from time import time

import numpy as np
import pandas as pd


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Random clustering baseline")
    p.add_argument("--output_dir", "-o", required=True, help="output directory")
    p.add_argument("--name", "-n", required=True, help="name used for output files")
    p.add_argument("--data.h5ad", dest="data_h5ad", required=True, help="input h5ad")
    p.add_argument(
        "--data.clusters_truth_num",
        dest="clusters_truth_num_path",
        required=True,
        help="file holding the true number of clusters",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--d_cluster",
        type=int,
        default=0,
        help="delta number of clusters with respect to the true number",
    )
    p.add_argument(
        "--n_comp", type=int, default=30, help="number of PCs written to pca.tsv"
    )
    args, _ = p.parse_known_args(argv)
    return args


def random_labels(n_cells: int, n_cluster: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random cluster labels 0..n_cluster-1 for n_cells cells."""
    if n_cluster < 1:
        raise ValueError(f"number of clusters must be >= 1, got {n_cluster}")
    return rng.integers(0, n_cluster, size=n_cells)


def compute_pca(adata, n_comp: int) -> pd.DataFrame:
    """Log-normalised PCA, only used for the structure metrics (silhouette, ...)."""
    import scanpy as sc

    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_comp = min(n_comp, adata.n_obs - 1, adata.n_vars - 1)
    sc.tl.pca(adata, n_comps=n_comp)
    return pd.DataFrame(
        adata.obsm["X_pca"],
        index=adata.obs_names,
        columns=[f"PC{i + 1}" for i in range(adata.obsm["X_pca"].shape[1])],
    )


def write_outputs(output_dir, name, cell_ids, labels_leiden, labels_louvain,
                  pca_df, timings, clustering_info):
    clusters = pd.DataFrame(
        {"cell_id": cell_ids, "louvain": labels_louvain, "leiden": labels_leiden}
    )
    clusters.to_csv(os.path.join(output_dir, f"{name}.clusters.tsv"),
                    sep="\t", index=False)
    pca_df.to_csv(os.path.join(output_dir, f"{name}.pca.tsv"),
                  sep="\t", index=True, index_label="cell_id")
    with open(os.path.join(output_dir, f"{name}.timings.json"), "w") as f:
        json.dump(timings, f, indent=2)
    with open(os.path.join(output_dir, f"{name}.clustering_info.json"), "w") as f:
        json.dump(clustering_info, f, indent=2)


def main(argv=None):
    args = parse_args(argv)

    import anndata as ad

    adata = ad.read_h5ad(args.data_h5ad)
    adata.var_names_make_unique()
    with open(args.clusters_truth_num_path) as f:
        n_cluster = int(f.read().strip()) + args.d_cluster

    rng = np.random.default_rng(args.seed)
    start = time()
    leiden = random_labels(adata.n_obs, n_cluster, rng)
    louvain = random_labels(adata.n_obs, n_cluster, rng)
    t_cluster = time() - start

    start = time()
    pca_df = compute_pca(adata, args.n_comp)
    t_pca = time() - start

    timings = {
        "gpu_load": None, "find_mit_gene": None, "filter": None,
        "normalization": None, "hvg": None, "scaling": None, "pca": t_pca,
        "t_sne": None, "umap": None, "louvain": t_cluster, "leiden": t_cluster,
    }
    clustering_info = {
        "resolutions": {"louvain": None, "leiden": None},
        "num_runs": {"louvain": 1, "leiden": 1},
    }

    write_outputs(args.output_dir, args.name, adata.obs_names.to_numpy(),
                  leiden, louvain, pca_df, timings, clustering_info)
    print(f"random baseline: {adata.n_obs} cells, {n_cluster} clusters, seed {args.seed}")


if __name__ == "__main__":
    main()
