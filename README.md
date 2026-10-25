# omni-random-clusters

Random-clustering **baseline** method module for the STA426 scRNA pipelines
Omnibenchmark (`methods` stage).

Each cell is assigned uniformly at random to one of K clusters, with
K = true number of clusters + `d_cluster`. Both the `leiden` and `louvain`
columns are random draws, so any agreement metric (ARI, NMI, ...) should be ~0.
A log-normalised PCA is also written so the structure metrics can be computed.

Parameters: `--seed` (RNG seed, reproducibility), `--d_cluster` (default 0),
`--n_comp` (number of PCs written, default 30).

Runs in the benchmark's `scanpy` software environment (needs numpy, pandas,
anndata, scanpy).
