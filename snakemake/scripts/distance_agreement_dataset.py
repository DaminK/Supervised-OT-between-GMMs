"""
Compare patient distances under the exact and the diagonal Bures term (one dataset).

Reads the per-split ground metrics trained by the benchmark and recomputes the
patient distance matrices on all patients, as ARI and silhouette use them:

four matrices per split, named "<trained with>/<evaluated with>": exact/exact,
exact/diag (the map of the exact-Bures method), diag/diag, diag/exact (the map of the
diagonal-Bures method). Records the Spearman correlation over all patient pairs (which
ignores a global rescaling) for all six pairs of them, plus one pair for a fixed
Euclidean ground metric, none/exact vs none/diag (nothing is trained).

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.h5ad           — fitted GMMs used by the benchmark
  snakemake.input.exact_metric   — trained_metric.pkl of the exact-Bures method
  snakemake.input.diag_metric    — trained_metric.pkl of the diagonal-Bures method
  snakemake.output.per_split     — CSV: first, second (matrix names), split, spearman
  snakemake.output.matrices      — npz of all distance matrices
  snakemake.params.config        — benchmark config subset (dataset construction)
  snakemake.params.method_cfg    — method entry of the exact-Bures method
  snakemake.params.ds_meta       — dataset metadata dict
  snakemake.params.gmm_key       — GMM key
  snakemake.log[0]
"""

import itertools
import pickle
import sys

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _distance_agreement import patient_distances, spearman  # noqa: E402
from _utils import apply_runtime_settings, build_dataset, setup_logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402

# "<trained with>/<evaluated with>" -> (which map, diag_bures_approx)
MATRICES = {
    "exact/exact": ("exact", False),
    "exact/diag": ("exact", True),
    "diag/diag": ("diag", True),
    "diag/exact": ("diag", False),
}


def load_ground_metrics(path: str) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)["Ws"]


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    apply_runtime_settings(snakemake.params.config)

    dataset = build_dataset(
        sc.read_h5ad(snakemake.input.h5ad),
        snakemake.params.config,
        snakemake.params.ds_meta,
        snakemake.params.method_cfg,
        snakemake.params.gmm_key,
    )
    exact_maps = load_ground_metrics(snakemake.input.exact_metric)
    diag_maps = load_ground_metrics(snakemake.input.diag_metric)
    if len(exact_maps) != len(diag_maps):
        raise ValueError(f"split counts differ: {len(exact_maps)} exact vs {len(diag_maps)} diagonal maps")

    matrices = {
        "none/exact": patient_distances(dataset, "euclidean", False),
        "none/diag": patient_distances(dataset, "euclidean", True),
    }
    rows = [{"first": "none/exact", "second": "none/diag", "split": -1,
             "spearman": spearman(matrices["none/exact"], matrices["none/diag"])}]
    for split, maps in enumerate(zip(exact_maps, diag_maps)):
        ground_metrics = dict(zip(("exact", "diag"), maps))
        split_matrices = {
            name: patient_distances(dataset, ground_metrics[trained], diag_bures_approx)
            for name, (trained, diag_bures_approx) in MATRICES.items()
        }
        for first, second in itertools.combinations(MATRICES, 2):
            rows.append({"first": first, "second": second, "split": split,
                         "spearman": spearman(split_matrices[first], split_matrices[second])})
        matrices.update({f"{name}_split{split}": matrix for name, matrix in split_matrices.items()})
        logger.info(f"split {split}: " + ", ".join(f"{r['first']} vs {r['second']} {r['spearman']:.3f}" for r in rows[-6:]))

    pd.DataFrame(rows).to_csv(snakemake.output.per_split, index=False)
    # npz keys cannot hold "/" portably: exact/diag -> exact_diag
    np.savez(snakemake.output.matrices, **{name.replace("/", "_"): matrix for name, matrix in matrices.items()})
    logger.info(f"wrote {len(rows)} comparisons for {matrices['none/exact'].shape[0]} patients")


run(snakemake)  # noqa: F821
