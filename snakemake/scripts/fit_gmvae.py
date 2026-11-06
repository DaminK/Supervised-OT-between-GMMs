#!/usr/bin/env python
"""PILOT-GM-VAE GMM provider (GM-VAE stage).

Trains a PILOT GM-VAE (CostaLab/PILOT-GM-VAE, ``pip install pilotgm``) on a
prepared AnnData and emits the assembled per-sample Gaussian mixture tensors to
a plain ``.npz`` sidecar. The canonical schema is then written in the main
pipeline env by ``store_gmvae_gmm.py`` (which calls ggml_ot's
``_store_gmm_in_adata``). The split matters: this stage runs in an isolated
environment (``.venv_gmvae``) whose anndata version differs from the pipeline
env, so it must not serialise the consumed AnnData itself. Emitting only numpy
arrays avoids the cross-version encoding hazard.

The GM-VAE only provides the clustering (a hard ``component_assignment`` and
soft ``weights`` per cell); the per-sample Gaussians are assembled here directly
from those outputs in the PCA representation, which avoids pilotgm's fragile
obs-index handling and its EMD computation (we use our own learned metric
downstream).

Label/weight pairing. pilotgm 0.1.1 does not take
the hard labels from the GM-VAE itself: ``train_gmvae`` fits a separate sklearn
``GaussianMixture`` on the soft assignments and stores its labels + 1
(``pilotgm/core.py:225-227``). These labels are an arbitrary permutation of the
columns of ``obsm['weights']``, so label ``c`` does not correspond to column
``c - 1`` (and pilotgm's own per-sample routine additionally compares the
1-indexed labels with 0-indexed components, ``core.py:794``). Pairing them by
index gives each component the weight of another one. We match each label to
the soft-assignment column it shares most cells with (Hungarian matching on the
label x argmax counts) and take the weight from that column, which keeps
pilotgm's definition of the weights (a sample's mean soft assignment).

CLI (invoked by the snakemake rule via the venv interpreter)::

    fit_gmvae.py --input prepared.h5ad --out-npz sidecar.npz \
        --patient-col PAT --use-rep X_pca \
        --num-components K --epochs N [--gaussian-size G] [--seed S]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

# Cap BLAS/OpenMP threads before numpy/torch load. On many-core hosts OpenBLAS
# otherwise spawns one memory region per core during the per-sample covariance
# assembly and aborts ("Program is Terminated. Because you tried to allocate too
# many memory regions"). setdefault leaves any caller-provided values intact.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np


def _add_pilotgm_shim() -> None:
    """pilotgm 0.1.1 uses top-level imports (``from networks.Networks import *``).

    Putting the package's own directory on ``sys.path`` makes its ``networks``,
    ``losses`` and ``metrics`` subpackages resolve as top-level modules.
    """
    spec = importlib.util.find_spec("pilotgm")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("pilotgm is not installed in this interpreter.")
    sys.path.insert(0, os.path.dirname(spec.origin))


def _match_labels_to_columns(component_labels: np.ndarray, soft_weights: np.ndarray) -> tuple[dict[int, int], float]:
    """Match each hard label to the soft-assignment column it shares most cells with.

    One-to-one (Hungarian) matching on the counts of label x argmax(soft_weights).
    Returns the mapping and the fraction of cells whose label and argmax agree under it.
    """
    from scipy.optimize import linear_sum_assignment

    labels = np.unique(component_labels)
    counts = np.zeros((len(labels), soft_weights.shape[1]))
    np.add.at(counts, (np.searchsorted(labels, component_labels), soft_weights.argmax(axis=1)), 1)
    rows, cols = linear_sum_assignment(-counts)
    overlap = float(counts[rows, cols].sum() / counts.sum())
    return {int(labels[r]): int(c) for r, c in zip(rows, cols)}, overlap


def _assemble_sample_gmms(
    *,
    x: np.ndarray,
    component_labels: np.ndarray,
    soft_weights: np.ndarray,
    patient_values: np.ndarray,
    distribution_ids: list[str],
    eps: float,
    label_to_column: dict[int, int],
):
    """Build canonical per-sample GMM tensors from GM-VAE cluster outputs.

    For each sample and each hard component present in it, the component mean and
    covariance are the empirical mean/covariance of that sample's cells assigned
    to the component in the PCA space ``x``. The mixture weight is the sample's
    average soft assignment probability for that component (PILOT-GM-VAE
    convention), read from the soft-assignment column matched to the label
    (``label_to_column``), and renormalised over the components retained for the sample.
    Components with fewer than two assigned cells (degenerate covariance) are
    dropped, matching the anchored provider's variable per-distribution
    component count.
    """
    n_features = x.shape[1]
    per_mu: list[np.ndarray] = []
    per_var: list[np.ndarray] = []
    per_pi: list[np.ndarray] = []
    per_k: list[int] = []
    n_labels = soft_weights.shape[1]

    for sid in distribution_ids:
        sample_mask = patient_values == sid
        labels_s = component_labels[sample_mask]
        x_s = x[sample_mask]
        w_s = soft_weights[sample_mask]  # (n_cells_s, n_labels)

        mus: list[np.ndarray] = []
        covs: list[np.ndarray] = []
        pis: list[float] = []
        for c in np.unique(labels_s):
            col = label_to_column.get(int(c), -1)
            if col < 0 or col >= n_labels:
                continue
            cells = x_s[labels_s == c]
            if cells.shape[0] < 2:
                # Too few cells for a covariance; skip this degenerate component.
                continue
            mu_c = cells.mean(axis=0)
            cov_c = np.cov(cells, rowvar=False)
            cov_c = 0.5 * (cov_c + cov_c.T)  # symmetrise
            cov_c = cov_c + eps * np.eye(n_features)  # PD floor
            if not np.isfinite(cov_c).all() or not np.isfinite(mu_c).all():
                continue
            mus.append(mu_c)
            covs.append(cov_c)
            pis.append(float(w_s[:, col].mean()))

        if not mus:
            raise ValueError(
                f"Sample {sid!r} produced no valid GM-VAE components; "
                "check the number of assigned cells and components."
            )
        pi_arr = np.asarray(pis, dtype=np.float64)
        pi_arr = pi_arr / pi_arr.sum() if pi_arr.sum() > 0 else np.full(len(pis), 1.0 / len(pis))

        per_mu.append(np.stack(mus, axis=0))
        per_var.append(np.stack(covs, axis=0))
        per_pi.append(pi_arr)
        per_k.append(len(mus))

    k_max = int(max(per_k))
    n_dist = len(distribution_ids)
    mu = np.zeros((n_dist, k_max, n_features), dtype=np.float64)
    var = np.zeros((n_dist, k_max, n_features, n_features), dtype=np.float64)
    pi = np.zeros((n_dist, k_max), dtype=np.float64)
    for d in range(n_dist):
        k_d = per_k[d]
        mu[d, :k_d] = per_mu[d]
        var[d, :k_d] = per_var[d]
        pi[d, :k_d] = per_pi[d]

    return {
        "n_components": k_max,
        "distribution_n_components": np.asarray(per_k, dtype=int),
        "model_payload": {"mu": mu, "var": var, "pi": pi},
        "distribution_weights": pi.copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PILOT-GM-VAE GMM provider (GM-VAE stage).")
    parser.add_argument("--input", required=True, help="Prepared AnnData (.h5ad).")
    parser.add_argument("--out-npz", required=True, help="Output sidecar (.npz) with assembled GMM tensors.")
    parser.add_argument("--patient-col", required=True)
    parser.add_argument("--use-rep", default="X_pca", help="obsm key of the PCA representation.")
    parser.add_argument("--num-components", type=int, default=10, help="GM-VAE num_classes.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256, help="GM-VAE training batch size.")
    parser.add_argument("--gaussian-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-6, help="Covariance PD floor.")
    args = parser.parse_args()

    _add_pilotgm_shim()

    import scanpy as sc  # noqa: E402
    from pilotgm import core  # noqa: E402

    adata = sc.read_h5ad(args.input)
    if args.use_rep not in adata.obsm:
        raise KeyError(f"use_rep {args.use_rep!r} not found in adata.obsm ({list(adata.obsm)}).")

    x = np.asarray(adata.obsm[args.use_rep], dtype=np.float64)
    n_features = x.shape[1]
    print(f"[fit_gmvae] adata={adata.shape}, use_rep={args.use_rep}, dim={n_features}, "
          f"patients={adata.obs[args.patient_col].nunique()}, num_components={args.num_components}",
          flush=True)

    # Train the GM-VAE. It reads obsm[pca_key] as a numpy array and writes
    # obs['component_assignment'] (1-indexed hard labels) and obsm['weights'].
    core.train_gmvae(
        adata,
        dataset_name="gmvae",
        pca_key=args.use_rep,
        num_classes=int(args.num_components),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        gaussian_size=int(args.gaussian_size),
        input_size=n_features,
        seed=int(args.seed),
        cuda=0,
        save_model=False,
        verbose=1,
    )

    component_labels = np.asarray(adata.obs["component_assignment"]).astype(int)
    soft_weights = np.asarray(adata.obsm["weights"], dtype=np.float64)
    patient_values = adata.obs[args.patient_col].astype(str).to_numpy()

    # Canonical distribution order: first appearance in the obs table.
    distribution_ids = list(dict.fromkeys(patient_values.tolist()))

    label_to_column, overlap = _match_labels_to_columns(component_labels, soft_weights)
    print(f"[fit_gmvae] label/soft-assignment agreement: index pairing "
          f"{np.mean(component_labels - 1 == soft_weights.argmax(axis=1)):.3f}, matched {overlap:.3f}", flush=True)

    outputs = _assemble_sample_gmms(
        x=x,
        component_labels=component_labels,
        soft_weights=soft_weights,
        patient_values=patient_values,
        distribution_ids=distribution_ids,
        eps=float(args.eps),
        label_to_column=label_to_column,
    )
    print(f"[fit_gmvae] assembled GMMs: K_max={outputs['n_components']}, "
          f"per-sample components min/median/max="
          f"{int(outputs['distribution_n_components'].min())}/"
          f"{int(np.median(outputs['distribution_n_components']))}/"
          f"{int(outputs['distribution_n_components'].max())}", flush=True)

    # Emit only numpy arrays. The canonical schema is written in the main
    # pipeline env by store_gmvae_gmm.py, avoiding cross-anndata-version
    # serialisation hazards (this venv's anndata differs from the pipeline's).
    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        mu=outputs["model_payload"]["mu"],
        var=outputs["model_payload"]["var"],
        pi=outputs["model_payload"]["pi"],
        distribution_n_components=outputs["distribution_n_components"],
        distribution_weights=outputs["distribution_weights"],
        distribution_ids=np.asarray(distribution_ids, dtype=object),
        n_components=int(outputs["n_components"]),
        use_rep=args.use_rep,
        num_components=int(args.num_components),
        epochs=int(args.epochs),
        gaussian_size=int(args.gaussian_size),
        seed=int(args.seed),
        eps=float(args.eps),
        label_to_column=np.asarray(sorted(label_to_column.items()), dtype=int),
        label_column_overlap=overlap,
    )
    print(f"[fit_gmvae] wrote {args.out_npz}", flush=True)


if __name__ == "__main__":
    main()
