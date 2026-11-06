#!/usr/bin/env python
"""PILOT-GM-VAE GMM provider (storage stage).

Reads the prepared AnnData plus the ``.npz`` sidecar emitted by ``fit_gmvae.py``
(the isolated-env GM-VAE stage) and persists the per-sample Gaussian mixtures
under ``adata.uns[gmm_key]`` in the canonical ggml_ot schema, so downstream
tune / train_test / benchmark consume it unchanged.

Runs in the main pipeline env (``.venv``) so the AnnData is serialised with the
same anndata version the rest of the pipeline reads.

CLI::

    store_gmvae_gmm.py --input prepared.h5ad --npz sidecar.npz \
        --output gmm.h5ad --gmm-key KEY [--ggml-path /path/to/ggml-ot]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def store_gmvae_npz(adata, npz_path: str, gmm_key: str) -> None:
    """Persist the canonical GMM schema from a fit_gmvae.py sidecar into ``adata``.

    Imports the ggml_ot storage helper lazily so callers already inside the
    pipeline env (e.g. fit_gmm.py) can reuse this without a subprocess.
    """
    from ggml_ot.gmm._anndata_dataset_interface import _store_gmm_in_adata

    side = np.load(npz_path, allow_pickle=True)
    distribution_ids = [str(x) for x in side["distribution_ids"].tolist()]

    _store_gmm_in_adata(
        adata=adata,
        gmm_key=gmm_key,
        component_sharing="sample_specific",
        use_rep=str(side["use_rep"]),
        covariance_type="full",
        n_components=int(side["n_components"]),
        distribution_n_components=np.asarray(side["distribution_n_components"], dtype=int),
        fit_params={
            "provider": "pilotgm",
            "num_components": int(side["num_components"]),
            "epochs": int(side["epochs"]),
            "gaussian_size": int(side["gaussian_size"]),
            "seed": int(side["seed"]),
            "eps": float(side["eps"]),
        },
        selection=None,
        backend="pilotgm",
        backend_metadata={
            "pilotgm_version": "0.1.1",
            "weight_convention": "soft_mean",
            "label_column_overlap": float(side["label_column_overlap"]),
        },
        model_payload={
            "mu": np.asarray(side["mu"], dtype=np.float64),
            "var": np.asarray(side["var"], dtype=np.float64),
            "pi": np.asarray(side["pi"], dtype=np.float64),
        },
        distribution_weights=np.asarray(side["distribution_weights"], dtype=np.float64),
        distribution_ids=distribution_ids,
    )
    adata.uns[gmm_key]["component_definition"] = "gmvae"


def main() -> None:
    parser = argparse.ArgumentParser(description="PILOT-GM-VAE GMM provider (storage stage).")
    parser.add_argument("--input", required=True, help="Prepared AnnData (.h5ad).")
    parser.add_argument("--npz", required=True, help="Sidecar (.npz) from fit_gmvae.py.")
    parser.add_argument("--output", required=True, help="Output AnnData with uns[gmm_key].")
    parser.add_argument("--gmm-key", required=True)
    parser.add_argument("--ggml-path", default=None, help="Path to the ggml_ot source tree.")
    args = parser.parse_args()

    if args.ggml_path:
        sys.path.insert(0, args.ggml_path)

    import scanpy as sc  # noqa: E402

    adata = sc.read_h5ad(args.input)
    store_gmvae_npz(adata, args.npz, args.gmm_key)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    adata.write_h5ad(args.output)
    print(f"[store_gmvae_gmm] wrote {args.output} with uns[{args.gmm_key!r}]", flush=True)


if __name__ == "__main__":
    main()
