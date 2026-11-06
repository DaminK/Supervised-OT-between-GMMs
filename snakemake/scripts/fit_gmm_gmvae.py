"""
Fit per-patient GMMs with PILOT-GM-VAE instead of EM (``gmm.provider: gmvae``).

Called via Snakemake ``script:`` (rule ``fit_gmm_gmvae``). Applies the same patient
filter as the EM rule, then
  1. trains the GM-VAE in its isolated environment (``fit_gmvae.py``) on a temporary
     copy of the data, which writes the per-patient GMMs to a numpy ``.npz``;
  2. stores them in the canonical GMM schema (``store_gmvae_gmm.py``).
Tuning and benchmarking read the result like an EM fit.

The ``snakemake`` object provides:
  snakemake.input.h5ad      — prepared AnnData
  snakemake.output.h5ad     — AnnData with the GMMs in ``uns[gmm_key]``
  snakemake.params.config   — GMM config subset plus ``random_seed``
  snakemake.params.ds_meta  — dataset metadata
  snakemake.params.gmm_key  — GMM key
  snakemake.log[0]
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _utils import (  # noqa: E402
    filter_small_distributions_for_gmm,
    get_dataset_source,
    get_dataset_use_rep,
    setup_logging,
)
from store_gmvae_gmm import store_gmvae_npz  # noqa: E402

import scanpy as sc  # noqa: E402


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    cfg = snakemake.params.config
    ds_meta = snakemake.params.ds_meta
    gmm_key = snakemake.params.gmm_key
    gmm_cfg = cfg["gmm"]
    gmvae_cfg = gmm_cfg["gmvae"]

    if get_dataset_source(ds_meta) == "synthetic_gmm":
        raise ValueError("The GM-VAE provider applies to scRNA datasets, not to synthetic ones.")
    if gmm_cfg.get("output_scale", "none") != "none":
        raise ValueError("gmm.output_scale is not supported with the GM-VAE provider.")
    venv_python = gmvae_cfg.get("venv_python")
    if not venv_python or not Path(venv_python).exists():
        raise ValueError(f"gmm.gmvae.venv_python must point to the pilotgm interpreter (got {venv_python!r}).")

    adata = sc.read_h5ad(snakemake.input.h5ad)

    # Same patients as the EM fit
    min_cells_filter = None
    if gmm_cfg.get("component_sharing", "auto") == "sample_specific" and gmm_cfg.get("min_cells") is not None:
        adata, min_cells_filter = filter_small_distributions_for_gmm(
            adata, ds_meta["patient_col"], int(gmm_cfg["min_cells"]), logger
        )

    # PILOT-GM-VAE sets the number of components to the number of cell types in the data
    if gmvae_cfg["num_components"] == "cell_types":
        num_components = int(adata.obs[ds_meta["cell_subtype_col"]].nunique())
    else:
        num_components = int(gmvae_cfg["num_components"])
    logger.info(f"GM-VAE components: {num_components}")

    with tempfile.TemporaryDirectory() as td:
        prep_tmp = str(Path(td) / "prep.h5ad")
        npz = str(Path(td) / "gmvae_side.npz")
        adata.write_h5ad(prep_tmp)
        cmd = [
            venv_python,
            str(Path(snakemake.scriptdir) / "fit_gmvae.py"),
            "--input", prep_tmp,
            "--out-npz", npz,
            "--patient-col", ds_meta["patient_col"],
            "--use-rep", get_dataset_use_rep(cfg, ds_meta),
            "--num-components", str(num_components),
            "--epochs", str(gmvae_cfg["epochs"]),
            "--batch-size", str(gmvae_cfg["batch_size"]),
            "--gaussian-size", str(gmvae_cfg["gaussian_size"]),
            "--seed", str(cfg["random_seed"]),
        ]
        logger.info("GM-VAE: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        store_gmvae_npz(adata, npz, gmm_key)

    if min_cells_filter is not None:
        adata.uns[gmm_key]["min_cells_filter"] = min_cells_filter

    out_path = Path(snakemake.output.h5ad)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    logger.info(f"Wrote GM-VAE GMMs under uns['{gmm_key}']: {out_path}")


run(snakemake)  # noqa: F821
