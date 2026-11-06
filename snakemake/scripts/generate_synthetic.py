"""
Generate one synthetic dataset directly as the prepared h5ad artifact.

Called via Snakemake ``script:`` directive.  The ``snakemake`` object provides:
  snakemake.output.h5ad   — target path
  snakemake.params.ds_meta
  snakemake.params.gmm_key
  snakemake.log[0]
"""

import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _utils import get_dataset_source, get_synthetic_gmm_source, setup_logging  # noqa: E402

import ggml_ot  # noqa: E402


def _materialize_from_synth_to_anndata(ds_meta: dict, synth_cfg: dict):
  """Generate an AnnData artifact from ggml_ot.data.from_synth output."""
  synthetic_cfg = dict(ds_meta.get("synthetic", {}))
  seed = synthetic_cfg.get("random_seed")
  if seed is not None:
    np.random.seed(int(seed))

  dataset = ggml_ot.data.from_synth(**synth_cfg)
  supports = dataset.supports
  supports_np = supports.detach().cpu().numpy() if hasattr(supports, "detach") else np.asarray(supports)

  if supports_np.ndim != 3:
    raise ValueError(
      "from_synth materialization expects fixed-size supports with shape (n_distributions, n_cells, n_features)."
    )

  n_distributions, n_cells, n_features = supports_np.shape
  X = supports_np.reshape(n_distributions * n_cells, n_features)

  patient_col = ds_meta["patient_col"]
  label_col = ds_meta["label_col"]
  mode_col = ds_meta["cell_subtype_col"]

  symbols = getattr(dataset, "symbols", np.arange(n_distributions))
  dist_labels = np.asarray(dataset.distribution_labels)
  sample_ids = np.repeat(np.asarray(symbols).astype(str), n_cells)
  group_labels = np.repeat(dist_labels.astype(str), n_cells)
  mode_values = np.repeat("mode_0", n_distributions * n_cells)

  obs = pd.DataFrame(
    {
      patient_col: sample_ids,
      label_col: group_labels,
      mode_col: mode_values,
    }
  )
  obs.index = obs.index.astype(str)

  adata = anndata.AnnData(X=X, obs=obs)
  adata.var_names = [f"feature_{i}" for i in range(adata.n_vars)]
  adata.uns["n_modes"] = 1
  adata.uns["synthetic_generator"] = "from_synth"
  return adata


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    out_path = Path(snakemake.output.h5ad)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds_meta = snakemake.params.ds_meta
    if get_dataset_source(ds_meta) != "synthetic_gmm":
        raise ValueError("generate_synthetic only supports synthetic_gmm datasets.")

    synth_cfg = dict(ds_meta.get("synth", {}))
    synthetic_cfg = dict(ds_meta.get("synthetic", {}))
    generator = synthetic_cfg.get("generator", "synth_gmm_anndata")
    gmm_source = get_synthetic_gmm_source(ds_meta)
    gmm_key = snakemake.params.gmm_key if gmm_source == "ground_truth" else None

    logger.info(
      f"Generating synthetic dataset -> {out_path} "
      f"(generator={generator}, gmm_source={gmm_source})"
    )

    if generator == "from_synth":
      if gmm_source == "ground_truth":
        raise ValueError("synthetic.gmm_source='ground_truth' is not supported with synthetic.generator='from_synth'.")
      adata = _materialize_from_synth_to_anndata(ds_meta=ds_meta, synth_cfg=synth_cfg)
    else:
      adata = ggml_ot.data.synth_gmm_anndata(gmm_key=gmm_key, **synth_cfg)

    adata.write_h5ad(out_path)
    logger.info(f"Done: {out_path} (n_obs={adata.n_obs}, n_vars={adata.n_vars})")


run(snakemake)  # noqa: F821
