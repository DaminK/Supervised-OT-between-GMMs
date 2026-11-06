"""
Compute patient-wise train/validation NLL diagnostics for one fitted GMM dataset.

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.h5ad        — fitted GMM h5ad path
  snakemake.output.csv        — per-patient diagnostic CSV
  snakemake.params.config     — resolved GMM-relevant config subset
  snakemake.params.ds_meta    — dataset metadata dict
  snakemake.params.dataset    — dataset name
  snakemake.params.gmm_key    — stored adata.uns GMM key
  snakemake.log[0]
"""

import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _dataset_metadata import dataset_display_name  # noqa: E402
from _utils import apply_runtime_settings, setup_logging  # noqa: E402

import anndata as ad  # noqa: E402
import ggml_ot  # noqa: E402


def _is_score_trace(value) -> bool:
    """Return whether a mapping is one stored per-distribution score table."""
    if not isinstance(value, Mapping) or "k" not in value:
        return False
    k_values = value["k"]
    return hasattr(k_values, "__len__") and not isinstance(k_values, (str, bytes, Mapping))


def _flatten_hdf5_path_mapping(mapping, *, prefix="", stop_at_score_trace=False):
    """Flatten AnnData-uns dict keys split by HDF5 path separators.

    AnnData persists nested ``uns`` mappings through HDF5 groups.  Metadata keys
    such as donor id ``SD48/16`` therefore round-trip as ``{"SD48": {"16": ...}}``.
    Flatten those paths back before handing the selection metadata to ggml_ot.
    """
    for key, value in dict(mapping).items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping) and not (stop_at_score_trace and _is_score_trace(value)):
            yield from _flatten_hdf5_path_mapping(
                value,
                prefix=path,
                stop_at_score_trace=stop_at_score_trace,
            )
        else:
            yield path, value


def _normalize_selection_path_keys(adata, gmm_key: str) -> None:
    """Repair slash-containing distribution ids in stored GMM selection metadata."""
    if gmm_key not in adata.uns:
        return

    gmm_cfg = adata.uns[gmm_key]
    selection = gmm_cfg.get("selection")
    if not isinstance(selection, Mapping):
        return

    if isinstance(selection.get("best_k_by_distribution"), Mapping):
        selection["best_k_by_distribution"] = dict(
            _flatten_hdf5_path_mapping(selection["best_k_by_distribution"])
        )

    if isinstance(selection.get("scores_by_distribution"), Mapping):
        selection["scores_by_distribution"] = dict(
            _flatten_hdf5_path_mapping(
                selection["scores_by_distribution"],
                stop_at_score_trace=True,
            )
        )


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    cfg = snakemake.params.config
    ds_meta = dict(snakemake.params.ds_meta)
    dataset_name = str(snakemake.params.dataset)
    gmm_key = str(snakemake.params.gmm_key)
    out_path = Path(snakemake.output.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    apply_runtime_settings(cfg)
    adata = ad.read_h5ad(snakemake.input.h5ad)
    _normalize_selection_path_keys(adata, gmm_key)

    logger.info("Extracting stored hold-out NLL diagnostics: dataset=%s, gmm_key=%s", dataset_name, gmm_key)

    results = ggml_ot.gmm.evaluate_holdout_nll(
        adata,
        gmm_key=gmm_key,
        patient_col=ds_meta["patient_col"],
        label_col=ds_meta.get("label_col"),
    )
    results.insert(0, "dataset", dataset_name)
    results.insert(1, "dataset_display_name", dataset_display_name(dataset_name, ds_meta))
    results.to_csv(out_path, index=False)

    logger.info("Wrote: %s (%d distributions)", out_path, len(results))


run(snakemake)  # noqa: F821
