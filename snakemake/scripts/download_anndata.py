"""
Materialize one raw CELLxGENE dataset artifact and save as raw.h5ad.

Called via Snakemake ``script:`` directive.  The ``snakemake`` object provides:
  snakemake.output.h5ad   — target path
  snakemake.params.ds_meta
  snakemake.log[0]
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _utils import get_dataset_source, setup_logging  # noqa: E402

import ggml_ot  # noqa: E402
import h5py  # noqa: E402


def _dataset_cache_path(root: Path, dataset_id: str) -> Path:
    """Return the raw downloaded CELLxGENE cache path for one dataset id."""
    path = root / dataset_id
    return path if path.suffix else path.with_suffix(".h5ad")


def _is_valid_h5ad(path: Path) -> bool:
    """Return whether a candidate h5ad file can be opened successfully."""
    try:
        with h5py.File(path, "r"):
            return True
    except OSError:
        return False


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    out_path = Path(snakemake.output.h5ad)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds_meta = snakemake.params.ds_meta
    if get_dataset_source(ds_meta) != "cellxgene":
        raise ValueError("download_anndata only supports CELLxGENE-backed datasets.")

    dataset_id = ds_meta["id"]
    logger.info(f"Downloading dataset id={dataset_id} -> {out_path}")

    downloaded = _dataset_cache_path(out_path.parent, dataset_id)
    if downloaded.exists() and not _is_valid_h5ad(downloaded):
        logger.warning("Removing corrupted cached dataset before re-download: %s", downloaded)
        downloaded.unlink()

    adata = ggml_ot.data.load_cellxgene(dataset_id, path=out_path.parent, load=True)

    if downloaded.exists() and downloaded != out_path:
        downloaded.rename(out_path)
    elif not out_path.exists():
        adata.write_h5ad(out_path)

    logger.info(f"Done: {out_path} (n_obs={adata.n_obs}, n_vars={adata.n_vars})")


run(snakemake)  # noqa: F821
