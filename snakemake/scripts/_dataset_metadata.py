"""
Shared dataset metadata helpers used by tables and aggregate plots.
"""

from _utils import get_dataset_source, get_dataset_use_rep


def dataset_display_name(dataset_name: str, ds_meta: dict) -> str:
    """Return the paper-facing dataset label."""
    name = ds_meta.get("display_name")
    if name:
        return str(name)

    label = dataset_name.replace("_", " ")
    return label if any(char.isupper() for char in label) else label.capitalize()


def unique_count(adata, column_name: str) -> int:
    """Count unique entries in an AnnData obs column."""
    if column_name not in adata.obs:
        raise KeyError(f"Column '{column_name}' not found in AnnData obs.")
    return int(adata.obs[column_name].nunique())


def used_dimension(prepared_adata, cfg: dict, ds_meta: dict) -> int:
    """Return the active representation dimension used for one dataset."""
    use_rep = get_dataset_use_rep(cfg, ds_meta)
    if use_rep is None:
        return int(prepared_adata.n_vars)
    if use_rep in prepared_adata.obsm:
        return int(prepared_adata.obsm[use_rep].shape[1])
    raise KeyError(
        f"Resolved use_rep='{use_rep}' not found in prepared artifact. "
        "Expected preprocessing/materialization to persist the active representation."
    )


def distribution_count(adata, ds_meta: dict) -> int:
    """Return the number of patient-level distributions for one dataset."""
    patient_col = ds_meta["patient_col"]
    if patient_col not in adata.obs:
        synth_cfg = ds_meta.get("synth", {})
        if "n_patients" in synth_cfg:
            return int(synth_cfg["n_patients"])
        raise KeyError(f"Column '{patient_col}' not found in AnnData obs.")
    return int(adata.obs[patient_col].nunique())


def cell_type_count(adata, ds_meta: dict) -> int:
    """Return the number of cell types or synthetic modes for one dataset."""
    source = get_dataset_source(ds_meta)
    if source == "synthetic_gmm":
        n_modes = adata.uns.get("n_modes")
        if n_modes is None:
            n_modes = ds_meta.get("synth", {}).get("n_modes")
        if n_modes is not None:
            return int(n_modes)
    return unique_count(adata, ds_meta["cell_subtype_col"])
