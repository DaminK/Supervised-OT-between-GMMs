"""
Fit or materialize a GMM artifact from a preprocessed h5ad using the ggml_ot dataset API.

Called via Snakemake ``script:`` directive.  The ``snakemake`` object provides:
  snakemake.input.h5ad       — preprocessed adata path
  snakemake.output.h5ad      — output path (gmm key stored in adata.uns)
  snakemake.params.config    — GMM-relevant config subset
  snakemake.params.ds_meta   — dataset metadata dict (label_col, patient_col, ...)
  snakemake.params.gmm_key   — precomputed gmm_key string
  snakemake.log[0]
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _utils import (  # noqa: E402
    get_dataset_use_rep,
    get_dataset_source,
    get_synthetic_gmm_source,
    setup_logging,
)

import numpy as np  # noqa: E402
import scanpy as sc  # noqa: E402
import ggml_ot  # noqa: E402


_OUTPUT_SCALE_ALIASES = {
    "global_trace": "max_trace",
    "global_max_trace": "max_trace",
    "max_trace": "max_trace",
}


def _resolve_output_scale_mode(value) -> str:
    """Resolve the configured post-fit GMM output scaling mode."""
    if value is None:
        return "none"
    mode = str(value).strip().lower()
    if mode in {"", "none", "false", "off", "0"}:
        return "none"
    if mode in _OUTPUT_SCALE_ALIASES:
        return _OUTPUT_SCALE_ALIASES[mode]
    raise ValueError(
        f"Unsupported gmm.output_scale={value!r}. Expected 'none' or 'max_trace' "
        "(alias: 'global_trace')."
    )


def _resolve_trace_bound(value, *, model_dim: int) -> tuple[float, str]:
    """Return the positive trace target used by max-trace output scaling."""
    source = "n_feats" if value is None else str(value)
    if value is None:
        bound = float(model_dim)
    elif isinstance(value, str) and value.strip().lower() in {"n_feats", "dimension", "dim", "d"}:
        bound = float(model_dim)
    else:
        bound = float(value)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError(f"gmm.trace_bound must be positive and finite, got {value!r}.")
    return bound, source


def _covariance_traces(model: dict) -> np.ndarray:
    """Compute covariance traces from a stored GMM model payload."""
    mu = np.asarray(model["mu"], dtype=np.float64)
    var = np.asarray(model["var"], dtype=np.float64)
    if mu.ndim < 2:
        raise ValueError(f"GMM model['mu'] must have at least 2 dimensions, got shape {mu.shape}.")
    if var.ndim == mu.ndim:
        traces = np.sum(var, axis=-1)
    elif var.ndim == mu.ndim + 1 and var.shape[-2] == var.shape[-1]:
        traces = np.trace(var, axis1=-2, axis2=-1)
    else:
        raise ValueError(
            "GMM model['var'] shape is incompatible with model['mu'] or is not a square covariance: "
            f"mu={mu.shape}, var={var.shape}."
        )
    expected_shape = mu.shape[:-1]
    if traces.shape != expected_shape:
        raise ValueError(
            "GMM covariance trace shape is incompatible with model['mu']: "
            f"traces={traces.shape}, expected={expected_shape}."
        )
    if not np.isfinite(traces).all():
        raise ValueError("GMM covariance traces contain non-finite values.")
    return traces


def _active_covariance_traces(gmm_schema: dict) -> np.ndarray:
    """Return traces for active components only, ignoring sample-specific padding."""
    traces = _covariance_traces(gmm_schema["model"])
    if str(gmm_schema.get("component_sharing", "global")) != "sample_specific":
        return traces.reshape(-1)
    if traces.ndim != 2:
        raise ValueError(
            "Sample-specific GMM output scaling expects traces with shape (n_distributions, K_max); "
            f"got {traces.shape}."
        )
    distribution_n_components = np.asarray(
        gmm_schema.get("distribution_n_components", np.full(traces.shape[0], traces.shape[1])),
        dtype=int,
    )
    if distribution_n_components.shape != (traces.shape[0],):
        raise ValueError(
            "distribution_n_components must contain one value per distribution for output scaling. "
            f"Got {distribution_n_components.shape}, expected {(traces.shape[0],)}."
        )
    if np.any(distribution_n_components < 0) or np.any(distribution_n_components > traces.shape[1]):
        raise ValueError(
            "distribution_n_components contains values outside the stored component range "
            f"[0, {traces.shape[1]}]."
        )
    active_mask = np.zeros(traces.shape, dtype=bool)
    for dist_idx, n_active in enumerate(distribution_n_components.tolist()):
        active_mask[dist_idx, : int(n_active)] = True
    active_traces = traces[active_mask]
    if active_traces.size == 0:
        raise ValueError("Cannot scale GMM output: no active covariance components found.")
    return active_traces


def _apply_gmm_output_scale(adata, gmm_key: str, cfg: dict, logger) -> None:
    """Apply one post-fit scalar to GMM means/covariances for trace-bounded geometry."""
    gmm_cfg = cfg.get("gmm", {}) or {}
    mode = _resolve_output_scale_mode(gmm_cfg.get("output_scale", "none"))
    if mode == "none":
        return
    if gmm_key not in adata.uns:
        raise KeyError(f"Cannot apply GMM output scale: {gmm_key!r} not found in adata.uns.")

    gmm_schema = adata.uns[gmm_key]
    model = gmm_schema.get("model")
    if not isinstance(model, dict) or "mu" not in model or "var" not in model:
        raise KeyError(f"adata.uns[{gmm_key!r}] does not contain a complete GMM model payload.")

    model_dim = int(np.asarray(model["mu"]).shape[-1])
    trace_bound, trace_bound_source = _resolve_trace_bound(
        gmm_cfg.get("trace_bound", "n_feats"),
        model_dim=model_dim,
    )
    active_traces = _active_covariance_traces(gmm_schema)
    pre_max_trace = float(np.max(active_traces))
    if not np.isfinite(pre_max_trace) or pre_max_trace <= 0.0:
        raise ValueError(f"Cannot apply GMM output scale with max covariance trace {pre_max_trace}.")

    scale = float(np.sqrt(pre_max_trace / trace_bound))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Computed invalid GMM output scale {scale}.")

    model["mu"] = np.asarray(model["mu"], dtype=np.float64) / scale
    model["var"] = np.asarray(model["var"], dtype=np.float64) / (scale * scale)
    post_max_trace = float(pre_max_trace / (scale * scale))
    gmm_schema["output_scale"] = {
        "mode": mode,
        "scale": scale,
        "trace_bound": float(trace_bound),
        "trace_bound_source": trace_bound_source,
        "pre_max_trace": pre_max_trace,
        "post_max_trace": post_max_trace,
        "model_dim": model_dim,
    }
    logger.info(
        "Applied GMM output scaling (%s): global scale=%s, max trace %s -> %s",
        mode,
        scale,
        pre_max_trace,
        post_max_trace,
    )


def _fit_init_n_cells(adata, patient_col: str) -> int:
    """Return a safe constructor-only n_cells value for AnnData-backed GMM fitting.

    This is only used to initialize ``AnnData_TripletDataset`` without dropping
    smaller patients during ``from_anndata(...)`` construction. The actual GMM
    fit later reads the full ``adata`` matrix directly inside ``dataset.fit_gmm``.
    """
    counts = adata.obs[patient_col].value_counts()
    counts = counts[counts > 0]
    if counts.empty:
        raise ValueError(f"No distributions found in adata.obs[{patient_col!r}].")
    return int(counts.min())


def _filter_small_distributions_for_gmm(adata, patient_col: str, min_cells: int, logger):
    counts = adata.obs[patient_col].value_counts()
    counts = counts[counts > 0]
    too_small = counts[counts < min_cells]
    if too_small.empty:
        return adata, None

    preview_items = [f"{dist}={int(count)}" for dist, count in too_small.head(20).items()]
    if len(too_small) > len(preview_items):
        preview_items.append(f"... {len(too_small) - len(preview_items)} more")
    logger.warning(
        "Dropping %d distributions with fewer than %d cells before sample_specific GMM fitting: %s",
        len(too_small),
        min_cells,
        ", ".join(preview_items),
    )

    keep_distributions = set(counts[counts >= min_cells].index)
    keep_mask = adata.obs[patient_col].isin(keep_distributions).to_numpy()
    filtered = adata[keep_mask].copy()
    if filtered.n_obs == 0:
        raise ValueError(
            f"No distributions remain after filtering obs[{patient_col!r}] to at least {min_cells} cells."
        )
    return filtered, {
        "patient_col": patient_col,
        "min_cells": min_cells,
        "n_dropped": int(len(too_small)),
        "dropped_distributions": {str(dist): int(count) for dist, count in too_small.items()},
    }


def _fit_gmm_via_gmvae(adata, *, gmm_key, use_rep, patient_col, gmvae_cfg, seed, scriptdir, logger):
    """PILOT-GM-VAE GMM provider.

    Trains the GM-VAE in the isolated ``.venv_gmvae`` interpreter (heavy pilotgm
    dependency stack), which emits assembled per-sample GMM tensors to a temp
    ``.npz``. The canonical schema is then written into ``adata`` in-process via
    ``store_gmvae_npz``, so ``adata.uns[gmm_key]`` matches the native provider.
    """
    import subprocess
    import tempfile

    from store_gmvae_gmm import store_gmvae_npz

    venv_python = gmvae_cfg.get("venv_python")
    if not venv_python or not Path(venv_python).exists():
        raise ValueError(
            "gmm.gmvae.venv_python must point to the isolated pilotgm interpreter "
            f"(got {venv_python!r})."
        )

    with tempfile.TemporaryDirectory() as td:
        prep_tmp = str(Path(td) / "prep.h5ad")
        npz = str(Path(td) / "gmvae_side.npz")
        adata.write_h5ad(prep_tmp)
        cmd = [
            venv_python,
            str(Path(scriptdir) / "fit_gmvae.py"),
            "--input", prep_tmp,
            "--out-npz", npz,
            "--patient-col", patient_col,
            "--use-rep", use_rep,
            "--num-components", str(gmvae_cfg.get("num_components", 10)),
            "--epochs", str(gmvae_cfg.get("epochs", 100)),
            "--batch-size", str(gmvae_cfg.get("batch_size", 256)),
            "--gaussian-size", str(gmvae_cfg.get("gaussian_size", 64)),
            "--seed", str(seed),
        ]
        logger.info("GM-VAE provider: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        store_gmvae_npz(adata, npz, gmm_key)
    logger.info(f"GM-VAE provider stored GMM under uns['{gmm_key}']")


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    cfg = snakemake.params.config
    ds_meta = snakemake.params.ds_meta
    gmm_key = snakemake.params.gmm_key

    out_path = Path(snakemake.output.h5ad)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_rep_key = get_dataset_use_rep(cfg, ds_meta)

    logger.info(f"Fitting GMM: key={gmm_key}, use_rep={use_rep_key}")

    adata = sc.read_h5ad(snakemake.input.h5ad)

    if get_dataset_source(ds_meta) == "synthetic_gmm" and get_synthetic_gmm_source(ds_meta) == "ground_truth":
        if gmm_key not in adata.uns:
            raise KeyError(
                f"Synthetic ground-truth GMM key {gmm_key!r} not found in preprocessed AnnData. "
                "Expected the raw materialization step to persist it."
            )
        _apply_gmm_output_scale(adata, gmm_key, cfg, logger)
        adata.write_h5ad(out_path)
        logger.info(f"Reused synthetic ground-truth GMM: {out_path}")
        return

    gmm_cfg = cfg.get("gmm", {})

    component_sharing = gmm_cfg.get("component_sharing", "auto")
    min_cells = gmm_cfg.get("min_cells")
    min_cells_filter = None
    if component_sharing == "sample_specific" and min_cells is not None:
        adata, min_cells_filter = _filter_small_distributions_for_gmm(
            adata,
            ds_meta["patient_col"],
            int(min_cells),
            logger,
        )

    # PILOT-GM-VAE provider: trains an external GM-VAE (isolated env) and stores
    # the per-sample mixtures in the canonical schema, bypassing the native EM fit.
    if gmm_cfg.get("provider") == "gmvae":
        gmvae_cfg = gmm_cfg.get("gmvae", {})
        seed = int(gmvae_cfg.get("seed", cfg.get("random_seed", 42)))
        _fit_gmm_via_gmvae(
            adata,
            gmm_key=gmm_key,
            use_rep=use_rep_key,
            patient_col=ds_meta["patient_col"],
            gmvae_cfg=gmvae_cfg,
            seed=seed,
            scriptdir=snakemake.scriptdir,
            logger=logger,
        )
        if min_cells_filter is not None:
            adata.uns[gmm_key]["min_cells_filter"] = min_cells_filter
        _apply_gmm_output_scale(adata, gmm_key, cfg, logger)
        adata.write_h5ad(out_path)
        logger.info(f"Wrote (GM-VAE provider): {out_path}")
        return

    # Resolve k_comps. A configured ``anchor_means_from_col`` switches to
    # cell-type-anchored fitting: k_comps becomes the resolved obs column name
    # (a string), so each component is a cell type with a fixed mean and an
    # estimated covariance (closed-form, sample_specific). This parallels the
    # ``k_comps_from_col`` path, which instead resolves to a component count.
    anchor_col_key = gmm_cfg.get("anchor_means_from_col")
    if anchor_col_key is not None:
        anchor_col_name = ds_meta[anchor_col_key]
        k_comps = anchor_col_name
        logger.info(f"Anchoring GMM means to cell types in obs['{anchor_col_name}'] (component={anchor_col_key})")
    else:
        k_comps_val = gmm_cfg.get("k_comps")
        if k_comps_val is None:
            col_key = gmm_cfg.get("k_comps_from_col", "cell_subtype_col")
            col_name = ds_meta[col_key]
            k_comps = adata.obs[col_name].nunique()
            logger.info(f"Inferred k_comps={k_comps} from obs['{col_name}']")
        else:
            k_comps = k_comps_val
            logger.info(f"Using explicit k_comps={k_comps}")

    # Build dataset (empirical — GMM fitting does not require a specific dataset_type)
    init_from_anndata_kwargs = dict(cfg.get("from_anndata", {}))
    init_n_cells = _fit_init_n_cells(adata, ds_meta["patient_col"])
    init_from_anndata_kwargs["n_cells"] = init_n_cells

    dataset = ggml_ot.from_anndata(
        adata,
        patient_col=ds_meta["patient_col"],
        label_col=ds_meta["label_col"],
        use_rep=use_rep_key,
        **init_from_anndata_kwargs,
    )

    # Legacy dataset-level normalisation. For AnnData-backed GMM fitting,
    # fit_gmm reads the matrix from adata directly, so preprocessing owns scale.
    if gmm_cfg.get("normalize", False):
        logger.warning(
            "gmm.normalize is legacy for AnnData-backed GMM fitting; use preprocess.mode "
            "for input scaling and gmm.output_scale for post-fit geometry scaling."
        )
        dataset.normalize()

    fit_kwargs = dict(gmm_cfg.get("fit_gmm_kwargs", {}))

    dataset = dataset.fit_gmm(
        component_sharing=component_sharing,
        k_comps=k_comps,
        gmm_key=gmm_key,
        **fit_kwargs,
    )

    assert gmm_key in dataset.adata.uns, f"fit_gmm did not write '{gmm_key}' to adata.uns"
    if min_cells_filter is not None:
        dataset.adata.uns[gmm_key]["min_cells_filter"] = min_cells_filter

    _apply_gmm_output_scale(dataset.adata, gmm_key, cfg, logger)
    dataset.adata.write_h5ad(out_path)
    logger.info(f"Wrote: {out_path}")


run(snakemake)  # noqa: F821
