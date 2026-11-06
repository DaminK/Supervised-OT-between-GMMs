"""
Preprocess a raw CELLxGENE h5ad: compute PCA (use_rep=X_pca) or HVG selection.

Called via Snakemake ``script:`` directive.  The ``snakemake`` object provides:
  snakemake.input.h5ad
  snakemake.output.h5ad
  snakemake.log[0]
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _utils import resolve_use_rep, setup_logging  # noqa: E402

import numpy as np  # noqa: E402
import scanpy as sc  # noqa: E402
from scipy import sparse  # noqa: E402


def _matrix_min(matrix) -> float:
    """Return the minimum value in a dense or sparse matrix."""
    if sparse.issparse(matrix):
        data = matrix.data
        explicit_min = float(np.min(data)) if data.size else 0.0
        return min(0.0, explicit_min)
    return float(np.nanmin(np.asarray(matrix)))


def _matrix_has_nan(matrix) -> bool:
    """Return whether a dense or sparse matrix contains NaN values."""
    if sparse.issparse(matrix):
        return bool(np.isnan(matrix.data).any())
    return bool(np.isnan(np.asarray(matrix)).any())


def _matrix_is_finite(matrix) -> bool:
    """Return whether a dense or sparse matrix has only finite stored values."""
    if sparse.issparse(matrix):
        return bool(np.isfinite(matrix.data).all())
    return bool(np.isfinite(np.asarray(matrix)).all())


def _ensure_log1p_safe_matrix(adata, logger):
    """Use raw counts when adata.X is already transformed in a way log1p cannot handle."""
    if _matrix_has_nan(adata.X):
        raise ValueError("adata.X contains NaN before preprocessing.")

    x_min = _matrix_min(adata.X)
    if x_min >= 0:
        return adata

    if adata.raw is None:
        raise ValueError(
            "adata.X contains negative values before log1p and adata.raw is unavailable. "
            "Use a nonnegative expression matrix or an existing embedding."
        )

    raw_adata = adata.raw.to_adata()
    if _matrix_has_nan(raw_adata.X):
        raise ValueError("adata.raw.X contains NaN and cannot be used for preprocessing.")
    raw_min = _matrix_min(raw_adata.X)
    if raw_min < 0:
        raise ValueError(
            "adata.X contains negative values before log1p, but adata.raw.X is also negative. "
            "Use a nonnegative expression matrix or an existing embedding."
        )

    logger.warning(
        "adata.X contains negative values (min=%s); using nonnegative adata.raw.X for preprocessing.",
        x_min,
    )
    return raw_adata


def _matrix_is_integer_like(matrix, sample_size: int = 200_000) -> bool:
    """Return whether sampled nonzero values are effectively integer counts."""
    if sparse.issparse(matrix):
        values = matrix.data
    else:
        values = np.asarray(matrix).ravel()
    if values.size == 0:
        return True
    values = values[:sample_size]
    return bool(np.allclose(values, np.round(values), rtol=0.0, atol=1e-8))


def _ensure_count_matrix(adata, logger):
    """Prefer raw count data for count-model preprocessing such as Pearson residuals."""
    candidates = [("adata.X", adata)]
    if adata.raw is not None:
        candidates.insert(0, ("adata.raw.X", adata.raw.to_adata()))

    diagnostics = []
    for label, candidate in candidates:
        if _matrix_has_nan(candidate.X):
            diagnostics.append(f"{label}: contains NaN")
            continue
        matrix_min = _matrix_min(candidate.X)
        if matrix_min < 0:
            diagnostics.append(f"{label}: min={matrix_min}")
            continue
        if not _matrix_is_integer_like(candidate.X):
            diagnostics.append(f"{label}: sampled values are not integer-like")
            continue
        if label != "adata.X":
            logger.info("Using %s for count-model preprocessing.", label)
        return candidate

    raise ValueError(
        "Could not find a nonnegative integer-like count matrix for Pearson residual preprocessing. "
        "Checked: " + "; ".join(diagnostics)
    )


def _canonicalize_qc_covariates(adata, ds_meta: dict, requested: list[str], logger) -> list[str]:
    """Copy dataset-specific QC columns to canonical names used by regress_out."""
    qc_covariates = dict(ds_meta.get("qc_covariates", {}) or {})
    canonical_cols = []
    for covariate in requested:
        source_col = qc_covariates.get(covariate)
        if source_col is None:
            raise KeyError(
                f"Requested QC covariate {covariate!r}, but dataset metadata has no "
                f"qc_covariates entry for it."
            )
        if source_col not in adata.obs:
            raise KeyError(
                f"Dataset metadata maps QC covariate {covariate!r} to obs column "
                f"{source_col!r}, but that column is absent."
            )
        canonical_col = f"_ggml_qc_{covariate}"
        values = np.asarray(adata.obs[source_col], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"QC covariate obs[{source_col!r}] contains non-finite values.")
        adata.obs[canonical_col] = values
        canonical_cols.append(canonical_col)
        logger.info("Mapped QC covariate %s <- obs[%s]", canonical_col, source_col)
    return canonical_cols


def _compute_pca(adata, n_feats: int, logger):
    """Compute PCA with a safe component count for the current matrix shape."""
    max_n_comps = min(adata.n_obs, adata.n_vars)
    if max_n_comps < 1:
        raise ValueError("Cannot compute PCA on an empty AnnData object.")
    pca_n_comps = min(n_feats, max_n_comps)
    if pca_n_comps < n_feats:
        logger.warning(
            "Requested n_comps=%s but only %s are possible for shape (%s, %s); capping PCA output.",
            n_feats,
            pca_n_comps,
            adata.n_obs,
            adata.n_vars,
        )
    svd_solver = "full" if pca_n_comps >= max_n_comps else "arpack"
    sc.pp.pca(adata, n_comps=pca_n_comps, svd_solver=svd_solver)
    assert "X_pca" in adata.obsm, "PCA failed"


def _select_highly_variable_genes(adata, n_hvg: int, logger):
    """Select HVGs when the matrix has more genes than requested."""
    if n_hvg <= 0:
        raise ValueError(f"preprocess.n_hvg must be positive, got {n_hvg}.")
    if adata.n_vars <= n_hvg:
        logger.info(
            "Skipping HVG subsetting: n_vars=%s <= n_hvg=%s",
            adata.n_vars,
            n_hvg,
        )
        return adata
    logger.info("Selecting %s highly variable genes", n_hvg)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)
    return adata


def _scale_genes(adata, preprocess_cfg: dict, logger):
    """Scale genes before PCA, optionally clipping extreme z-scores."""
    max_value = preprocess_cfg.get("scale_max_value", 10)
    if max_value is None:
        logger.info("Scaling genes without clipping")
        sc.pp.scale(adata, max_value=None)
    else:
        max_value = float(max_value)
        logger.info("Scaling genes with max_value=%s", max_value)
        sc.pp.scale(adata, max_value=max_value)
    if not _matrix_is_finite(adata.X):
        raise ValueError("Gene scaling produced non-finite values.")
    return adata


def _normalize_and_log1p(adata, logger, *, skip_if_logged=False):
    """Library-size normalize to 1e4 and apply log1p, shared by all log-based modes."""
    if skip_if_logged and "log1p" in adata.uns:
        logger.info("Skipping normalize/log1p: adata.uns['log1p'] already present.")
        return adata
    adata = _ensure_log1p_safe_matrix(adata, logger)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def _apply_pearson_residuals(adata, preprocess_cfg, logger):
    """Analytic Pearson-residual normalization on HVG-restricted counts."""
    adata = _ensure_count_matrix(adata, logger)
    before_shape = adata.shape
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_counts=1)
    if adata.shape != before_shape:
        logger.info(
            "Filtered empty cells/genes before Pearson residuals: %s -> %s",
            before_shape,
            adata.shape,
        )
    experimental_pp = getattr(getattr(sc, "experimental", None), "pp", None)
    hvg_pr = getattr(experimental_pp, "highly_variable_genes", None)
    normalize_pr = getattr(experimental_pp, "normalize_pearson_residuals", None)
    if hvg_pr is None or normalize_pr is None:
        raise RuntimeError(
            "scanpy.experimental.pp (highly_variable_genes / normalize_pearson_residuals) "
            "is unavailable in this environment."
        )
    # Select HVGs on the (sparse) counts and subset before computing residuals.
    # Pearson residuals densify every entry, so restricting to a few thousand
    # genes avoids materialising the full n_cells x n_genes dense matrix
    # (e.g. 33k genes -> ~95 GB). Full genes remain in .raw.
    n_hvg = int(preprocess_cfg.get("n_hvg", 5000))
    if adata.n_vars > n_hvg:
        logger.info("Selecting %s highly variable genes (flavor=pearson_residuals)", n_hvg)
        hvg_pr(adata, flavor="pearson_residuals", n_top_genes=n_hvg)
        adata = adata[:, adata.var["highly_variable"]].copy()
    normalize_pr(adata)
    if not _matrix_is_finite(adata.X):
        raise ValueError("Pearson residual preprocessing produced non-finite values.")
    return adata


def _apply_expression_preprocessing(
    adata, use_rep_key, preprocess_mode, preprocess_cfg, n_feats, ds_meta, logger
):
    """Apply normalization and mode-specific gene-space steps shared by the
    gene-space (``use_rep='X'``) and PCA (``use_rep='X_pca'``) representations.

    PCA itself is applied by the caller only when ``use_rep_key == 'X_pca'``.
    ``n_feats`` always sets the final model dimension: it is the number of PCA
    components for ``X_pca``, and the number of highly variable genes kept for
    gene-space (``X``) representations. In gene space there is no PCA, so both
    ``log1p_pca`` and ``log1p_hvg_scale_pca`` select ``n_feats`` HVGs and scale
    them; ``n_hvg`` only acts as the pre-PCA filter when PCA follows.
    """
    gene_space = use_rep_key is None  # 'X' -> GMM fits directly on adata.X

    if preprocess_mode in {"log1p_pca", "default"}:
        adata = _normalize_and_log1p(adata, logger, skip_if_logged=True)
        if gene_space:
            # No PCA follows: reduce to n_feats HVGs and scale so the gene-space
            # matrix is a tractable, standardized model input.
            adata = _select_highly_variable_genes(adata, n_feats, logger)
            adata = _scale_genes(adata, preprocess_cfg, logger)
    elif preprocess_mode == "log1p_hvg_scale_pca":
        adata = _normalize_and_log1p(adata, logger)
        if gene_space:
            # No PCA follows, so n_feats is the final model dimension: select
            # exactly that many highly variable genes.
            n_genes = n_feats
        else:
            # PCA follows: n_hvg is a pre-filter and PCA reduces to n_feats.
            n_genes = int(preprocess_cfg.get("n_hvg", 5000))
        adata = _select_highly_variable_genes(adata, n_genes, logger)
        adata = _scale_genes(adata, preprocess_cfg, logger)
    elif preprocess_mode == "log1p_regress_pca":
        if gene_space:
            raise ValueError(
                "preprocess.mode='log1p_regress_pca' keeps all genes and is only supported "
                "for use_rep='X_pca'. Use 'log1p_hvg_scale_pca' for a gene-space representation."
            )
        adata = _normalize_and_log1p(adata, logger)
        requested = list(preprocess_cfg.get("regress_out", []))
        if not requested:
            raise ValueError("preprocess.mode='log1p_regress_pca' requires preprocess.regress_out.")
        regress_cols = _canonicalize_qc_covariates(adata, ds_meta, requested, logger)
        logger.info("Regressing out canonical QC covariates: %s", regress_cols)
        sc.pp.regress_out(adata, regress_cols)
        adata = _scale_genes(adata, preprocess_cfg, logger)
    elif preprocess_mode == "pearson_residual_pca":
        adata = _apply_pearson_residuals(adata, preprocess_cfg, logger)
    else:
        raise ValueError(
            "Unsupported preprocess.mode=%r. Expected one of: log1p_pca, "
            "log1p_hvg_scale_pca, log1p_regress_pca, pearson_residual_pca." % preprocess_mode
        )
    return adata


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    out_path = Path(snakemake.output.h5ad)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_rep = snakemake.config.get("use_rep")
    n_feats = int(snakemake.config.get("n_feats", 50))
    preprocess_cfg = dict(snakemake.config.get("preprocess", {}) or {})
    preprocess_mode = str(preprocess_cfg.get("mode", "log1p_pca"))
    use_rep_key = resolve_use_rep(use_rep)
    dataset_name = str(snakemake.wildcards.dataset)
    ds_meta = snakemake.config.get("datasets", {}).get(dataset_name, {})

    logger.info(
        "Preprocessing AnnData: use_rep=%s, n_feats=%s, mode=%s",
        use_rep,
        n_feats,
        preprocess_mode,
    )

    adata = sc.read_h5ad(snakemake.input.h5ad)

    if use_rep_key in (None, "X_pca"):
        adata = _apply_expression_preprocessing(
            adata, use_rep_key, preprocess_mode, preprocess_cfg, n_feats, ds_meta, logger
        )
        if use_rep_key == "X_pca":
            logger.info("Computing PCA (n_comps=%s)", n_feats)
            _compute_pca(adata, n_feats, logger)
        else:
            logger.info(
                "Using gene-space representation (use_rep='X') with %s genes after preprocessing.",
                adata.n_vars,
            )
    else:
        if preprocess_mode not in {"log1p_pca", "default"}:
            raise ValueError(
                f"preprocess.mode={preprocess_mode!r} cannot be applied to existing embedding {use_rep_key!r}."
            )
        if use_rep_key not in adata.obsm:
            raise KeyError(
                f"adata.obsm['{use_rep_key}'] not found. "
                "Compute it upstream or use X_pca."
            )
        logger.info(f"Using existing embedding: {use_rep_key}")

    adata.write_h5ad(out_path)
    logger.info(f"Wrote: {out_path}")


run(snakemake)  # noqa: F821
