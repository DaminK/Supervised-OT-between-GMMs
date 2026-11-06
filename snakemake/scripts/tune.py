"""
Hyperparameter tuning for a single tunable GGML method on one dataset.

Called via Snakemake ``script:`` directive.  The ``snakemake`` object provides:
  snakemake.input.h5ad           — preprocessed or GMM-fitted adata
  snakemake.output.best_params   — JSON with best hyperparameters
  snakemake.output.scores        — CSV of full grid-search scores
  snakemake.output.manifest      — JSON manifest for provenance
  snakemake.params.config        — tune-relevant config subset for this method
  snakemake.params.ds_meta       — dataset metadata dict
  snakemake.params.method_name   — name of the method being tuned
  snakemake.params.gmm_key       — GMM key (empty string for non-GMM methods)
  snakemake.log[0]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _utils import (  # noqa: E402
    apply_runtime_settings,
    build_dataset,
    get_tune_plot_config,
    get_method_tune_kwargs,
    infer_tune_contour_axes,
    list_artifact_files,
    method_uses_fitted_gmm,
    resolve_tune_log_axis,
    setup_logging,
    slim_adata_after_build,
    temporary_plot_settings,
    to_jsonable,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import scanpy as sc  # noqa: E402
import ggml_ot  # noqa: E402
from ggml_ot.benchmark._metrics import _best_index_by_scoring  # noqa: E402
from ggml_ot.plot._utils import savefig_or_show  # noqa: E402


def _flatten_score_row(row) -> dict:
    """Convert a selected tune-score row into JSON-friendly flat keys."""
    flattened = {}
    for key, value in row.items():
        if isinstance(key, tuple):
            name = ".".join(str(part) for part in key if part is not None)
        else:
            name = str(key)
        flattened[name] = value.item() if hasattr(value, "item") else value
    return flattened


def _render_tune_contours(
    *,
    scores_df,
    best_params: dict,
    cfg: dict,
    out_dir: Path,
    logger,
) -> list[str]:
    """Render contour plots for all 2D pairs of varying hyperparameters."""
    plot_cfg = get_tune_plot_config(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not plot_cfg["enabled"]:
        return []

    pairs = infer_tune_contour_axes(scores_df, plot_cfg)
    if not pairs:
        logger.info("Skipping tune contours: fewer than two varying hyperparameters")
        return []

    import matplotlib.pyplot as plt  # noqa: E402

    written = []
    for x, y in pairs:
        if x not in scores_df.index.names or y not in scores_df.index.names:
            logger.warning(f"Skipping contour: axes ({x}, {y}) not in tuning scores")
            continue

        fixed_params = dict(plot_cfg["fixed_params"])
        for name in scores_df.index.names:
            if name in {x, y}:
                continue
            if name not in fixed_params:
                fixed_params[name] = best_params.get(name)

        # Place each pair in its own subfolder
        pair_dir = out_dir / f"{x}_vs_{y}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        filename = f"contour_{x}_vs_{y}.{plot_cfg['format']}"

        log_axis = resolve_tune_log_axis(scores_df, x, y, plot_cfg["log_axis"])
        with temporary_plot_settings(
            figdir=pair_dir,
            figformat=plot_cfg["format"],
            figdpi=plot_cfg["dpi"],
        ):
            fig, ax = ggml_ot.pl.contour_hyperparams(
                scores_df,
                x=x,
                y=y,
                fixed_params=fixed_params,
                value_col=plot_cfg["value_col"],
                log_axis=log_axis,
                label_style=plot_cfg.get("label_style", "paper"),
                levels=plot_cfg["levels"],
                show=False,
                save=False,
            )
            # Mark the selected hyperparameters when they lie on this slice
            if all(fixed_params.get(name) == best_params.get(name) for name in fixed_params):
                bx, by = float(best_params[x]), float(best_params[y])
                if log_axis is True or log_axis == "x":
                    bx = np.log10(bx)
                if log_axis is True or log_axis == "y":
                    by = np.log10(by)
                ax.plot(bx, by, "x", color="black", markersize=8, markeredgewidth=1.2)
            savefig_or_show(fig, default_name="contour_hyperparams", show=False, save=filename)

        if fig is not None:
            plt.close(fig)

        written.append(str(pair_dir / filename))
        logger.info(f"Wrote contour: {pair_dir / filename}")

    return written


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    cfg = snakemake.params.config
    ds_meta = snakemake.params.ds_meta
    method_name = snakemake.params.method_name
    gmm_key = snakemake.params.gmm_key

    method = cfg["methods"][method_name]

    out_dir = Path(snakemake.output.best_params).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(snakemake.output.plots)
    plot_dir.mkdir(parents=True, exist_ok=True)

    apply_runtime_settings(cfg)

    logger.info(f"Tuning method={method_name}, dataset_type={method['dataset_type']}")

    adata = sc.read_h5ad(snakemake.input.h5ad)
    dataset = build_dataset(adata, cfg, ds_meta, method, gmm_key)

    # Free large AnnData payloads now baked into the dataset tensors (e.g. the
    # full GMM covariances) before cross-validation. Fitted-GMM methods also drop
    # the dense expression matrix, which their supports don't depend on.
    slim_adata_after_build(
        dataset,
        gmm_key,
        drop_expression=method_uses_fitted_gmm(method),
        logger=logger,
    )

    # Build tune kwargs from the resolved experiment + method config.
    tune_cfg = get_method_tune_kwargs(cfg, method_name)
    tune_cfg["plot_contour"] = False   # always disable interactive plots
    tune_cfg["verbose"] = tune_cfg.get("verbose", False)

    logger.info(f"Tune kwargs: {tune_cfg}")

    Ws, scores_df = dataset.tune(**tune_cfg)

    scoring = tune_cfg.get("scoring")
    if scoring is None:
        scoring = ["knn", "ari"]
    # tune.tie_break: rule among combinations that tie on the scores (ggml_ot TIE_BREAKS)
    raw_best_idx = _best_index_by_scoring(scores_df, scoring, tie_break=tune_cfg.get("tie_break", "first"))
    best_idx = raw_best_idx if isinstance(raw_best_idx, tuple) else (raw_best_idx,)
    best_params = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in zip(scores_df.index.names, best_idx)
    }
    selected_scores = _flatten_score_row(scores_df.loc[raw_best_idx])

    written_plots = _render_tune_contours(
        scores_df=scores_df,
        best_params=best_params,
        cfg=cfg,
        out_dir=plot_dir,
        logger=logger,
    )

    # Outputs
    Path(snakemake.output.best_params).write_text(
        json.dumps(best_params, indent=2, sort_keys=True)
    )
    scores_df.to_csv(snakemake.output.scores)

    manifest = {
        "schema": "snakemake.tune.v2",
        "method_name": method_name,
        "dataset_type": method["dataset_type"],
        "tune_kwargs": to_jsonable(tune_cfg),
        "selection_scoring": to_jsonable(scoring),
        "selection_index": to_jsonable(best_params),
        "selection_scores": to_jsonable(selected_scores),
        "best_params": to_jsonable(best_params),
        "plot_artifacts": to_jsonable(
            {
                "plot_dir": str(plot_dir),
                "files": list_artifact_files(plot_dir) if written_plots else [],
            }
        ),
        "ggml_ot_version": getattr(ggml_ot, "__version__", "unknown"),
    }
    Path(snakemake.output.manifest).write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    logger.info(f"Best params: {best_params}")
    logger.info(f"Wrote: {snakemake.output.best_params}")


run(snakemake)  # noqa: F821
