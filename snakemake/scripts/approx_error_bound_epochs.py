"""
Generate the Figure 5 epoch-wise approximation-error ablation.

Called via Snakemake ``script:`` directive (one invocation per dataset).
The ``snakemake`` object provides:
  snakemake.input.h5ad            — fitted GMM-backed AnnData for this dataset
  snakemake.output.metrics_csv    — long-form per-split/per-epoch metrics
  snakemake.output.metrics_json   — summary metrics JSON
  snakemake.output.figure_pdf     — rendered linear-scale PDF figure
  snakemake.output.figure_log_pdf — rendered log-scale PDF figure
  snakemake.output.relative_figure_pdf     — rendered relative-error PDF figure
  snakemake.output.relative_figure_log_pdf — rendered relative-error log-scale PDF figure
  snakemake.output.bures_gap_figure_pdf           — absolute Bures error PDF
  snakemake.output.bures_gap_figure_log_pdf       — absolute Bures error log PDF
  snakemake.output.bures_relative_figure_pdf     — Bures-only relative error PDF
  snakemake.output.bures_relative_figure_log_pdf — Bures-only relative error log PDF
  snakemake.output.mean_fraction_figure_pdf      — mean fraction of total distance PDF
  snakemake.output.trained_maps   — final learned map per init, variant and split (npz,
                                    keys "<init>__<variant>__split<i>")
  snakemake.output.manifest       — provenance JSON manifest
  snakemake.params.config         — resolved experiment config subset
  snakemake.params.dataset_name   — dataset name (from wildcard)
  snakemake.params.ds_meta        — dataset metadata dict
  snakemake.params.gmm_key        — GMM key for this dataset
  snakemake.log[0]
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _approx_error import (  # noqa: E402
    flatten_gmm_model_arrays,
    gaussian_emd_approx_errors,
    map_metrics,
)
from _utils import (  # noqa: E402
    apply_runtime_settings,
    build_dataset,
    list_artifact_files,
    setup_logging,
    to_jsonable,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import ggml_ot  # noqa: E402

try:  # noqa: E402
    import torch
except Exception:  # pragma: no cover - optional runtime dependency detail
    torch = None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    ggml_ot.settings.random_seed = seed


def _variant_plot_kwargs(variant_cfg: dict, variant_label: str) -> dict:
    style = dict(variant_cfg.get("style") or {})
    style.setdefault("linewidth", 3.0)
    style.setdefault("linestyle", "--" if variant_label == "None" else "-")
    return style


def _format_lambda_value(value) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if value.is_integer() and abs(value) >= 1.0:
        return f"{value:.1f}"
    return f"{value:g}"


def _dataset_variant_label(variant_name: str, variant_cfg: dict, variant_train: dict) -> str:
    base = str(variant_cfg.get("display_name", variant_name))
    reg = _format_lambda_value(variant_train.get("reg", 0.0))
    mi_reg = _format_lambda_value(variant_train.get("mi_reg", 0.0))

    if variant_name == "NoReg":
        return base
    if variant_name == "Frobenius":
        return rf"{base} ($\lambda_F = {reg}$)"
    if variant_name == "MutualInformation":
        return rf"{base} ($\lambda_{{\mathrm{{MI}}}} = {mi_reg}$)"
    if variant_name == "Both":
        return rf"{base} ($\lambda_F = {reg}, \lambda_{{\mathrm{{MI}}}} = {mi_reg}$)"
    return base


def _load_benchmark_train_kwargs(snakemake) -> dict | None:
    benchmark_manifest = snakemake.input.get("benchmark_manifest", [])
    if not benchmark_manifest:
        return None
    manifest_path = benchmark_manifest[0] if isinstance(benchmark_manifest, list) else benchmark_manifest
    if not manifest_path:
        return None
    manifest = json.loads(Path(manifest_path).read_text())
    train_kwargs = manifest.get("train_kwargs")
    if not isinstance(train_kwargs, dict):
        raise ValueError(f"Expected train_kwargs in benchmark manifest: {manifest_path}")
    return train_kwargs


def _resolve_variant_train_kwargs(
    base_train_kwargs: dict,
    variant_cfg: dict,
    benchmark_train_kwargs: dict | None,
) -> dict:
    variant_train = {**base_train_kwargs, **dict(variant_cfg.get("train") or {})}

    reg_from_benchmark = variant_train.pop("reg_from_benchmark", None)
    mi_reg_from_benchmark = variant_train.pop("mi_reg_from_benchmark", None)

    if reg_from_benchmark is not None:
        if benchmark_train_kwargs is None:
            raise ValueError("Variant requested reg_from_benchmark but no benchmark manifest was provided.")
        variant_train["reg"] = benchmark_train_kwargs[str(reg_from_benchmark)]

    if mi_reg_from_benchmark is not None:
        if benchmark_train_kwargs is None:
            raise ValueError("Variant requested mi_reg_from_benchmark but no benchmark manifest was provided.")
        variant_train["mi_reg"] = benchmark_train_kwargs[str(mi_reg_from_benchmark)]

    return variant_train


def _save_figure(
    metrics_df: pd.DataFrame,
    variants: dict,
    *,
    legend_labels: dict[str, str] | None,
    value_col: str,
    figsize: tuple[float, float],
    tick_step: int,
    ylabel: str,
    title: str,
    max_epochs_seen: int,
    base_train_kwargs: dict,
    out_pdf: Path,
    dpi: int,
    log_scale: bool,
    logger,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    for variant_name, variant_cfg in variants.items():
        display_name = (
            legend_labels.get(variant_name)
            if legend_labels is not None
            else str(variant_cfg.get("display_name", variant_name))
        )
        variant_mean = (
            metrics_df[
                (metrics_df["variant_name"] == variant_name)
                & (metrics_df["split"] == "mean")
            ]
            .sort_values("epoch")
        )
        ax.plot(
            variant_mean["epoch"].to_numpy(),
            variant_mean[value_col].to_numpy(),
            label=display_name,
            **_variant_plot_kwargs(dict(variant_cfg or {}), display_name),
        )

    epochs = int(base_train_kwargs.get("max_iter", max_epochs_seen - 1))
    tick_end = min(max_epochs_seen - 1, epochs)
    ax.set_xticks(list(range(0, tick_end + 1, max(1, tick_step))))
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        mean_values = metrics_df.loc[metrics_df["split"] == "mean", value_col].dropna()
        if len(mean_values) > 0 and (mean_values > 0).all():
            ax.set_yscale("log")
        else:
            logger.warning(f"Skipping log scale for {value_col!r}: contains non-positive values.")
    ax.legend(title="Regularization")
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote: {out_pdf}")


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    cfg = snakemake.params.config
    dataset_name = str(snakemake.params.dataset_name)
    ds_meta = dict(snakemake.params.ds_meta)
    gmm_key = str(snakemake.params.gmm_key)
    fig_cfg = dict(cfg.get("approx_error_bound") or {})
    if not fig_cfg:
        raise ValueError("Missing required top-level `approx_error_bound:` config block.")

    gmm_path = Path(snakemake.input.h5ad)

    method_cfg = dict(fig_cfg.get("method") or {})
    method_cfg.setdefault("dataset_type", "gmm")
    method_cfg.setdefault("ground_metric", "learned")

    split_cfg = dict(fig_cfg.get("split") or {})
    n_splits = int(split_cfg.get("n_splits", 3))
    train_size = float(split_cfg.get("train_size", 0.6))

    base_train_kwargs = dict(fig_cfg.get("train") or {})
    variants = dict(fig_cfg.get("variants") or {})
    if not variants:
        raise ValueError("`approx_error_bound.variants` must define at least one ablation setup.")

    plot_cfg = dict(fig_cfg.get("plot") or {})
    tick_step = int(plot_cfg.get("tick_step", 5))
    dpi = int(plot_cfg.get("dpi", 300))
    title = plot_cfg.get(
        "title",
        r"Ablation Study: Approximation Error Gap $\Delta_{i,j}(\mathbf{A})$",
    )
    ylabel = plot_cfg.get(
        "ylabel",
        r"Approximation Error $\Delta$",
    )
    relative_title = plot_cfg.get(
        "relative_title",
        r"Relative Approximation Error",
    )
    relative_ylabel = plot_cfg.get(
        "relative_ylabel",
        r"Relative Approximation Error $\Delta_{\mathrm{rel}}$",
    )
    figsize = tuple(plot_cfg.get("figsize", [6, 3]))

    # Error computation options
    error_cfg = dict(fig_cfg.get("error") or {})
    min_distance_threshold = float(error_cfg.get("min_distance_threshold", 0.0))
    min_active_fraction = float(error_cfg.get("min_active_fraction", 0.5))
    weight_by_component = bool(error_cfg.get("weight_by_component", False))

    apply_runtime_settings(cfg)
    base_seed = int(cfg.get("random_seed", 42))
    benchmark_train_kwargs = _load_benchmark_train_kwargs(snakemake)
    logger.info(f"Figure 5 dataset={dataset_name}, gmm_key={gmm_key}, gmm_path={gmm_path}")
    logger.info(f"Variants: {list(variants)}")
    if benchmark_train_kwargs is not None:
        logger.info(
            "Using benchmark-selected regularization values: "
            f"reg={benchmark_train_kwargs.get('reg')}, mi_reg={benchmark_train_kwargs.get('mi_reg')}"
        )

    adata = sc.read_h5ad(gmm_path)
    dataset = build_dataset(adata, cfg, ds_meta, method_cfg, gmm_key)
    split_indices = dataset._train_test_split(n_splits, train_size)
    train_sets = [dataset._subset(train_idx) for train_idx, _ in split_indices]
    inits = list(fig_cfg.get("inits", ["orthonormal"]))

    try:
        model_payload = adata.uns[gmm_key]["model"]
    except KeyError as exc:
        raise KeyError(f"Could not find gmm model payload under adata.uns[{gmm_key!r}]['model']") from exc
    flat_means, flat_covariances, flat_weights = flatten_gmm_model_arrays(model_payload)

    records: list[dict] = []
    summary: dict[str, dict] = {}
    final_maps: dict[str, np.ndarray] = {}
    max_epochs_seen = 0

    METRIC_KEYS = [
        "approx_error", "relative_approx_error",
        "bures_gap", "bures_relative_error", "bures_fraction",
        "n_pairs_active", "n_pairs_total",
        "fro_norm", "projected_mi_weighted", "projected_mi_total",
        "axis_redundancy", "gram_dependence", "effective_rank", "stable_rank",
        "latent_correlation", "latent_correlation_within", "latent_participation_ratio",
    ]

    for init in inits:
        # every variant from both initializations of the map (ggml-ot init_strategy)
        ggml_ot.settings.init_strategy = init
        for variant_index, (variant_name, variant_cfg_raw) in enumerate(variants.items()):
            variant_cfg = dict(variant_cfg_raw or {})
            display_name = str(variant_cfg.get("display_name", variant_name))
            variant_train = _resolve_variant_train_kwargs(base_train_kwargs, variant_cfg, benchmark_train_kwargs)
            legend_label = _dataset_variant_label(variant_name, variant_cfg, variant_train)
            # Per-split histories for each metric
            variant_histories: dict[str, list[list[float]]] = {k: [] for k in METRIC_KEYS}
            split_seconds: list[float] = []
            logger.info(
                f"Running variant={variant_name} ({legend_label}) with train kwargs: {variant_train}"
            )

            for split_idx, train_set in enumerate(train_sets):
                seed = base_seed + split_idx
                _set_seed(seed)
                start = time.perf_counter()
                history = ggml_ot.train_history(train_set, **variant_train)
                elapsed = float(time.perf_counter() - start)
                split_seconds.append(elapsed)
                final_map = np.asarray(history[-1])
                final_maps[f"{init}__{variant_name}__split{split_idx}"] = final_map

                split_series: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
                for epoch_idx, linear_map in enumerate(history):
                    lm = np.asarray(linear_map)
                    metrics = gaussian_emd_approx_errors(
                        flat_means,
                        flat_covariances,
                        lm,
                        weights=flat_weights if weight_by_component else None,
                        min_distance_threshold=min_distance_threshold,
                        min_active_fraction=min_active_fraction,
                    )
                    metrics.update(map_metrics(flat_covariances, flat_weights, lm, mu=flat_means))
                    for k in METRIC_KEYS:
                        split_series[k].append(metrics[k])
                    record = {
                        "dataset": dataset_name,
                        "init": init,
                        "variant_name": variant_name,
                        "display_name": display_name,
                        "split": str(split_idx),
                        "epoch": epoch_idx,
                        "train_seconds": elapsed,
                        "seed": seed,
                        "reg_type": variant_train.get("reg_type"),
                        "reg": variant_train.get("reg"),
                        "mi_reg": variant_train.get("mi_reg", 0.0),
                        "diag_bures_approx": bool(variant_train.get("diag_bures_approx", False)),
                    }
                    record.update(metrics)
                    records.append(record)

                for k in METRIC_KEYS:
                    variant_histories[k].append(split_series[k])
                max_epochs_seen = max(max_epochs_seen, len(split_series[METRIC_KEYS[0]]))

            # Aggregate across splits: compute mean and std for each metric
            aggregated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for k in METRIC_KEYS:
                hists = variant_histories[k]
                max_len = max(len(h) for h in hists)
                padded = np.full((len(hists), max_len), np.nan, dtype=float)
                for row_idx, hist in enumerate(hists):
                    padded[row_idx, : len(hist)] = hist
                aggregated[k] = (np.nanmean(padded, axis=0), np.nanstd(padded, axis=0))

            n_epochs = len(aggregated[METRIC_KEYS[0]][0])
            for epoch_idx in range(n_epochs):
                record = {
                    "dataset": dataset_name,
                    "init": init,
                    "variant_name": variant_name,
                    "display_name": display_name,
                    "split": "mean",
                    "epoch": epoch_idx,
                    "train_seconds": float(np.mean(split_seconds)),
                    "seed": None,
                    "reg_type": variant_train.get("reg_type"),
                    "reg": variant_train.get("reg"),
                    "mi_reg": variant_train.get("mi_reg", 0.0),
                    "diag_bures_approx": bool(variant_train.get("diag_bures_approx", False)),
                }
                for k in METRIC_KEYS:
                    record[k] = float(aggregated[k][0][epoch_idx])
                    record[f"{k}_std"] = float(aggregated[k][1][epoch_idx])
                records.append(record)

            variant_summary = {
                "display_name": display_name,
                "legend_label": legend_label,
                "train_kwargs": to_jsonable(variant_train),
                "plot_kwargs": to_jsonable(_variant_plot_kwargs(variant_cfg, display_name)),
                "split_train_seconds": split_seconds,
            }
            for k in METRIC_KEYS:
                mean_vals, std_vals = aggregated[k]
                variant_summary[f"mean_{k}_by_epoch"] = [float(v) for v in mean_vals]
                variant_summary[f"std_{k}_by_epoch"] = [float(v) for v in std_vals]
            summary[f"{init}/{variant_name}"] = variant_summary

    metrics_df = pd.DataFrame.from_records(records)
    metrics_df.sort_values(["init", "variant_name", "split", "epoch"], inplace=True)
    # the per-dataset diagnostic figures show the first init; per-init figures are made by
    # the aggregate rules
    figure_df = metrics_df[metrics_df["init"] == inits[0]]

    out_metrics_csv = Path(snakemake.output.metrics_csv)
    out_metrics_json = Path(snakemake.output.metrics_json)
    out_figure_pdf = Path(snakemake.output.figure_pdf)
    out_figure_log_pdf = Path(snakemake.output.figure_log_pdf)
    out_relative_figure_pdf = Path(snakemake.output.relative_figure_pdf)
    out_relative_figure_log_pdf = Path(snakemake.output.relative_figure_log_pdf)
    out_bures_gap_pdf = Path(snakemake.output.bures_gap_figure_pdf)
    out_bures_gap_log_pdf = Path(snakemake.output.bures_gap_figure_log_pdf)
    out_bures_rel_pdf = Path(snakemake.output.bures_relative_figure_pdf)
    out_bures_rel_log_pdf = Path(snakemake.output.bures_relative_figure_log_pdf)
    out_bures_fraction_pdf = Path(snakemake.output.bures_fraction_figure_pdf)
    out_fro_norm_pdf = Path(snakemake.output.fro_norm_figure_pdf)
    out_projected_mi_pdf = Path(snakemake.output.projected_mi_figure_pdf)
    out_manifest = Path(snakemake.output.manifest)
    for path in [
        out_metrics_csv,
        out_metrics_json,
        out_figure_pdf,
        out_figure_log_pdf,
        out_relative_figure_pdf,
        out_relative_figure_log_pdf,
        out_bures_gap_pdf,
        out_bures_gap_log_pdf,
        out_bures_rel_pdf,
        out_bures_rel_log_pdf,
        out_bures_fraction_pdf,
        out_fro_norm_pdf,
        out_projected_mi_pdf,
        out_manifest,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(out_metrics_csv, index=False)
    np.savez(snakemake.output.trained_maps, **final_maps)
    out_metrics_json.write_text(
        json.dumps(
            {
                "schema": "snakemake.approx_error_bound.metrics.v3",
                "dataset": dataset_name,
                "n_splits": n_splits,
                "train_size": train_size,
                "variants": to_jsonable(summary),
            },
            indent=2,
            sort_keys=True,
        )
    )

    bures_gap_ylabel = plot_cfg.get(
        "bures_gap_ylabel",
        r"Absolute Bures Error $|\hat{B} - B|$",
    )
    bures_gap_title = plot_cfg.get("bures_gap_title", "Absolute Covariance Approximation Error")
    bures_rel_ylabel = plot_cfg.get(
        "bures_relative_ylabel",
        r"Bures Relative Error $|\Delta_{\mathrm{Bures}}| / B_{\mathrm{exact}}$",
    )
    bures_rel_title = plot_cfg.get("bures_relative_title", "Covariance Approximation Quality")
    bures_frac_ylabel = plot_cfg.get("bures_fraction_ylabel", "Bures Fraction of Total Distance")
    bures_frac_title = plot_cfg.get("bures_fraction_title", "Covariance vs. Mean Dominance")
    legend_labels = {
        variant_name: str(summary[f"{inits[0]}/{variant_name}"]["legend_label"])
        for variant_name in variants
    }

    _save_figure(
        figure_df,
        variants,
        legend_labels=legend_labels,
        value_col="approx_error",
        figsize=figsize,
        tick_step=tick_step,
        ylabel=ylabel,
        title=title,
        max_epochs_seen=max_epochs_seen,
        base_train_kwargs=base_train_kwargs,
        out_pdf=out_figure_pdf,
        dpi=dpi,
        log_scale=False,
        logger=logger,
    )
    _save_figure(
        figure_df,
        variants,
        legend_labels=legend_labels,
        value_col="approx_error",
        figsize=figsize,
        tick_step=tick_step,
        ylabel=ylabel,
        title=title,
        max_epochs_seen=max_epochs_seen,
        base_train_kwargs=base_train_kwargs,
        out_pdf=out_figure_log_pdf,
        dpi=dpi,
        log_scale=True,
        logger=logger,
    )
    _save_figure(
        figure_df,
        variants,
        legend_labels=legend_labels,
        value_col="relative_approx_error",
        figsize=figsize,
        tick_step=tick_step,
        ylabel=relative_ylabel,
        title=relative_title,
        max_epochs_seen=max_epochs_seen,
        base_train_kwargs=base_train_kwargs,
        out_pdf=out_relative_figure_pdf,
        dpi=dpi,
        log_scale=False,
        logger=logger,
    )
    _save_figure(
        figure_df,
        variants,
        legend_labels=legend_labels,
        value_col="relative_approx_error",
        figsize=figsize,
        tick_step=tick_step,
        ylabel=relative_ylabel,
        title=relative_title,
        max_epochs_seen=max_epochs_seen,
        base_train_kwargs=base_train_kwargs,
        out_pdf=out_relative_figure_log_pdf,
        dpi=dpi,
        log_scale=True,
        logger=logger,
    )
    _save_figure(
        figure_df,
        variants,
        legend_labels=legend_labels,
        value_col="bures_gap",
        figsize=figsize,
        tick_step=tick_step,
        ylabel=bures_gap_ylabel,
        title=bures_gap_title,
        max_epochs_seen=max_epochs_seen,
        base_train_kwargs=base_train_kwargs,
        out_pdf=out_bures_gap_pdf,
        dpi=dpi,
        log_scale=False,
        logger=logger,
    )
    _save_figure(
        figure_df,
        variants,
        legend_labels=legend_labels,
        value_col="bures_gap",
        figsize=figsize,
        tick_step=tick_step,
        ylabel=bures_gap_ylabel,
        title=bures_gap_title,
        max_epochs_seen=max_epochs_seen,
        base_train_kwargs=base_train_kwargs,
        out_pdf=out_bures_gap_log_pdf,
        dpi=dpi,
        log_scale=True,
        logger=logger,
    )
    _save_figure(
        figure_df,
        variants,
        legend_labels=legend_labels,
        value_col="bures_relative_error",
        figsize=figsize,
        tick_step=tick_step,
        ylabel=bures_rel_ylabel,
        title=bures_rel_title,
        max_epochs_seen=max_epochs_seen,
        base_train_kwargs=base_train_kwargs,
        out_pdf=out_bures_rel_pdf,
        dpi=dpi,
        log_scale=False,
        logger=logger,
    )
    _save_figure(
        figure_df,
        variants,
        legend_labels=legend_labels,
        value_col="bures_relative_error",
        figsize=figsize,
        tick_step=tick_step,
        ylabel=bures_rel_ylabel,
        title=bures_rel_title,
        max_epochs_seen=max_epochs_seen,
        base_train_kwargs=base_train_kwargs,
        out_pdf=out_bures_rel_log_pdf,
        dpi=dpi,
        log_scale=True,
        logger=logger,
    )
    _save_figure(
        figure_df,
        variants,
        legend_labels=legend_labels,
        value_col="bures_fraction",
        figsize=figsize,
        tick_step=tick_step,
        ylabel=bures_frac_ylabel,
        title=bures_frac_title,
        max_epochs_seen=max_epochs_seen,
        base_train_kwargs=base_train_kwargs,
        out_pdf=out_bures_fraction_pdf,
        dpi=dpi,
        log_scale=False,
        logger=logger,
    )
    _save_figure(
        figure_df,
        variants,
        legend_labels=legend_labels,
        value_col="fro_norm",
        figsize=figsize,
        tick_step=tick_step,
        ylabel=r"$\|\mathbf{A}\|_F$",
        title="Frobenius Norm of Learned Map",
        max_epochs_seen=max_epochs_seen,
        base_train_kwargs=base_train_kwargs,
        out_pdf=out_fro_norm_pdf,
        dpi=dpi,
        log_scale=False,
        logger=logger,
    )
    _save_figure(
        figure_df,
        variants,
        legend_labels=legend_labels,
        value_col="projected_mi_weighted",
        figsize=figsize,
        tick_step=tick_step,
        ylabel=r"Weighted MI($\mathbf{A}\Sigma\mathbf{A}^\top$)",
        title="Projected MI (Component-Weighted)",
        max_epochs_seen=max_epochs_seen,
        base_train_kwargs=base_train_kwargs,
        out_pdf=out_projected_mi_pdf,
        dpi=dpi,
        log_scale=False,
        logger=logger,
    )

    manifest = {
        "schema": "snakemake.approx_error_bound.v2",
        "dataset": dataset_name,
        "gmm_path": str(gmm_path),
        "gmm_key": gmm_key,
        "n_splits": n_splits,
        "train_size": train_size,
        "config": to_jsonable(cfg),
        "source_artifacts": [str(gmm_path)],
        "generated_artifacts": list_artifact_files(out_figure_pdf.parent),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    logger.info(f"Wrote: {out_metrics_csv}")
    logger.info(f"Wrote: {out_metrics_json}")
    logger.info(f"Wrote: {out_manifest}")


run(snakemake)  # noqa: F821
