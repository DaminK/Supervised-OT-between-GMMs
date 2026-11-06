"""
Aggregate Figure 5 per-dataset CSVs into cross-dataset average plots.

Only relative error metrics are aggregated (absolute errors are not comparable
across datasets with different scale).

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.dataset_csvs     — per-dataset approx_error_epochs.csv files
  snakemake.output.csv             — concatenated + aggregated CSV
  snakemake.output.relative_pdf    — cross-dataset mean relative approx error
  snakemake.output.relative_log_pdf
  snakemake.output.bures_rel_pdf   — cross-dataset mean Bures relative error
  snakemake.output.bures_rel_log_pdf
  snakemake.output.manifest        — provenance JSON manifest
  snakemake.params.config          — resolved experiment config subset
  snakemake.params.dataset_names   — ordered dataset names
  snakemake.log[0]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _utils import (  # noqa: E402
    list_artifact_files,
    setup_logging,
    to_jsonable,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _variant_plot_kwargs(variant_cfg: dict, variant_label: str) -> dict:
    style = dict(variant_cfg.get("style") or {})
    style.setdefault("linewidth", 3.0)
    style.setdefault("linestyle", "--" if variant_label == "None" else "-")
    return style


def _save_figure(
    metrics_df: pd.DataFrame,
    variants: dict,
    *,
    value_col: str,
    figsize: tuple[float, float],
    tick_step: int,
    ylabel: str,
    title: str,
    max_epochs_seen: int,
    max_iter: int,
    out_pdf: Path,
    dpi: int,
    log_scale: bool,
    logger,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    for variant_name, variant_cfg in variants.items():
        display_name = str(variant_cfg.get("display_name", variant_name))
        variant_mean = (
            metrics_df[metrics_df["variant_name"] == variant_name]
            .sort_values("epoch")
        )
        if variant_mean.empty:
            continue
        ax.plot(
            variant_mean["epoch"].to_numpy(),
            variant_mean[value_col].to_numpy(),
            label=display_name,
            **_variant_plot_kwargs(dict(variant_cfg or {}), display_name),
        )

    tick_end = min(max_epochs_seen - 1, max_iter)
    ax.set_xticks(list(range(0, tick_end + 1, max(1, tick_step))))
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        mean_values = metrics_df[value_col]
        if (mean_values > 0).all():
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
    dataset_names = list(snakemake.params.dataset_names)
    fig_cfg = dict(cfg.get("approx_error_bound") or {})
    if not fig_cfg:
        raise ValueError("Missing required top-level `approx_error_bound:` config block.")

    variants = dict(fig_cfg.get("variants") or {})
    plot_cfg = dict(fig_cfg.get("plot") or {})
    base_train_kwargs = dict(fig_cfg.get("train") or {})
    tick_step = int(plot_cfg.get("tick_step", 5))
    dpi = int(plot_cfg.get("dpi", 300))
    figsize = tuple(plot_cfg.get("figsize", [6, 4]))
    max_iter = int(base_train_kwargs.get("max_iter", 50))

    # Output paths
    out_csv = Path(snakemake.output.csv)
    out_relative_pdf = Path(snakemake.output.relative_pdf)
    out_relative_log_pdf = Path(snakemake.output.relative_log_pdf)
    out_bures_rel_pdf = Path(snakemake.output.bures_rel_pdf)
    out_bures_rel_log_pdf = Path(snakemake.output.bures_rel_log_pdf)
    out_manifest = Path(snakemake.output.manifest)
    for path in [out_csv, out_relative_pdf, out_relative_log_pdf,
                 out_bures_rel_pdf, out_bures_rel_log_pdf, out_manifest]:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Read and concat per-dataset CSVs
    frames = []
    for csv_path in snakemake.input.dataset_csvs:
        # Preserve the literal variant label "None" instead of treating it as NA.
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
        frames.append(df)
        logger.info(f"Read {csv_path}: {len(df)} rows")

    if not frames:
        logger.warning("No per-dataset CSVs found; writing empty outputs.")
        out_csv.write_text("")
        for path in [out_relative_pdf, out_relative_log_pdf,
                     out_bures_rel_pdf, out_bures_rel_log_pdf]:
            path.touch()
        return

    all_df = pd.concat(frames, ignore_index=True)
    logger.info(f"Concatenated {len(frames)} datasets, {len(all_df)} total rows")

    # Filter to cross-split mean rows and average across datasets. These diagnostic figures show
    # the first initialization; the per-init paper figures come from approx_error_bound_figures.
    mean_rows = all_df[all_df["split"] == "mean"].copy()
    if "init" in mean_rows.columns:
        first_init = list((snakemake.params.config.get("approx_error_bound") or {}).get("inits", ["orthonormal"]))[0]
        mean_rows = mean_rows[mean_rows["init"] == first_init]

    AGG_METRICS = [
        "relative_approx_error", "bures_relative_error", "bures_fraction",
        "fro_norm", "projected_mi_weighted", "projected_mi_total",
    ]
    agg_cols = {col: "mean" for col in AGG_METRICS if col in mean_rows.columns}
    aggregated = (
        mean_rows
        .groupby(["variant_name", "display_name", "epoch"], sort=False)
        .agg(agg_cols)
        .reset_index()
    )

    max_epochs_seen = int(aggregated["epoch"].max()) + 1 if len(aggregated) > 0 else 0
    logger.info(f"Aggregated across {len(dataset_names)} datasets, "
                f"{len(aggregated)} rows, max_epochs={max_epochs_seen}")

    # Write aggregated CSV
    aggregated.to_csv(out_csv, index=False)
    logger.info(f"Wrote: {out_csv}")

    # Plot relative approximation error
    relative_ylabel = plot_cfg.get(
        "relative_ylabel",
        r"Average Relative $\Delta_{ij}(\mathbf{A})$",
    )
    relative_title = plot_cfg.get(
        "relative_title",
        "Relative Approximation Error",
    )
    aggregate_title_suffix = f" (avg. over {len(dataset_names)} datasets)"

    plot_kwargs = dict(
        figsize=figsize,
        tick_step=tick_step,
        max_epochs_seen=max_epochs_seen,
        max_iter=max_iter,
        dpi=dpi,
        logger=logger,
    )

    if "relative_approx_error" in aggregated.columns:
        _save_figure(
            aggregated, variants,
            value_col="relative_approx_error",
            ylabel=relative_ylabel,
            title=relative_title + aggregate_title_suffix,
            out_pdf=out_relative_pdf,
            log_scale=False,
            **plot_kwargs,
        )
        _save_figure(
            aggregated, variants,
            value_col="relative_approx_error",
            ylabel=relative_ylabel,
            title=relative_title + aggregate_title_suffix,
            out_pdf=out_relative_log_pdf,
            log_scale=True,
            **plot_kwargs,
        )

    # Plot Bures relative error
    bures_rel_ylabel = plot_cfg.get(
        "bures_relative_ylabel",
        r"Bures Relative Error $|\Delta_{\mathrm{Bures}}| / B_{\mathrm{exact}}$",
    )
    bures_rel_title = plot_cfg.get("bures_relative_title", "Covariance Approximation Quality")

    if "bures_relative_error" in aggregated.columns:
        _save_figure(
            aggregated, variants,
            value_col="bures_relative_error",
            ylabel=bures_rel_ylabel,
            title=bures_rel_title + aggregate_title_suffix,
            out_pdf=out_bures_rel_pdf,
            log_scale=False,
            **plot_kwargs,
        )
        _save_figure(
            aggregated, variants,
            value_col="bures_relative_error",
            ylabel=bures_rel_ylabel,
            title=bures_rel_title + aggregate_title_suffix,
            out_pdf=out_bures_rel_log_pdf,
            log_scale=True,
            **plot_kwargs,
        )

    # Manifest
    manifest = {
        "schema": "snakemake.approx_error_bound_aggregate.v1",
        "datasets": dataset_names,
        "n_datasets": len(dataset_names),
        "config": to_jsonable(fig_cfg),
        "source_csvs": [str(p) for p in snakemake.input.dataset_csvs],
        "generated_artifacts": list_artifact_files(out_csv.parent),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info(f"Wrote: {out_manifest}")


run(snakemake)  # noqa: F821
