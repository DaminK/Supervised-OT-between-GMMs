"""
Aggregate patient-wise GMM hold-out NLL diagnostics into appendix-ready outputs.

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.dataset_csvs   — per-dataset hold-out NLL CSVs
  snakemake.output.csv           — concatenated patient-wise CSV
  snakemake.output.table         — LaTeX summary table
  snakemake.output.plot          — per-dataset train-vs-validation panel plot
  snakemake.params.config        — dataset metadata config subset
  snakemake.params.dataset_names — dataset order
  snakemake.log[0]
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _dataset_metadata import dataset_display_name  # noqa: E402
from _utils import setup_logging  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402
import ggml_ot  # noqa: E402
from ggml_ot.plot.eval import gmm_fit_validation_boxplot  # noqa: E402


def _format_median_quartiles(
    median_value: float,
    q25_value: float,
    q75_value: float,
    *,
    digits: int = 3,
) -> str:
    """Format one median [Q1, Q3] entry for LaTeX."""
    return rf"{median_value:.{digits}f} [{q25_value:.{digits}f}, {q75_value:.{digits}f}]"


def _robust_summary(results: pd.DataFrame, *, groupby_cols: list[str] | None = None) -> pd.DataFrame:
    """Summarize hold-out NLL results with robust median/IQR statistics."""
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return pd.DataFrame()

    group_cols = list(groupby_cols or ["group"])
    if groupby_cols is None:
        results_df = results_df.assign(group="Overall")

    grouped = results_df.groupby(group_cols, dropna=False, sort=False)
    summary = grouped.agg(
        count=("nll_gap", "size"),
        median_train_nll=("train_nll", "median"),
        q25_train_nll=("train_nll", lambda s: s.quantile(0.25)),
        q75_train_nll=("train_nll", lambda s: s.quantile(0.75)),
        median_validation_nll=("validation_nll", "median"),
        q25_validation_nll=("validation_nll", lambda s: s.quantile(0.25)),
        q75_validation_nll=("validation_nll", lambda s: s.quantile(0.75)),
        median_nll_gap=("nll_gap", "median"),
        q25_nll_gap=("nll_gap", lambda s: s.quantile(0.25)),
        q75_nll_gap=("nll_gap", lambda s: s.quantile(0.75)),
    )
    if len(group_cols) == 1:
        summary.index = pd.Index(summary.index.tolist(), name=group_cols[0])
    return summary


def _summary_rows(summary: pd.DataFrame, dataset_cfg: dict, dataset_names: list[str]) -> list[dict[str, str | int]]:
    """Convert summary dataframe into ordered row dictionaries."""
    rows: list[dict[str, str | int]] = []
    for dataset_name in dataset_names:
        if dataset_name not in summary.index:
            continue
        row = summary.loc[dataset_name]
        ds_meta = dataset_cfg[dataset_name]
        rows.append(
            {
                "Dataset": dataset_display_name(dataset_name, ds_meta),
                r"$n_{\mathrm{dist}}$": int(row["count"]),
                "Train NLL": _format_median_quartiles(
                    float(row["median_train_nll"]),
                    float(row["q25_train_nll"]),
                    float(row["q75_train_nll"]),
                ),
                "Validation NLL": _format_median_quartiles(
                    float(row["median_validation_nll"]),
                    float(row["q25_validation_nll"]),
                    float(row["q75_validation_nll"]),
                ),
                "Gap": _format_median_quartiles(
                    float(row["median_nll_gap"]),
                    float(row["q25_nll_gap"]),
                    float(row["q75_nll_gap"]),
                ),
            }
        )
    return rows


def _render_latex_table(rows: list[dict[str, str | int]]) -> str:
    """Render the hold-out NLL summary table as LaTeX."""
    df = pd.DataFrame(rows)
    body = df.to_latex(
        index=False,
        escape=False,
        column_format=r"@{}lrrrr@{}",
    )
    return "\n".join(
        [
            r"% GMM hold-out NLL summary table",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{\textbf{Patient-wise GMM hold-out negative log-likelihood.} "
            r"Values report the median with first and third quartiles $[Q_1, Q_3]$ across patient-level distributions. "
            r"The gap is defined as validation NLL minus train NLL, so values near zero indicate limited overfitting.}",
            r"\label{tab:gmm_holdout_nll}",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            body,
            r"\end{table}",
            "",
        ]
    )


def _ordered_present_datasets(results: pd.DataFrame, dataset_names: list[str]) -> list[str]:
    """Return datasets present in the results, preserving config order."""
    present = set(results["dataset"].dropna().astype(str))
    ordered = [dataset_name for dataset_name in dataset_names if dataset_name in present]
    extras = sorted(present.difference(ordered))
    return ordered + extras


def _plot_dataset_panels(
    results: pd.DataFrame,
    *,
    dataset_cfg: dict,
    dataset_names: list[str],
    out_plot: Path,
) -> None:
    """Render one train-vs-validation panel per dataset with independent y-scales."""
    ordered_datasets = _ordered_present_datasets(results, dataset_names)
    if not ordered_datasets:
        raise ValueError("No datasets available for GMM hold-out validation plotting.")

    n_panels = len(ordered_datasets)
    n_cols = min(4, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 4.0 * n_rows),
        squeeze=False,
        sharey=False,
    )
    axes_flat = list(axes.ravel())

    for panel_idx, dataset_name in enumerate(ordered_datasets):
        ax = axes_flat[panel_idx]
        dataset_results = results.loc[results["dataset"].astype(str) == str(dataset_name)].copy()
        ds_meta = dataset_cfg.get(dataset_name, {})
        title = f"{dataset_display_name(dataset_name, ds_meta)}\n$n_{{\\mathrm{{dist}}}}={len(dataset_results)}$"

        gmm_fit_validation_boxplot(
            dataset_results,
            title=title,
            ax=ax,
            show=False,
            save=None,
        )
        if panel_idx % n_cols != 0:
            ax.set_ylabel("")

    for ax in axes_flat[n_panels:]:
        ax.remove()

    fig.suptitle("Patient-wise GMM hold-out NLL by dataset", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_plot, dpi=ggml_ot.settings.figdpi, bbox_inches="tight")
    plt.close(fig)


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    cfg = dict(snakemake.params.config)
    dataset_cfg = cfg["datasets"]
    dataset_names = list(snakemake.params.dataset_names)

    out_csv = Path(snakemake.output.csv)
    out_table = Path(snakemake.output.table)
    out_plot = Path(snakemake.output.plot)
    for path in [out_csv, out_table, out_plot]:
        path.parent.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(csv_path) for csv_path in snakemake.input.dataset_csvs]
    if not frames:
        logger.warning("No hold-out NLL inputs found; writing empty outputs.")
        out_csv.write_text("")
        out_table.write_text("% No hold-out NLL data available\n")
        out_plot.touch()
        return

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(out_csv, index=False)
    logger.info("Wrote: %s (%d patient rows)", out_csv, len(results))

    summary = _robust_summary(results, groupby_cols=["dataset"])
    overall = _robust_summary(results)
    rows = _summary_rows(summary, dataset_cfg, dataset_names)
    if not overall.empty:
        overall_row = overall.iloc[0]
        rows.append(
            {
                "Dataset": "Overall",
                r"$n_{\mathrm{dist}}$": int(overall_row["count"]),
                "Train NLL": _format_median_quartiles(
                    float(overall_row["median_train_nll"]),
                    float(overall_row["q25_train_nll"]),
                    float(overall_row["q75_train_nll"]),
                ),
                "Validation NLL": _format_median_quartiles(
                    float(overall_row["median_validation_nll"]),
                    float(overall_row["q25_validation_nll"]),
                    float(overall_row["q75_validation_nll"]),
                ),
                "Gap": _format_median_quartiles(
                    float(overall_row["median_nll_gap"]),
                    float(overall_row["q25_nll_gap"]),
                    float(overall_row["q75_nll_gap"]),
                ),
            }
        )
    out_table.write_text(_render_latex_table(rows))
    logger.info("Wrote: %s", out_table)

    _plot_dataset_panels(
        results,
        dataset_cfg=dataset_cfg,
        dataset_names=dataset_names,
        out_plot=out_plot,
    )
    logger.info("Wrote: %s", out_plot)


run(snakemake)  # noqa: F821
