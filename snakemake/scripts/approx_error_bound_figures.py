"""
Figure 5 as in the paper: approximation error per regularizer with the bound overlay.

Ported from notebooks/fig5_regularizer_ablation.ipynb, with the paper's settings: log scale,
bound proxy ||A||_F^2 sqrt(MI) on the right axis (scaled with the tightest constant, so that it
stays above the error), geometric mean over the datasets of each group. Written for the absolute
(approx_error) and the relative (relative_approx_error) error, each for every initialization
alone and for all together (shade by initialization).

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.dataset_csvs     — per-dataset approx_error_epochs.csv
  snakemake.input.gmm_h5ads        — per-dataset fitted GMMs (for the analytical constant)
  snakemake.output.grouped         — per error column and init set: scRNA-seq and synthetic panels
  snakemake.output.per_dataset     — per error column and init set: one panel per dataset
  snakemake.output.constants       — CSV: empirical vs analytical bound constant per error column and init
  snakemake.params.dataset_names   — dataset order, matching the inputs
  snakemake.params.inits           — initializations
  snakemake.params.error_columns   — error columns; the outputs run over error columns, then
                                     init sets (_error_plots.init_sets)
  snakemake.params.synthetic       — datasets of the synthetic group
  snakemake.params.gmm_key         — per-dataset GMM keys
  snakemake.log[0]
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _error_plots import (  # noqa: E402
    DS_DISPLAY_NAMES,
    ERROR_YLABELS,
    analytical_bound_constant,
    compute_bound_constant,
    compute_bound_proxy,
    init_sets,
    load_dataset_df,
    load_group,
    plot_error_panels,
    plot_per_dataset_grid,
    resolve_init_set,
)
from _utils import setup_logging  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402

SUPTITLES = {"approx_error": "Absolute W₂ Approx. Error", "relative_approx_error": "Relative W₂ Approx. Error"}


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    datasets = list(snakemake.params.dataset_names)
    synthetic = set(snakemake.params.synthetic)
    csv_base = Path(snakemake.input.dataset_csvs[0]).parents[1]

    inits = list(snakemake.params.inits)
    combos = list(itertools.product(snakemake.params.error_columns, init_sets(inits)))
    assert len(combos) == len(snakemake.output.grouped) == len(snakemake.output.per_dataset)
    groups = [
        ([ds for ds in datasets if ds not in synthetic], "scRNA-seq Datasets"),
        ([ds for ds in datasets if ds in synthetic], "Synthetic Datasets"),
    ]
    analytical = {}
    for ds, h5ad_path in zip(datasets, snakemake.input.gmm_h5ads):
        model = sc.read_h5ad(h5ad_path).uns[snakemake.params.gmm_key[ds]]["model"]
        analytical[ds] = analytical_bound_constant(np.asarray(model["var"]), np.asarray(model["pi"]))

    rows = []
    for (error_col, init_set), grouped_path, per_dataset_path in zip(combos, snakemake.output.grouped,
                                                                     snakemake.output.per_dataset):
        shown = resolve_init_set(init_set, inits)
        fig, constants = plot_error_panels(
            [load_group(csv_base, members, agg="geometric", init=shown) for members, _ in groups],
            [title for _, title in groups],
            error_col=error_col, ylabel=ERROR_YLABELS[error_col], suptitle=SUPTITLES[error_col],
            log_scale=True, show_bound=True,
            legend_layout="regularization_only",
            line_style_axis_labels=True, outer_axis_labels=True,
            fig5_layout=True,
        )
        # no tight cropping: the shared layout keeps the panels the size of Figure 5 (b)
        fig.savefig(grouped_path, dpi=300)
        plt.close(fig)
        for (_, title), constant in zip(groups, constants or []):
            logger.info(f"{error_col}, {init_set}, {title}: empirical constant (right-axis scale) = {constant:.4f}")

        per_dataset = {ds: load_dataset_df(csv_base, ds, init=shown) for ds in datasets}
        fig, _ = plot_per_dataset_grid(
            per_dataset, datasets,
            error_col=error_col, ylabel=ERROR_YLABELS[error_col], suptitle=SUPTITLES[error_col],
            log_scale=True, show_bound=True,
            # regularization and y-axis key in two rows (the stacked layout has no initialization row)
            legend_layout="stacked" if len(shown) == 1 else "horizontal",
        )
        fig.savefig(per_dataset_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        if len(shown) > 1:
            continue
        for ds in datasets:
            df = per_dataset[ds]
            rows.append(
                {
                    "error": error_col,
                    "init": init_set,
                    "dataset": ds,
                    "display_name": DS_DISPLAY_NAMES.get(ds, ds),
                    "C_empirical": compute_bound_constant(df[error_col].to_numpy(), compute_bound_proxy(df).to_numpy()),
                    # the analytical constant bounds the absolute error only
                    "C_analytical": analytical[ds] if error_col == "approx_error" else np.nan,
                }
            )
    constants_df = pd.DataFrame(rows)
    constants_df["C_analytical_over_empirical"] = constants_df["C_analytical"] / constants_df["C_empirical"]
    constants_df.to_csv(snakemake.output.constants, index=False)
    logger.info("\n" + constants_df.round(4).to_string(index=False))


run(snakemake)  # noqa: F821
