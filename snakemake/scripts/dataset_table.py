"""
Generate a LaTeX dataset summary table for the active Snakemake dataset config.

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.source_h5ads    — raw.h5ad for cellxgene datasets; prepared h5ad for synthetic datasets
  snakemake.input.prepared_h5ads  — prepared/generated h5ad for all datasets
  snakemake.output.table          — output LaTeX table path
  snakemake.params.dataset_labels — paper-facing dataset names (optional per dataset)
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _dataset_metadata import (  # noqa: E402
    cell_type_count,
    dataset_display_name,
    distribution_count,
    unique_count,
    used_dimension,
)
from _utils import get_dataset_source, setup_logging  # noqa: E402

import pandas as pd  # noqa: E402
import anndata as ad  # noqa: E402


def _render_latex_table(df: pd.DataFrame) -> str:
    body = df.to_latex(
        index=False,
        escape=False,
        column_format=r"@{}l|cccccc@{}",
    )
    return "\n".join(
        [
            r"% Dataset summary table (active Snakemake dataset config)",
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Summary of the datasets used in the active Snakemake benchmark configuration. "
            r"Patients / dist. ($n_{\text{dist}}$) reports the number of patient-level distributions. "
            r"Dim. ($n_{\text{genes}}$) reports the original feature dimension when applicable; "
            r"Dim. used ($d$) reports the active representation dimension used by the workflow. "
            r"Synthetic datasets do not have an original gene dimension and are shown as ``-''; "
            r"for them, Cell types reports the total number of generating modes (signal + noise).}",
            r"\label{tab:dataset_summary}",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            body,
            r"\end{table}",
            "",
        ]
    )


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    out_path = Path(snakemake.output.table)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = snakemake.params.config
    dataset_names = list(snakemake.params.dataset_names)
    # Paper-facing labels shared with the other tables; fall back to the dataset metadata.
    dataset_labels = dict(snakemake.params.dataset_labels)

    rows = []
    for dataset_name, source_path_str, prepared_path_str in zip(
        dataset_names,
        snakemake.input.source_h5ads,
        snakemake.input.prepared_h5ads,
    ):
        ds_meta = cfg["datasets"][dataset_name]
        source = get_dataset_source(ds_meta)

        source_path = Path(source_path_str)
        prepared_path = Path(prepared_path_str)

        logger.info(f"Collecting dataset summary for {dataset_name}: source={source}")
        source_adata = ad.read_h5ad(source_path)
        prepared_adata = source_adata if source_path == prepared_path else ad.read_h5ad(prepared_path)

        original_dim = "-" if source == "synthetic_gmm" else f"{int(source_adata.n_vars)}"
        used_dim = used_dimension(prepared_adata, cfg, ds_meta)

        rows.append(
            {
                "Dataset": dataset_labels.get(dataset_name) or dataset_display_name(dataset_name, ds_meta),
                r"\shortstack{Size\\($n_{\text{cells}}$)}": int(source_adata.n_obs),
                r"\shortstack{Patients / dist.\\($n_{\text{dist}}$)}": distribution_count(source_adata, ds_meta),
                r"\shortstack{Dim.\\($n_{\text{genes}}$)}": original_dim,
                r"\shortstack{Dim. used\\($d$)}": used_dim,
                r"\shortstack{Cell\\types}": cell_type_count(source_adata, ds_meta),
                r"\shortstack{Disease\\states}": unique_count(source_adata, ds_meta["label_col"]),
            }
        )

    df = pd.DataFrame(rows)
    out_path.write_text(_render_latex_table(df))
    logger.info(f"Wrote: {out_path}")


run(snakemake)  # noqa: F821
