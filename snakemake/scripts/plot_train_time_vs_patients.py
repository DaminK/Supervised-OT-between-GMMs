"""
Aggregate training/OT-time-vs-patient-count plots.

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.scores               — list of all benchmark scores.csv files
  snakemake.input.source_h5ads         — one dataset source h5ad per dataset
  snakemake.output.plot                — training-time PDF output
  snakemake.params.plot_config         — display config subset (methods + splits)
  snakemake.params.table_config        — dataset metadata config subset
  snakemake.params.method_dataset_pairs — list of (method_name, dataset)
  snakemake.params.method_names        — configured benchmark methods
  snakemake.params.dataset_names       — configured datasets
  snakemake.log[0]
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

import anndata as ad  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from _benchmark_display import (  # noqa: E402
    load_score_summary,
    make_score_record,
    records_to_frame,
    render_ot_time_vs_patients_plot,
    render_train_time_vs_patients_plot,
    render_time_vs_patients_panel,
)
from _dataset_metadata import distribution_count  # noqa: E402
from _utils import setup_logging  # noqa: E402


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    plot_cfg = snakemake.params.plot_config
    table_cfg = snakemake.params.table_config
    method_names = list(snakemake.params.method_names)
    method_dataset_pairs = snakemake.params.method_dataset_pairs
    dataset_names = list(snakemake.params.dataset_names)

    out_plot = Path(snakemake.output.plot)
    out_plot_log = Path(snakemake.output.plot_log)
    out_ot_plot = Path(snakemake.output.ot_plot)
    out_ot_plot_log = Path(snakemake.output.ot_plot_log)
    out_panel = Path(snakemake.output.panel)
    out_panel_log = Path(snakemake.output.panel_log)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    score_records = []
    for (method_name, dataset), score_path in zip(method_dataset_pairs, snakemake.input.scores):
        try:
            summary = load_score_summary(score_path, extra_metrics=["full_ot_time(s)"])
            score_records.append(
                make_score_record(
                    dataset=dataset,
                    method_name=method_name,
                    method_cfg=plot_cfg["methods"][method_name],
                    summary=summary,
                )
            )
        except Exception as exc:
            logger.warning(f"Could not load {score_path}: {exc}")

    patient_counts = {}
    for dataset_name, source_path_str in zip(dataset_names, snakemake.input.source_h5ads):
        ds_meta = table_cfg["datasets"][dataset_name]
        source_path = Path(source_path_str)
        logger.info(f"Loading dataset metadata for {dataset_name} from {source_path}")
        source_adata = ad.read_h5ad(source_path)
        patient_counts[dataset_name] = distribution_count(source_adata, ds_meta)

    data = records_to_frame(score_records)
    render_train_time_vs_patients_plot(
        data,
        plot_cfg,
        method_names,
        patient_counts,
        str(out_plot),
    )
    logger.info(f"Wrote: {out_plot}")
    render_train_time_vs_patients_plot(
        data,
        plot_cfg,
        method_names,
        patient_counts,
        str(out_plot_log),
        log_scale=True,
    )
    logger.info(f"Wrote: {out_plot_log}")
    render_ot_time_vs_patients_plot(
        data,
        plot_cfg,
        method_names,
        patient_counts,
        str(out_ot_plot),
    )
    logger.info(f"Wrote: {out_ot_plot}")
    render_ot_time_vs_patients_plot(
        data,
        plot_cfg,
        method_names,
        patient_counts,
        str(out_ot_plot_log),
        log_scale=True,
    )
    logger.info(f"Wrote: {out_ot_plot_log}")
    render_time_vs_patients_panel(
        data,
        plot_cfg,
        method_names,
        patient_counts,
        str(out_panel),
    )
    logger.info(f"Wrote: {out_panel}")
    render_time_vs_patients_panel(
        data,
        plot_cfg,
        method_names,
        patient_counts,
        str(out_panel_log),
        log_scale=True,
    )
    logger.info(f"Wrote: {out_panel_log}")


run(snakemake)  # noqa: F821
