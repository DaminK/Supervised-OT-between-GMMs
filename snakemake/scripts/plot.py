"""
Per-dataset notebook-style LaTeX benchmark tables.

Called via Snakemake ``script:`` directive.  The ``snakemake`` object provides:
  snakemake.input.scores             — list of scores.csv (one per method)
  snakemake.output.table             — legacy per-dataset LaTeX table path
  snakemake.output.benchmark_table   — notebook-style per-dataset table path
  snakemake.output.pareto            — notebook-style per-dataset Pareto plot
  snakemake.params.config            — resolved experiment config
  snakemake.params.method_names      — list of method names (parallel to scores)
  snakemake.log[0]
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from _benchmark_display import (  # noqa: E402
    load_score_summary,
    make_score_record,
    records_to_frame,
    render_pareto_plot,
    render_latex_table,
)
from _utils import setup_logging  # noqa: E402


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    cfg = snakemake.params.config
    method_names = snakemake.params.method_names
    dataset_name = str(snakemake.wildcards.dataset)

    out_table = Path(snakemake.output.table)
    out_benchmark_table = Path(snakemake.output.benchmark_table)
    out_pareto = Path(snakemake.output.pareto)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_benchmark_table.parent.mkdir(parents=True, exist_ok=True)
    out_pareto.parent.mkdir(parents=True, exist_ok=True)

    score_records = []
    for method_name, score_path in zip(method_names, snakemake.input.scores):
        try:
            summary = load_score_summary(score_path)
            score_records.append(
                make_score_record(
                    dataset=dataset_name,
                    method_name=method_name,
                    method_cfg=cfg["methods"][method_name],
                    summary=summary,
                )
            )
        except Exception as exc:
            logger.warning(f"Could not load {score_path}: {exc}")

    if not score_records:
        empty = "% No benchmark data\n"
        out_table.write_text(empty)
        out_benchmark_table.write_text(empty)
        render_pareto_plot(
            records_to_frame([]),
            cfg,
            method_names,
            dataset_name,
            str(out_pareto),
            n_splits=int(cfg.get("train_test", {}).get("n_splits", 5)),
        )
        logger.warning("No scores found — wrote empty tables and placeholder Pareto plot")
        return

    data = records_to_frame(score_records)
    latex = render_latex_table(
        data,
        cfg,
        method_names,
        dataset_name,
        table_kind="dataset",
    )
    out_table.write_text(latex)
    out_benchmark_table.write_text(latex)
    render_pareto_plot(
        data,
        cfg,
        method_names,
        dataset_name,
        str(out_pareto),
        n_splits=int(cfg.get("train_test", {}).get("n_splits", 5)),
    )
    logger.info(f"Wrote: {out_table}")
    logger.info(f"Wrote: {out_benchmark_table}")
    logger.info(f"Wrote: {out_pareto}")


run(snakemake)  # noqa: F821
