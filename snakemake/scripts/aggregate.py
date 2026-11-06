"""
Cross-dataset aggregation: main LaTeX benchmark table + Pareto plot.

Called via Snakemake ``script:`` directive.  The ``snakemake`` object provides:
  snakemake.input.scores          — list of scores.csv (datasets in group × methods)
  snakemake.output.table          — LaTeX benchmark table for the group
  snakemake.output.pareto         — Pareto plot PDF for the group
  snakemake.params.config         — resolved experiment config
  snakemake.params.method_dataset_pairs  — list of (method_name, dataset) tuples;
                                           also fixes the table's row order
  snakemake.log[0]
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _benchmark_display import (  # noqa: E402
    aggregate_scores,
    load_score_summary,
    make_score_record,
    records_to_frame,
    render_pareto_plot,
    render_latex_table,
)
from _utils import setup_logging  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    cfg = snakemake.params.config
    pairs = snakemake.params.method_dataset_pairs   # [(method, dataset), ...]
    # Row order comes from the pairs (built from ALL_METHODS, i.e. YAML order and
    # honouring run_methods), not from cfg["methods"]: rule configs are key-sorted
    # by project_config, so that mapping is alphabetical.
    method_names = list(dict.fromkeys(method_name for method_name, _ in pairs))
    # Methods with `main_table: false` (e.g. Table 2 rows) are skipped before loading so
    # they do not affect the ranking of the shown methods.
    method_names = [m for m in method_names if cfg["methods"][m].get("main_table", True)]
    n_splits = int(cfg.get("train_test", {}).get("n_splits", 5))

    out_table = Path(snakemake.output.table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_pareto = Path(snakemake.output.pareto)

    # ── Load all scores ────────────────────────────────────────────────────

    records = []
    for (method_name, dataset), score_path in zip(pairs, snakemake.input.scores):
        if method_name not in method_names:
            continue
        try:
            summary = load_score_summary(score_path)
            records.append(
                make_score_record(
                    dataset=dataset,
                    method_name=method_name,
                    method_cfg=cfg["methods"][method_name],
                    summary=summary,
                )
            )
        except Exception as e:
            logger.warning(f"Could not load {score_path}: {e}")

    if not records:
        logger.warning("No benchmark scores found — writing empty outputs")
        out_table.write_text("% No data\n")
        out_pareto.write_text("")
        return

    raw_df = records_to_frame(records)
    averaged_df = aggregate_scores(raw_df, n_splits=n_splits)

    # ── Main benchmark table (notebook layout) ─────────────────────────────

    latex = render_latex_table(
        averaged_df,
        cfg,
        method_names,
        "Average",
        table_kind="main",
    )
    out_table.write_text(latex)
    logger.info(f"Wrote: {out_table}")

    # ── Pareto plot (accuracy vs OT computation time) ─────────────────────
    n_datasets = len({dataset for _, dataset in pairs})
    render_pareto_plot(
        averaged_df,
        cfg,
        method_names,
        "Average",
        str(out_pareto),
        n_splits=n_splits,
        stats_label=rf"Mean Acc. $\pm$ SD over" + "\n" + f"{n_datasets} Datasets, {n_splits} Splits",
    )
    logger.info(f"Wrote: {out_pareto}")


run(snakemake)  # noqa: F821
