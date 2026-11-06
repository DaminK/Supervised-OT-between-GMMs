"""
Raw-cell table (appendix): the point-cloud baselines on N points sampled from each patient's
fitted GMM (as in the main benchmark) next to the same methods on N raw cells per patient.

One row per point-cloud method; KNN accuracy, ARI and silhouette width for both point
sources. Mean ± SD pooled over cohorts and splits, as in the main benchmark table. No bold:
the table checks the point source, it does not rank methods. Writes the tabular only; the
table environment, caption and label are in the paper.

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.scores          — scores.csv of all sampled and raw-cell methods
  snakemake.output.csv            — per-row, per-source, per-dataset scores
  snakemake.output.table          — LaTeX tabular, pooled over cohorts
  snakemake.params.cells          — [{row, source, method, dataset, scores}]
  snakemake.params.rows           — row definitions (label, sampled, raw)
  snakemake.params.n_splits       — splits per cohort (for the pooled SD)
  snakemake.log[0]
"""

import sys

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _benchmark_display import load_score_summary, pooled_mean_std  # noqa: E402
from _utils import setup_logging  # noqa: E402

import pandas as pd  # noqa: E402

SOURCES = {"sampled": "GMM-sampled points", "raw": "Raw cells"}
TABLE_METRICS = {"knn": "KNN", "ari": "ARI", "silhouette": "Silh."}


def load_cells(cells: list[dict]) -> pd.DataFrame:
    records = []
    for cell in cells:
        summary = load_score_summary(cell["scores"])
        record = dict(cell)
        for metric in TABLE_METRICS:
            record[metric] = summary[(metric, "mean")]
            record[f"{metric}_sd"] = summary[(metric, "SD")]
        records.append(record)
    return pd.DataFrame(records)


def pooled(per_dataset: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """Mean and pooled SD over cohorts and splits per (row, source, metric)."""
    rows = []
    for (row, source), frame in per_dataset.groupby(["row", "source"]):
        entry = {"row": row, "source": source}
        for metric in TABLE_METRICS:
            mean, sd = pooled_mean_std(frame[metric].to_numpy(), frame[f"{metric}_sd"].to_numpy(), n=n_splits)
            entry[metric], entry[f"{metric}_sd"] = mean, sd
        rows.append(entry)
    return pd.DataFrame(rows).set_index(["row", "source"])


def write_tabular(path: str, rows: list[dict], stats: pd.DataFrame) -> None:
    n = len(TABLE_METRICS)
    headers = " & ".join(TABLE_METRICS.values())
    body = []
    for i, row in enumerate(rows):
        cells = [
            f"${stats.loc[(i, source), metric]:.2f} \\pm {stats.loc[(i, source), f'{metric}_sd']:.2f}$"
            for source in SOURCES
            for metric in TABLE_METRICS
        ]
        body.append(f"{row['label']} & " + " & ".join(cells) + " \\\\")
    groups = " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{label}}}" for label in SOURCES.values())
    lines = [
        f"\\begin{{tabular}}{{l{'c' * len(SOURCES) * n}}}",
        "\\toprule",
        f" & {groups} \\\\",
        f"\\cmidrule(lr){{2-{n + 1}}} \\cmidrule(lr){{{n + 2}-{2 * n + 1}}}",
        f"Method & {headers} & {headers} \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    rows = list(snakemake.params.rows)
    per_dataset = load_cells(list(snakemake.params.cells))
    per_dataset.to_csv(snakemake.output.csv, index=False)
    summary = pooled(per_dataset, int(snakemake.params.n_splits))
    logger.info("\n" + summary.round(3).to_string())
    write_tabular(snakemake.output.table, rows, summary)


run(snakemake)  # noqa: F821
