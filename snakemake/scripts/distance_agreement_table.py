"""
Tables: agreement of patient distances under the exact and the diagonal Bures term.

The four matrices per split are named "<trained with>/<evaluated with>" (see
distance_agreement_dataset.py). Two tables of Spearman correlations:

  table               main text: lower triangle of the symmetric 4 x 4 matrix of all pairs,
                      mean ± SD pooled over datasets and splits (as in the main benchmark
                      table)
  table_per_dataset   appendix: one row per dataset, one column per pair, mean ± SD over
                      splits

The untrained pair (fixed Euclidean ground metric) is left out of both tables, since a
reader could take it for a comparison of the untrained with the trained metric; it stays
in the CSV, for the text.

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.per_split         — per-dataset CSVs, in table row order
  snakemake.output.csv              — summary per dataset and pair (mean, sd), incl. the
                                      untrained pair and the pooled rows (dataset "all")
  snakemake.output.table            — LaTeX matrix table (main text)
  snakemake.output.table_per_dataset — LaTeX per-dataset table (appendix)
  snakemake.params.dataset_names    — dataset order, matching the inputs
  snakemake.params.dataset_labels   — paper-facing dataset names
  snakemake.params.n_splits         — splits per dataset (for the pooled SD)
  snakemake.log[0]
"""

import sys

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _benchmark_display import pooled_mean_std  # noqa: E402
from _utils import setup_logging  # noqa: E402

import pandas as pd  # noqa: E402

MATRICES = ["exact/exact", "exact/diag", "diag/diag", "diag/exact"]
# per-dataset column order: same map first, then across maps
PAIRS = [
    ("exact/exact", "exact/diag"),
    ("diag/diag", "diag/exact"),
    ("exact/exact", "diag/diag"),
    ("exact/exact", "diag/exact"),
    ("exact/diag", "diag/diag"),
    ("exact/diag", "diag/exact"),
]
UNTRAINED = ("none/exact", "none/diag")


def pair_key(a: str, b: str) -> tuple[str, str]:
    """Pairs are unordered; key them in MATRICES order."""
    order = {name: i for i, name in enumerate(MATRICES + list(UNTRAINED))}
    return (a, b) if order[a] < order[b] else (b, a)


def summarise(per_split: dict[str, pd.DataFrame], n_splits: int) -> pd.DataFrame:
    rows = []
    for dataset, frame in per_split.items():
        for (first, second), group in frame.groupby(["first", "second"]):
            values = group["spearman"]
            rows.append({"dataset": dataset, "first": first, "second": second, "mean": values.mean(),
                         "sd": values.std(ddof=0) if len(values) > 1 else float("nan"), "n_splits": len(values)})
    summary = pd.DataFrame(rows)
    pooled = []
    for (first, second), group in summary.groupby(["first", "second"]):
        mean, sd = pooled_mean_std(group["mean"].to_numpy(), group["sd"].to_numpy(), n=n_splits)
        pooled.append({"dataset": "all", "first": first, "second": second, "mean": mean, "sd": sd,
                       "n_splits": int(group["n_splits"].sum())})
    return pd.concat([summary, pd.DataFrame(pooled)], ignore_index=True)


def lookup(summary: pd.DataFrame, dataset: str, a: str, b: str) -> tuple[float, float]:
    first, second = pair_key(a, b)
    row = summary[(summary["dataset"] == dataset) & (summary["first"] == first) & (summary["second"] == second)]
    return float(row["mean"].iloc[0]), float(row["sd"].iloc[0])


def fmt(mean: float, sd: float) -> str:
    return f"{mean:.2f} \\pm {sd:.2f}"


def matrix_table(summary: pd.DataFrame, n_datasets: int, n_splits: int) -> list[str]:
    # Lower triangle only: the matrix is symmetric with a unit diagonal.
    rows, cols = MATRICES[1:], MATRICES[:-1]
    body = []
    for i, row_name in enumerate(rows):
        cells = [
            f"${fmt(*lookup(summary, 'all', row_name, col_name))}$" if j <= i else ""
            for j, col_name in enumerate(cols)
        ]
        body.append(f"{row_name} & " + " & ".join(cells) + " \\\\")
    return [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Spearman correlation between patient distance matrices, each given as the Bures term "
        "used in training / in evaluation, reported as mean$\\pm$SD over "
        f"{n_datasets} datasets with {n_splits} train-test splits.}}",
        "\\label{tab:distance_agreement}",
        "\\footnotesize",
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        "Trained / evaluated & " + " & ".join(cols) + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]


def per_dataset_table(summary: pd.DataFrame, datasets: list[str], labels: dict, n_splits: int) -> list[str]:
    body = [
        f"{labels.get(ds, ds)} & " + " & ".join(f"${fmt(*lookup(summary, ds, a, b))}$" for a, b in PAIRS) + " \\\\"
        for ds in datasets
    ]
    return [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Exact vs.\\ diagonal Bures patient distances per dataset.} Spearman correlation "
        "between patient distance matrices over all patient pairs, each matrix given as the Bures term "
        "the metric was trained with / the term the distances were evaluated with. The first two "
        "columns compare the two terms under one learned map, the others compare the maps of Ours (Exact) "
        f"and Ours (Diag.). Mean $\\pm$ SD over {n_splits} train-test splits.}}",
        "\\label{tab:distance_agreement_per_dataset}",
        "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{l" + "c" * len(PAIRS) + "}",
        "\\toprule",
        "Trained / evaluated & " + " & ".join(a for a, _ in PAIRS) + " \\\\",
        "compared against & " + " & ".join(b for _, b in PAIRS) + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
    ]


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    datasets = list(snakemake.params.dataset_names)
    n_splits = int(snakemake.params.n_splits)
    per_split = {ds: pd.read_csv(path) for ds, path in zip(datasets, snakemake.input.per_split)}
    summary = summarise(per_split, n_splits)
    summary.to_csv(snakemake.output.csv, index=False)
    logger.info("\n" + summary[summary["dataset"] == "all"].round(3).to_string(index=False))

    with open(snakemake.output.table, "w") as f:
        f.write("\n".join(matrix_table(summary, len(datasets), n_splits)) + "\n")
    with open(snakemake.output.table_per_dataset, "w") as f:
        f.write("\n".join(per_dataset_table(summary, datasets, dict(snakemake.params.dataset_labels), n_splits)) + "\n")


run(snakemake)  # noqa: F821
