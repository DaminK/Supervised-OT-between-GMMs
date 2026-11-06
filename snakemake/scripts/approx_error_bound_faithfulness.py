"""
Faithfulness of learned maps under both geometries.

For every map of the tuned replay (rule approx_error_bound_tuned_dataset, training "tuned") and of
the long training (rule approx_error_bound_long_dataset, training "long"), the downstream
evaluation computes patient distances with the exact and the diagonal Bures term and their
Spearman correlation ``rho``. This script summarises ``rho``, the relative approximation error,
and KNN accuracy and ARI (average linkage) with the Bures term used in training (diagonal or
exact, from the reference method's config) per training term (reference method), training,
regularization setup and initialization: mean ± SD over the splits of each dataset, pooled over
datasets as in the main benchmark table.

Table rows per training term: with (setup Both, which for the exact reference equals Frobenius
since it has no MI weight) and without regularization (setup NoReg), from an orthonormal
initialization at the weights tuned for the main benchmark, and from a standard normal
initialization with the weights and max_iter of the long training. The CSV keeps every setup.

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.downstream_csvs   — per-dataset downstream CSVs of the tuned replay and the long training
  snakemake.params.reference_keys   — reference key of each input, in input order
  snakemake.params.trainings        — training ("tuned" or "long") of each input, in input order
  snakemake.params.references       — reference key -> {method_name, display_name, diag_bures_approx}
  snakemake.params.long_training    — approx_error_bound_long_training (for the caption)
  snakemake.params.n_datasets       — number of datasets (for the caption)
  snakemake.output.csv              — summary per reference, training, setup, init and dataset (incl. "all")
  snakemake.output.table            — LaTeX table
  snakemake.log[0]
"""

import sys

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _benchmark_display import pooled_mean_std  # noqa: E402
from _utils import setup_logging  # noqa: E402

import pandas as pd  # noqa: E402

# Table rows per reference: (training, setup, init, regularization label, init label). With and
# without regularization from an orthonormal start at the tuned weights, and from a standard
# normal start with the long training.
TABLE_ROWS = [
    ("tuned", "Both", "orthonormal", "with", "orthonormal"),
    ("tuned", "NoReg", "orthonormal", "without", "orthonormal"),
    ("long", "Both", "random", "with", "standard normal"),
    ("long", "NoReg", "random", "without", "standard normal"),
]
# column -> scale (relative error in %); knn_train and ari_train: scores with the training term
VALUES = {"rho": 1.0, "relative_approx_error": 100.0, "knn_train": 1.0, "ari_train": 1.0}
# Table columns per init: (column, header, digits)
TABLE_COLUMNS = [("rho", "$\\rho$", 2), ("relative_approx_error", "Rel. error (\\%)", 1),
                 ("knn_train", "KNN", 2), ("ari_train", "ARI", 2)]


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over splits per dataset, then pooled over datasets (dataset "all")."""
    keys = ["reference", "training", "variant_name", "init"]
    rows = []
    for (reference, training, setup, init, dataset), group in frame.groupby(keys + ["dataset"]):
        row = {"reference": reference, "training": training, "variant_name": setup, "init": init,
               "dataset": dataset, "n_splits": len(group)}
        for column, scale in VALUES.items():
            values = group[column] * scale
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_sd"] = values.std(ddof=0) if len(values) > 1 else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)

    pooled = []
    for (reference, training, setup, init), group in summary.groupby(keys):
        n_splits = int(group["n_splits"].max())
        row = {"reference": reference, "training": training, "variant_name": setup, "init": init,
               "dataset": "all", "n_splits": int(group["n_splits"].sum())}
        for column in VALUES:
            mean, sd = pooled_mean_std(group[f"{column}_mean"].to_numpy(), group[f"{column}_sd"].to_numpy(), n=n_splits)
            row[f"{column}_mean"], row[f"{column}_sd"] = mean, sd
        pooled.append(row)
    return pd.concat([summary, pd.DataFrame(pooled)], ignore_index=True)


def cell(summary: pd.DataFrame, reference: str, training: str, setup: str, init: str, column: str,
         digits: int) -> str:
    row = summary[(summary["dataset"] == "all") & (summary["reference"] == reference)
                  & (summary["training"] == training) & (summary["variant_name"] == setup)
                  & (summary["init"] == init)]
    if row.empty:
        return "--"
    mean, sd = float(row[f"{column}_mean"].iloc[0]), float(row[f"{column}_sd"].iloc[0])
    return f"${mean:.{digits}f} \\pm {sd:.{digits}f}$"


def latex_table(summary: pd.DataFrame, references: dict, long_training: dict, n_datasets: int) -> list[str]:
    body = []
    for ref_idx, (reference, ref) in enumerate(references.items()):
        if ref_idx:
            body.append("\\midrule")
        for row_idx, (training, setup, init, setup_label, init_label) in enumerate(TABLE_ROWS):
            cells = [cell(summary, reference, training, setup, init, column, digits)
                     for column, _, digits in TABLE_COLUMNS]
            name = ref["display_name"] if row_idx == 0 else ""
            body.append(f"{name} & {setup_label} & {init_label} & " + " & ".join(cells) + " \\\\")
    n_splits = int(summary.loc[summary["dataset"] != "all", "n_splits"].max())
    headers = " & ".join(header for _, header, _ in TABLE_COLUMNS)
    long_weights = dict(long_training["setups"]["Both"])
    exact_names = [ref["display_name"] for ref in references.values() if not ref["diag_bures_approx"]]
    exact_note = f" ({', '.join(exact_names)}: no MI weight)" if exact_names else ""
    return [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Spearman correlation $\\rho$ between exact and diagonal patient distances of the same "
        "learned map, relative approximation error, and KNN accuracy and ARI (average linkage) with the "
        "Bures term used in training. Maps are trained with or without regularization, from an orthonormal "
        "initialization at the weights tuned for the main benchmark, and from a standard normal "
        f"initialization with $\\lambda_F = {float(long_weights['reg']):g}$, "
        f"$\\lambda_{{\\MI}} = {float(long_weights['mi_reg']):g}${exact_note} and "
        f"{int(long_training['max_iter'])} epochs. Reported as "
        f"mean$\\pm$SD over {n_datasets} datasets with {n_splits} train-test splits.}}",
        "\\label{tab:map_faithfulness}",
        "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{lll" + "c" * len(TABLE_COLUMNS) + "}",
        "\\toprule",
        f"Trained as & Regularization & Init. & {headers} \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
    ]


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    references = dict(snakemake.params.references)

    frames = []
    for reference, training, path in zip(snakemake.params.reference_keys, snakemake.params.trainings,
                                         snakemake.input.downstream_csvs):
        frame = pd.read_csv(path)
        frame["reference"], frame["training"] = reference, training
        suffix = "" if references[reference]["diag_bures_approx"] else "_exact"
        frame["knn_train"], frame["ari_train"] = frame[f"knn{suffix}"], frame[f"ari{suffix}"]
        frames.append(frame)
    summary = summarise(pd.concat(frames, ignore_index=True))
    summary.to_csv(snakemake.output.csv, index=False)

    with open(snakemake.output.table, "w") as f:
        f.write("\n".join(latex_table(summary, references, dict(snakemake.params.long_training),
                                      int(snakemake.params.n_datasets))) + "\n")
    logger.info(f"Wrote: {snakemake.output.csv}, {snakemake.output.table}")


run(snakemake)  # noqa: F821
