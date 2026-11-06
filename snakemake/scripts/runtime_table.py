"""
Runtime table (appendix): wall-clock time of the main benchmark's pipeline jobs per cohort.

Rows: PCA (preprocess job), GMM fitting (GMM job), training and evaluation per method
(benchmark job, all splits), and end to end per method: PCA + training and evaluation, plus
GMM fitting for GMM methods. Hyperparameter tuning is not included. Times in h:mm, rounded
to the nearest minute after summing the seconds.

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.output.csv              — seconds per row and cohort
  snakemake.output.table            — LaTeX table environment
  snakemake.params.timings          — {dataset: {"preprocess"|"gmm"|method: timing TSV}}
  snakemake.params.datasets         — cohort (column) order
  snakemake.params.dataset_labels   — cohort column headers
  snakemake.params.preprocess_label — label of the PCA row
  snakemake.params.gmm_label        — label of the GMM fitting row
  snakemake.params.rows             — [{method, label, uses_gmm}]
  snakemake.params.caption          — table caption
  snakemake.log[0]
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _utils import setup_logging  # noqa: E402

import pandas as pd  # noqa: E402


def wall_seconds(path: str) -> float:
    """Wall-clock seconds (column s) of a Snakemake benchmark: file, last line."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Timing file missing: {path}. Run the main benchmark first.")
    return float(pd.read_csv(path, sep="\t")["s"].iloc[-1])


def hmm(seconds: float) -> str:
    minutes = int(seconds / 60 + 0.5)
    return f"{minutes // 60}:{minutes % 60:02d}"


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    p = snakemake.params
    datasets = list(p.datasets)
    seconds = {ds: {job: wall_seconds(path) for job, path in p.timings[ds].items()} for ds in datasets}

    # (section, label, seconds per dataset)
    lines = [
        ("Preprocessing", p.preprocess_label, [seconds[ds]["preprocess"] for ds in datasets]),
        ("Preprocessing", p.gmm_label, [seconds[ds]["gmm"] for ds in datasets]),
    ]
    for row in p.rows:
        lines.append(("Training and evaluation", row["label"], [seconds[ds][row["method"]] for ds in datasets]))
    for row in p.rows:
        jobs = ["preprocess"] + (["gmm"] if row["uses_gmm"] else []) + [row["method"]]
        lines.append(("End to end", row["label"], [sum(seconds[ds][job] for job in jobs) for ds in datasets]))

    frame = pd.DataFrame(
        [[section, label, *values] for section, label, values in lines],
        columns=["section", "row", *datasets],
    )
    frame.round(2).to_csv(snakemake.output.csv, index=False)
    logger.info("\n" + frame.round(1).to_string())

    n_columns = 1 + len(datasets)
    body, previous = [], "Preprocessing"
    for section, label, values in lines:
        if section != previous:
            body += ["\\midrule", f"\\multicolumn{{{n_columns}}}{{l}}{{\\textit{{{section}}}}} \\\\"]
            previous = section
        body.append(f"{label} & " + " & ".join(hmm(v) for v in values) + " \\\\")

    header = " & ".join(p.dataset_labels.get(ds, ds) for ds in datasets)
    table = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{p.caption}}}",
        "\\label{tab:runtime}",
        "\\footnotesize",
        f"\\begin{{tabular}}{{l{'c' * len(datasets)}}}",
        "\\toprule",
        f" & {header} \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    with open(snakemake.output.table, "w") as f:
        f.write("\n".join(table) + "\n")


run(snakemake)  # noqa: F821
