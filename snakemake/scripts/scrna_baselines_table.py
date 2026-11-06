"""
scRNA baselines table (main text): each scRNA-specific baseline next to our learned metric
on the same representation.

KNN accuracy, ARI, silhouette width and OT time per patient pair of the baseline (fixed
ground metric) and of our method on that representation. Mean ± SD pooled over cohorts and
splits, as in the main benchmark table; times as means without SD. Bold: the better of
baseline and ours at the displayed precision (neither on a tie).

Main table: narrow, to sit next to the main benchmark table. Rows under the group headers
of the main benchmark table; per representation a pair of rows, baseline (with its fixed
ground metric below its name) and ours, labelled by the representation and separated by
rules; SD below the mean. The appendix table shows the same rows per cohort (mean ± SD
over that cohort's splits), one wide line per representation. Writes the tabulars only;
table environments, captions and labels are in the paper.

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.scores          — scores.csv of the rows of this run
  snakemake.output.csv            — per-row, per-role, per-dataset scores
  snakemake.output.table          — LaTeX tabular, pooled over cohorts
  snakemake.output.table_per_dataset — LaTeX tabular, one block per cohort
  snakemake.params.cells          — [{row, role, run, method, dataset, scores}], incl. the
                                    GM-VAE rows, which are read from params only
  snakemake.params.rows           — row definitions (baseline, ours, representation, group,
                                    sublabel, ground_metric, run)
  snakemake.params.gmvae_overlay  — overlay config of the GM-VAE run (for the error message)
  snakemake.params.n_splits       — splits per cohort (for the pooled SD)
  snakemake.params.datasets       — cohort order of the per-cohort table
  snakemake.params.dataset_labels — cohort display names
  snakemake.log[0]
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _benchmark_display import load_score_summary, pooled_mean_std  # noqa: E402
from _utils import setup_logging  # noqa: E402

import pandas as pd  # noqa: E402

METRICS = ["knn", "ari", "ari_complete", "silhouette", "inf_time(ms)"]
# column header and direction (+1: higher is better); times as in the main benchmark table
TABLE_METRICS = {
    "knn": ("KNN", 1),
    "ari": ("ARI", 1),
    "silhouette": ("Silh.", 1),
    "inf_time(ms)": ("OT $t$", -1),
}
MAIN_HEADERS = {
    "knn": "\\textbf{KNN} $\\uparrow$",
    "ari": "\\textbf{ARI} $\\uparrow$",
    "silhouette": "\\textbf{Silh.} $\\uparrow$",
    "inf_time(ms)": "\\textbf{OT} $t$ $\\downarrow$",
}
# Cell macros of the main table. Each row is two lines high (mean over SD); the representation
# label is lowered to the middle of its row pair (7.6pt: half a row at \footnotesize).
MAIN_PREAMBLE = [
    "% Cell helpers: mean over SD; method over ground metric; sub-label centred on its row pair.",
    "\\providecommand{\\ScMs}[2]{\\begin{tabular}[c]{@{}c@{}}$#1$\\\\[-3pt]{\\scriptsize$\\pm#2$}\\end{tabular}}",
    "\\providecommand{\\ScMsB}[2]{\\begin{tabular}[c]{@{}c@{}}$\\mathbf{#1}$\\\\[-3pt]"
    "{\\scriptsize$\\mathbf{\\pm#2}$}\\end{tabular}}",
    "\\providecommand{\\ScMeth}[2]{\\begin{tabular}[c]{@{}l@{}}#1\\\\[-3pt]{\\scriptsize\\itshape #2}\\end{tabular}}",
    "\\providecommand{\\ScSub}[1]{\\hspace*{2\\tabcolsep}\\smash{\\raisebox{-7.6pt}{#1}}}",
    "\\setlength{\\tabcolsep}{3.5pt}",
    "\\renewcommand{\\arraystretch}{1.0}",
]


def load_cells(cells: list[dict], gmvae_overlay: str) -> pd.DataFrame:
    missing = [cell["scores"] for cell in cells if not Path(cell["scores"]).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} result(s) missing, e.g. {missing[0]}. The GM-VAE rows need the run of "
            f"config/{gmvae_overlay} (target benchmark_only) before this table."
        )
    records = []
    for cell in cells:
        summary = load_score_summary(cell["scores"], extra_metrics=["ari_complete"])
        record = dict(cell)
        for metric in METRICS:
            record[metric] = summary[(metric, "mean")]
            record[f"{metric}_sd"] = summary[(metric, "SD")]
        records.append(record)
    return pd.DataFrame(records)


def pooled(per_dataset: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """Mean and pooled SD over cohorts and splits per (row, role, metric)."""
    rows = []
    for (row, role), frame in per_dataset.groupby(["row", "role"]):
        entry = {"row": row, "role": role}
        for metric in METRICS:
            mean, sd = pooled_mean_std(frame[metric].to_numpy(), frame[f"{metric}_sd"].to_numpy(), n=n_splits)
            entry[metric], entry[f"{metric}_sd"] = mean, sd
        rows.append(entry)
    return pd.DataFrame(rows).set_index(["row", "role"])


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    rows = list(snakemake.params.rows)

    per_dataset = load_cells(list(snakemake.params.cells), snakemake.params.gmvae_overlay)
    per_dataset.to_csv(snakemake.output.csv, index=False)
    summary = pooled(per_dataset, int(snakemake.params.n_splits))
    logger.info("\n" + summary.round(3).to_string())
    write_main_tabular(snakemake.output.table, main_rows(rows, summary))

    # Appendix: the same rows per cohort, mean ± SD over that cohort's splits
    per_cohort = []
    for i, dataset in enumerate(snakemake.params.datasets):
        stats = per_dataset[per_dataset["dataset"] == dataset].set_index(["row", "role"])
        label = snakemake.params.dataset_labels.get(dataset, dataset)
        n_columns = 1 + 2 * len(TABLE_METRICS)
        per_cohort += (["\\midrule"] if i else []) + [f"\\multicolumn{{{n_columns}}}{{l}}{{\\textit{{{label}}}}} \\\\"]
        per_cohort += body_rows(rows, stats)
    write_tabular(snakemake.output.table_per_dataset, per_cohort)


def compare(stats: pd.DataFrame, row_index: int, role: str, metric: str) -> tuple[float, float, bool]:
    """Mean, SD and whether the role beats the other role of its row at the displayed
    precision (so a tie is bold for neither); stats indexed by (row, role)."""
    mean, sd = stats.loc[(row_index, role), [metric, f"{metric}_sd"]]
    other = stats.loc[(row_index, "ours" if role == "baseline" else "baseline"), metric]
    digits = 1 if metric.endswith("(ms)") else 2
    better = TABLE_METRICS[metric][1] * (round(mean, digits) - round(other, digits)) > 0
    return mean, sd, better


def time_text(mean: float) -> str:
    """Mean time per patient pair, without SD, as in the main benchmark table."""
    return "{<}0.1\\,ms" if 0 < mean < 0.05 else f"{mean:.1f}\\,ms"


def main_rows(rows: list[dict], stats: pd.DataFrame) -> list[str]:
    """Group headers, then per row a baseline line and an ours line, pairs separated by rules."""

    def cell(row_index: int, role: str, metric: str) -> str:
        mean, sd, better = compare(stats, row_index, role, metric)
        if metric.endswith("(ms)"):
            text = time_text(mean)
            return f"$\\mathbf{{{text}}}$" if better else f"${text}$"
        macro = "\\ScMsB" if better else "\\ScMs"
        return f"{macro}{{{mean:.2f}}}{{{sd:.2f}}}"

    def cells(row_index: int, role: str) -> str:
        return " & ".join(cell(row_index, role, metric) for metric in TABLE_METRICS)

    n_columns = 2 + len(TABLE_METRICS)
    lines, group = [], None
    for i, row in enumerate(rows):
        if row["group"] != group:
            header = f"\\multicolumn{{{n_columns}}}{{l}}{{\\textsc{{{row['group']}}}}} \\\\"
            if group is None:
                # the first header sits below \midrule, so it gets space below it as well
                lines += [header, "\\addlinespace[2pt]"]
            else:
                lines += [f"\\cmidrule(lr){{1-{n_columns}}}", "\\addlinespace[4pt]", header]
            group = row["group"]
        else:
            lines.append(f"\\cmidrule(lr){{1-{n_columns}}}")
        lines.append(
            f"\\ScSub{{{row['sublabel']}}} & \\ScMeth{{{row['baseline']['label']}}}{{{row['ground_metric']}}} & "
            + cells(i, "baseline") + " \\\\"
        )
        lines.append(f" & {row['ours']['label']} & " + cells(i, "ours") + " \\\\")
    return lines


def write_main_tabular(path: str, body: list[str]) -> None:
    lines = [
        *MAIN_PREAMBLE,
        f"\\begin{{tabular}}{{ll{'c' * len(TABLE_METRICS)}}}",
        "\\toprule",
        "\\multicolumn{2}{l}{\\textbf{Method}} & " + " & ".join(MAIN_HEADERS[m] for m in TABLE_METRICS) + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def body_rows(rows: list[dict], stats: pd.DataFrame) -> list[str]:
    """Appendix: one line per row, baseline and ours side by side; stats indexed by (row, role)."""

    def cell(row_index: int, role: str, metric: str) -> str:
        mean, sd, better = compare(stats, row_index, role, metric)
        text = time_text(mean) if metric.endswith("(ms)") else f"{mean:.2f} \\pm {sd:.2f}"
        return f"$\\mathbf{{{text}}}$" if better else f"${text}$"

    return [
        f"{row['representation']} ({row['baseline']['label']}) & "
        + " & ".join(cell(i, "baseline", metric) for metric in TABLE_METRICS)
        + " & "
        + " & ".join(cell(i, "ours", metric) for metric in TABLE_METRICS)
        + " \\\\"
        for i, row in enumerate(rows)
    ]


def write_tabular(path: str, body: list[str]) -> None:
    n = len(TABLE_METRICS)
    headers = " & ".join(header for header, _ in TABLE_METRICS.values())
    lines = [
        f"\\begin{{tabular}}{{l{'c' * 2 * n}}}",
        "\\toprule",
        f" & \\multicolumn{{{n}}}{{c}}{{Baseline}} & \\multicolumn{{{n}}}{{c}}{{Ours}} \\\\",
        f"\\cmidrule(lr){{2-{n + 1}}} \\cmidrule(lr){{{n + 2}-{2 * n + 1}}}",
        f"Representation & {headers} & {headers} \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


run(snakemake)  # noqa: F821
