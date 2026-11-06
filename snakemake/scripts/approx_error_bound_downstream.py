"""
Figure 5 (b): downstream scores of the Figure 5 maps, with diagonal and with exact distances.

Each map trained with the diagonal Bures term is evaluated twice: with its own diagonal patient
distances and with exact ones (approx_error_bound_evaluate_dataset). Per dataset group
(scRNA-seq, synthetic), the scores are averaged over splits and then over the datasets of the
group (errors: geometric mean). The diagonal score is filled, the exact score hollow, joined by
a line, so the line length is the mismatch between the diagonal metric and its exact evaluation;
colour = regularization, marker = initialization (several initializations: also shade).

Figures, each for the absolute (approx_error) and the relative (relative_approx_error) error and
for each initialization alone and all together:
  main         KNN, ARI (average linkage), silhouette over the approximation error (high to low);
               one row per dataset group (main text)
  all_scores   x = regularization: error, KNN, ARI (average, complete linkage), silhouette
               (appendix)
  per_dataset  as all_scores, one row per dataset (appendix)
  gap_scatter  diagonal - exact silhouette and complete-linkage ARI per split over the error,
               binned trend and Spearman correlation per group (appendix)

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.downstream_csvs  — per-dataset downstream.csv (one row per init, variant, split)
  snakemake.output.main / all_scores / per_dataset / gap_scatter — figure lists over error
                                     columns, then init sets (_error_plots.init_sets)
  snakemake.output.csv             — means and SDs over splits per dataset, plus the group means
  snakemake.output.table           — LaTeX table (appendix), group means
  snakemake.params.dataset_names   — dataset order, matching the inputs
  snakemake.params.variants        — variant name -> display name, in plot order
  snakemake.params.inits           — initializations, in plot order
  snakemake.params.error_columns   — error columns of the figures
  snakemake.params.caption        — what the maps are (fixed weights or tuned), for the table caption
  snakemake.params.label           — LaTeX label of the table
  snakemake.log[0]
"""

import itertools
import sys

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _error_plots import (  # noqa: E402
    DISPLAY_RENAME,
    DS_DISPLAY_NAMES,
    INIT_LABELS,
    VARIANT_STYLES,
    init_color,
    init_sets,
    resolve_init_set,
)
from _utils import setup_logging  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

SCORES = ["knn", "ari", "ari_complete", "silhouette"]
SCORE_LABELS = {
    "knn": "KNN accuracy",
    "ari": "ARI (average linkage)",
    "ari_complete": "ARI (complete linkage)",
    "silhouette": "Silhouette",
}
MAIN_SCORES = ["knn", "ari", "silhouette"]
AGREEMENT = ["rho", "rho_top10", "rho_within"]
ERRORS = ["approx_error", "relative_approx_error"]  # group means: geometric
VALUE_COLUMNS = ERRORS + SCORES + [f"{score}_exact" for score in SCORES] + AGREEMENT
INIT_MARKERS = {"orthonormal": "o", "xavier": "^", "random": "s"}
# error axis: label, factor (relative error in percent), labelled ticks (None: default log ticks)
ERROR_AXES = {
    "approx_error": ("Abs. approx. error", 1.0, None),
    "relative_approx_error": ("Rel. approx. error (%)", 100.0, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10]),
}
GROUPS = ["scRNA-seq", "Synthetic"]
GAP_SCORES = ["silhouette", "ari_complete"]
FONT_SIZE = 17
AXIS_NAMES = {"Frobenius": "Frob."}  # short x tick labels
MARKER_SIZE = 120


def group_of(dataset: str) -> str:
    return "Synthetic" if dataset.startswith("synth") else "scRNA-seq"


def summarise(per_split: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over splits per dataset, init and variant; the group rows (dataset = group
    name) average the dataset means, each dataset once, the errors geometrically."""
    keys = ["init", "variant_name"]
    grouped = per_split.groupby(["dataset"] + keys, sort=False)[VALUE_COLUMNS]
    means = grouped.mean().reset_index()
    sds = grouped.std(ddof=0).add_suffix("_sd").reset_index(drop=True)
    per_dataset = pd.concat([means, sds], axis=1)
    means = means.assign(group=means["dataset"].map(group_of))
    means[ERRORS] = np.log(means[ERRORS])
    groups = means.groupby(["group"] + keys, sort=False)[VALUE_COLUMNS].mean().reset_index()
    groups[ERRORS] = np.exp(groups[ERRORS])
    groups["n_datasets"] = means.groupby(["group"] + keys, sort=False).size().to_numpy()
    groups = groups.rename(columns={"group": "dataset"})
    return pd.concat([per_dataset, groups], ignore_index=True)


def short_name(display_name: str) -> str:
    return DISPLAY_RENAME.get(display_name, display_name)


def style_color(display_name: str, init: str | None = None, inits: list[str] = ()) -> str:
    """Regularization colour; with several *inits*, in the shade of *init*."""
    return init_color(VARIANT_STYLES[short_name(display_name)]["color"], init, shaded=len(inits) > 1)


def legend(fig, variants: dict, inits: list[str], pairs: bool = True, colors: bool = True, y: float = -0.02):
    """Legend below the figure; the regularization colours only where the x-axis does not name them."""
    # regularization as line segments, so that no colour swatch looks like an init marker
    handles = [Line2D([], [], lw=5, color=style_color(d), label=short_name(d)) for d in variants.values()] if colors else []
    handles += [Line2D([], [], marker=INIT_MARKERS[i], ls="", ms=10, color=init_color("#555555", i, shaded=len(inits) > 1),
                       label=INIT_LABELS[i]) for i in inits]
    if pairs:
        handles += [Line2D([], [], marker="o", ls="", ms=10, color="#555555", label="Diagonal distances"),
                    Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec="#555555", mew=1.8, label="Exact distances")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=FONT_SIZE - 1,
               bbox_to_anchor=(0.5, y), handletextpad=0.2, columnspacing=1.0)


def category_panel(ax, rows: pd.DataFrame, column: str, variants: dict, inits: list[str], log: bool = False):
    """x = regularization, one marker per init; for a score, the diagonal (filled) and exact
    (hollow) value joined by a line; for an error, the value alone."""
    width = 0.24
    for i, (variant_name, display_name) in enumerate(variants.items()):
        for j, init in enumerate(inits):
            row = rows[(rows["variant_name"] == variant_name) & (rows["init"] == init)]
            if row.empty:
                continue
            row = row.iloc[0]
            color = style_color(display_name, init, inits)
            x = i + (j - (len(inits) - 1) / 2) * width
            marker = INIT_MARKERS[init]
            if column in ERROR_AXES:
                ax.scatter(x, ERROR_AXES[column][1] * row[column], marker=marker, s=MARKER_SIZE, color=color,
                           edgecolor="white", lw=1.2, zorder=3)
                continue
            ax.plot([x, x], [row[column], row[f"{column}_exact"]], color=color, lw=2.2, zorder=1)
            ax.scatter(x, row[column], marker=marker, s=MARKER_SIZE, color=color, edgecolor="white", lw=1.2, zorder=3)
            ax.scatter(x, row[f"{column}_exact"], marker=marker, s=MARKER_SIZE, facecolor="white", edgecolor=color, lw=1.8, zorder=2)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(variants)), [AXIS_NAMES.get(short_name(d), short_name(d)) for d in variants.values()],
                  fontsize=FONT_SIZE - 1)
    ax.set_xlim(-0.6, len(variants) - 0.4)
    ax.tick_params(axis="y", labelsize=FONT_SIZE - 2)
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)


def category_figure(summary: pd.DataFrame, row_names: list[str], columns: list[str], variants: dict, inits: list[str]):
    labels = {**{error: axis[0] for error, axis in ERROR_AXES.items()}, **SCORE_LABELS}
    fig, axes = plt.subplots(len(row_names), len(columns), figsize=(3.6 * len(columns), 3.0 * len(row_names)),
                             squeeze=False, sharex=True)
    for r, name in enumerate(row_names):
        rows = summary[summary["dataset"] == name]
        for ax, column in zip(axes[r], columns):
            category_panel(ax, rows, column, variants, inits, log=column in ERROR_AXES)
            if r == 0:
                ax.set_title(labels[column], fontsize=FONT_SIZE)
        n = rows["n_datasets"].max() if "n_datasets" in rows and rows["n_datasets"].notna().any() else None
        label = DS_DISPLAY_NAMES.get(name, name) + (f" ({int(n)})" if n else "")
        axes[r, 0].set_ylabel(label, fontsize=FONT_SIZE)
    fig.tight_layout()
    legend(fig, variants, inits, colors=False, y=0.0)
    return fig


def log_x_high_to_low(ax, error: str):
    ax.set_xscale("log")
    ticks = ERROR_AXES[error][2]
    ax.xaxis.set_major_locator(FixedLocator(ticks) if ticks else LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.invert_xaxis()


def error_figure(summary: pd.DataFrame, row_names: list[str], columns: list[str], variants: dict, inits: list[str],
                 error: str):
    """x = approximation error (high to low), one point pair per regularization (colour) and
    initialization (marker): the diagonal score filled, the exact score hollow, joined."""
    label, factor, _ = ERROR_AXES[error]
    fig, axes = plt.subplots(len(row_names), len(columns), figsize=(4.0 * len(columns), 3.2 * len(row_names)),
                             squeeze=False)
    for r, name in enumerate(row_names):
        rows = summary[summary["dataset"] == name]
        for ax, column in zip(axes[r], columns):
            for variant_name, display_name in variants.items():
                for init in inits:
                    row = rows[(rows["variant_name"] == variant_name) & (rows["init"] == init)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    color = style_color(display_name, init, inits)
                    x, marker = factor * row[error], INIT_MARKERS[init]
                    ax.plot([x, x], [row[column], row[f"{column}_exact"]], color=color, lw=2.2, zorder=1)
                    ax.scatter(x, row[column], marker=marker, s=MARKER_SIZE, color=color, edgecolor="white", lw=1.2, zorder=3)
                    ax.scatter(x, row[f"{column}_exact"], marker=marker, s=MARKER_SIZE, facecolor="white",
                               edgecolor=color, lw=1.8, zorder=2)
            log_x_high_to_low(ax, error)
            ax.tick_params(labelsize=FONT_SIZE - 2)
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(SCORE_LABELS[column], fontsize=FONT_SIZE)
            if r == len(row_names) - 1:
                ax.set_xlabel(label, fontsize=FONT_SIZE - 1)
        n = rows["n_datasets"].max() if "n_datasets" in rows and rows["n_datasets"].notna().any() else None
        axes[r, 0].set_ylabel(DS_DISPLAY_NAMES.get(name, name) + (f" ({int(n)})" if n else ""), fontsize=FONT_SIZE)
    fig.tight_layout()
    legend(fig, variants, inits, y=0.0)
    return fig


def gap_scatter(per_split: pd.DataFrame, variants: dict, inits: list[str], error: str):
    """Diagonal - exact score per split over the error; silhouette with the binned median of the
    gap, complete-linkage ARI (gaps mostly 0 with large outliers) with the binned mean absolute
    gap."""
    label, factor, _ = ERROR_AXES[error]
    data = per_split[per_split["init"].isin(inits)]
    data = data.assign(group=data["dataset"].map(group_of), error=factor * data[error])
    data = data[np.isfinite(data["error"])]
    groups = [g for g in GROUPS if g in set(data["group"])]
    fig, axes = plt.subplots(len(groups), len(GAP_SCORES), figsize=(5.4 * len(GAP_SCORES), 3.6 * len(groups)), squeeze=False)
    for r, group in enumerate(groups):
        rows = data[data["group"] == group]
        for ax, score in zip(axes[r], GAP_SCORES):
            gap = rows[score] - rows[f"{score}_exact"]
            for (init, variant_name), g in rows.groupby(["init", "variant_name"]):
                ax.scatter(g["error"], g[score] - g[f"{score}_exact"], marker=INIT_MARKERS[init], s=26,
                           color=style_color(variants[variant_name], init, inits), edgecolor="white", lw=0.6,
                           alpha=0.85, zorder=3)
            edges = np.unique(np.quantile(np.log(rows["error"]), np.linspace(0, 1, 9)))
            bins = np.clip(np.digitize(np.log(rows["error"]), edges[1:-1]), 0, len(edges) - 2)
            binned = pd.DataFrame({"x": rows["error"].to_numpy(), "y": gap.to_numpy(), "a": gap.abs().to_numpy(), "b": bins}).groupby("b")
            if score == "silhouette":
                ax.plot(binned["x"].median(), binned["y"].median(), color="#222222", lw=2, zorder=4)
                rho, what = spearmanr(rows["error"], gap).statistic, "gap"
            else:
                ax.plot(binned["x"].median(), binned["a"].mean(), color="#222222", lw=2, ls="--", zorder=4)
                rho, what = spearmanr(rows["error"], gap.abs()).statistic, "|gap|"
            ax.axhline(0, color="#999999", lw=1, zorder=0)
            log_x_high_to_low(ax, error)
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{group}: {SCORE_LABELS[score]}, diagonal $-$ exact\nSpearman(error, {what}) = {rho:.2f}",
                         fontsize=FONT_SIZE - 1)
            if r == len(groups) - 1:
                ax.set_xlabel(f"{label}, high $\\to$ low", fontsize=FONT_SIZE - 1)
    fig.tight_layout()
    legend(fig, variants, inits, pairs=False, y=0.0)
    return fig


def table(summary: pd.DataFrame, variants: dict, inits: list[str], n_splits: int, caption: str, label: str) -> list[str]:
    body = []
    for group in GROUPS:
        rows = summary[summary["dataset"] == group]
        if rows.empty:
            continue
        n_datasets = int(rows["n_datasets"].max())
        body.append(f"\\multicolumn{{{3 + 2 * len(SCORES) + len(AGREEMENT)}}}{{l}}{{\\textit{{{group} ({n_datasets} datasets)}}}} \\\\")
        for init in inits:
            for i, (variant_name, display_name) in enumerate(variants.items()):
                row = rows[(rows["init"] == init) & (rows["variant_name"] == variant_name)].iloc[0]
                init_cell = f"\\multirow{{{len(variants)}}}{{*}}{{{INIT_LABELS[init]}}}" if i == 0 else ""
                cells = [f"{100 * row['relative_approx_error']:.2f}"]
                for score in SCORES:
                    cells += [f"{row[score]:.3f}", f"{row[f'{score}_exact']:.3f}"]
                cells += [f"{row[name]:.3f}" for name in AGREEMENT]
                body.append(f"{init_cell} & {short_name(display_name)} & " + " & ".join(cells) + " \\\\")
            body.append("\\cmidrule(lr){1-" + str(3 + 2 * len(SCORES) + len(AGREEMENT)) + "}")
        body[-1] = "\\midrule"
    body[-1] = "\\bottomrule"
    score_header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{SCORE_LABELS[score]}}}" for score in SCORES)
    rules = " ".join(f"\\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(SCORES)))
    return [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{\\textbf{{Downstream scores with diagonal and exact distances.}} {caption} Each map is "
        "evaluated with its own diagonal (Diag.) and with exact (Exact) patient distances; relative approximation "
        "error in percent (geometric mean); $\\rho$: Spearman correlation of the two distance matrices over all "
        "patient pairs, over each patient's 10 nearest neighbours, and over pairs within a class. Mean over the "
        f"datasets of a group with {n_splits} train-test splits each.}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{ll" + "c" * (1 + 2 * len(SCORES) + len(AGREEMENT)) + "}",
        "\\toprule",
        f"Init. & Regularization & Rel. error & {score_header} & $\\rho$ & $\\rho_{{\\text{{top-10}}}}$ "
        "& $\\rho_{\\text{within}}$ \\\\",
        rules,
        " & & (\\%) & " + " & ".join(["Diag. & Exact"] * len(SCORES)) + " & & & \\\\",
        "\\midrule",
        *body,
        "\\end{tabular}}",
        "\\end{table}",
    ]


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    datasets = list(snakemake.params.dataset_names)
    variants = dict(snakemake.params.variants)
    inits = list(snakemake.params.inits)

    per_split = pd.concat([pd.read_csv(path) for path in snakemake.input.downstream_csvs], ignore_index=True)
    summary = summarise(per_split)
    summary["display_name"] = summary["dataset"].map(lambda ds: DS_DISPLAY_NAMES.get(ds, ds))
    summary.to_csv(snakemake.output.csv, index=False)
    logger.info("\n" + summary[summary["dataset"].isin(GROUPS)].round(3).to_string(index=False))

    groups = [g for g in GROUPS if g in set(summary["dataset"])]
    present = [ds for ds in datasets if ds in set(per_split["dataset"])]
    combos = list(itertools.product(snakemake.params.error_columns, init_sets(inits)))
    outputs = [snakemake.output.main, snakemake.output.all_scores, snakemake.output.per_dataset, snakemake.output.gap_scatter]
    assert all(len(paths) == len(combos) for paths in outputs)
    for (error, init_set), main, all_scores, per_dataset, gaps in zip(combos, *outputs):
        shown = resolve_init_set(init_set, inits)
        save(error_figure(summary, groups, MAIN_SCORES, variants, shown, error), main)
        save(category_figure(summary, groups, [error] + SCORES, variants, shown), all_scores)
        save(category_figure(summary, present, [error] + SCORES, variants, shown), per_dataset)
        save(gap_scatter(per_split, variants, shown, error), gaps)

    n_splits = int(per_split.groupby(["dataset", "init", "variant_name"])["split"].nunique().max())
    with open(snakemake.output.table, "w") as f:
        f.write("\n".join(table(summary, variants, inits, n_splits, str(snakemake.params.caption),
                                str(snakemake.params.label))) + "\n")


run(snakemake)  # noqa: F821
