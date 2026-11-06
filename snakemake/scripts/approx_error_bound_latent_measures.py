"""
Figure 5 (b): measures of dimensional collapse and redundancy of the latent space vs the
approximation error.

Frobenius regularization alone makes the diagonal Bures term faithful by making the latent
dimensions redundant (strongly correlated within the components), MI regularization keeps them
distinct; only both together reach a small error with low redundancy. Each measure (x, per-epoch
metric of approx_error_bound_epochs.py, defined in _approx_error.map_metrics) is plotted against
the absolute (approx_error, the y-axis of Figure 5 (a)) or relative (relative_approx_error)
approximation error, from the initialization (epoch 0) to the trained map (last epoch).

Map measures (effective_dof) are map_metrics the training does not store per epoch: computed here
from the stored trained maps and the initial maps, regenerated with the training's seeds
(_approx_error.initial_map). To make sure these are the maps of the run, latent_correlation is
recomputed from them and must equal the stored epoch-0 and last-epoch value of every split.

Per measure, one folder <measure>/ with one figure per error column, kind and init set
(_error_plots.init_sets), named <error>_<kind>_<init>.pdf:
  joined        one panel, both dataset groups (marker = group); mean over splits per dataset,
                then over the datasets of the group (error: geometric mean), arrow from the start
  means         as joined, one panel per dataset group (sized as Figure 5 (a))
  splits        one panel per dataset group, every dataset and train-test split: the trained map
                filled, the split's initialization hollow
  split_arrows  as splits, each split joined to its initialization by a thin line

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.dataset_csvs   — per-dataset approx_error_epochs.csv
  snakemake.input.trained_maps   — per-dataset trained_maps.npz (same order)
  snakemake.input.gmms           — per-dataset fitted GMM (h5ad, same order)
  snakemake.output.figures       — output folder: <measure>/ folders, means.csv and splits.csv
                                   (start and end values per group / per dataset and split)
  snakemake.params.measures      — measure columns, one folder each
  snakemake.params.map_measures  — map measures, one folder each
  snakemake.params.datasets      — dataset names (order of the inputs)
  snakemake.params.gmm_keys      — per-dataset GMM key in adata.uns (same order)
  snakemake.params.random_seed   — seed of the training (split i: random_seed + i)
  snakemake.params.variants      — variant name -> display name
  snakemake.params.inits         — initializations
  snakemake.params.error_columns — error columns
  snakemake.log[0]
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _error_plots import (  # noqa: E402
    DISPLAY_RENAME,
    ERROR_YLABELS,
    FONT_SIZE as PANEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    TICK_LABEL_SIZE,
    TITLE_PAD,
    TITLE_SIZE,
    VARIANT_ORDER,
    VARIANT_STYLES,
    X_TICK_LABEL_SIZE,
    _configure_y_ticks,
    fig5_axes,
    fig5_legend_rows,
    init_color,
    init_legend_handles,
    init_sets,
    present_inits,
    resolve_init_set,
)
from _approx_error import flatten_gmm_model_arrays, initial_map, map_metrics  # noqa: E402
from _utils import setup_logging  # noqa: E402

import anndata  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MEASURE_LABELS = {
    "axis_redundancy": "Mean |cos| between axes",
    "gram_dependence": "Gram dependence",
    "effective_rank": "Effective rank of $\\mathbf{A}$",
    "stable_rank": "Stable rank of $\\mathbf{A}$",
    "latent_participation_ratio": "Latent participation ratio",
    "latent_correlation": "Latent correlation",
    "latent_correlation_within": "Within-component correlation",
    "effective_dof": "Effective dimensionality",
}
# stored per epoch by the training and recomputed from the maps, to check the regenerated maps
CHECK_MEASURE = "latent_correlation"
GROUP_MARKERS = {"scRNA-seq": "o", "Synthetic": "^"}
FONT_SIZE = 15  # joined panel; the per-group figures use the Figure 5 (a) sizes of _error_plots


def group_of(dataset: str) -> str:
    return "Synthetic" if dataset.startswith("synth") else "scRNA-seq"


def start_end(paths, inits: list[str], columns: list[str], error_columns: list[str]) -> pd.DataFrame:
    """Start (epoch 0) and end (last epoch) per dataset, init and variant, mean over splits;
    then the group means (errors: geometric)."""
    rows = []
    for path in paths:
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        df = df[(df["split"].astype(str) == "mean") & df["init"].isin(inits)]
        for (dataset, init, variant), curve in df.groupby(["dataset", "init", "variant_name"]):
            curve = curve.sort_values("epoch")
            rows.append({"dataset": dataset, "group": group_of(dataset), "init": init, "variant_name": variant,
                         **{f"{c}_start": curve[c].iloc[0] for c in columns},
                         **{f"{c}_end": curve[c].iloc[-1] for c in columns}})
    per_dataset = pd.DataFrame(rows)
    error_cols = [f"{c}_{end}" for c in error_columns for end in ("start", "end")]
    per_dataset[error_cols] = np.log(per_dataset[error_cols])
    value_cols = [c for c in per_dataset if c.endswith(("_start", "_end"))]
    groups = per_dataset.groupby(["group", "init", "variant_name"])[value_cols].mean()
    groups[error_cols] = np.exp(groups[error_cols])
    return groups.reset_index()


def split_start_end(paths, inits: list[str], columns: list[str]) -> pd.DataFrame:
    """Start (epoch 0, shared by the regularizations of a split) and end (last epoch) per dataset,
    init, variant and train-test split."""
    rows = []
    for path in paths:
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        df = df[(df["split"].astype(str) != "mean") & df["init"].isin(inits)]
        for (dataset, init, variant, split), curve in df.groupby(["dataset", "init", "variant_name", "split"]):
            curve = curve.sort_values("epoch")
            rows.append({"dataset": dataset, "group": group_of(dataset), "init": init, "variant_name": variant,
                         "split": int(split),
                         **{f"{c}_start": curve[c].iloc[0] for c in columns},
                         **{f"{c}_end": curve[c].iloc[-1] for c in columns}})
    return pd.DataFrame(rows)


def map_measure_splits(maps_paths, gmm_paths, datasets, gmm_keys, inits: list[str], seed: int,
                       measures: list[str]) -> pd.DataFrame:
    """Start (regenerated initial map) and end (stored trained map) of the map measures per
    dataset, init, variant and train-test split, and of CHECK_MEASURE as check_start / check_end."""
    rows = []
    for dataset, maps_path, gmm_path, gmm_key in zip(datasets, maps_paths, gmm_paths, gmm_keys):
        mu, sigma, pi = flatten_gmm_model_arrays(anndata.read_h5ad(gmm_path, backed="r").uns[gmm_key]["model"])
        maps = np.load(maps_path)
        for key in maps.files:
            init, variant, split = key.split("__")
            if init not in inits:
                continue
            split = int(split.removeprefix("split"))
            trained = maps[key].astype(float)
            row = {"dataset": dataset, "init": init, "variant_name": variant, "split": split}
            for end, linear_map in (("start", initial_map(trained.shape, init, seed + split)), ("end", trained)):
                metrics = map_metrics(sigma, pi, linear_map, mu=mu)
                row.update({f"{m}_{end}": metrics[m] for m in measures})
                row[f"check_{end}"] = metrics[CHECK_MEASURE]
            rows.append(row)
    return pd.DataFrame(rows)


def check_maps(splits: pd.DataFrame, logger) -> None:
    """The recomputed CHECK_MEASURE must equal the one stored by the training for every split, at
    the start (regenerated initial maps) and at the end (stored trained maps)."""
    for end in ("start", "end"):
        stored, recomputed = splits[f"{CHECK_MEASURE}_{end}"], splits[f"check_{end}"]
        mismatch = ~np.isclose(recomputed, stored, rtol=1e-6, atol=1e-9)
        if mismatch.any():
            raise ValueError(
                f"{CHECK_MEASURE} at the {end} recomputed from the maps differs from the stored training metric "
                f"for {int(mismatch.sum())} splits; the {'regenerated initial' if end == 'start' else 'stored'} maps "
                f"are not the maps of the run:\n{splits.loc[mismatch, ['dataset', 'init', 'variant_name', 'split']]}")
        logger.info(f"Map check ({end}): max |diff| {CHECK_MEASURE} = {float((recomputed - stored).abs().max()):.2e}")


def map_measure_means(map_splits: pd.DataFrame, measures: list[str]) -> pd.DataFrame:
    """Group means of the map measures as in start_end: mean over splits per dataset, then over
    the datasets of the group."""
    columns = [f"{m}_{end}" for m in measures for end in ("start", "end")]
    per_dataset = map_splits.groupby(["dataset", "init", "variant_name"])[columns].mean().reset_index()
    per_dataset["group"] = per_dataset["dataset"].map(group_of)
    return per_dataset.groupby(["group", "init", "variant_name"])[columns].mean().reset_index()


def variant_color(variants: dict, variant_name: str, init: str, shaded: bool):
    display_name = variants[variant_name]
    return init_color(VARIANT_STYLES[DISPLAY_RENAME.get(display_name, display_name)]["color"], init, shaded=shaded)


def draw_means(ax, rows: pd.DataFrame, variants: dict, x: str, y: str, marker_of_group: bool, scale: float = 1.0):
    """Arrows from the initialization (hollow) to the trained map (filled); *scale* enlarges
    the markers and arrows (per-group figure)."""
    shaded = len(present_inits(rows)) > 1
    for _, row in rows.iterrows():
        color = variant_color(variants, row["variant_name"], row["init"], shaded)
        marker = GROUP_MARKERS[row["group"]] if marker_of_group else "o"
        ax.annotate("", xy=(row[f"{x}_end"], row[f"{y}_end"]), xytext=(row[f"{x}_start"], row[f"{y}_start"]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0 * scale, alpha=0.75,
                                    mutation_scale=15 * scale, shrinkA=4 * scale, shrinkB=6 * scale))
        ax.scatter(row[f"{x}_start"], row[f"{y}_start"], marker=marker, s=80 * scale ** 2, facecolor="white",
                   edgecolor="#777777", lw=1.4 * scale, zorder=3)
        ax.scatter(row[f"{x}_end"], row[f"{y}_end"], marker=marker, s=150 * scale ** 2, color=color,
                   edgecolor="white", lw=1.2 * scale, zorder=4)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def draw_splits(ax, rows: pd.DataFrame, variants: dict, x: str, y: str, arrows: bool):
    """One filled point per dataset and split (trained map) and one hollow point per split start
    (shared by the regularizations); with *arrows*, a thin line joins each split to its start."""
    shaded = len(present_inits(rows)) > 1
    starts = rows.drop_duplicates(["dataset", "init", "split"])
    ax.scatter(starts[f"{x}_start"], starts[f"{y}_start"], s=34, facecolor="white", edgecolor="#888888",
               lw=1.1, zorder=3)
    for (init, variant_name), variant_rows in rows.groupby(["init", "variant_name"]):
        color = variant_color(variants, variant_name, init, shaded)
        if arrows:
            for _, row in variant_rows.iterrows():
                ax.plot([row[f"{x}_start"], row[f"{x}_end"]], [row[f"{y}_start"], row[f"{y}_end"]],
                        color=color, lw=1.0, alpha=0.35, zorder=1)
        ax.scatter(variant_rows[f"{x}_end"], variant_rows[f"{y}_end"], s=60, color=color, edgecolor="white",
                   lw=0.7, alpha=0.9, zorder=4)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def group_legend(fig, inits: list[str]):
    """Framed legend as in Figure 5 (a), with points as in the panels: regularization (trained
    map) and the start marker; with several inits, their shades in a second row."""
    invisible = Line2D([], [], linestyle="None", marker="None")
    handles = [invisible] + [Line2D([], [], color=VARIANT_STYLES[n]["color"], ls="", marker="o", ms=12)
                             for n in VARIANT_ORDER]
    handles += [Line2D([], [], marker="o", ls="", ms=11, mfc="white", mec="#777777", mew=1.8)]
    labels = ["Regularization:"] + list(VARIANT_ORDER) + ["Initialization"]
    rows = [(handles, labels)]
    if len(inits) > 1:
        init_handles, init_labels = init_legend_handles(inits, marker="o", markersize=12)
        rows.append(([invisible] + init_handles, ["Initialization:"] + init_labels))
    fig5_legend_rows(fig, rows, handlelength=1.0, columnspacing=1.0, handletextpad=0.4)


def joined_legend(fig, variants: dict, groups: list[str], inits: list[str], y: float):
    handles = [Line2D([], [], ls="", marker="o", ms=10, color=VARIANT_STYLES[DISPLAY_RENAME.get(d, d)]["color"],
                      label=DISPLAY_RENAME.get(d, d)) for d in variants.values()]
    handles += [Line2D([], [], marker=GROUP_MARKERS[g], ls="", ms=10, color="#555555", label=g) for g in groups]
    if len(inits) > 1:
        init_handles, init_labels = init_legend_handles(inits, marker="o", markersize=10)
        for handle, label in zip(init_handles, init_labels):
            handle.set_label(label)
            handles.append(handle)
    handles += [Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec="#777777", label="Initialization")]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 7), frameon=False,
               fontsize=FONT_SIZE - 2, bbox_to_anchor=(0.5, y), handletextpad=0.3, columnspacing=1.0)


def joined_figure(path, rows: pd.DataFrame, variants: dict, groups: list[str], inits: list[str], x: str, y: str):
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    draw_means(ax, rows, variants, x, y, marker_of_group=True)
    ax.set_xlabel(MEASURE_LABELS[x], fontsize=FONT_SIZE - 1)
    ax.set_ylabel(f"Error {ERROR_YLABELS[y]}", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE - 2)
    fig.tight_layout()
    joined_legend(fig, variants, groups, inits, y=0.0)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def group_figure(path, rows: pd.DataFrame, groups: list[str], inits: list[str], x: str, y: str, draw_panel):
    """One panel per dataset group in the shared Figure 5 layout (panels of the same size as in
    Figure 5 (a)); *draw_panel(ax, rows)*."""
    fig, axes = fig5_axes(legend_rows=2 if len(inits) > 1 else 1)
    for ax, group in zip(axes, groups):
        draw_panel(ax, rows[rows["group"] == group])
        ax.set_title(f"{group} Datasets", fontsize=TITLE_SIZE, pad=TITLE_PAD)
        ax.set_xlabel(MEASURE_LABELS[x], fontsize=PANEL_FONT_SIZE)
        ax.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
        ax.margins(x=0.08, y=0.08)
        fig.canvas.draw()
        _configure_y_ticks(ax, log_scale=True)
    axes[0].set_ylabel(f"Error {ERROR_YLABELS[y]}", fontsize=PANEL_FONT_SIZE)
    group_legend(fig, inits)
    fig.savefig(path, dpi=300)  # no tight cropping, see fig5_axes
    plt.close(fig)


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    variants = dict(snakemake.params.variants)
    inits = list(snakemake.params.inits)
    stored_measures = list(snakemake.params.measures)
    map_measures = list(snakemake.params.map_measures)
    error_columns = list(snakemake.params.error_columns)
    out = Path(snakemake.output.figures)
    out.mkdir(parents=True, exist_ok=True)

    columns = list(dict.fromkeys(stored_measures + [CHECK_MEASURE] + error_columns))
    means = start_end(snakemake.input.dataset_csvs, inits, columns, error_columns)
    splits = split_start_end(snakemake.input.dataset_csvs, inits, columns)
    if map_measures:
        map_splits = map_measure_splits(snakemake.input.trained_maps, snakemake.input.gmms,
                                        list(snakemake.params.datasets), list(snakemake.params.gmm_keys),
                                        inits, int(snakemake.params.random_seed), map_measures)
        # left join: a split without a stored map fails the check below
        splits = splits.merge(map_splits, on=["dataset", "init", "variant_name", "split"], how="left",
                              validate="one_to_one")
        check_maps(splits, logger)
        splits = splits.drop(columns=["check_start", "check_end"])
        means = means.merge(map_measure_means(map_splits, map_measures), on=["group", "init", "variant_name"],
                            validate="one_to_one")
    means.to_csv(out / "means.csv", index=False)
    splits.to_csv(out / "splits.csv", index=False)
    logger.info("\n" + means.round(4).to_string(index=False))
    groups = [g for g in GROUP_MARKERS if g in set(means["group"])]

    for x in stored_measures + map_measures:
        folder = out / x
        folder.mkdir(exist_ok=True)
        for y, init_set in itertools.product(error_columns, init_sets(inits)):
            shown = resolve_init_set(init_set, inits)
            mean_rows = means[means["init"].isin(shown)]
            split_rows = splits[splits["init"].isin(shown)]
            joined_figure(folder / f"{y}_joined_{init_set}.pdf", mean_rows, variants, groups, shown, x, y)
            group_figure(folder / f"{y}_means_{init_set}.pdf", mean_rows, groups, shown, x, y,
                         lambda ax, r: draw_means(ax, r, variants, x, y, marker_of_group=False, scale=1.25))
            for kind, arrows in (("splits", False), ("split_arrows", True)):
                group_figure(folder / f"{y}_{kind}_{init_set}.pdf", split_rows, groups, shown, x, y,
                             lambda ax, r, arrows=arrows: draw_splits(ax, r, variants, x, y, arrows))
        logger.info(f"{x}: written to {folder}")


run(snakemake)  # noqa: F821
