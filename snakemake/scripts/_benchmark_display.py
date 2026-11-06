"""
Shared notebook-style display helpers for benchmark tables and Pareto plots.
"""

from collections import OrderedDict
import re

import numpy as np
import pandas as pd
from matplotlib import colormaps


DISPLAY_METRICS = OrderedDict(
    [
        ("knn", {"direction": 1, "unit": None, "header": r"\textbf{KNN Acc.} $\uparrow$"}),
        ("ari", {"direction": 1, "unit": None, "header": r"\textbf{ARI} $\uparrow$"}),
        ("silhouette", {"direction": 1, "unit": None, "header": r"\textbf{Silh.} $\uparrow$"}),
        ("inf_time(ms)", {"direction": -1, "unit": "ms", "header": r"\textbf{OT} $t$ $\downarrow$"}),
        ("epoch_time(s)", {"direction": -1, "unit": "s", "header": r"\textbf{Train} $t$ $\downarrow$"}),
    ]
)
"""Table columns in display order. Runs without a metric render it as ``$-$``."""

SECTION_BY_DATASET_TYPE = OrderedDict(
    [
        ("empirical", "Point Clouds"),
        ("centroid", "Weighted Centroids"),
        ("gmm", "GMMs"),
    ]
)

_TAB20_REORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 18, 19, 16, 17]

_TAB20 = colormaps["tab20"].colors

_METHOD_COLORS = {
    # Section-coherent tab20 mapping:
    # isotropic uniform = cool, isotropic weighted = green, anisotropic = warm.
    "Empirical_euclidean": _TAB20[0],
    "Empirical_cosine": _TAB20[1],
    "Empirical_sinkhorn": _TAB20[18],
    "FDA": _TAB20[8],
    "WDA": _TAB20[9],
    "Empirical_GML": _TAB20[12],
    "Centroid_euclidean": _TAB20[4],
    "Centroid_cosine": _TAB20[5],
    "Centroid_GML": _TAB20[16],
    # Distinct related tab20 shades, fixed across method subsets and ordering.
    "GMM_means_only": _TAB20[17],  # light olive
    "Centroid_raw_cosine": _TAB20[19],  # light cyan
    "GMM_KL_divergence": _TAB20[14],
    "GMM_euclidean": _TAB20[2],
    "GMM_euclidean_diag": _TAB20[3],
    "GMM_DiagBures_NoMI": _TAB20[10],
    "GMM_DiagBures": _TAB20[7],
    "GMM_FullBures": _TAB20[6],
}


def section_for_method(method_cfg: dict) -> str:
    """Map dataset_type to the paper/notebook section title."""
    dataset_type = str(method_cfg.get("dataset_type", "")).lower()
    return SECTION_BY_DATASET_TYPE.get(dataset_type, dataset_type or "Other")


def get_sectioned_methods(config: dict, method_names: list[str]) -> OrderedDict[str, list[str]]:
    """Return configured methods grouped by display section, preserving config order."""
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for section in SECTION_BY_DATASET_TYPE.values():
        grouped[section] = []

    for method_name in method_names:
        section = section_for_method(config["methods"][method_name])
        grouped.setdefault(section, []).append(method_name)

    return OrderedDict((section, methods) for section, methods in grouped.items() if methods)


def get_display_name(config: dict, method_name: str) -> str:
    """Human-facing label for tables/plots."""
    return str(config["methods"][method_name].get("display_name", method_name))


def is_ours_method(config: dict, method_name: str) -> bool:
    """Highlight the proposed methods consistently with the notebook."""
    return get_display_name(config, method_name).startswith("Ours")


def is_supervised_method(config: dict, method_name: str) -> bool:
    """Whether labels are used to learn a metric/projection."""
    method_cfg = config["methods"][method_name]
    if str(method_cfg.get("ground_metric", "")).lower() == "learned":
        return True
    return str(method_cfg.get("external", "")).lower() in {"fda", "wda"}


def build_color_map(config: dict, method_names: list[str]) -> dict[str, tuple[float, ...]]:
    """Notebook-like color ordering based on a reordered tab20 palette."""
    base_palette = list(colormaps["tab20"].colors)
    palette = [base_palette[i] for i in _TAB20_REORDER if i < len(base_palette)]
    palette += list(colormaps["tab20b"].colors)
    palette += list(colormaps["tab20c"].colors)

    flat_methods = [
        method_name
        for methods in get_sectioned_methods(config, method_names).values()
        for method_name in methods
    ]
    return {
        method_name: _METHOD_COLORS.get(method_name, palette[idx % len(palette)])
        for idx, method_name in enumerate(flat_methods)
    }


def load_score_summary(
    score_path: str,
    *,
    extra_metrics: list[str] | tuple[str, ...] | None = None,
) -> dict[tuple[str, str], float]:
    """Load the first benchmark row from a 2-level header scores.csv."""
    df = pd.read_csv(score_path, header=[0, 1], index_col=0)
    if df.empty:
        raise ValueError(f"Empty scores file: {score_path}")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    row = df.iloc[0]
    values: dict[tuple[str, str], float] = {}
    metrics = list(DISPLAY_METRICS)
    for metric in extra_metrics or []:
        if metric not in metrics:
            metrics.append(metric)

    for metric in metrics:
        for stat in ("mean", "SD"):
            col = (metric, stat)
            value = row[col] if col in row.index else np.nan
            values[col] = float(value) if pd.notna(value) else np.nan
    return values


def make_score_record(
    *,
    dataset: str,
    method_name: str,
    method_cfg: dict,
    summary: dict[tuple[str, str], float],
) -> dict:
    """Create a row record for a benchmark result."""
    record = {
        "dataset": dataset,
        "section": section_for_method(method_cfg),
        "method_name": method_name,
    }
    record.update(summary)
    return record


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Convert benchmark records into a MultiIndex dataframe."""
    metric_cols = pd.MultiIndex.from_tuples(
        [(metric, stat) for metric in DISPLAY_METRICS for stat in ("mean", "SD")]
    )
    if not records:
        return pd.DataFrame(columns=metric_cols)

    df = pd.DataFrame(records)
    df = df.set_index(["dataset", "section", "method_name"])
    metric_columns = [col for col in df.columns if isinstance(col, tuple)]
    df = df[metric_columns]
    df.columns = pd.MultiIndex.from_tuples(metric_columns)
    return df.sort_index()


def pooled_mean_std(means, stds, n: int = 5) -> tuple[float, float]:
    """Notebook-style pooled mean and standard deviation across datasets."""
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    valid_means = ~np.isnan(means)
    if not np.any(valid_means):
        return np.nan, np.nan

    grand_mean = float(np.mean(means[valid_means]))

    valid_both = valid_means & ~np.isnan(stds)
    means, stds = means[valid_both], stds[valid_both]
    k = len(means)
    if k == 0:
        return grand_mean, np.nan

    ss_within = np.sum((n - 1) * (stds**2))
    ss_between = np.sum(n * (means - grand_mean) ** 2)
    total_dof = (k * n) - 1

    if total_dof <= 0:
        return grand_mean, 0.0

    pooled_std = float(np.sqrt((ss_within + ss_between) / total_dof))
    return grand_mean, pooled_std


def aggregate_scores(df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """Aggregate raw per-dataset scores into the notebook's global average view."""
    if df.empty:
        return records_to_frame([])

    agg_data = {}
    for metric in DISPLAY_METRICS:
        mean_col = (metric, "mean")
        sd_col = (metric, "SD")
        if mean_col not in df.columns:
            continue

        grouped_means = df[mean_col].groupby(level=["section", "method_name"])
        grouped_stds = df[sd_col].groupby(level=["section", "method_name"])

        pooled_means = []
        pooled_stds = []
        indices = []
        for key, val_means in grouped_means:
            try:
                val_stds = grouped_stds.get_group(key)
            except KeyError:
                val_stds = pd.Series(np.nan, index=val_means.index)

            pooled_mean, pooled_std = pooled_mean_std(
                val_means.values,
                val_stds.values,
                n=n_splits,
            )
            indices.append(key)
            pooled_means.append(pooled_mean)
            pooled_stds.append(pooled_std)

        idx = pd.MultiIndex.from_tuples(indices, names=["section", "method_name"])
        agg_data[(metric, "mean")] = pd.Series(pooled_means, index=idx)
        agg_data[(metric, "SD")] = pd.Series(pooled_stds, index=idx)

    agg_df = pd.DataFrame(agg_data)
    agg_df["dataset"] = "Average"
    agg_df.set_index("dataset", append=True, inplace=True)
    return agg_df.reorder_levels(["dataset", "section", "method_name"]).sort_index()


def get_dataset_slice(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Extract one dataset from a raw or aggregated benchmark dataframe."""
    if "dataset" in df.index.names:
        return df.xs(dataset_name, level="dataset")
    return df


def render_pareto_plot(
    df: pd.DataFrame,
    config: dict,
    method_names: list[str],
    dataset_name: str,
    out_path: str,
    *,
    n_splits: int,
    stats_label: str | None = None,
) -> None:
    """Render the notebook-style Pareto plot for one dataset slice."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = get_dataset_slice(df, dataset_name)
    sectioned_methods = get_sectioned_methods(config, method_names)
    color_map = build_color_map(config, method_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    plotted_methods = set()
    valid_x_vals = []
    points_for_pareto = []

    for section, methods in sectioned_methods.items():
        for method_name in methods:
            section_key = (section, method_name)
            if section_key not in data.index:
                continue

            row = data.loc[section_key]
            x_mean = row.get(("inf_time(ms)", "mean"), np.nan)
            y_mean = row.get(("knn", "mean"), np.nan)
            y_std = row.get(("knn", "SD"), np.nan)
            if np.isnan(x_mean) or np.isnan(y_mean):
                continue

            ours = is_ours_method(config, method_name)
            marker = "s" if is_supervised_method(config, method_name) else "o"
            color = color_map.get(method_name, "gray")

            ax.errorbar(
                x=x_mean,
                y=y_mean,
                xerr=None,
                yerr=y_std,
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                capsize=0,
                alpha=0.6 if ours else 0.4,
                zorder=9 if ours else 4,
            )
            ax.scatter(
                x_mean,
                y_mean,
                s=180 if ours else 120,
                color=color,
                edgecolor="black" if ours else "white",
                linewidth=1.5 if ours else 0.8,
                marker=marker,
                zorder=10 if ours else 5,
            )

            points_for_pareto.append((x_mean, y_mean))
            plotted_methods.add(method_name)
            valid_x_vals.append(x_mean)

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("OT computation time (ms)", fontsize=10, labelpad=5)
    ax.set_ylabel("KNN Accuracy", fontsize=10, labelpad=5)
    ax.set_ylim(0.45, 1.0)

    if valid_x_vals:
        x_min_data, x_max_data = min(valid_x_vals), max(valid_x_vals)
        ax.set_xlim(x_max_data * 2.5, x_min_data * 0.5)
    else:
        ax.set_xlim(10, 1)
        ax.text(0.5, 0.5, "No benchmark data", ha="center", va="center", transform=ax.transAxes)

    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="-", color="#E6E6E6", linewidth=1.0)
    ax.grid(True, which="minor", linestyle="-", color="#F5F5F5", linewidth=0.5)
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    if points_for_pareto:
        pts = sorted(points_for_pareto, key=lambda point: point[0])
        pareto_front = [pts[0]]
        for point in pts[1:]:
            if point[1] > pareto_front[-1][1]:
                pareto_front.append(point)

        px, py = list(zip(*pareto_front))
        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()
        px_final = [px[0]] + list(px) + [x_lims[0]]
        py_final = [y_lims[0]] + list(py) + [py[-1]]
        ax.step(
            px_final,
            py_final,
            where="post",
            color="#333333",
            linestyle="-",
            alpha=0.8,
            zorder=1,
            linewidth=2.5,
        )

    legend_handles = []
    for section, methods in sectioned_methods.items():
        section_methods = [method_name for method_name in methods if method_name in plotted_methods]
        if not section_methods:
            continue
        legend_handles.append(
            Line2D([0], [0], marker="", color="none", label=section)
        )
        for method_name in section_methods:
            label = get_display_name(config, method_name)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="s" if is_supervised_method(config, method_name) else "o",
                    color="w",
                    markerfacecolor=color_map.get(method_name, "gray"),
                    markeredgecolor="black" if is_ours_method(config, method_name) else "white",
                    markeredgewidth=1.5 if is_ours_method(config, method_name) else 0.5,
                    markersize=9,
                    label=label,
                )
            )
        legend_handles.append(Line2D([0], [0], marker="", color="none", label=""))
    if legend_handles and legend_handles[-1].get_label() == "":
        legend_handles.pop()

    extra_artists = []
    if legend_handles:
        main_legend = ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            fontsize=9,
            edgecolor="#D0D0D0",
            labelspacing=0.4,
            handlelength=1.6,
            handletextpad=0.8,
            borderpad=0.6,
        )
        section_labels = set(sectioned_methods.keys())
        for text in main_legend.get_texts():
            label = text.get_text()
            if label in section_labels or label.startswith("Ours"):
                text.set_fontweight("bold")
        main_legend.get_frame().set_linewidth(0.8)
        main_legend.get_frame().set_alpha(1.0)
        ax.add_artist(main_legend)
        extra_artists.append(main_legend)

    ground_metric_header = Line2D(
        [0],
        [0],
        marker="",
        color="none",
        label="Ground Metric",
    )
    supervised_handle = Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=9,
        label="Supervised",
    )
    unsupervised_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=9,
        label="Unsupervised",
    )

    stats_legend = ax.legend(
        handles=[ground_metric_header, supervised_handle, unsupervised_handle],
        loc="upper left",
        bbox_to_anchor=(1.02, 0.15),
        frameon=True,
        fontsize=9,
        edgecolor="#D0D0D0",
        labelspacing=0.4,
        handlelength=1.6,
        handletextpad=0.8,
        borderpad=0.6,
    )
    stats_legend.get_texts()[0].set_fontweight("bold")
    stats_legend.get_frame().set_linewidth(0.8)
    stats_legend.get_frame().set_alpha(1.0)
    extra_artists.append(stats_legend)

    plt.subplots_adjust(right=0.6)
    fig.savefig(out_path, bbox_extra_artists=tuple(extra_artists), bbox_inches="tight")
    plt.close(fig)


def render_train_time_vs_patients_plot(
    df: pd.DataFrame,
    config: dict,
    method_names: list[str],
    dataset_patient_counts: dict[str, int],
    out_path: str,
    *,
    log_scale: bool = False,
) -> None:
    """Render training time against dataset patient count for trained methods."""
    render_time_vs_patients_plot(
        df,
        config,
        method_names,
        dataset_patient_counts,
        out_path,
        metric="epoch_time(s)",
        ylabel="Training Time (s)",
        empty_message="No training-time benchmark data",
        log_scale=log_scale,
    )


def render_ot_time_vs_patients_plot(
    df: pd.DataFrame,
    config: dict,
    method_names: list[str],
    dataset_patient_counts: dict[str, int],
    out_path: str,
    *,
    log_scale: bool = False,
) -> None:
    """Render full OT computation time against dataset patient count."""
    render_time_vs_patients_plot(
        df,
        config,
        method_names,
        dataset_patient_counts,
        out_path,
        metric="full_ot_time(s)",
        ylabel="Full OT Time (s)",
        empty_message="No OT-time benchmark data",
        log_scale=log_scale,
    )


def _time_line_style(config: dict, method_name: str) -> tuple[str, str]:
    if is_supervised_method(config, method_name):
        return "s", "-"
    return "o", "--"


def _style_time_axis(ax, valid_x_vals: list[float], valid_y_vals: list[float], *, log_scale: bool) -> None:
    from matplotlib.ticker import MaxNLocator

    if valid_x_vals and valid_y_vals:
        x_min, x_max = min(valid_x_vals), max(valid_x_vals)
        x_pad = max(2.0, 0.06 * (x_max - x_min))
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

        if log_scale:
            ax.set_yscale("log")
        else:
            y_min, y_max = min(valid_y_vals), max(valid_y_vals)
            y_pad = 0.08 * (y_max - y_min) if y_max > y_min else max(0.5, 0.1 * y_max)
            ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="-", color="#E6E6E6", linewidth=1.0)
    ax.grid(True, which="minor", linestyle="-", color="#F5F5F5", linewidth=0.5)


def _draw_time_vs_patients_axis(
    ax,
    df: pd.DataFrame,
    config: dict,
    method_names: list[str],
    dataset_patient_counts: dict[str, int],
    *,
    metric: str,
    ylabel: str,
    empty_message: str,
    log_scale: bool,
) -> set[str]:
    sectioned_methods = get_sectioned_methods(config, method_names)
    color_map = build_color_map(config, method_names)

    plotted_methods = set()
    valid_x_vals = []
    valid_y_vals = []
    ordered_datasets = sorted(dataset_patient_counts.items(), key=lambda item: item[1])

    for section, methods in sectioned_methods.items():
        for method_name in methods:
            x_vals = []
            y_vals = []

            for dataset_name, patient_count in ordered_datasets:
                key = (dataset_name, section, method_name)
                if key not in df.index:
                    continue

                row = df.loc[key]
                y_value = row.get((metric, "mean"), np.nan)
                if np.isnan(y_value) or (log_scale and y_value <= 0):
                    continue

                x_vals.append(patient_count)
                y_vals.append(y_value)

            if not x_vals:
                continue

            ours = is_ours_method(config, method_name)
            color = color_map.get(method_name, "gray")
            marker, linestyle = _time_line_style(config, method_name)

            ax.plot(
                x_vals,
                y_vals,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=6.8 if ours else 5.8,
                markerfacecolor=color,
                markeredgecolor="black" if ours else "white",
                markeredgewidth=1.2 if ours else 0.6,
                linewidth=2.4 if ours else 1.7,
                alpha=0.95 if ours else 0.82,
                zorder=9 if ours else 4,
            )

            plotted_methods.add(method_name)
            valid_x_vals.extend(x_vals)
            valid_y_vals.extend(y_vals)

    ax.set_ylabel(ylabel, fontsize=10, labelpad=5)
    if valid_x_vals and valid_y_vals:
        _style_time_axis(ax, valid_x_vals, valid_y_vals, log_scale=log_scale)
    else:
        ax.text(0.5, 0.5, empty_message, ha="center", va="center", transform=ax.transAxes)
        _style_time_axis(ax, [], [], log_scale=log_scale)

    return plotted_methods


def _time_legend_handles(
    config: dict,
    method_names: list[str],
    plotted_methods: set[str],
):
    from matplotlib.lines import Line2D

    sectioned_methods = get_sectioned_methods(config, method_names)
    color_map = build_color_map(config, method_names)

    legend_handles = []
    for section, methods in sectioned_methods.items():
        section_methods = [method_name for method_name in methods if method_name in plotted_methods]
        if not section_methods:
            continue
        legend_handles.append(Line2D([0], [0], marker="", color="none", label=section))
        for method_name in section_methods:
            marker, linestyle = _time_line_style(config, method_name)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color_map.get(method_name, "gray"),
                    linestyle=linestyle,
                    linewidth=2.4 if is_ours_method(config, method_name) else 1.7,
                    marker=marker,
                    markerfacecolor=color_map.get(method_name, "gray"),
                    markeredgecolor="black" if is_ours_method(config, method_name) else "white",
                    markeredgewidth=1.2 if is_ours_method(config, method_name) else 0.6,
                    markersize=6.8 if is_ours_method(config, method_name) else 5.8,
                    label=get_display_name(config, method_name),
                )
            )
    return legend_handles


def _add_time_legend(ax, config: dict, method_names: list[str], plotted_methods: set[str]):
    legend_handles = _time_legend_handles(config, method_names, plotted_methods)
    if not legend_handles:
        return None

    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=9,
        edgecolor="#D0D0D0",
        labelspacing=0.35,
        handlelength=1.8,
        handletextpad=0.7,
        borderpad=0.6,
    )
    section_labels = set(get_sectioned_methods(config, method_names).keys())
    for text in legend.get_texts():
        label = text.get_text()
        if label in section_labels or label.startswith("Ours"):
            text.set_fontweight("bold")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)
    ax.add_artist(legend)
    return legend


def render_time_vs_patients_plot(
    df: pd.DataFrame,
    config: dict,
    method_names: list[str],
    dataset_patient_counts: dict[str, int],
    out_path: str,
    *,
    metric: str,
    ylabel: str,
    empty_message: str,
    log_scale: bool = False,
) -> None:
    """Render one time metric against dataset patient count."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    plotted_methods = _draw_time_vs_patients_axis(
        ax,
        df,
        config,
        method_names,
        dataset_patient_counts,
        metric=metric,
        ylabel=ylabel,
        empty_message=empty_message,
        log_scale=log_scale,
    )
    ax.set_xlabel("Number of Patients / Distributions", fontsize=10, labelpad=5)

    extra_artists = []
    legend = _add_time_legend(ax, config, method_names, plotted_methods)
    if legend is not None:
        extra_artists.append(legend)

    plt.subplots_adjust(right=0.66)
    fig.savefig(out_path, bbox_extra_artists=tuple(extra_artists), bbox_inches="tight")
    plt.close(fig)


def render_time_vs_patients_panel(
    df: pd.DataFrame,
    config: dict,
    method_names: list[str],
    dataset_patient_counts: dict[str, int],
    out_path: str,
    *,
    log_scale: bool = False,
) -> None:
    """Render full OT time above training time with a shared patient-count axis."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.8), sharex=True)

    ot_methods = _draw_time_vs_patients_axis(
        axes[0],
        df,
        config,
        method_names,
        dataset_patient_counts,
        metric="full_ot_time(s)",
        ylabel="Full OT Time (s)",
        empty_message="No OT-time benchmark data",
        log_scale=log_scale,
    )
    train_methods = _draw_time_vs_patients_axis(
        axes[1],
        df,
        config,
        method_names,
        dataset_patient_counts,
        metric="epoch_time(s)",
        ylabel="Training Time (s)",
        empty_message="No training-time benchmark data",
        log_scale=log_scale,
    )
    axes[1].set_xlabel("Number of Patients / Distributions", fontsize=10, labelpad=5)

    plotted_methods = ot_methods | train_methods
    extra_artists = []
    legend = _add_time_legend(axes[0], config, method_names, plotted_methods)
    if legend is not None:
        extra_artists.append(legend)

    plt.subplots_adjust(right=0.66, hspace=0.12)
    fig.savefig(out_path, bbox_extra_artists=tuple(extra_artists), bbox_inches="tight")
    plt.close(fig)


def _metric_thresholds(data: pd.DataFrame) -> dict[str, tuple[float | None, float | None]]:
    thresholds = {}
    for metric, meta in DISPLAY_METRICS.items():
        mean_col = (metric, "mean")
        best = second = None
        if mean_col in data.columns:
            values = data[mean_col].dropna().unique()
            if len(values) > 0:
                ordered = sorted(values, reverse=(meta["direction"] == 1))
                best = ordered[0]
                second = ordered[1] if len(ordered) > 1 else None
        thresholds[metric] = (best, second)
    return thresholds


def _format_value(mean: float, sd: float, *, rank: int, unit: str | None) -> str:
    if pd.isna(mean):
        return "$-$"

    if unit:
        # A positive time below the display precision would otherwise read as "0.0".
        # "{<}" keeps math mode from spacing "<" as a relation.
        content = f"{{<}}0.1\\,{unit}" if 0 < mean < 0.05 else f"{mean:.1f}\\,{unit}"
    else:
        if pd.isna(sd):
            content = f"{mean:.2f}"
        else:
            content = f"{mean:.2f} \\pm {sd:.2f}"

    if rank == 1:
        return f"$\\mathbf{{{content}}}$"
    if rank == 2:
        return f"$\\underline{{{content}}}$"
    return f"${content}$"


def _escape_latex(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
    )


def _label_fragment(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()


def render_latex_table(
    df: pd.DataFrame,
    config: dict,
    method_names: list[str],
    dataset_name: str,
    *,
    table_kind: str,
) -> str:
    """Render a notebook-style LaTeX benchmark table."""
    try:
        data = get_dataset_slice(df, dataset_name)
    except KeyError:
        return f"% Error: Dataset '{dataset_name}' not found.\n"

    thresholds = _metric_thresholds(data)
    sections = get_sectioned_methods(config, method_names)

    n_metrics = len(DISPLAY_METRICS)
    n_columns = 2 + n_metrics
    column_spec = "ll" + "c" * n_metrics
    header = " & ".join(meta["header"] for meta in DISPLAY_METRICS.values())
    latex_rows = [
        r"\toprule",
        r"\multicolumn{2}{l}{\textbf{Method}} & " + header + r" \\",
        r"\midrule",
    ]

    for section, methods in sections.items():
        latex_rows.append(f"\\multicolumn{{{n_columns}}}{{l}}{{\\textsc{{{section}}}}} \\\\")
        for method_name in methods:
            label = get_display_name(config, method_name)
            section_key = (section, method_name)
            if section_key not in data.index:
                latex_rows.append(f"& {label} & " + " & ".join(["-"] * n_metrics) + r" \\")
                continue

            row = data.loc[section_key]

            def _rank(metric: str) -> int:
                mean_col = (metric, "mean")
                if mean_col not in row.index:
                    return 0
                value = row[mean_col]
                best, second = thresholds[metric]
                if best is not None and value == best:
                    return 1
                if second is not None and value == second:
                    return 2
                return 0

            values = []
            for metric, meta in DISPLAY_METRICS.items():
                values.append(
                    _format_value(
                        row.get((metric, "mean"), np.nan),
                        row.get((metric, "SD"), np.nan),
                        rank=_rank(metric),
                        unit=meta["unit"],
                    )
                )

            latex_rows.append(f"& {label} & " + " & ".join(values) + r" \\")

    latex_rows.append(r"\bottomrule")

    if table_kind == "main":
        wrapper_start = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{\textbf{Benchmark Performance.} Aggregated results. \textbf{Bold}=Best, \underline{Underline}=Second.}",
            r"\label{tab:main_results}",
            r"\resizebox{\linewidth}{!}{",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3.5pt}",
            f"\\begin{{tabular}}{{{column_spec}}}",
        ]
        wrapper_end = [
            r"\end{tabular}}",
            r"\end{table}",
        ]
    elif table_kind == "subtable":
        wrapper_start = [
            r"\begin{subtable}{0.48\textwidth}",
            r"\centering",
            f"\\caption{{{_escape_latex(dataset_name)}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{1.2pt}",
            r"\renewcommand{\arraystretch}{1.1}",
            f"\\begin{{tabular}}{{{column_spec}}}",
        ]
        wrapper_end = [
            r"\end{tabular}",
            r"\end{subtable}%",
        ]
    elif table_kind == "dataset":
        wrapper_start = [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{{_escape_latex(dataset_name)}}}",
            f"\\label{{tab:{_label_fragment(dataset_name)}_benchmark}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{1.2pt}",
            r"\renewcommand{\arraystretch}{1.1}",
            f"\\begin{{tabular}}{{{column_spec}}}",
        ]
        wrapper_end = [
            r"\end{tabular}",
            r"\end{table}",
        ]
    else:
        raise ValueError(f"Unknown table_kind: {table_kind!r}")

    return "\n".join(wrapper_start + latex_rows + wrapper_end) + "\n"
