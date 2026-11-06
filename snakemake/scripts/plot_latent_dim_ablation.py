"""
Summarize latent-dimension tuning by selecting the best hyperparameters per d.

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.scores        — tuning scores.csv for one dataset/method pair
  snakemake.output.summary_csv  — selected-best-per-d summary CSV
  snakemake.output.accuracy_plot — line plot of best kNN accuracy vs latent dim
  snakemake.output.train_time_plot — line plot of train time vs latent dim
  snakemake.params.plot_cfg     — latent_dim_ablation_plot config block
  snakemake.log[0]
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _utils import setup_logging  # noqa: E402


def _metric_col_name(metric: str, stat: str) -> str:
    return f"{metric}_{stat}"


def _load_tune_scores(path: str | Path) -> pd.DataFrame:
    preview = pd.read_csv(path, header=[0, 1], nrows=1)

    index_names = []
    for col in preview.columns:
        value = preview.iloc[0][col]
        if pd.isna(value):
            break
        index_names.append(str(value))

    if not index_names:
        raise ValueError(f"Could not infer tuning index columns from {path}")

    df = pd.read_csv(path, header=[0, 1], index_col=list(range(len(index_names))))
    df.index.names = index_names

    drop_cols = [col for col in df.columns if str(col[0]).startswith("Unnamed:")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _select_best_rows_per_x(
    scores_df: pd.DataFrame,
    *,
    x_param: str,
    fixed_params: dict,
    selection_col: tuple[str, str],
    tie_break_col: tuple[str, str] | None,
    logger,
) -> pd.DataFrame:
    work = scores_df

    for level, value in (fixed_params or {}).items():
        if level not in work.index.names:
            logger.warning(f"Skipping unknown fixed_params level: {level!r}")
            continue
        try:
            work = work.xs(value, level=level, drop_level=False)
        except KeyError as exc:
            raise ValueError(f"fixed_params selects missing value {value!r} for level {level!r}") from exc

    if x_param not in work.index.names:
        raise ValueError(f"x_param {x_param!r} not found in tuning index {list(work.index.names)!r}")
    if selection_col not in work.columns:
        raise ValueError(f"Selection metric {selection_col!r} not found in tuning scores")

    x_values = pd.Index(work.index.get_level_values(x_param).unique())
    x_values = sorted(x_values, key=lambda value: float(value))

    records = []
    for x_value in x_values:
        group = work.xs(x_value, level=x_param, drop_level=False)
        selection_scores = pd.to_numeric(group[selection_col], errors="coerce")
        selection_scores = selection_scores.dropna()
        if selection_scores.empty:
            logger.warning(f"Skipping {x_param}={x_value}: no finite {selection_col!r} values")
            continue

        best_score = selection_scores.max()
        candidates = group.loc[selection_scores[selection_scores == best_score].index]

        best_idx = candidates.index[0]
        if tie_break_col is not None and tie_break_col in candidates.columns:
            tie_scores = pd.to_numeric(candidates[tie_break_col], errors="coerce")
            tie_scores = tie_scores.dropna()
            if not tie_scores.empty:
                best_idx = tie_scores.idxmin()

        row = group.loc[best_idx]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if not isinstance(best_idx, tuple):
            best_idx = (best_idx,)

        record = {
            level_name: (value.item() if hasattr(value, "item") else value)
            for level_name, value in zip(work.index.names, best_idx)
        }
        for metric, stat in row.index:
            record[_metric_col_name(str(metric), str(stat))] = row[(metric, stat)]
        records.append(record)

    if not records:
        raise ValueError("No best-per-dimension rows could be selected from tuning scores")

    summary_df = pd.DataFrame(records)
    summary_df[x_param] = pd.to_numeric(summary_df[x_param], errors="coerce")
    return summary_df.sort_values(x_param).reset_index(drop=True)


def _render_line_plot(
    summary_df: pd.DataFrame,
    *,
    x_param: str,
    y_metric: str,
    out_path: str | Path,
    title: str,
    ylabel: str,
    x_label: str,
    dpi: int,
    figsize: tuple[float, float],
    log_y: bool,
) -> None:
    y_col = _metric_col_name(y_metric, "mean")
    yerr_col = _metric_col_name(y_metric, "SD")

    if y_col not in summary_df.columns:
        raise KeyError(f"Missing summary column: {y_col}")

    x = pd.to_numeric(summary_df[x_param], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(summary_df[y_col], errors="coerce").to_numpy(dtype=float)
    yerr = None
    if yerr_col in summary_df.columns:
        yerr = pd.to_numeric(summary_df[yerr_col], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    color = "#1f77b4"

    if yerr is not None and np.isfinite(yerr).any():
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            markersize=6,
            linewidth=2.0,
            capsize=4,
            color=color,
        )
    else:
        ax.plot(x, y, marker="o", markersize=6, linewidth=2.0, color=color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if y_metric == "knn":
        ax.set_ylim(0.0, 1.05)
    if log_y:
        positive = np.isfinite(y) & (y > 0)
        if positive.any():
            ax.set_yscale("log")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    plot_cfg = dict(snakemake.params.plot_cfg or {})

    x_param = str(plot_cfg.get("x_param", "n_comps"))
    x_label = str(plot_cfg.get("x_label", "Latent dimension $d$"))
    selection_col = (
        str(plot_cfg.get("selection_metric", "knn")),
        str(plot_cfg.get("selection_stat", "mean")),
    )

    tie_break_metric = plot_cfg.get("tie_break_metric", "epoch_time(s)")
    tie_break_stat = plot_cfg.get("tie_break_stat", "mean")
    tie_break_col = None
    if tie_break_metric is not None and tie_break_stat is not None:
        tie_break_col = (str(tie_break_metric), str(tie_break_stat))

    figsize_cfg = plot_cfg.get("figsize", [6, 4])
    figsize = (float(figsize_cfg[0]), float(figsize_cfg[1]))
    dpi = int(plot_cfg.get("dpi", 300))

    logger.info(f"Loading tuning scores from {snakemake.input.scores}")
    scores_df = _load_tune_scores(snakemake.input.scores)
    logger.info(f"Selection metric: {selection_col}; tie-break: {tie_break_col}")

    summary_df = _select_best_rows_per_x(
        scores_df,
        x_param=x_param,
        fixed_params=dict(plot_cfg.get("fixed_params", {})),
        selection_col=selection_col,
        tie_break_col=tie_break_col,
        logger=logger,
    )

    summary_path = Path(snakemake.output.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Wrote: {summary_path}")

    _render_line_plot(
        summary_df,
        x_param=x_param,
        y_metric="knn",
        out_path=snakemake.output.accuracy_plot,
        title=str(plot_cfg.get("accuracy_title", "Best $k$-NN Accuracy vs Latent Dimension")),
        ylabel=str(plot_cfg.get("accuracy_ylabel", "$k$-NN accuracy")),
        x_label=x_label,
        dpi=dpi,
        figsize=figsize,
        log_y=bool(plot_cfg.get("accuracy_log_y", False)),
    )
    logger.info(f"Wrote: {snakemake.output.accuracy_plot}")

    _render_line_plot(
        summary_df,
        x_param=x_param,
        y_metric="epoch_time(s)",
        out_path=snakemake.output.train_time_plot,
        title=str(plot_cfg.get("train_time_title", "Training Time vs Latent Dimension")),
        ylabel=str(plot_cfg.get("train_time_ylabel", "Train time (s)")),
        x_label=x_label,
        dpi=dpi,
        figsize=figsize,
        log_y=bool(plot_cfg.get("train_time_log_y", False)),
    )
    logger.info(f"Wrote: {snakemake.output.train_time_plot}")


run(snakemake)  # noqa: F821
