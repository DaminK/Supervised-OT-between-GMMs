"""Plotting utilities for approximation error figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

# ── Display constants ────────────────────────────────────────────────────────

VARIANT_ORDER = ["None", "Frobenius", "MI", "Both"]

VARIANT_STYLES = {
    "None":      {"color": "#666666", "linestyle": "--", "linewidth": 5.5},
    "Frobenius": {"color": "#E69F00", "linestyle": "-",  "linewidth": 5.5},
    "MI":        {"color": "#4A90D9", "linestyle": "-",  "linewidth": 5.5},
    "Both":      {"color": "#5BAA5B", "linestyle": "-",  "linewidth": 5.5},
}

DISPLAY_RENAME = {
    "None": "None",
    "Frobenius": "Frobenius",
    "Mutual Information": "MI",
    "Both": "Both",
}

DS_DISPLAY_NAMES = {
    "kidney": "Kidney Injury",
    "COVID": "COVID-19",
    "breastcancer": "Breast Cancer",
    "myocardial_infarction": "Myocardial Infarction",
    "synth_recovery2_jitter": "Synthetic (Mean & Cov)",
    "synth_recovery2_jitter_onlycov": "Synthetic (Cov only)",
}

INIT_ORDER = ["orthonormal", "xavier", "random"]
INIT_LABELS = {"orthonormal": "Orthonormal init.", "xavier": "Xavier init.", "random": "Standard normal init."}
# Figures with several initializations: orthonormal in the saturated regularization colour, the
# others in lighter tints of it (share of white mixed in)
INIT_TINT = {"orthonormal": 0.0, "xavier": 0.35, "random": 0.55}
COMBINED_INITS = "combined"

# Error columns of the Figure 5 figures, each written to its own files
ERROR_COLUMNS = ["approx_error", "relative_approx_error"]
# relative error: sum_ij w_ij Delta_ij / sum_ij w_ij W_2,ij (exact distances; _approx_error.py)
ERROR_YLABELS = {"approx_error": r"$\Delta_{ij}(\mathbf{A})$", "relative_approx_error": r"$\Delta_{ij}(\mathbf{A})\,/\,W_2$"}


def init_sets(inits: list[str]) -> list[str]:
    """Initialization sets of the Figure 5 figures: each initialization alone, then all together."""
    return list(inits) + ([COMBINED_INITS] if len(inits) > 1 else [])


def resolve_init_set(init_set: str, inits: list[str]) -> list[str]:
    return list(inits) if init_set == COMBINED_INITS else [init_set]


def present_inits(df: pd.DataFrame) -> list:
    """Initializations in *df*, in INIT_ORDER; [None] without an init column."""
    if "init" not in df.columns:
        return [None]
    present = set(df["init"])
    return [i for i in INIT_ORDER if i in present] + sorted(present - set(INIT_ORDER))


def init_color(color, init: str | None, shaded: bool = True):
    """*color* lightened towards white by INIT_TINT[init]; unchanged unless *shaded*."""
    if not shaded or init is None:
        return color
    rgb = np.array(to_rgb(color))
    return tuple(rgb + (1 - rgb) * INIT_TINT.get(init, 0.0))


def init_legend_handles(inits: list[str], marker: str | None = None, markersize: float = 11) -> tuple[list, list[str]]:
    """Grey handles in each initialization's shade: lines, or points with *marker* (scatter plots)."""
    if marker is None:
        handles = [Line2D([0], [0], color=init_color("#555555", i), linestyle="-", linewidth=LEGEND_LINEWIDTH) for i in inits]
    else:
        handles = [Line2D([], [], color=init_color("#555555", i), linestyle="", marker=marker, markersize=markersize)
                   for i in inits]
    return handles, [INIT_LABELS.get(i, i) for i in inits]


# Figure 5 (a) and (b) share one fixed layout and are saved without tight cropping, so that their
# panels have the same size and position in the paper when both are included at the same width
# (inches; two square panels, room for the bound axis of (a) and one-line x labels)
FIG5_FIGSIZE = (11.6, 5.7)
FIG5_PANEL_SIDE = 3.7
FIG5_PANEL_LEFT = (1.45, 6.65)
FIG5_PANEL_BOTTOM = 1.35
FIG5_LEGEND_TOP = 0.62


FIG5_LEGEND_ROW = 1.9 * 19 / 72  # height of one legend row at LEGEND_FONT_SIZE (inches)


def fig5_axes(legend_rows: int = 1) -> tuple[plt.Figure, list]:
    """Figure and the two square panel axes of the shared Figure 5 layout; each further legend
    row adds height below the panels (the panels keep their size)."""
    extra = (legend_rows - 1) * FIG5_LEGEND_ROW
    width, height = FIG5_FIGSIZE[0], FIG5_FIGSIZE[1] + extra
    fig = plt.figure(figsize=(width, height))
    bottom = FIG5_PANEL_BOTTOM + extra
    axes = [
        fig.add_axes([left / width, bottom / height, FIG5_PANEL_SIDE / width, FIG5_PANEL_SIDE / height])
        for left in FIG5_PANEL_LEFT
    ]
    return fig, axes


def fig5_legend_y(fig=None) -> float:
    """Top of the legend below the panels, as a figure fraction of *fig* (default: one row)."""
    height = fig.get_figheight() if fig is not None else FIG5_FIGSIZE[1]
    return (FIG5_LEGEND_TOP + height - FIG5_FIGSIZE[1]) / height


def fig5_legend_rows(fig, rows: list[tuple[list, list[str]]], **legend_kwargs):
    """One framed, centred legend per row below the panels of the shared Figure 5 layout."""
    top = fig5_legend_y(fig)
    step = FIG5_LEGEND_ROW / fig.get_figheight()
    for i, (handles, labels) in enumerate(rows):
        legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, top - i * step),
                            ncol=len(handles), frameon=True, fontsize=LEGEND_FONT_SIZE, **legend_kwargs)
        fig.add_artist(legend)


def interleave_rows(rows: list[tuple[list, list[str]]]) -> tuple[list, list[str]]:
    """Handles and labels of several legend rows, padded to one width and ordered column by
    column (as matplotlib fills a multi-column legend), so that each list forms one row."""
    invisible = Line2D([], [], linestyle="None", marker="None")
    width = max(len(handles) for handles, _ in rows)
    padded = [(handles + [invisible] * (width - len(handles)), labels + [""] * (width - len(labels)))
              for handles, labels in rows]
    handles = [h for column in zip(*(handles for handles, _ in padded)) for h in column]
    labels = [lab for column in zip(*(labels for _, labels in padded)) for lab in column]
    return handles, labels


BOUND_LABEL = r"Bound $\|\mathbf{A}\|_F^2 \sqrt{\overline{\mathrm{MI}}}$"
BOUND_LEGEND_LABEL = r"Bound $\|\mathbf{A}\|_F^2 \sqrt{\overline{\mathrm{MI}}}$"

TICK_STEP = 10
LEFT_LABEL_PAD = 2
RIGHT_LABEL_PAD = 2
Y_TICK_PAD = 2
FONT_SIZE = 19
TITLE_SIZE = 20
TITLE_PAD = 9
X_TICK_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 19
ERROR_LINEWIDTH = 4.8
BOUND_LINEWIDTH = 4.3
LEGEND_LINEWIDTH = 4.8


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset_df(csv_base: Path, ds_name: str, init: str | list[str] | None = None) -> pd.DataFrame:
    """Load a single dataset's approx-error CSV (mean split only; the given initialization(s) if any)."""
    csv_path = csv_base / ds_name / "approx_error_epochs.csv"
    assert csv_path.exists(), f"Missing: {csv_path}"
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    df = df[df["split"] == "mean"].copy()
    if init is not None:
        df = df[df["init"].isin([init] if isinstance(init, str) else init)]
    df["display_name"] = df["display_name"].map(DISPLAY_RENAME)
    return df


ALL_METRICS = [
    "relative_approx_error", "bures_relative_error", "bures_fraction",
    "approx_error", "bures_gap",
    "fro_norm", "projected_mi_weighted", "projected_mi_total",
]


def load_group(
    csv_base: Path,
    datasets: list[str],
    agg: str = "arithmetic",
    init: str | list[str] | None = None,
) -> pd.DataFrame:
    """Load and aggregate multiple datasets into a single DataFrame (one curve per
    initialization and regularization).

    Parameters
    ----------
    init : str or list of str, optional
        Keep only these initializations (``init`` column of the per-dataset CSVs).
    agg : str
        ``"arithmetic"`` (default), ``"geometric"`` (exp-of-mean-log),
        or ``"normalized"`` (divide each dataset curve by its epoch-0 value,
        then arithmetic mean).
    """
    frames = [load_dataset_df(csv_base, ds, init=init) for ds in datasets]
    combined = pd.concat(frames, ignore_index=True)
    available = [c for c in ALL_METRICS if c in combined.columns]
    keys = (["init"] if "init" in combined.columns else []) + ["display_name", "epoch"]

    if agg == "geometric":
        def geomean(s):
            vals = s.clip(lower=1e-30)
            return np.exp(np.mean(np.log(vals)))
        agg_cols = {col: geomean for col in available}
    elif agg == "normalized":
        normed = []
        for df in frames:
            ndf = df.copy()
            for col in available:
                for idx in ndf.groupby(keys[:-1]).groups.values():
                    mask = ndf.index.isin(idx)
                    vals = ndf.loc[mask, col].values
                    if len(vals) > 0 and vals[0] != 0:
                        ndf.loc[mask, col] = vals / vals[0]
                    else:
                        ndf.loc[mask, col] = np.nan
            normed.append(ndf)
        combined = pd.concat(normed, ignore_index=True)
        agg_cols = {col: "mean" for col in available}
    else:
        agg_cols = {col: "mean" for col in available}

    return (
        combined
        .groupby(keys, sort=False)
        .agg(agg_cols)
        .reset_index()
    )


# ── Bound-proxy helpers ──────────────────────────────────────────────────────

def compute_bound_proxy(df: pd.DataFrame) -> pd.Series:
    """||A||_F^2 * sqrt(weighted MI) — matches the bound's non-constant part."""
    return df["fro_norm"] ** 2 * np.sqrt(df["projected_mi_weighted"].clip(lower=0))


def compute_bound_constant(errors: np.ndarray, proxy: np.ndarray) -> float:
    """Tightest C such that error <= C * proxy for all data points."""
    mask = (proxy > 0) & np.isfinite(errors) & np.isfinite(proxy) & (errors > 0)
    if not mask.any():
        return 1.0
    return float(np.max(errors[mask] / proxy[mask]))


def analytical_bound_constant(covariances: np.ndarray, weights: np.ndarray) -> float:
    """Constant C of the absolute-error bound, averaged over patient pairs.

    For patients s, t with component weights w and covariance traces tr, the pair's
    constant is 4 * sqrt(2) * sum_ij w_si w_tj max(tr_si, tr_tj).
    """
    traces = np.trace(covariances, axis1=-2, axis2=-1)
    pair_constants = []
    n_patients = covariances.shape[0]
    for s in range(n_patients):
        for t in range(s + 1, n_patients):
            active_s, active_t = weights[s] > 0, weights[t] > 0
            if not active_s.any() or not active_t.any():
                continue
            w_s = weights[s][active_s] / weights[s][active_s].sum()
            w_t = weights[t][active_t] / weights[t][active_t].sum()
            max_trace = np.maximum(traces[s][active_s][:, None], traces[t][active_t][None, :])
            pair_constants.append(4 * np.sqrt(2) * np.sum(w_s[:, None] * w_t[None, :] * max_trace))
    return float(np.mean(pair_constants)) if pair_constants else np.nan


def scale_right_axis(
    ax_left, ax_right, C: float,
    all_errors: np.ndarray, all_proxy: np.ndarray,
    log_scale: bool = False, pad: float = 0.05,
):
    """Scale right axis so left = C * right (shared origin, possibly off-screen)."""
    e_min, e_max = np.nanmin(all_errors), np.nanmax(all_errors)
    p_min, p_max = np.nanmin(all_proxy), np.nanmax(all_proxy)

    left_lo = min(e_min, C * p_min)
    left_hi = max(e_max, C * p_max)

    if log_scale:
        log_range = np.log10(left_hi) - np.log10(left_lo)
        left_lo = 10 ** (np.log10(left_lo) - pad * log_range)
        left_hi = 10 ** (np.log10(left_hi) + pad * log_range)
    else:
        span = left_hi - left_lo
        left_lo = left_lo - pad * span
        left_hi = left_hi + pad * span

    ax_left.set_ylim(left_lo, left_hi)
    ax_right.set_ylim(left_lo / C, left_hi / C)


def _compact_log_tick(value, _pos=None) -> str:
    """Format 1/2/5 log ticks compactly, e.g. 5x10^0 without wide math spacing."""
    if value <= 0 or not np.isfinite(value):
        return ""

    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10 ** exponent)
    candidate = round(mantissa * 2) / 2  # 1/2/5 ticks; narrow ranges: every half step
    if np.isclose(mantissa, candidate, rtol=1e-5, atol=1e-8):
        if candidate == 1.0:
            return rf"$10^{{{exponent}}}$"
        return rf"${candidate:g}{{\times}}10^{{{exponent}}}$"
    return ""


def _configure_y_ticks(ax, log_scale: bool = False, min_ticks: int = 3):
    """Use matplotlib locators/formatters that keep log ticks readable."""
    if log_scale:
        # 1/2/5 ticks; for narrow ranges with fewer than two of them, every integer mantissa,
        # then every half step
        low, high = sorted(ax.get_ylim())
        decades = np.arange(np.floor(np.log10(low)), np.ceil(np.log10(high)) + 1)
        for subs in ((1.0, 2.0, 5.0), tuple(np.arange(1.0, 10.0)), tuple(np.arange(1.0, 10.0, 0.5))):
            if np.sum([low <= s * 10 ** d <= high for d in decades for s in subs]) >= 2:
                break
        ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=subs, numticks=max(6, min_ticks + 2))
        )
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_compact_log_tick))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(1, 10), numticks=12))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=min_ticks))
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())


def _error_label(ylabel: str) -> str:
    return f"Error {ylabel}"


def _error_legend_label(ylabel: str) -> str:
    return f"Error {ylabel}"


def _show_right_label(panel_idx: int, n_panels: int) -> bool:
    return panel_idx == n_panels - 1


def _show_right_label_grid(panel_idx: int, n_panels: int, ncols: int) -> bool:
    row = panel_idx // ncols
    row_start = row * ncols
    row_len = min(ncols, n_panels - row_start)
    return (panel_idx % ncols) == row_len - 1


def _apply_axis_labels(
    ax,
    ax2,
    *,
    left_label: str,
    show_left: bool,
    show_right: bool,
    line_style_labels: bool = False,
):
    error_label = _error_label(left_label)
    bound_label = BOUND_LABEL
    if line_style_labels:
        error_label = f"━━━━  {error_label}"
        bound_label = f"━━ ━━  {bound_label}"

    ax.set_ylabel(error_label if show_left else "")
    ax.yaxis.label.set_size(FONT_SIZE)
    ax.yaxis.labelpad = LEFT_LABEL_PAD
    ax.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", pad=Y_TICK_PAD)

    if ax2 is None:
        return

    ax2.set_ylabel(bound_label if show_right else "")
    ax2.yaxis.label.set_size(FONT_SIZE)
    ax2.yaxis.labelpad = RIGHT_LABEL_PAD
    ax2.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax2.tick_params(axis="y", pad=Y_TICK_PAD)


# ── Legend builder ───────────────────────────────────────────────────────────

def _build_legend(
    fig,
    ylabel: str,
    show_bound: bool,
    layout: str = "horizontal",
    anchor_y: float = -0.01,
    inits: list | None = None,
):
    """Build a two-section bottom legend.

    Section 1 ("Regularization:"): color handles for each variant.
    Section 2 ("Quantity:"): solid = Error, dashed = Bound  (only when show_bound).
    With several *inits*, an "Initialization:" section shows their shades.

    Section headers are embedded as text legend entries with invisible
    handles so they stay inside the legend box regardless of figure layout.
    """
    _invis = Line2D([], [], linestyle="None", marker="None")

    # Regularization section
    reg_handles = [_invis] + [
        Line2D([0], [0], color=VARIANT_STYLES[n]["color"], linestyle="-", linewidth=LEGEND_LINEWIDTH)
        for n in VARIANT_ORDER
    ]
    reg_labels = ["Regularization:"] + list(VARIANT_ORDER)
    init_handles, init_labels = [], []
    if inits is not None and len(inits) > 1:
        init_handles, init_labels = init_legend_handles(inits)
        init_handles, init_labels = [_invis] + init_handles, ["Initialization:"] + init_labels

    if show_bound:
        if layout == "none":
            return None

        if layout == "regularization_only":
            if init_handles:
                # regularization in the first row, initialization in the second
                handles, labels = interleave_rows([(reg_handles, reg_labels), (init_handles, init_labels)])
                return fig.legend(
                    handles, labels,
                    loc="upper center", bbox_to_anchor=(0.5, anchor_y),
                    ncol=max(len(reg_handles), len(init_handles)), frameon=True,
                    fontsize=LEGEND_FONT_SIZE, handlelength=2.2, columnspacing=1.2,
                )
            return fig.legend(
                reg_handles, reg_labels,
                loc="upper center", bbox_to_anchor=(0.5, anchor_y),
                ncol=len(reg_handles), frameon=True,
                fontsize=LEGEND_FONT_SIZE, handlelength=2.2, columnspacing=1.2,
            )

        qty_handles = [_invis] + [
            Line2D([0], [0], color="black", linestyle="-", linewidth=LEGEND_LINEWIDTH),
            Line2D([0], [0], color="black", linestyle="--", linewidth=BOUND_LINEWIDTH),
        ]
        qty_labels = [
            r"$\bf{y-Axis}$",
            _error_legend_label(ylabel),
            BOUND_LEGEND_LABEL,
        ]

        if layout == "stacked":
            row_width = len(reg_handles)
            blank_count = row_width - len(qty_handles)
            top_handles = reg_handles
            top_labels = reg_labels
            bottom_handles = qty_handles + [_invis] * blank_count
            bottom_labels = qty_labels + [""] * blank_count
            all_handles = [
                handle
                for col in zip(top_handles, bottom_handles, strict=True)
                for handle in col
            ]
            all_labels = [
                label
                for col in zip(top_labels, bottom_labels, strict=True)
                for label in col
            ]
            return fig.legend(
                all_handles, all_labels,
                loc="upper center", bbox_to_anchor=(0.5, anchor_y),
                ncol=row_width, frameon=True,
                fontsize=LEGEND_FONT_SIZE, handlelength=2.2, columnspacing=1.2,
            )

        all_handles = reg_handles + qty_handles + init_handles
        all_labels = reg_labels + qty_labels + init_labels

        return fig.legend(
            all_handles, all_labels,
            loc="upper center", bbox_to_anchor=(0.5, anchor_y),
            ncol=len(all_handles), frameon=True,
            fontsize=LEGEND_FONT_SIZE, handlelength=2.2, columnspacing=1.2,
        )
    else:
        if layout == "none":
            return None

        return fig.legend(
            reg_handles + init_handles, reg_labels + init_labels,
            loc="upper center", bbox_to_anchor=(0.5, anchor_y),
            ncol=len(reg_handles) + len(init_handles), frameon=True,
            fontsize=LEGEND_FONT_SIZE, handlelength=2.2,
        )


# ── Core plotting ────────────────────────────────────────────────────────────

def _curves(df: pd.DataFrame):
    """(curve, colour, regularization) per initialization and regularization, sorted by epoch;
    with several initializations, each in its INIT_TINT shade of the regularization colour."""
    inits = present_inits(df)
    for init in inits:
        for name in VARIANT_ORDER:
            sub = df[df["display_name"] == name]
            if init is not None:
                sub = sub[sub["init"] == init]
            if sub.empty:
                continue
            yield sub.sort_values("epoch"), init_color(VARIANT_STYLES[name]["color"], init, shaded=len(inits) > 1), name

def plot_error_panels(
    panel_dfs: list[pd.DataFrame],
    panel_titles: list[str],
    error_col: str,
    ylabel: str,
    suptitle: str,
    log_scale: bool = False,
    show_bound: bool = False,
    figsize: tuple[float, float] | None = None,
    square_axes: bool = False,
    legend_layout: str = "horizontal",
    legend_y: float = -0.01,
    line_style_axis_labels: bool = False,
    outer_axis_labels: bool = False,
    wspace: float | None = None,
    fig5_layout: bool = False,
) -> tuple[plt.Figure, list[float] | None]:
    """Plot 1×N panels of error curves with optional bound proxy overlay.

    *suptitle* is NOT rendered on the figure (caller should print it). With *fig5_layout*
    (two panels), the shared fixed Figure 5 layout (fig5_axes) replaces figsize, square_axes,
    wspace and legend_y; save such figures without tight cropping.

    Returns (fig, panel_Cs) where panel_Cs is a list of empirical C values
    (one per panel) when show_bound=True, else None.
    """
    n = len(panel_dfs)
    shown_inits = present_inits(pd.concat(panel_dfs, ignore_index=True))
    if fig5_layout:
        fig, axes = fig5_axes(legend_rows=2 if len(shown_inits) > 1 else 1)
        square_axes = False
    else:
        if figsize is None:
            width_per_panel = 6.0 if (show_bound and n == 2) else 5.0
            figsize = (width_per_panel * n, 3.5)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]
        panel_wspace = wspace if wspace is not None else (0.55 if (show_bound and n == 2) else 0.35)
        fig.subplots_adjust(wspace=panel_wspace)

    panel_Cs = [] if show_bound else None

    for panel_idx, (ax, df, title) in enumerate(zip(axes, panel_dfs, panel_titles)):
        df = df.copy()
        ax2 = None
        if show_bound:
            df["bound_proxy"] = compute_bound_proxy(df)
            ax2 = ax.twinx()

        if square_axes:
            ax.set_box_aspect(1)
            if ax2 is not None:
                ax2.set_box_aspect(1)

        all_errors, all_proxy = [], []
        for sub, color, name in _curves(df):
            if error_col not in sub.columns:
                continue
            epochs = sub["epoch"].to_numpy()
            err = sub[error_col].to_numpy()
            ax.plot(epochs, err, color=color, linestyle="-", linewidth=ERROR_LINEWIDTH, label=name)
            all_errors.append(err)

            if show_bound:
                bp = sub["bound_proxy"].to_numpy()
                ax2.plot(epochs, bp, color=color, linestyle="--", linewidth=BOUND_LINEWIDTH)
                all_proxy.append(bp)

        max_epoch = int(df["epoch"].max())
        tick_end = max_epoch
        ax.set_xticks(list(range(0, tick_end + 1, TICK_STEP)))
        ax.set_xlabel("Training Iterations", fontsize=FONT_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE, pad=TITLE_PAD)
        _apply_axis_labels(
            ax,
            ax2,
            left_label=ylabel,
            show_left=(not outer_axis_labels or panel_idx == 0),
            show_right=(ax2 is not None and (not outer_axis_labels or panel_idx == n - 1)),
            line_style_labels=line_style_axis_labels,
        )

        if log_scale:
            ax.set_yscale("log")
            if ax2 is not None:
                ax2.set_yscale("log")

        if show_bound and all_errors and all_proxy:
            all_e = np.concatenate(all_errors)
            all_p = np.concatenate(all_proxy)
            C = compute_bound_constant(all_e, all_p)
            panel_Cs.append(C)
            fig.canvas.draw()
            scale_right_axis(ax, ax2, C, all_e, all_p, log_scale=log_scale)

        _configure_y_ticks(ax, log_scale)
        if ax2 is not None:
            _configure_y_ticks(ax2, log_scale)

    if fig5_layout:
        # regularization row; with several inits, their shades in a second row
        invisible = Line2D([], [], linestyle="None", marker="None")
        rows = [([invisible] + [Line2D([0], [0], color=VARIANT_STYLES[n]["color"], linewidth=LEGEND_LINEWIDTH)
                                for n in VARIANT_ORDER], ["Regularization:"] + list(VARIANT_ORDER))]
        if len(shown_inits) > 1:
            init_handles, init_labels = init_legend_handles(shown_inits)
            rows.append(([invisible] + init_handles, ["Initialization:"] + init_labels))
        fig5_legend_rows(fig, rows, handlelength=2.2, columnspacing=1.2)
    else:
        _build_legend(fig, ylabel, show_bound, layout=legend_layout, anchor_y=legend_y, inits=shown_inits)
    return fig, panel_Cs


def plot_per_dataset_grid(
    per_ds_dfs: dict[str, pd.DataFrame],
    ds_order: list[str],
    error_col: str,
    ylabel: str,
    suptitle: str,
    log_scale: bool = False,
    show_bound: bool = False,
    ncols: int = 3,
    legend_layout: str = "horizontal",
) -> tuple[plt.Figure, dict[str, float] | None]:
    """Plot a 2×3 (or similar) grid with one subplot per dataset.

    *suptitle* is NOT rendered on the figure (caller should print it).
    *legend_layout* is passed to ``_build_legend`` ("stacked": regularization and y-axis key in two rows).
    """
    n_ds = len(ds_order)
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows))
    axes_flat = axes.flatten()
    dataset_Cs = {} if show_bound else None
    fig.subplots_adjust(wspace=0.45 if show_bound else 0.3, hspace=0.35)

    for i, ds_name in enumerate(ds_order):
        ax = axes_flat[i]
        df = per_ds_dfs[ds_name].copy()
        if error_col not in df.columns:
            ax.set_visible(False)
            continue

        ax2 = None
        if show_bound:
            df["bound_proxy"] = compute_bound_proxy(df)
            ax2 = ax.twinx()

        all_errors, all_proxy = [], []
        for sub, color, name in _curves(df):
            epochs = sub["epoch"].to_numpy()
            err = sub[error_col].to_numpy()
            ax.plot(epochs, err, color=color, linestyle="-", linewidth=ERROR_LINEWIDTH, label=name)
            all_errors.append(err)

            if show_bound:
                bp = sub["bound_proxy"].to_numpy()
                ax2.plot(epochs, bp, color=color, linestyle="--", linewidth=BOUND_LINEWIDTH)
                all_proxy.append(bp)

        max_epoch = int(df["epoch"].max())
        tick_end = max_epoch
        ax.set_xticks(list(range(0, tick_end + 1, TICK_STEP)))
        ax.set_xlabel("Training Iterations", fontsize=FONT_SIZE)
        _apply_axis_labels(
            ax,
            ax2,
            left_label=ylabel,
            show_left=True,
            show_right=ax2 is not None,
        )
        ax.set_title(DS_DISPLAY_NAMES.get(ds_name, ds_name), fontsize=TITLE_SIZE, pad=TITLE_PAD)

        if log_scale:
            ax.set_yscale("log")
            if ax2 is not None:
                ax2.set_yscale("log")

        if show_bound and all_errors and all_proxy:
            all_e = np.concatenate(all_errors)
            all_p = np.concatenate(all_proxy)
            C = compute_bound_constant(all_e, all_p)
            dataset_Cs[ds_name] = C
            fig.canvas.draw()
            scale_right_axis(ax, ax2, C, all_e, all_p, log_scale=log_scale)

        _configure_y_ticks(ax, log_scale)
        if ax2 is not None:
            _configure_y_ticks(ax2, log_scale)

    for j in range(n_ds, len(axes_flat)):
        axes_flat[j].set_visible(False)

    _build_legend(fig, ylabel, show_bound, layout=legend_layout,
                  inits=present_inits(pd.concat(per_ds_dfs.values(), ignore_index=True)))
    fig.tight_layout()
    return fig, dataset_Cs
