"""Aggregate per-patient held-out accuracy against selected GMM component count.

Second step of the per-patient number-of-components check (appendix, GMM fitting).
Consumes the per-pair prediction tables from ``patient_predictions.py`` and emits:
  - a per-patient summary (accuracy, selected components, dataset-centered accuracy),
  - a within-stratum association test (weighted slope, permutation p, bootstrap CI),
  - a two-panel figure (one per method), and a LaTeX summary table.

The claim is observational: it tests whether patients assigned few/many components
perform systematically worse within their cohort and class, not the causal effect
of changing k (that is the fixed-k sweep in scripts/gmm_sensitivity/).

Inputs (snakemake):
  input.predictions  — list of per-pair prediction CSVs
  params.display_names, params.pcs
  output.patients, output.associations, output.plot, output.table
"""


import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _utils import setup_logging  # noqa: E402


def patient_summary(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, method, patient); dataset-centered accuracy."""
    g = obs.groupby(["dataset", "method", "patient_id"], as_index=False)
    summ = g.agg(
        true_label=("true_label", "first"),
        selected_n_components=("selected_n_components", "first"),
        n_test_splits=("correct", "size"),
        n_correct=("correct", "sum"),
    )
    summ["accuracy"] = summ["n_correct"] / summ["n_test_splits"]
    # dataset x method mean accuracy (exposure-weighted over patients' appearances)
    dm = obs.groupby(["dataset", "method"])["correct"].mean().rename("dataset_accuracy")
    summ = summ.merge(dm, on=["dataset", "method"], how="left")
    summ["centered_accuracy"] = summ["accuracy"] - summ["dataset_accuracy"]
    return summ


def _weighted_slope(x, y, w):
    w = np.asarray(w, float)
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    dx = x - xm
    denom = np.sum(w * dx * dx)
    return float(np.sum(w * dx * (y - ym)) / denom) if denom > 0 else np.nan


def association(summ: pd.DataFrame, method: str, *, n_perm: int, n_boot: int, seed: int) -> dict:
    """Within dataset x label stratum, weighted slope of centered accuracy on k."""
    d = summ[summ["method"] == method].copy()
    d = d[d["n_test_splits"] > 0]
    if d["selected_n_components"].nunique() < 2 or len(d) < 5:
        return {"method": method, "n_patients": int(len(d)), "slope_pp_per_comp": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "perm_p": np.nan}
    # residualize k and accuracy within dataset x label strata
    strata = d.groupby(["dataset", "true_label"])
    d["k_res"] = d["selected_n_components"] - strata["selected_n_components"].transform("mean")
    d["y_res"] = d["centered_accuracy"] - strata["centered_accuracy"].transform("mean")
    w = d["n_test_splits"].to_numpy(float)
    k = d["k_res"].to_numpy(float)
    y = d["y_res"].to_numpy(float)
    slope = _weighted_slope(k, y, w) * 100.0  # percentage points per component

    rng = np.random.default_rng(seed)
    codes = strata.ngroup().to_numpy()
    # permutation: shuffle k residuals within strata
    perm_slopes = np.empty(n_perm)
    for i in range(n_perm):
        kp = k.copy()
        for c in np.unique(codes):
            m = codes == c
            kp[m] = rng.permutation(kp[m])
        perm_slopes[i] = _weighted_slope(kp, y, w) * 100.0
    perm_p = float((np.sum(np.abs(perm_slopes) >= abs(slope)) + 1) / (n_perm + 1))
    # patient bootstrap CI
    boot = np.empty(n_boot)
    n = len(d)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = _weighted_slope(k[idx], y[idx], w[idx]) * 100.0
    ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
    return {"method": method, "n_patients": int(n), "slope_pp_per_comp": slope,
            "ci_low": float(ci_low), "ci_high": float(ci_high), "perm_p": perm_p}


def make_plot(summ, methods, display_names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(methods), figsize=(6.2 * len(methods), 4.6), squeeze=False)
    datasets = sorted(summ["dataset"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {ds: cmap(i % 10) for i, ds in enumerate(datasets)}
    for ax, method in zip(axes[0], methods):
        d = summ[summ["method"] == method]
        for ds in datasets:
            sub = d[d["dataset"] == ds]
            if sub.empty:
                continue
            ax.scatter(sub["selected_n_components"], 100 * sub["centered_accuracy"],
                       s=6 + 6 * sub["n_test_splits"], color=colors[ds], alpha=0.45,
                       edgecolors="none", label=ds)
        # exposure-weighted mean per component count
        if not d.empty:
            for kc, grp in d.groupby("selected_n_components"):
                m = np.average(100 * grp["centered_accuracy"], weights=grp["n_test_splits"])
                ax.plot(kc, m, "D", color="black", markersize=6, zorder=5)
        ax.axhline(0, ls="--", color="gray", lw=1)
        ax.set_xlabel("selected GMM components")
        ax.set_ylabel("patient accuracy − cohort accuracy (pp)")
        ax.set_title(display_names.get(method, method))
    axes[0][0].legend(title="cohort", fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_table(assoc: pd.DataFrame, display_names, path):
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Method & Patients & Slope (pp/comp), 95\% CI & Perm. $p$ \\", r"\midrule"]
    for _, r in assoc.iterrows():
        name = display_names.get(r["method"], r["method"])
        if np.isnan(r["slope_pp_per_comp"]):
            lines.append(f"{name} & {int(r['n_patients'])} & -- & -- \\\\")
        else:
            lines.append(
                f"{name} & {int(r['n_patients'])} & "
                f"{r['slope_pp_per_comp']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] & "
                f"{r['perm_p']:.3f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines))


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    pcs = dict(snakemake.params.pcs)
    display_names = dict(snakemake.params.display_names)

    obs = pd.concat([pd.read_csv(p) for p in snakemake.input.predictions], ignore_index=True)
    summ = patient_summary(obs)
    summ.to_csv(snakemake.output.patients, index=False)

    methods = list(display_names) or sorted(summ["method"].unique())
    assoc = pd.DataFrame([
        association(summ, m, n_perm=int(pcs.get("n_permutations", 9999)),
                    n_boot=int(pcs.get("n_bootstrap", 2000)), seed=int(pcs.get("random_seed", 42)))
        for m in methods
    ])
    assoc.to_csv(snakemake.output.associations, index=False)

    make_plot(summ, methods, display_names, snakemake.output.plot)
    make_table(assoc, display_names, snakemake.output.table)
    # coverage diagnostic
    cov = summ.groupby(["dataset", "method"])["n_test_splits"].agg(["min", "median", "max", "size"])
    logger.info(f"patients={len(summ)}; coverage(min/med/max appearances):\n{cov}")
    logger.info(f"associations:\n{assoc.to_string(index=False)}")


run(snakemake)  # noqa: F821
