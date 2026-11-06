"""
Paired significance tests of our methods against every baseline (appendix, tab:paired_tests_all).

For each reference method (ours), baseline and metric, the paired differences are the
per-split score gaps on identical train/test splits, pooled over the datasets of the
group (datasets x splits). Two tests per comparison:

  wilcoxon  two-sided Wilcoxon signed-rank test on the pooled gaps. The training sets of
            the splits overlap, so the gaps are not independent and the p-values may be
            anti-conservative.
  lmm       linear mixed model gap ~ 1 with a random intercept per dataset (REML); the
            test is on the intercept. It models the dataset grouping, not the dependence
            from overlapping training sets.

Holm correction over all comparisons, separately for each metric and test. An LMM that
cannot be fitted (e.g. constant gaps) is reported without p-value and left out of the
correction.

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.scores                — scores_per_split.csv, aligned with the pairs
  snakemake.output.gaps                 — one paired gap per dataset, split and comparison
  snakemake.output.tests                — one row per comparison, metric and test
  snakemake.output.table                — LaTeX grids, one table per metric (appendix)
  snakemake.params.method_dataset_pairs — (method, dataset) per input
  snakemake.params.dataset_names        — datasets of the group
  snakemake.params.references           — our methods
  snakemake.params.baselines            — compared methods, in table order
  snakemake.params.metrics              — metric columns of scores_per_split.csv
  snakemake.params.alpha                — family-wise level after Holm correction
  snakemake.params.display_names        — paper-facing method names
  snakemake.params.group                — dataset group (all / scRNA / synth)
  snakemake.log[0]
"""

import sys
import warnings

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from significance import _load_split_frames, collect_paired_gaps  # noqa: E402
from _utils import setup_logging  # noqa: E402
from _benchmark_display import section_for_method  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

METRIC_LABELS = {"knn": "KNN", "ari": "ARI (average linkage)", "ari_complete": "ARI (complete linkage)",
                 "silhouette": "silhouette width"}


def wilcoxon_row(gaps: np.ndarray) -> dict:
    result = stats.wilcoxon(gaps, zero_method="wilcox", alternative="two-sided", method="auto")
    return {"test": "wilcoxon", "statistic": result.statistic, "p_value": result.pvalue}


def lmm_row(frame: pd.DataFrame) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = smf.mixedlm("gap ~ 1", data=frame, groups=frame["dataset"]).fit(reml=True, method="powell")
        except np.linalg.LinAlgError as error:
            # e.g. constant gaps (both methods perfect on every split), or too few datasets
            return {"test": "lmm", "p_value": np.nan, "converged": False, "warnings": f"fit failed: {error}"}
    ci_low, ci_high = fit.conf_int(alpha=0.05).loc["Intercept"]
    return {
        "test": "lmm",
        "estimate": fit.params["Intercept"],
        "standard_error": fit.bse["Intercept"],
        "statistic": fit.tvalues["Intercept"],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": fit.pvalues["Intercept"],
        "converged": bool(fit.converged),
        "warnings": " | ".join(dict.fromkeys(str(item.message) for item in caught)),
    }


def run_tests(gaps: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for (reference, baseline, metric), frame in gaps.groupby(
        ["reference_method", "baseline_method", "metric"], sort=False
    ):
        values = frame["gap"].to_numpy(dtype=float)
        common = {
            "reference_method": reference,
            "baseline_method": baseline,
            "metric": metric,
            "n_datasets": frame["dataset"].nunique(),
            "n_paired_gaps": len(values),
            "mean_gap": values.mean(),
            "median_gap": np.median(values),
        }
        rows.append({**common, **wilcoxon_row(values)})
        rows.append({**common, **lmm_row(frame)})
    tests = pd.DataFrame(rows)
    tests["p_value_holm"] = np.nan
    for index in tests.groupby(["metric", "test"], sort=False).groups.values():
        finite = [i for i in index if np.isfinite(tests.loc[i, "p_value"])]
        if finite:
            tests.loc[finite, "p_value_holm"] = multipletests(tests.loc[finite, "p_value"], method="holm")[1]
    tests["significant_holm"] = tests["p_value_holm"] < alpha
    return tests


def format_p(p_value: float, significant: bool) -> str:
    if not np.isfinite(p_value):
        return "--"
    text = f"{p_value:.1e}" if p_value < 1e-3 else f"{p_value:.3f}"
    return f"\\textbf{{{text}}}" if significant else text


def latex_grids(tests: pd.DataFrame, references, baselines, metrics, display_names, group, alpha) -> str:
    """Two-column grid of Holm-adjusted p-value tables, one panel per metric."""
    blocks = [
        r"\begin{table}[ht!]",
        r"\centering",
        f"\\caption{{\\textbf{{Paired statistical comparisons}} ({group} datasets). "
        "Holm-adjusted $p$-values comparing each baseline with Ours (Exact) and Ours (Diag.). "
        f"Bold: $p_{{\\mathrm{{Holm}}}} < {alpha:g}$.}}",
        f"\\label{{tab:paired_tests_{group}}}",
    ]
    columns = [(test, ref) for test in ("wilcoxon", "lmm") for ref in references]
    n_refs = len(references)
    for index, metric in enumerate(metrics):
        subset = tests[tests["metric"] == metric].set_index(
            ["test", "reference_method", "baseline_method"]
        ).sort_index()
        counts = [
            f"{int(subset.loc[(test, ref), 'significant_holm'].sum())}/{len(baselines)}"
            for test, ref in columns
        ]
        lines = [
            f"\\subfloat[{METRIC_LABELS.get(metric, metric)}"
            f"\\label{{tab:paired_tests_{group}_{metric}}}]{{%",
            r"\begin{minipage}[t]{0.48\textwidth}",
            "\\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{1.2pt}",
            r"\renewcommand{\arraystretch}{1.1}",
            r"\resizebox{\linewidth}{!}{%",
            "\\begin{tabular}{l" + "c" * len(columns) + "}",
            "\\toprule",
            f" & \\multicolumn{{{n_refs}}}{{c}}{{Wilcoxon}} & \\multicolumn{{{n_refs}}}{{c}}{{LMM}}" + r" \\",
            f"\\cmidrule(lr){{2-{1 + n_refs}}} \\cmidrule(lr){{{2 + n_refs}-{1 + 2 * n_refs}}}",
            "Method & " + " & ".join(display_names.get(ref, ref) for _, ref in columns) + r" \\",
            "\\midrule",
            "Significant / total & " + " & ".join(counts) + r" \\",
            "\\midrule",
        ]
        for baseline in baselines:
            cells = []
            for test, reference in columns:
                row = subset.loc[(test, reference, baseline)]
                cells.append(format_p(row["p_value_holm"], row["significant_holm"]))
            lines.append(f"{display_names.get(baseline, baseline)} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{minipage}}%"]
        blocks.append("\n".join(lines))
        if index < len(metrics) - 1:
            blocks.append(r"\hfill" if index % 2 == 0 else r"\par\vspace{0.3cm}")
    blocks.append(r"\end{table}")
    return "\n".join(blocks) + "\n"


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    references = list(snakemake.params.references)
    baselines = list(snakemake.params.baselines)
    metrics = list(snakemake.params.metrics)
    alpha = float(snakemake.params.alpha)

    frames = _load_split_frames(snakemake.input.scores, snakemake.params.method_dataset_pairs)
    method_pairs = [(reference, baseline) for reference in references for baseline in baselines
                    if baseline != reference]
    gaps = collect_paired_gaps(
        frames, datasets=list(snakemake.params.dataset_names), method_pairs=method_pairs, metrics=metrics
    )
    tests = run_tests(gaps, alpha)
    gaps.to_csv(snakemake.output.gaps, index=False)
    tests.to_csv(snakemake.output.tests, index=False)

    display_names = dict(snakemake.params.display_names)
    baseline_names = [display_names.get(method, method) for method in baselines]
    for method, name in zip(baselines, baseline_names):
        if baseline_names.count(name) > 1:
            section = section_for_method(snakemake.config["methods"][method]).lower()
            display_names[method] = f"{name} ({section})"

    with open(snakemake.output.table, "w") as f:
        f.write(latex_grids(tests, references, baselines, metrics, display_names,
                            snakemake.params.group, alpha))

    counts = tests.groupby(["reference_method", "metric", "test"])["significant_holm"].sum().unstack("test")
    logger.info(f"{len(method_pairs)} comparisons x {len(metrics)} metrics; significant after Holm:\n{counts}")
    not_converged = tests[(tests["test"] == "lmm") & (~tests["converged"].astype(bool))]
    if len(not_converged):
        logger.warning(f"{len(not_converged)} LMM fits did not converge")


run(snakemake)  # noqa: F821
