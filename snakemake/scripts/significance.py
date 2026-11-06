"""Prototype significance analyses from persisted benchmark split scores.

Dataset-level tests treat cohorts as independent units. Within each cohort, the
Nadeau--Bengio corrected resampled t interval accounts approximately for the
overlap between repeated train/test splits. A dependence-ignoring pooled-split
Wilcoxon is available only as an opt-in diagnostic and is never included in the
primary table.
"""

from __future__ import annotations

import itertools
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SPLIT_IDENTITY_COLUMNS = (
    "schema",
    "random_seed",
    "split_strategy",
    "requested_train_size",
    "requested_test_size",
    "n_distributions",
    "n_train",
    "n_test",
    "train_idx",
    "test_idx",
    "ordered_patient_id_sha256",
    "ordered_label_sha256",
    "train_patient_ids",
    "test_patient_ids",
    "train_labels",
    "test_labels",
)


def _as_method_pairs(
    method_names: Sequence[str],
    display_names: Mapping[str, str],
    config: Mapping,
) -> list[tuple[str, str]]:
    configured = config.get("method_pairs")
    if configured:
        pairs: list[tuple[str, str]] = []
        for item in configured:
            if isinstance(item, Mapping):
                pairs.append((str(item["reference"]), str(item["baseline"])))
            else:
                reference, baseline = item
                pairs.append((str(reference), str(baseline)))
        unknown = sorted(
            {method for pair in pairs for method in pair} - set(method_names)
        )
        if unknown:
            raise ValueError(f"Unknown methods in significance.method_pairs: {unknown}")
        if any(reference == baseline for reference, baseline in pairs):
            raise ValueError("A significance method pair cannot compare a method to itself.")
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate entries in significance.method_pairs.")
        return pairs

    if config.get("require_explicit_pairs", True):
        raise ValueError(
            "Configure significance.method_pairs explicitly before running the "
            "analysis; data-driven comparison inference is disabled."
        )

    # Legacy exploratory fallback. Never use this branch for reported results.
    references = list(config.get("reference_methods", []))
    if not references:
        references = [
            method
            for method in method_names
            if "ours" in display_names.get(method, method).lower()
        ]
    baselines = list(config.get("baseline_methods", []))
    if not baselines:
        baselines = [method for method in method_names if method not in references]
    if not references or not baselines:
        raise ValueError(
            "Could not infer significance comparisons. Configure method_pairs or "
            "reference_methods/baseline_methods."
        )
    return list(itertools.product(references, baselines))


def _load_split_frames(
    paths: Sequence[str | Path],
    method_dataset_pairs: Sequence[Sequence[str]],
) -> dict[tuple[str, str], pd.DataFrame]:
    if len(paths) != len(method_dataset_pairs):
        raise ValueError("Input paths do not align with method_dataset_pairs.")

    frames = {}
    for path, pair in zip(paths, method_dataset_pairs):
        method, dataset = pair
        frame = pd.read_csv(path)
        if "split_idx" not in frame or frame["split_idx"].duplicated().any():
            raise ValueError(f"Invalid or duplicate split_idx values in {path}.")
        frames[(str(dataset), str(method))] = frame.sort_values(
            "split_idx"
        ).reset_index(drop=True)
    return frames


def _normalized_identity(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _assert_same_splits(
    reference: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    dataset: str,
    reference_method: str,
    baseline_method: str,
) -> pd.DataFrame:
    merged = reference.merge(
        baseline,
        on="split_idx",
        how="outer",
        suffixes=("_reference", "_baseline"),
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError(
            f"Split indices differ for {dataset}: {reference_method} vs {baseline_method}."
        )

    for column in SPLIT_IDENTITY_COLUMNS:
        left = f"{column}_reference"
        right = f"{column}_baseline"
        if left not in merged or right not in merged:
            continue
        matches = _normalized_identity(merged[left]) == _normalized_identity(
            merged[right]
        )
        if not matches.all():
            bad_splits = merged.loc[~matches, "split_idx"].tolist()
            raise ValueError(
                f"Split identity mismatch for {dataset}, {reference_method} vs "
                f"{baseline_method}, column={column}, splits={bad_splits}."
            )
    return merged.drop(columns="_merge")


def collect_paired_gaps(
    frames: Mapping[tuple[str, str], pd.DataFrame],
    *,
    datasets: Sequence[str],
    method_pairs: Sequence[tuple[str, str]],
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Validate split identity and return one paired difference per split."""
    rows = []
    for dataset in datasets:
        for reference_method, baseline_method in method_pairs:
            try:
                reference = frames[(dataset, reference_method)]
                baseline = frames[(dataset, baseline_method)]
            except KeyError as exc:
                raise ValueError(f"Missing split scores for {exc.args[0]}.") from exc
            merged = _assert_same_splits(
                reference,
                baseline,
                dataset=dataset,
                reference_method=reference_method,
                baseline_method=baseline_method,
            )
            for metric in metrics:
                ref_column = f"{metric}_reference"
                base_column = f"{metric}_baseline"
                if ref_column not in merged or base_column not in merged:
                    raise ValueError(
                        f"Metric {metric!r} is unavailable for {dataset}: "
                        f"{reference_method} vs {baseline_method}."
                    )
                for row in merged.itertuples(index=False):
                    reference_value = float(getattr(row, ref_column))
                    baseline_value = float(getattr(row, base_column))
                    rows.append(
                        {
                            "dataset": dataset,
                            "reference_method": reference_method,
                            "baseline_method": baseline_method,
                            "metric": metric,
                            "split_idx": int(row.split_idx),
                            "n_train": int(row.n_train_reference),
                            "n_test": int(row.n_test_reference),
                            "reference_score": reference_value,
                            "baseline_score": baseline_value,
                            "gap": reference_value - baseline_value,
                        }
                    )
    return pd.DataFrame(rows)


def corrected_resampled_t(
    gaps: Iterable[float],
    *,
    n_train: Iterable[int],
    n_test: Iterable[int],
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Nadeau--Bengio corrected resampled t-test for paired split gaps."""
    values = np.asarray(list(gaps), dtype=float)
    train_sizes = np.asarray(list(n_train), dtype=float)
    test_sizes = np.asarray(list(n_test), dtype=float)
    if len(values) < 2:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "correction": np.nan,
        }
    if np.any(train_sizes <= 0) or np.any(test_sizes <= 0):
        raise ValueError("Corrected resampled t-test requires positive split sizes.")

    mean_gap = float(values.mean())
    variance = float(values.var(ddof=1))
    correction = float(1.0 / len(values) + np.mean(test_sizes / train_sizes))
    standard_error = math.sqrt(correction * variance)
    if standard_error == 0:
        statistic = 0.0 if mean_gap == 0 else math.copysign(math.inf, mean_gap)
        p_value = 1.0 if mean_gap == 0 else 0.0
        return {
            "statistic": statistic,
            "p_value": p_value,
            "ci_low": mean_gap,
            "ci_high": mean_gap,
            "correction": correction,
        }

    degrees_freedom = len(values) - 1
    statistic = mean_gap / standard_error
    p_value = 2 * stats.t.sf(abs(statistic), df=degrees_freedom)
    alpha = 1 - confidence_level
    critical = stats.t.ppf(1 - alpha / 2, df=degrees_freedom)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "ci_low": float(mean_gap - critical * standard_error),
        "ci_high": float(mean_gap + critical * standard_error),
        "correction": correction,
    }


def _wilcoxon(values: np.ndarray) -> tuple[float, float]:
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 0.0, 1.0
    result = stats.wilcoxon(
        values, zero_method="wilcox", alternative="two-sided", method="auto"
    )
    return float(result.statistic), float(result.pvalue)


def _sign_flip_mean(
    values: np.ndarray, *, rng: np.random.Generator, max_exact: int = 20
) -> tuple[float, float, int, float]:
    """Test a mean gap by flipping independent cohort-level effect signs."""
    nonzero = values[values != 0]
    n_nonzero = len(nonzero)
    observed = abs(float(values.mean()))
    if n_nonzero == 0:
        return observed, 1.0, 0, 1.0
    min_attainable_p = min(1.0, 2.0 / (2**n_nonzero))

    if n_nonzero <= max_exact:
        statistics = []
        for signs in itertools.product((-1.0, 1.0), repeat=n_nonzero):
            signed_mean = float(
                np.sum(nonzero * np.asarray(signs)) / len(values)
            )
            statistics.append(abs(signed_mean))
        statistics = np.asarray(statistics)
    else:
        signs = rng.choice((-1.0, 1.0), size=(100_000, n_nonzero))
        statistics = np.abs(
            (signs * nonzero).sum(axis=1) / len(values)
        )
    tolerance = np.finfo(float).eps * max(1.0, observed) * 8
    p_value = float(np.mean(statistics >= observed - tolerance))
    return observed, p_value, n_nonzero, min_attainable_p


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm family-wise-error correction while preserving NaN entries."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    running = 0.0
    n_tests = len(order)
    for rank, index in enumerate(order):
        candidate = (n_tests - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def analyze_gaps(
    gaps: pd.DataFrame,
    *,
    random_seed: int = 42,
    confidence_level: float = 0.95,
    include_pooled_split_sensitivity: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cross-dataset prototype tests and per-dataset diagnostics."""
    rng = np.random.default_rng(random_seed)
    aggregate_rows = []
    dataset_rows = []
    keys = ["reference_method", "baseline_method", "metric"]

    for key, comparison in gaps.groupby(keys, sort=True):
        reference_method, baseline_method, metric = key
        dataset_means = comparison.groupby("dataset", sort=True)["gap"].mean()
        dataset_values = dataset_means.to_numpy(dtype=float)
        (
            sign_flip_statistic,
            sign_flip_p_value,
            n_nonzero_datasets,
            min_attainable_p,
        ) = _sign_flip_mean(dataset_values, rng=rng)

        statistic, p_value = _wilcoxon(dataset_values)
        common = {
            "reference_method": reference_method,
            "baseline_method": baseline_method,
            "metric": metric,
            "n_datasets": len(dataset_values),
            "n_split_pairs": len(comparison),
            "mean_gap": float(dataset_values.mean()),
            "median_gap": float(np.median(dataset_values)),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_nonzero_datasets": n_nonzero_datasets,
            "min_attainable_p": min_attainable_p,
        }
        aggregate_rows.append(
            {
                **common,
                "analysis": "wilcoxon_dataset_means",
                "analysis_unit": "dataset",
                "dependence_handling": "splits averaged within dataset",
                "statistic": statistic,
                "p_value": p_value,
            }
        )

        aggregate_rows.append(
            {
                **common,
                "analysis": "sign_flip_dataset_means",
                "analysis_unit": "dataset",
                "dependence_handling": "splits averaged within dataset",
                "statistic": sign_flip_statistic,
                "p_value": sign_flip_p_value,
            }
        )

        if include_pooled_split_sensitivity:
            pooled_values = comparison["gap"].to_numpy(dtype=float)
            statistic, p_value = _wilcoxon(pooled_values)
            aggregate_rows.append(
                {
                    **common,
                    "mean_gap": float(pooled_values.mean()),
                    "median_gap": float(np.median(pooled_values)),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "analysis": "wilcoxon_pooled_splits_sensitivity_only",
                    "analysis_unit": "split",
                    "dependence_handling": "none; overlapping splits are not independent",
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )

        for dataset, dataset_frame in comparison.groupby("dataset", sort=True):
            corrected = corrected_resampled_t(
                dataset_frame["gap"],
                n_train=dataset_frame["n_train"],
                n_test=dataset_frame["n_test"],
                confidence_level=confidence_level,
            )
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "reference_method": reference_method,
                    "baseline_method": baseline_method,
                    "metric": metric,
                    "n_splits": len(dataset_frame),
                    "reference_mean": float(dataset_frame["reference_score"].mean()),
                    "baseline_mean": float(dataset_frame["baseline_score"].mean()),
                    "mean_gap": float(dataset_frame["gap"].mean()),
                    "median_gap": float(dataset_frame["gap"].median()),
                    "corrected_t_statistic": corrected["statistic"],
                    "corrected_t_p_value": corrected["p_value"],
                    "corrected_t_ci_low": corrected["ci_low"],
                    "corrected_t_ci_high": corrected["ci_high"],
                    "variance_correction": corrected["correction"],
                }
            )

    aggregate = pd.DataFrame(aggregate_rows)
    for _, indices in aggregate.groupby(
        ["analysis", "metric"], sort=False
    ).groups.items():
        aggregate.loc[indices, "p_value_holm"] = holm_adjust(
            aggregate.loc[indices, "p_value"]
        )
    per_dataset = pd.DataFrame(dataset_rows)
    for _, indices in per_dataset.groupby("metric", sort=False).groups.items():
        per_dataset.loc[indices, "corrected_t_p_value_holm"] = holm_adjust(
            per_dataset.loc[indices, "corrected_t_p_value"]
        )
    return aggregate, per_dataset


def _write_latex_table(
    per_dataset: pd.DataFrame,
    output_path: str | Path,
    display_names: Mapping[str, str],
) -> None:
    primary = per_dataset.copy()
    primary["Dataset"] = primary["dataset"]
    primary["Comparison"] = [
        f"{display_names.get(ref, ref)} $-$ {display_names.get(base, base)}"
        for ref, base in zip(primary["reference_method"], primary["baseline_method"])
    ]
    primary["Metric"] = primary["metric"].str.upper()
    primary["Paired gap (corrected 95% CI)"] = [
        f"{mean:.3f} [{low:.3f}, {high:.3f}]"
        for mean, low, high in zip(
            primary["mean_gap"],
            primary["corrected_t_ci_low"],
            primary["corrected_t_ci_high"],
        )
    ]
    primary["Corrected $p$"] = primary["corrected_t_p_value"]
    primary["Holm $p$"] = primary["corrected_t_p_value_holm"]
    table = primary[
        [
            "Dataset",
            "Comparison",
            "Metric",
            "Paired gap (corrected 95% CI)",
            "Corrected $p$",
            "Holm $p$",
        ]
    ]
    Path(output_path).write_text(
        table.to_latex(
            index=False, escape=False, float_format=lambda value: f"{value:.3g}"
        )
    )


def run(snakemake) -> None:
    sys.path.insert(0, snakemake.scriptdir)
    from _utils import setup_logging

    logger = setup_logging(snakemake.log[0])
    config = dict(snakemake.params.config)
    method_names = list(snakemake.params.method_names)
    datasets = list(snakemake.params.dataset_names)
    display_names = dict(snakemake.params.display_names)
    method_pairs = _as_method_pairs(method_names, display_names, config)
    metrics = list(config.get("metrics", ["knn", "ari"]))

    frames = _load_split_frames(
        snakemake.input.scores, snakemake.params.method_dataset_pairs
    )
    gaps = collect_paired_gaps(
        frames,
        datasets=datasets,
        method_pairs=method_pairs,
        metrics=metrics,
    )
    aggregate, per_dataset = analyze_gaps(
        gaps,
        random_seed=int(config.get("random_seed", 42)),
        confidence_level=float(config.get("confidence_level", 0.95)),
        include_pooled_split_sensitivity=bool(
            config.get("include_pooled_split_sensitivity", False)
        ),
    )

    for output in snakemake.output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(snakemake.output.gaps, index=False)
    aggregate.to_csv(snakemake.output.summary, index=False)
    per_dataset.to_csv(snakemake.output.per_dataset, index=False)
    _write_latex_table(per_dataset, snakemake.output.table, display_names)
    logger.info("Wrote significance prototypes for %d method pairs", len(method_pairs))


if "snakemake" in globals():
    run(globals()["snakemake"])
