"""Build a validated, serialization-friendly table of repeated-split scores."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


SPLIT_SCORE_SCHEMA = "snakemake.scores_per_split.v1"


def _json_array(values: Sequence) -> str:
    """Serialize a one-dimensional sequence deterministically for a CSV cell."""
    normalized = []
    for value in values:
        if isinstance(value, np.generic):
            value = value.item()
        normalized.append(value)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _string_array(values: Sequence | None) -> list[str] | None:
    if values is None:
        return None
    return [str(value) for value in values]


def _ordered_fingerprint(values: Sequence[str] | None) -> str | None:
    if values is None:
        return None
    payload = _json_array(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dataset_sequence(dataset, attribute: str) -> list | None:
    try:
        values = getattr(dataset, attribute)
    except (AttributeError, NotImplementedError):
        return None
    if values is None:
        return None
    return np.asarray(values).tolist()


def _dataset_patient_ids(dataset, n_distributions: int) -> list[str] | None:
    """Return one stable patient ID per distribution, in dataset order.

    AnnData-backed datasets are constructed in sorted patient-ID order. Prefer
    that source when it has exactly one ID per distribution. In particular,
    ``AnnData_TripletDataset.patient_labels`` cannot currently reconstruct IDs
    after empirical supports have been replaced by parametric GMM components:
    it then slices the cell-level index mask using component counts and may
    repeat the first patient ID for every distribution.

    The dataset property remains the fallback for representations that can
    legitimately omit patients (for example, raw empirical distributions below
    a minimum cell count).
    """
    adata = getattr(dataset, "adata", None)
    patient_col = getattr(dataset, "patient_col", None)
    if (
        adata is not None
        and patient_col is not None
        and patient_col in adata.obs.columns
    ):
        obs_ids = _string_array(
            np.unique(adata.obs[patient_col].to_numpy()).tolist()
        )
        if obs_ids is not None and len(obs_ids) == n_distributions:
            return obs_ids

    property_ids = _string_array(_dataset_sequence(dataset, "patient_labels"))
    if property_ids is None:
        return None
    if len(property_ids) != n_distributions:
        raise ValueError("Patient IDs do not align with the dataset ordering.")
    if len(set(property_ids)) != n_distributions:
        raise ValueError(
            "Patient IDs are not unique per distribution; the dataset does not "
            "expose a reliable ordered patient identity."
        )
    return property_ids


def _validate_aggregate_means(
    scores_df: pd.DataFrame, per_split: Mapping[str, Sequence[float]]
) -> None:
    """Ensure the persisted values reproduce every matching aggregate mean."""
    for metric, values in per_split.items():
        column = (metric, "mean")
        if column not in scores_df.columns:
            continue
        aggregate_mean = float(scores_df[column].iloc[0])
        split_mean = float(np.mean(np.asarray(values, dtype=float)))
        if not np.isclose(aggregate_mean, split_mean, equal_nan=True):
            raise ValueError(
                f"Per-split {metric!r} mean ({split_mean}) does not match "
                f"the aggregate score ({aggregate_mean})."
            )


def build_split_scores_frame(
    scores_df: pd.DataFrame,
    dataset,
    *,
    random_seed: int | None,
    split_config: Mapping | None = None,
) -> pd.DataFrame:
    """Convert ``scores_df.attrs`` into one validated row per evaluation split.

    Besides the metric values, the output retains train/test positions, stable
    patient identifiers when the dataset exposes them, label composition, and
    realized split sizes. This supports several later uncertainty analyses
    without committing the benchmark step to one statistical test.
    """
    attrs = scores_df.attrs
    required = {"per_split", "split_train_idx", "split_test_idx"}
    missing = sorted(required - attrs.keys())
    if missing:
        raise ValueError(
            "ggml_ot did not return required split metadata: " + ", ".join(missing)
        )

    per_split = dict(attrs["per_split"])
    train_splits = list(attrs["split_train_idx"])
    test_splits = list(attrs["split_test_idx"])
    n_splits = int(attrs.get("n_splits", len(test_splits)))

    if not per_split:
        raise ValueError("ggml_ot returned an empty per_split score mapping.")
    if len(train_splits) != n_splits or len(test_splits) != n_splits:
        raise ValueError(
            "Split metadata length mismatch: "
            f"n_splits={n_splits}, train={len(train_splits)}, test={len(test_splits)}."
        )
    bad_metrics = {
        metric: len(values)
        for metric, values in per_split.items()
        if len(values) != n_splits
    }
    if bad_metrics:
        raise ValueError(
            f"Per-split metric lengths do not match n_splits={n_splits}: {bad_metrics}"
        )

    _validate_aggregate_means(scores_df, per_split)

    distribution_labels = _string_array(
        _dataset_sequence(dataset, "distribution_labels_str")
    )
    n_distributions = len(dataset.distribution_labels)
    if distribution_labels is not None and len(distribution_labels) != n_distributions:
        raise ValueError("Distribution labels do not align with the dataset ordering.")

    patient_ids = _dataset_patient_ids(dataset, n_distributions)

    split_config = dict(split_config or {})
    fixed = {
        "schema": SPLIT_SCORE_SCHEMA,
        "random_seed": random_seed,
        "split_strategy": split_config.get("split_strategy", "auto"),
        "requested_train_size": split_config.get("train_size"),
        "requested_test_size": split_config.get("test_size"),
        "n_distributions": n_distributions,
        "ordered_patient_id_sha256": _ordered_fingerprint(patient_ids),
        "ordered_label_sha256": _ordered_fingerprint(distribution_labels),
    }

    rows = []
    for split_idx, (train_values, test_values) in enumerate(
        zip(train_splits, test_splits)
    ):
        train_idx = [int(value) for value in train_values]
        test_idx = [int(value) for value in test_values]
        if set(train_idx) & set(test_idx):
            raise ValueError(f"Train/test overlap within split {split_idx}.")
        if any(idx < 0 or idx >= n_distributions for idx in train_idx + test_idx):
            raise ValueError(f"Out-of-range distribution index in split {split_idx}.")

        row = {
            **fixed,
            "split_idx": split_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_idx": _json_array(train_idx),
            "test_idx": _json_array(test_idx),
            "train_patient_ids": (
                _json_array([patient_ids[idx] for idx in train_idx])
                if patient_ids is not None
                else None
            ),
            "test_patient_ids": (
                _json_array([patient_ids[idx] for idx in test_idx])
                if patient_ids is not None
                else None
            ),
            "train_labels": (
                _json_array([distribution_labels[idx] for idx in train_idx])
                if distribution_labels is not None
                else None
            ),
            "test_labels": (
                _json_array([distribution_labels[idx] for idx in test_idx])
                if distribution_labels is not None
                else None
            ),
        }
        row.update(
            {metric: float(values[split_idx]) for metric, values in per_split.items()}
        )
        rows.append(row)

    return pd.DataFrame(rows)
