"""Notebook helpers for the GMM goodness-of-fit checks (notebooks/supp_gmm_validation.ipynb)."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from ggml_ot._utils._splits import split_train_val_indices
from ggml_ot.distances import compute_OT
from ggml_ot.gmm._GaussianMixture import GaussianMixture


SupportSource = Literal["auto", "preprocessed", "raw"]
EXACT_OT_MAX_ITER = 100_000


@dataclass(frozen=True)
class DatasetEvaluationBundle:
    """Resolved per-dataset inputs for GOF evaluation."""

    dataset_name: str
    gmm_path: Path
    gmm_key: str
    distribution_ids: tuple[str, ...]
    labels_by_distribution: dict[str, str | None]
    supports_by_distribution: dict[str, np.ndarray]
    gmm_payload_by_distribution: dict[str, dict[str, np.ndarray | str]]
    feature_dim: int
    support_source: str
    train_fraction_default: float
    random_seed: int | None
    split_source: str
    split_n_components: int
    expected_split_counts: dict[str, tuple[int, int]] | None
    selected_k_by_distribution: dict[str, int]
    selected_k_source: str
    selected_factor_rank: int | None
    selection_metric: str | None


def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk upward until the repository root is found."""
    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "snakemake" / "config" / "defaults.yml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root from "
        f"{current}. Expected to find snakemake/config/defaults.yml in a parent directory."
    )


def deep_merge(base: dict, override: dict | None) -> dict:
    """Recursively merge config dictionaries."""
    if override is None:
        return dict(base)
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config_path(repo_root: Path, config_file: str | Path) -> Path:
    path = Path(config_file)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                repo_root / "snakemake" / path,
                repo_root / path,
                repo_root / "snakemake" / "config" / path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve config file {config_file!r}. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _load_pipeline_helpers(repo_root: Path):
    scripts_dir = (repo_root / "snakemake" / "scripts").resolve()

    def _load_module(module_name: str, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module {module_name!r} from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    pipeline_utils = _load_module("_notebook_snakemake_utils", scripts_dir / "_utils.py")

    def dataset_display_name(dataset_name: str, ds_meta: dict) -> str:
        name = ds_meta.get("display_name")
        if name:
            return str(name)
        label = dataset_name.replace("_", " ")
        return label if any(char.isupper() for char in label) else label.capitalize()

    return {
        "get_gmm_key": pipeline_utils.get_gmm_key,
        "dataset_display_name": dataset_display_name,
    }


def load_resolved_config(
    config_file: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> tuple[dict, Path, Path]:
    """Load defaults, overlay the requested config, and expand datasets_file."""
    repo = find_repo_root(repo_root)
    snakemake_dir = repo / "snakemake"
    resolved_config_path = _resolve_config_path(repo, config_file)

    with open(snakemake_dir / "config" / "defaults.yml") as handle:
        config = yaml.safe_load(handle)
    with open(resolved_config_path) as handle:
        config = deep_merge(config, yaml.safe_load(handle))

    datasets_path = _resolve_config_path(repo, config["datasets_file"])
    with open(datasets_path) as handle:
        config["datasets"] = yaml.safe_load(handle)["datasets"]

    return config, resolved_config_path, repo


def _config_data_root(config: dict, repo_root: Path) -> Path:
    data_root = Path(config.get("data_root", "../data/snakemake"))
    if data_root.is_absolute():
        return data_root
    return (repo_root / "snakemake" / data_root).resolve()


def _resolve_distribution_ids(adata, gmm_cfg: dict, patient_col: str) -> list[str]:
    stored = gmm_cfg.get("weight_inference", {}).get("distribution_ids")
    if stored is not None:
        return [str(item) for item in stored]
    patient_values = adata.obs[patient_col].astype(str).to_numpy()
    return pd.Index(patient_values).drop_duplicates().astype(str).tolist()


def _label_map(adata, patient_col: str, label_col: str | None) -> dict[str, str | None]:
    patient_values = adata.obs[patient_col].astype(str)
    if label_col is None:
        return {dist_id: None for dist_id in pd.Index(patient_values).drop_duplicates().tolist()}

    label_values = adata.obs[label_col].astype(str)
    frame = pd.DataFrame(
        {
            "distribution_id": patient_values.to_numpy(),
            "label": label_values.to_numpy(),
        }
    )
    out: dict[str, str | None] = {}
    for distribution_id, rows in frame.groupby("distribution_id", sort=False):
        labels = rows["label"].dropna().unique().tolist()
        out[str(distribution_id)] = None if not labels else str(labels[0])
    return out


def _extract_feature_matrix(adata, use_rep: str | None) -> np.ndarray:
    if use_rep is None:
        return np.asarray(adata.X)
    return np.asarray(adata.obsm[use_rep])


def _split_raw_supports(
    adata,
    *,
    patient_col: str,
    use_rep: str | None,
    distribution_ids: list[str],
) -> dict[str, np.ndarray]:
    x = _extract_feature_matrix(adata, use_rep)
    patient_values = adata.obs[patient_col].astype(str).to_numpy()
    supports: dict[str, np.ndarray] = {}
    for dist_id in distribution_ids:
        mask = patient_values == dist_id
        if not np.any(mask):
            raise ValueError(f"Distribution {dist_id!r} has no cells in adata.obs[{patient_col!r}].")
        supports[dist_id] = np.asarray(x[mask], dtype=np.float64)
    return supports


def _extract_supports(
    adata,
    *,
    patient_col: str,
    use_rep: str | None,
    distribution_ids: list[str],
    support_source: SupportSource,
) -> tuple[dict[str, np.ndarray], str]:
    preprocessed = adata.uns.get("ggml_preprocessed")
    has_preprocessed = (
        isinstance(preprocessed, dict)
        and "supports" in preprocessed
        and np.asarray(preprocessed["supports"]).shape[0] == len(distribution_ids)
    )

    if support_source == "preprocessed":
        if not has_preprocessed:
            raise ValueError("Requested support_source='preprocessed' but ggml_preprocessed['supports'] is missing.")
        supports_array = np.asarray(preprocessed["supports"], dtype=np.float64)
        return {
            dist_id: np.asarray(supports_array[idx], dtype=np.float64)
            for idx, dist_id in enumerate(distribution_ids)
        }, "ggml_preprocessed"

    if support_source == "auto" and has_preprocessed:
        supports_array = np.asarray(preprocessed["supports"], dtype=np.float64)
        return {
            dist_id: np.asarray(supports_array[idx], dtype=np.float64)
            for idx, dist_id in enumerate(distribution_ids)
        }, "ggml_preprocessed"

    return _split_raw_supports(
        adata,
        patient_col=patient_col,
        use_rep=use_rep,
        distribution_ids=distribution_ids,
    ), "raw_representation"


def _trim_distribution_payload(
    *,
    gmm_cfg: dict,
    distribution_index: int,
    distribution_n_components: np.ndarray,
) -> dict[str, np.ndarray | str]:
    covariance_type = str(gmm_cfg["covariance_type"])
    component_sharing = str(gmm_cfg["component_sharing"])
    model = gmm_cfg["model"]
    distribution_weights = np.asarray(gmm_cfg["distribution_weights"], dtype=np.float64)

    if component_sharing == "sample_specific":
        n_components = int(distribution_n_components[distribution_index])
        mu = np.asarray(model["mu"][distribution_index][:n_components], dtype=np.float64)
        var = np.asarray(model["var"][distribution_index][:n_components], dtype=np.float64)
        pi = np.asarray(distribution_weights[distribution_index][:n_components], dtype=np.float64)
    else:
        n_components = int(distribution_n_components[distribution_index])
        mu = np.asarray(model["mu"][:n_components], dtype=np.float64)
        var = np.asarray(model["var"][:n_components], dtype=np.float64)
        pi = np.asarray(distribution_weights[distribution_index][:n_components], dtype=np.float64)

    pi = pi / pi.sum()
    return {
        "mu": mu,
        "var": var,
        "pi": pi,
        "covariance_type": covariance_type,
    }


def load_dataset_bundle(
    config_file: str | Path,
    dataset_name: str,
    *,
    repo_root: str | Path | None = None,
    support_source: SupportSource = "raw",
) -> DatasetEvaluationBundle:
    """Resolve one dataset's GMM artifact, grouped supports, and per-distribution GMM payloads."""
    config, _, repo = load_resolved_config(config_file, repo_root=repo_root)
    helpers = _load_pipeline_helpers(repo)
    ds_meta = config["datasets"][dataset_name]
    gmm_key = helpers["get_gmm_key"](config, ds_meta)

    data_root = _config_data_root(config, repo)
    gmm_path = data_root / dataset_name / "gmm" / f"{gmm_key}.h5ad"
    if not gmm_path.exists():
        raise FileNotFoundError(f"Missing GMM artifact: {gmm_path}")

    adata = ad.read_h5ad(gmm_path, backed="r")
    gmm_cfg = adata.uns[gmm_key]
    patient_col = ds_meta["patient_col"]
    label_col = ds_meta.get("label_col")
    distribution_ids = _resolve_distribution_ids(adata, gmm_cfg, patient_col)
    labels_by_distribution = _label_map(adata, patient_col, label_col)

    supports_by_distribution, resolved_support_source = _extract_supports(
        adata,
        patient_col=patient_col,
        use_rep=gmm_cfg.get("use_rep"),
        distribution_ids=distribution_ids,
        support_source=support_source,
    )

    distribution_n_components = np.asarray(
        gmm_cfg.get(
            "distribution_n_components",
            np.full(len(distribution_ids), int(gmm_cfg["n_components"])),
        ),
        dtype=int,
    )
    persisted_k_by_distribution = {
        dist_id: int(distribution_n_components[idx])
        for idx, dist_id in enumerate(distribution_ids)
    }
    gmm_payload_by_distribution = {
        dist_id: _trim_distribution_payload(
            gmm_cfg=gmm_cfg,
            distribution_index=idx,
            distribution_n_components=distribution_n_components,
        )
        for idx, dist_id in enumerate(distribution_ids)
    }

    selection = gmm_cfg.get("selection") or {}
    train_fraction_default = float(
        selection.get(
            "train_frac",
            config.get("gmm", {}).get("fit_gmm_kwargs", {}).get("train_size", 0.5),
        )
    )
    selection_metric = selection.get("metric")
    if selection_metric == "heldout_nll" and selection.get("k_comps") is not None:
        split_source = "fit_selection_reconstructed"
        split_n_components = int(max(int(k) for k in selection["k_comps"]))
        expected_split_counts = {}
        for dist_id, scores in dict(selection.get("scores_by_distribution", {})).items():
            columnar = dict(scores)
            train_counts = columnar.get("n_train_cells", [])
            val_counts = columnar.get("n_val_cells", [])
            if len(train_counts) > 0 and len(val_counts) > 0:
                expected_split_counts[str(dist_id)] = (int(train_counts[0]), int(val_counts[0]))
    else:
        split_source = "config_fallback"
        split_n_components = int(
            max(
                np.asarray(
                    gmm_cfg.get("distribution_n_components", [gmm_cfg["n_components"]]),
                    dtype=int,
                )
            )
        )
        expected_split_counts = None
    best_k_by_distribution = selection.get("best_k_by_distribution") or {}
    if isinstance(best_k_by_distribution, dict) and len(best_k_by_distribution) > 0:
        selected_k_by_distribution = {
            str(dist_id): int(best_k_by_distribution.get(dist_id, persisted_k_by_distribution[dist_id]))
            for dist_id in distribution_ids
        }
        selected_k_source = "selection.best_k_by_distribution"
    else:
        selected_k_by_distribution = dict(persisted_k_by_distribution)
        selected_k_source = "distribution_n_components"
    best_factor_rank = selection.get("best_factor_rank", (gmm_cfg.get("fit_params") or {}).get("factor_rank"))
    selected_factor_rank = None if best_factor_rank is None else int(best_factor_rank)
    feature_dim = int(next(iter(supports_by_distribution.values())).shape[1])

    bundle = DatasetEvaluationBundle(
        dataset_name=dataset_name,
        gmm_path=gmm_path,
        gmm_key=gmm_key,
        distribution_ids=tuple(distribution_ids),
        labels_by_distribution=labels_by_distribution,
        supports_by_distribution=supports_by_distribution,
        gmm_payload_by_distribution=gmm_payload_by_distribution,
        feature_dim=feature_dim,
        support_source=resolved_support_source,
        train_fraction_default=train_fraction_default,
        random_seed=None if config.get("random_seed") is None else int(config["random_seed"]),
        split_source=split_source,
        split_n_components=split_n_components,
        expected_split_counts=expected_split_counts,
        selected_k_by_distribution=selected_k_by_distribution,
        selected_k_source=selected_k_source,
        selected_factor_rank=selected_factor_rank,
        selection_metric=None if selection_metric is None else str(selection_metric),
    )
    if getattr(adata, "isbacked", False) and getattr(adata, "file", None) is not None:
        adata.file.close()
    return bundle


def resolve_dataset_bundles(
    config_file: str | Path,
    *,
    repo_root: str | Path | None = None,
    dataset_names: list[str] | tuple[str, ...] | None = None,
    support_source: SupportSource = "raw",
) -> list[DatasetEvaluationBundle]:
    """Resolve all requested dataset bundles from a config file."""
    config, _, _ = load_resolved_config(config_file, repo_root=repo_root)
    names = list(dataset_names) if dataset_names is not None else list(config["datasets"].keys())
    return [
        load_dataset_bundle(
            config_file,
            dataset_name=name,
            repo_root=repo_root,
            support_source=support_source,
        )
        for name in names
    ]


def _rng_from_seed(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _derive_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _subsample_points(points: np.ndarray, max_points: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_points is None or len(points) <= max_points:
        return np.asarray(points, dtype=np.float64)
    indices = rng.choice(len(points), size=int(max_points), replace=False)
    return np.asarray(points[indices], dtype=np.float64)


def _train_val_split(
    points: np.ndarray,
    *,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_points = int(points.shape[0])
    if n_points < 4:
        raise ValueError(f"Need at least 4 points per distribution, got {n_points}.")

    n_train = int(np.floor(train_fraction * n_points))
    n_train = min(max(n_train, 2), n_points - 2)
    perm = rng.permutation(n_points)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    return np.asarray(points[train_idx], dtype=np.float64), np.asarray(points[val_idx], dtype=np.float64)


def reconstruct_bundle_train_val_splits(
    bundle: DatasetEvaluationBundle,
    *,
    train_fraction_override: float | None = None,
    max_points: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reconstruct one deterministic train/validation split per distribution."""
    if max_points is not None:
        raise ValueError(
            "max_points is incompatible with fit-time split reconstruction. "
            "Leave max_points=None to stay aligned with the fitted GMM split."
        )

    train_fraction = bundle.train_fraction_default if train_fraction_override is None else float(train_fraction_override)
    rng = _rng_from_seed(bundle.random_seed)
    split_by_distribution: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for dist_id in bundle.distribution_ids:
        points = np.asarray(bundle.supports_by_distribution[dist_id], dtype=np.float64)
        train_idx, val_idx = split_train_val_indices(
            len(points),
            n_components=int(bundle.split_n_components),
            train_frac=train_fraction,
            rng=rng,
        )
        train_points = np.asarray(points[train_idx], dtype=np.float64)
        val_points = np.asarray(points[val_idx], dtype=np.float64)
        expected = None if bundle.expected_split_counts is None else bundle.expected_split_counts.get(dist_id)
        if expected is not None and (len(train_points), len(val_points)) != expected:
            raise ValueError(
                "Reconstructed split counts do not match stored fit-time diagnostics for "
                f"{bundle.dataset_name}/{dist_id}: got {(len(train_points), len(val_points))}, expected {expected}."
            )
        split_by_distribution[dist_id] = (train_points, val_points)

    return split_by_distribution


def _validation_half_split(
    points: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_points = int(points.shape[0])
    if n_points < 2:
        raise ValueError(f"Need at least 2 validation points, got {n_points}.")
    perm = rng.permutation(n_points)
    split = n_points // 2
    split = min(max(split, 1), n_points - 1)
    left = np.asarray(points[perm[:split]], dtype=np.float64)
    right = np.asarray(points[perm[split:]], dtype=np.float64)
    return left, right


def exact_w2_squared(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Exact empirical W2^2 using ggml_ot's OT solver."""
    supports = [
        np.asarray(points_a, dtype=np.float32),
        np.asarray(points_b, dtype=np.float32),
    ]
    distances = compute_OT(
        supports,
        ground_metric="euclidean",
        squared_ground_cost=True,
        entropic_reg=0.0,
        # ggml_ot forwards this kwarg to the exact EMD solver's numItermax as well.
        sinkhorn_max_iter=EXACT_OT_MAX_ITER,
    )
    return float(np.asarray(distances)[0, 1])


def make_single_gaussian_payload(
    train_points: np.ndarray,
    *,
    covariance_ridge: float = 1.0e-6,
) -> dict[str, np.ndarray | str]:
    """Fit a single full-covariance Gaussian payload from point samples."""
    x = np.asarray(train_points, dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    if x.shape[0] == 1:
        cov = np.eye(x.shape[1], dtype=np.float64)
    else:
        cov = np.cov(x, rowvar=False, ddof=1)
        if cov.ndim == 0:
            cov = np.asarray([[float(cov)]], dtype=np.float64)
    ridge_scale = float(max(np.trace(cov) / cov.shape[0], 1.0)) * float(covariance_ridge)
    cov = np.asarray(cov, dtype=np.float64) + ridge_scale * np.eye(cov.shape[0], dtype=np.float64)
    return {
        "mu": mu,
        "var": cov[None, :, :],
        "pi": np.asarray([1.0], dtype=np.float64),
        "covariance_type": "full",
    }


def sample_gmm_points(
    payload: dict[str, np.ndarray | str],
    n_points: int,
    *,
    seed: int,
) -> np.ndarray:
    """Sample from a persisted ggml_ot GaussianMixture payload."""
    torch.manual_seed(int(seed))
    model = GaussianMixture.from_dict(payload)
    points, _ = model.sample(int(n_points))
    return np.asarray(points.detach().cpu().numpy(), dtype=np.float64)


def _distance_between(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> float:
    return exact_w2_squared(points_a, points_b)


def evaluate_distribution_gof(
    points: np.ndarray,
    gmm_payload: dict[str, np.ndarray | str],
    *,
    dataset_name: str,
    distribution_id: str,
    label: str | None,
    train_points: np.ndarray | None = None,
    val_points: np.ndarray | None = None,
    split_source: str = "resampled",
    train_fraction: float,
    repeats: int,
    max_points: int | None,
    covariance_ridge: float,
    seed: int | None,
) -> pd.DataFrame:
    """Evaluate lower/upper/GMM transport scores for one distribution."""
    base_rng = _rng_from_seed(seed)
    points = np.asarray(points, dtype=np.float64)

    records: list[dict[str, object]] = []
    for repeat_idx in range(int(repeats)):
        repeat_rng = _rng_from_seed(_derive_seed(base_rng))
        if train_points is None or val_points is None:
            repeat_points = _subsample_points(points, max_points=max_points, rng=repeat_rng)
            repeat_train_points, repeat_val_points = _train_val_split(
                repeat_points,
                train_fraction=train_fraction,
                rng=repeat_rng,
            )
            effective_split_source = "resampled"
        else:
            repeat_train_points = np.asarray(train_points, dtype=np.float64)
            repeat_val_points = np.asarray(val_points, dtype=np.float64)
            repeat_points = np.vstack([repeat_train_points, repeat_val_points])
            effective_split_source = split_source

        val_a, val_b = _validation_half_split(repeat_val_points, rng=repeat_rng)

        lower_bound = _distance_between(val_a, val_b)
        train_split_distance = _distance_between(repeat_train_points, val_b)

        baseline_payload = make_single_gaussian_payload(repeat_train_points, covariance_ridge=covariance_ridge)
        baseline_seed = _derive_seed(repeat_rng)
        baseline_points = sample_gmm_points(baseline_payload, len(val_b), seed=baseline_seed)
        upper_bound = _distance_between(baseline_points, val_b)

        gmm_seed = _derive_seed(repeat_rng)
        sampled_gmm_points = sample_gmm_points(gmm_payload, len(val_b), seed=gmm_seed)
        gmm_distance = _distance_between(sampled_gmm_points, val_b)

        denom = upper_bound - lower_bound
        relative_error = np.nan if denom <= 1.0e-12 else (gmm_distance - lower_bound) / denom

        records.append(
            {
                "dataset": dataset_name,
                "distribution_id": distribution_id,
                "label": label,
                "repeat": repeat_idx,
                "split_source": effective_split_source,
                "transport_mode": "exact",
                "feature_dim": int(points.shape[1]),
                "n_total": int(len(repeat_points)),
                "n_train": int(len(repeat_train_points)),
                "n_val": int(len(repeat_val_points)),
                "n_val_split_a": int(len(val_a)),
                "n_val_split_b": int(len(val_b)),
                "ground_truth_distance": float(lower_bound),
                "lower_bound": float(lower_bound),
                "train_split_distance": float(train_split_distance),
                "upper_bound": float(upper_bound),
                "gmm_distance": float(gmm_distance),
                "relative_error": float(relative_error) if np.isfinite(relative_error) else np.nan,
            }
        )

    return pd.DataFrame.from_records(records)


def summarize_results(
    results: pd.DataFrame,
    *,
    groupby: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Compute mean/sd summaries for the main GOF metrics."""
    if results.empty:
        return pd.DataFrame()

    summary = (
        results.groupby(list(groupby), dropna=False, sort=False)
        .agg(
            repeats=("repeat", "count"),
            mean_lower_bound=("lower_bound", "mean"),
            sd_lower_bound=("lower_bound", "std"),
            mean_train_split_distance=("train_split_distance", "mean"),
            sd_train_split_distance=("train_split_distance", "std"),
            mean_upper_bound=("upper_bound", "mean"),
            sd_upper_bound=("upper_bound", "std"),
            mean_gmm_distance=("gmm_distance", "mean"),
            sd_gmm_distance=("gmm_distance", "std"),
            mean_relative_error=("relative_error", "mean"),
            sd_relative_error=("relative_error", "std"),
        )
        .reset_index()
    )
    for column in summary.columns:
        if column.startswith("sd_"):
            summary[column] = summary[column].fillna(0.0)
    return summary


def evaluate_config_gof(
    config_file: str | Path,
    *,
    repo_root: str | Path | None = None,
    dataset_names: list[str] | tuple[str, ...] | None = None,
    distribution_ids: list[str] | tuple[str, ...] | None = None,
    support_source: SupportSource = "raw",
    train_fraction: float | None = None,
    repeats: int = 1,
    max_points: int | None = None,
    covariance_ridge: float = 1.0e-6,
    seed: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate the GMM goodness of fit for all requested datasets and distributions."""
    bundles = resolve_dataset_bundles(
        config_file,
        repo_root=repo_root,
        dataset_names=dataset_names,
        support_source=support_source,
    )

    results_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    base_rng = _rng_from_seed(seed)

    requested_distribution_ids = None if distribution_ids is None else set(str(item) for item in distribution_ids)

    for bundle in bundles:
        reconstructed_splits = reconstruct_bundle_train_val_splits(
            bundle,
            train_fraction_override=train_fraction,
            max_points=max_points,
        )
        metadata_rows.append(
            {
                "dataset": bundle.dataset_name,
                "gmm_path": str(bundle.gmm_path),
                "gmm_key": bundle.gmm_key,
                "support_source": bundle.support_source,
                "feature_dim": bundle.feature_dim,
                "n_distributions": len(bundle.distribution_ids),
                "train_fraction_default": bundle.train_fraction_default,
                "split_source": bundle.split_source,
            }
        )
        bundle_train_fraction = bundle.train_fraction_default if train_fraction is None else float(train_fraction)
        for dist_id in bundle.distribution_ids:
            if requested_distribution_ids is not None and dist_id not in requested_distribution_ids:
                continue
            points = bundle.supports_by_distribution[dist_id]
            payload = bundle.gmm_payload_by_distribution[dist_id]
            train_points, val_points = reconstructed_splits[dist_id]
            dist_seed = _derive_seed(base_rng)
            results_frames.append(
                evaluate_distribution_gof(
                    points,
                    payload,
                    dataset_name=bundle.dataset_name,
                    distribution_id=dist_id,
                    label=bundle.labels_by_distribution.get(dist_id),
                    train_points=train_points,
                    val_points=val_points,
                    split_source=bundle.split_source,
                    train_fraction=bundle_train_fraction,
                    repeats=repeats,
                    max_points=max_points,
                    covariance_ridge=covariance_ridge,
                    seed=dist_seed,
                )
            )

    results = (
        pd.concat(results_frames, ignore_index=True)
        if results_frames
        else pd.DataFrame(
            columns=[
                "dataset",
                "distribution_id",
                "label",
                "repeat",
                "transport_mode",
                "feature_dim",
                "n_total",
                "n_train",
                "n_val",
                "n_val_split_a",
                "n_val_split_b",
                "lower_bound",
                "train_split_distance",
                "upper_bound",
                "gmm_distance",
                "relative_error",
            ]
        )
    )
    distribution_summary = summarize_results(
        results,
        groupby=["dataset", "distribution_id", "label", "transport_mode", "feature_dim"],
    )
    dataset_summary = summarize_results(
        results,
        groupby=["dataset", "transport_mode", "feature_dim"],
    )
    metadata = pd.DataFrame.from_records(metadata_rows)
    return results, distribution_summary, dataset_summary, metadata


def collect_selected_k_summary(
    config_file: str | Path,
    *,
    repo_root: str | Path | None = None,
    dataset_names: list[str] | tuple[str, ...] | None = None,
    support_source: SupportSource = "raw",
) -> pd.DataFrame:
    """Collect the selected per-distribution k-component counts from fitted GMM artifacts."""
    config, _, resolved_repo_root = load_resolved_config(config_file, repo_root=repo_root)
    bundles = resolve_dataset_bundles(
        config_file,
        repo_root=resolved_repo_root,
        dataset_names=dataset_names,
        support_source=support_source,
    )
    helpers = _load_pipeline_helpers(resolved_repo_root)

    rows: list[dict[str, object]] = []
    for bundle in bundles:
        dataset_display = helpers["dataset_display_name"](
            bundle.dataset_name,
            config["datasets"].get(bundle.dataset_name, {}),
        )
        for dist_id in bundle.distribution_ids:
            rows.append(
                {
                    "dataset": bundle.dataset_name,
                    "dataset_display_name": dataset_display,
                    "distribution_id": dist_id,
                    "selected_k": int(bundle.selected_k_by_distribution[dist_id]),
                    "selected_k_source": bundle.selected_k_source,
                    "selected_q": bundle.selected_factor_rank,
                    "selection_metric": bundle.selection_metric,
                }
            )

    return pd.DataFrame.from_records(rows)


def plot_selected_k_histograms(
    selected_k_summary: pd.DataFrame,
    *,
    dataset_cfg: dict[str, dict],
    dataset_names: list[str],
):
    """Plot one selected-k histogram per dataset."""
    helpers = _load_pipeline_helpers(find_repo_root())
    ordered_datasets = _ordered_present_datasets(
        selected_k_summary,
        dataset_names,
        dataset_cfg=dataset_cfg,
        helpers=helpers,
    )
    if not ordered_datasets:
        raise ValueError("No datasets available for selected-k plotting.")

    n_panels = len(ordered_datasets)
    n_cols = min(4, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.9 * n_cols, 4.0 * n_rows),
        squeeze=False,
        sharey=False,
    )
    axes_flat = list(axes.ravel())

    for panel_idx, dataset_name in enumerate(ordered_datasets):
        ax = axes_flat[panel_idx]
        rows = selected_k_summary.loc[selected_k_summary["dataset"].astype(str) == str(dataset_name)].copy()
        counts = (
            pd.to_numeric(rows["selected_k"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
        display_name = helpers["dataset_display_name"](dataset_name, dataset_cfg.get(dataset_name, {}))
        title = f"{display_name}\n$n_{{\\mathrm{{dist}}}}={len(rows)}$"
        ax.bar(
            counts.index.to_numpy(dtype=int),
            counts.to_numpy(dtype=int),
            width=0.8,
            color="#4C72B0",
            alpha=0.8,
            edgecolor="#333333",
            linewidth=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel(r"Selected $k_{\mathrm{comp}}$")
        ax.set_ylabel("Number of patients")
        full_k_range = np.arange(int(counts.index.min()), int(counts.index.max()) + 1, dtype=int)
        ax.set_xticks(full_k_range)
        ax.set_xlim(full_k_range.min() - 0.5, full_k_range.max() + 0.5)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.8)
        ax.set_axisbelow(True)
        if panel_idx % n_cols != 0:
            ax.set_ylabel("")

    for ax in axes_flat[n_panels:]:
        ax.remove()

    fig.suptitle("Selected component counts of patient-specific GMMs", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig, axes


def explain_support_source(bundle: DatasetEvaluationBundle) -> str:
    """Return a short human-readable explanation of the resolved support source."""
    if bundle.support_source == "ggml_preprocessed":
        return (
            "Using persisted fit-time supports from adata.uns['ggml_preprocessed']['supports'], "
            "which reflect the exact representation cached for GGML/GMM fitting."
        )
    return (
        "Using grouped raw representation values reconstructed from the fitted artifact "
        "(adata.X or adata.obsm[use_rep]) because no persisted preprocessed supports were available."
    )


def _ordered_present_datasets(
    frame: pd.DataFrame,
    dataset_names: list[str],
    *,
    dataset_cfg: dict[str, dict],
    helpers: dict[str, object],
) -> list[str]:
    """Return datasets present in a frame, ordered alphabetically by display name."""
    present = set(frame["dataset"].dropna().astype(str))
    ordered = [dataset_name for dataset_name in dataset_names if dataset_name in present]
    extras = sorted(
        present.difference(ordered),
        key=lambda dataset_name: str(
            helpers["dataset_display_name"](dataset_name, dataset_cfg.get(dataset_name, {}))
        ).lower(),
    )
    ordered = ordered + extras
    ordered.sort(
        key=lambda dataset_name: str(
            helpers["dataset_display_name"](dataset_name, dataset_cfg.get(dataset_name, {}))
        ).lower()
    )
    return ordered


def _styled_boxplot(
    ax,
    values_by_label: list[np.ndarray],
    *,
    labels: list[str],
    colors: list[str],
    ylabel: str,
    title: str | None = None,
    show_points: bool = True,
    reference_line: float | None = None,
    ymin: float | None = None,
) -> None:
    """Render a boxplot using the same style as ggml_ot's GMM validation plot."""
    positions = list(range(1, len(values_by_label) + 1))
    box = ax.boxplot(
        values_by_label,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        whis=1.5,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.2)
    for median in box["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.4)
    for whisker in box["whiskers"]:
        whisker.set_color("#555555")
        whisker.set_linewidth(1.0)
    for cap in box["caps"]:
        cap.set_color("#555555")
        cap.set_linewidth(1.0)
    for flier in box["fliers"]:
        flier.set_markeredgecolor("#555555")
        flier.set_alpha(0.5)

    if show_points:
        rng = np.random.default_rng(0)
        for pos, values in zip(positions, values_by_label):
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                np.full(len(values), pos, dtype=float) + jitter,
                values,
                color="#2B2B2B",
                alpha=0.35,
                s=14,
                linewidths=0,
                zorder=3,
            )

    if reference_line is not None:
        ax.axhline(reference_line, color="#777777", linestyle="--", linewidth=1.0, zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ymin is not None:
        ax.set_ylim(bottom=float(ymin))
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_relative_error_panels(
    distribution_summary: pd.DataFrame,
    *,
    dataset_cfg: dict[str, dict],
    dataset_names: list[str],
):
    """Plot one relative-error boxplot per dataset."""
    helpers = _load_pipeline_helpers(find_repo_root())
    ordered_datasets = _ordered_present_datasets(
        distribution_summary,
        dataset_names,
        dataset_cfg=dataset_cfg,
        helpers=helpers,
    )
    if not ordered_datasets:
        raise ValueError("No datasets available for relative-error plotting.")

    n_panels = len(ordered_datasets)
    n_cols = min(4, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 4.0 * n_rows),
        squeeze=False,
        sharey=False,
    )
    axes_flat = list(axes.ravel())

    for panel_idx, dataset_name in enumerate(ordered_datasets):
        ax = axes_flat[panel_idx]
        rows = distribution_summary.loc[distribution_summary["dataset"].astype(str) == str(dataset_name)].copy()
        values = pd.to_numeric(rows["mean_relative_error"], errors="coerce").dropna().to_numpy()
        display_name = helpers["dataset_display_name"](dataset_name, dataset_cfg.get(dataset_name, {}))
        title = f"{display_name}\n$n_{{\\mathrm{{dist}}}}={len(rows)}$"
        _styled_boxplot(
            ax,
            [values],
            labels=["Relative error"],
            colors=["#4C72B0"],
            ylabel="Relative error",
            title=title,
            reference_line=0.0,
        )
        if panel_idx % n_cols != 0:
            ax.set_ylabel("")

    for ax in axes_flat[n_panels:]:
        ax.remove()

    fig.suptitle("Relative transport error by dataset", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig, axes


def plot_distance_triplet_panels(
    distribution_summary: pd.DataFrame,
    *,
    dataset_cfg: dict[str, dict],
    dataset_names: list[str],
):
    """Plot per-dataset train/GMM/baseline Wasserstein-2 distances to the validation split."""
    helpers = _load_pipeline_helpers(find_repo_root())
    ordered_datasets = _ordered_present_datasets(
        distribution_summary,
        dataset_names,
        dataset_cfg=dataset_cfg,
        helpers=helpers,
    )
    if not ordered_datasets:
        raise ValueError("No datasets available for distance plotting.")

    n_panels = len(ordered_datasets)
    n_cols = min(4, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.1 * n_rows),
        squeeze=False,
        sharey=False,
    )
    axes_flat = list(axes.ravel())

    for panel_idx, dataset_name in enumerate(ordered_datasets):
        ax = axes_flat[panel_idx]
        rows = distribution_summary.loc[distribution_summary["dataset"].astype(str) == str(dataset_name)].copy()
        display_name = helpers["dataset_display_name"](dataset_name, dataset_cfg.get(dataset_name, {}))
        title = f"{display_name}\n$n_{{\\mathrm{{dist}}}}={len(rows)}$"
        _styled_boxplot(
            ax,
            [
                pd.to_numeric(rows["mean_train_split_distance"], errors="coerce").dropna().to_numpy(),
                pd.to_numeric(rows["mean_gmm_distance"], errors="coerce").dropna().to_numpy(),
                pd.to_numeric(rows["mean_upper_bound"], errors="coerce").dropna().to_numpy(),
            ],
            labels=[
                "Train split\n(original cells)",
                "Fitted GMM\n(sampled)",
                "Single-Gaussian\nbaseline (sampled)",
            ],
            colors=["#4C72B0", "#55A868", "#DD8452"],
            ylabel="Wasserstein-2 distance",
            title=title,
            ymin=0.0,
        )
        if panel_idx % n_cols != 0:
            ax.set_ylabel("")

    for ax in axes_flat[n_panels:]:
        ax.remove()

    fig.suptitle("Per-dataset Wasserstein-2 distance to validation split", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig, axes


def plot_train_gmm_panels(
    distribution_summary: pd.DataFrame,
    *,
    dataset_cfg: dict[str, dict],
    dataset_names: list[str],
):
    """Plot per-dataset train/GMM Wasserstein-2 distances to the validation split."""
    helpers = _load_pipeline_helpers(find_repo_root())
    ordered_datasets = _ordered_present_datasets(
        distribution_summary,
        dataset_names,
        dataset_cfg=dataset_cfg,
        helpers=helpers,
    )
    if not ordered_datasets:
        raise ValueError("No datasets available for train/GMM plotting.")

    n_panels = len(ordered_datasets)
    n_cols = min(4, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.9 * n_cols, 4.1 * n_rows),
        squeeze=False,
        sharey=False,
    )
    axes_flat = list(axes.ravel())

    for panel_idx, dataset_name in enumerate(ordered_datasets):
        ax = axes_flat[panel_idx]
        rows = distribution_summary.loc[distribution_summary["dataset"].astype(str) == str(dataset_name)].copy()
        display_name = helpers["dataset_display_name"](dataset_name, dataset_cfg.get(dataset_name, {}))
        title = f"{display_name}\n$n_{{\\mathrm{{dist}}}}={len(rows)}$"
        _styled_boxplot(
            ax,
            [
                pd.to_numeric(rows["mean_train_split_distance"], errors="coerce").dropna().to_numpy(),
                pd.to_numeric(rows["mean_gmm_distance"], errors="coerce").dropna().to_numpy(),
            ],
            labels=[
                "Train split\n(original cells)",
                "Fitted GMM\n(sampled)",
            ],
            colors=["#4C72B0", "#55A868"],
            ylabel="Wasserstein-2 distance",
            title=title,
            ymin=0.0,
        )
        if panel_idx % n_cols != 0:
            ax.set_ylabel("")

    for ax in axes_flat[n_panels:]:
        ax.remove()

    fig.suptitle("Train split vs fitted GMM Wasserstein-2 distance to validation split", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig, axes
