"""
Shared utilities for the GGML-OT Snakemake pipeline.

All hashing and path-construction helpers live here.
Scripts import from this module; Snakefiles use it via `scripts/_utils.py`.
"""

import hashlib
import json
import logging
from copy import deepcopy
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
from collections.abc import Mapping

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────────────────────

def _config_hash(obj, length: int = 8) -> str:
    """Stable short hash of any JSON-serialisable object (sha256, hex)."""
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]


def sort_keys_deep(value):
    """Recursively order mapping keys; list order is left untouched.

    Snakemake compares rule params as text, so two configs that differ only in
    key order look like a changed parameter and retrigger the rule (and
    everything downstream). Ordering keys here makes params order-insensitive,
    so tidying a YAML file no longer refits every GMM.
    """
    if isinstance(value, Mapping):
        return {key: sort_keys_deep(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [sort_keys_deep(item) for item in value]
    return value


def project_config(config: Mapping, schema) -> dict:
    """Return a deep-copied, key-ordered config subset described by ``schema``.

    ``schema`` may be either:
    - an iterable of top-level keys, or
    - a nested mapping where ``True`` means "copy this key as-is".
    """
    if isinstance(schema, Mapping):
        result = {}
        for key, subschema in schema.items():
            if key not in config:
                continue
            value = config[key]
            if subschema is True:
                result[key] = deepcopy(value)
            else:
                if not isinstance(value, Mapping):
                    raise TypeError(f"Cannot project nested schema for non-mapping key {key!r}")
                result[key] = project_config(value, subschema)
        return sort_keys_deep(result)

    result = {}
    for key in schema:
        if key in config:
            result[key] = deepcopy(config[key])
    return sort_keys_deep(result)


_RUNTIME_CONFIG_SCHEMA = {
    "training_device": True,
    "n_threads": True,
    "random_seed": True,
    "init_strategy": True,
}

_DATASET_BUILD_CONFIG_SCHEMA = {
    "use_rep": True,
    "n_feats": True,
    "preprocess": True,
    "from_anndata": True,
}

_FIT_GMM_CONFIG_SCHEMA = {
    "use_rep": True,
    "n_feats": True,
    "preprocess": True,
    "gmm": True,
}

_TUNE_CONFIG_SCHEMA = {
    **_RUNTIME_CONFIG_SCHEMA,
    **_DATASET_BUILD_CONFIG_SCHEMA,
    "tune": True,
    "tune_plots": True,
}

_BENCHMARK_CONFIG_SCHEMA = {
    **_RUNTIME_CONFIG_SCHEMA,
    **_DATASET_BUILD_CONFIG_SCHEMA,
    "train_test": True,
    "benchmark_plots": True,
}

_APPROX_ERROR_BOUND_CONFIG_SCHEMA = {
    **_RUNTIME_CONFIG_SCHEMA,
    **_DATASET_BUILD_CONFIG_SCHEMA,
    "gmm": True,
    "datasets": True,
    "approx_error_bound": True,
}

_PLOT_DISPLAY_CONFIG_SCHEMA = {
    "methods": True,
    "train_test": {
        "n_splits": True,
    },
}

_DATASET_TABLE_CONFIG_SCHEMA = {
    "use_rep": True,
    "n_feats": True,
    "datasets": True,
}


def get_fit_gmm_rule_config(config: dict) -> dict:
    """Config subset that can affect the GMM fitting rule outputs."""
    return project_config(config, _FIT_GMM_CONFIG_SCHEMA)


# Dataset-entry keys that can reach a compute rule: obs-column references
# (``patient_col``/``label_col``/``cell_subtype_col``, plus the ones resolved via
# ``gmm.k_comps_from_col`` / ``gmm.anchor_means_from_col``) and the synthetic
# descriptors, which ``get_dataset_use_rep`` keys on. Everything else is
# irrelevant — ``id`` addresses the download, ``qc_covariates`` only fed the
# retired regress preprocessing mode.
_RULE_DS_META_KEYS = ("source", "synth", "synthetic")


def get_rule_ds_meta(ds_meta: Mapping) -> dict:
    """Subset of a dataset entry that can affect fit_gmm / tune / benchmark.

    Passing the whole entry as a rule param meant that editing an unrelated
    field (e.g. removing ``qc_covariates``) refit every GMM and re-ran every
    benchmark. The three rules consume the entry only through ``build_dataset``
    and the GMM fit, so they see the same subset.

    This does not affect output paths: ``gmm_key`` / ``bench_hash`` / ``tune_hash``
    are derived from the full config in ``common.smk``, not from this param.
    """
    kept = {
        key: value
        for key, value in ds_meta.items()
        if key.endswith("_col") or key in _RULE_DS_META_KEYS
    }
    return sort_keys_deep(deepcopy(kept))


def filter_small_distributions_for_gmm(adata, patient_col: str, min_cells: int, logger):
    """Drop patients with fewer than ``min_cells`` cells before per-patient GMM fitting.

    Returns ``(adata, info)``; ``info`` is ``None`` if nothing was dropped. Same as
    ``fit_gmm.py``'s private copy, which stays until the next full recompute (editing
    that rule script would invalidate every GMM fit).
    """
    counts = adata.obs[patient_col].value_counts()
    counts = counts[counts > 0]
    too_small = counts[counts < min_cells]
    if too_small.empty:
        return adata, None

    preview_items = [f"{dist}={int(count)}" for dist, count in too_small.head(20).items()]
    if len(too_small) > len(preview_items):
        preview_items.append(f"... {len(too_small) - len(preview_items)} more")
    logger.warning(
        "Dropping %d distributions with fewer than %d cells before sample_specific GMM fitting: %s",
        len(too_small),
        min_cells,
        ", ".join(preview_items),
    )

    keep_distributions = set(counts[counts >= min_cells].index)
    keep_mask = adata.obs[patient_col].isin(keep_distributions).to_numpy()
    filtered = adata[keep_mask].copy()
    if filtered.n_obs == 0:
        raise ValueError(
            f"No distributions remain after filtering obs[{patient_col!r}] to at least {min_cells} cells."
        )
    return filtered, {
        "patient_col": patient_col,
        "min_cells": min_cells,
        "n_dropped": int(len(too_small)),
        "dropped_distributions": {str(dist): int(count) for dist, count in too_small.items()},
    }


def get_tune_rule_config(config: dict, method_name: str) -> dict:
    """Config subset that can affect tuning outputs for one method."""
    projected = project_config(config, _TUNE_CONFIG_SCHEMA)
    projected["methods"] = {
        method_name: deepcopy(config["methods"][method_name])
    }
    return projected


def get_benchmark_rule_config(config: dict, method_name: str) -> dict:
    """Config subset that can affect benchmark outputs for one method."""
    projected = project_config(config, _BENCHMARK_CONFIG_SCHEMA)
    projected["methods"] = {
        method_name: deepcopy(config["methods"][method_name])
    }
    return projected


def get_plot_rule_config(config: dict) -> dict:
    """Config subset that can affect rendered benchmark display outputs."""
    return project_config(config, _PLOT_DISPLAY_CONFIG_SCHEMA)


def get_approx_error_bound_rule_config(config: dict) -> dict:
    """Config subset that can affect the Figure 5 epoch-error outputs."""
    return project_config(config, _APPROX_ERROR_BOUND_CONFIG_SCHEMA)


def get_dataset_table_rule_config(config: dict) -> dict:
    """Config subset that can affect the dataset summary table."""
    return project_config(config, _DATASET_TABLE_CONFIG_SCHEMA)


def _preprocess_label(config: dict) -> str:
    """Return a compact suffix for non-default preprocessing modes."""
    preprocess_cfg = config.get("preprocess", {}) or {}
    mode = str(preprocess_cfg.get("mode", "log1p_pca"))
    if mode in {"log1p_pca", "default"}:
        return ""
    if mode == "log1p_hvg_scale_pca":
        n_hvg = int(preprocess_cfg.get("n_hvg", 5000))
        scale_max_value = preprocess_cfg.get("scale_max_value", 10)
        scale_label = "noclip" if scale_max_value is None else f"clip{float(scale_max_value):g}"
        return f"hvg{n_hvg}-scale-{scale_label}"
    labels = {
        "log1p_regress_pca": "regress",
        "pearson_residual_pca": "pearson",
    }
    return labels.get(mode, mode.replace("_", "-"))


def format_representation_label(use_rep: str | None, n_feats: int, preprocess_label: str = "") -> str:
    """Return a filesystem-friendly representation label."""
    rep = resolve_use_rep(use_rep)
    suffix = f"_{preprocess_label}" if preprocess_label else ""
    if rep is None:
        return f"X_{n_feats}{suffix}"
    return f"{rep}_{n_feats}{suffix}"


def get_dataset_representation_label(config: dict, ds_meta: dict) -> str:
    """Return the concrete representation label used for one dataset."""
    if get_dataset_source(ds_meta) == "synthetic_gmm":
        synth_cfg = dict(ds_meta.get("synth", {}))
        synthetic_cfg = dict(ds_meta.get("synthetic", {}))
        generator = synthetic_cfg.get("generator", "synth_gmm_anndata")

        if generator == "from_synth":
            noise_dims = int(synth_cfg.get("noise_dims", 1))
            n_dim = 1 + noise_dims
        else:
            n_dim = int(synth_cfg["n_dim"])

        return f"X_{n_dim}"
    return format_representation_label(
        config.get("use_rep"),
        int(config.get("n_feats", 50)),
        _preprocess_label(config),
    )


def get_dataset_use_rep(config: dict, ds_meta: dict) -> str | None:
    """Resolve the ggml_ot use_rep value for one dataset."""
    if get_dataset_source(ds_meta) == "synthetic_gmm":
        return None
    return resolve_use_rep(config.get("use_rep"))


def get_gmm_key(config: dict, ds_meta: dict) -> str:
    """Build the dataset-specific gmm_key stored in adata.uns.

    Format: ``gmm_{representation_label}_{norm}_{cov}_{hex8}``

    The hex8 suffix hashes the full GMM config together with the resolved
    dataset representation label so different embeddings/dimensions cannot
    silently collide.
    """
    gmm = config.get("gmm", {})
    rep_label = get_dataset_representation_label(config, ds_meta)
    norm = "norm" if gmm.get("normalize", False) else "nonorm"
    cov = gmm.get("fit_gmm_kwargs", {}).get("covariance_type", "full")
    h = _config_hash({"gmm": gmm, "representation_label": rep_label})
    return f"gmm_{rep_label}_{norm}_{cov}_{h}"


def get_aggregate_experiment_label(config: dict, datasets: dict[str, dict]) -> str:
    """Build a readable aggregate output label plus a short collision hash."""
    counts = Counter()
    dataset_summary = {}
    for dataset_name, ds_meta in sorted(datasets.items()):
        source = get_dataset_source(ds_meta)
        source_label = "synth" if source == "synthetic_gmm" else "anndata"
        rep_label = get_dataset_representation_label(config, ds_meta)
        counts[(source_label, rep_label)] += 1
        dataset_summary[dataset_name] = {
            "source": source_label,
            "representation_label": rep_label,
        }

    parts = [
        f"{source}-{count}_{rep_label}"
        for (source, rep_label), count in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    h = _config_hash(
        {
            "datasets": dataset_summary,
            "methods": config.get("methods", {}),
            "use_rep": config.get("use_rep"),
            "n_feats": config.get("n_feats"),
            "preprocess": config.get("preprocess", {}),
            "gmm": config.get("gmm", {}),
            "train_test": config.get("train_test", {}),
            "tune": config.get("tune", {}),
            "approx_error_bound": config.get("approx_error_bound", {}),
        }
    )
    return "__".join(parts + [h])


def get_method_train_test_kwargs(
    config: dict,
    method_name: str,
    tuned_params: dict | None = None,
) -> dict:
    """Build train_test kwargs for a GGML method.

    Merge order (later wins):
      config["train_test"]  ←  method["train_test"]  ←  tuned_params

    Ground metric is added when ``method["ground_metric"] != "learned"``.

    Assumes ``config["train_test"]`` already contains the resolved values
    (defaults.yml merged with experiment overrides by Snakemake).
    """
    method = config["methods"][method_name]
    kw = dict(config.get("train_test", {}))
    kw.update(method.get("train_test", {}))
    if tuned_params:
        kw.update(tuned_params)
    # Keep timing metrics available in scores.csv unless explicitly disabled.
    kw.setdefault("time_inference", True)
    gm = method.get("ground_metric", "learned")
    if gm != "learned":
        kw["ground_metric"] = gm
    return kw


def get_method_tune_kwargs(config: dict, method_name: str) -> dict:
    """Build tune kwargs for a GGML method.

    Merge order (later wins):
      config["tune"]  ←  method["train_test"] (missing keys only)  ←  method["tune"]

    This keeps method-level fixed training kwargs (for example
    ``diag_bures_approx``) available during tuning without letting them
    accidentally overwrite the global tuning grid unless explicitly
    requested via ``method["tune"]``.
    """
    method = config["methods"][method_name]
    kw = dict(config.get("tune", {}))
    kw.pop("enabled", None)
    for key, value in method.get("train_test", {}).items():
        kw.setdefault(key, value)
    kw.update(method.get("tune", {}))
    return kw


def is_tunable_method(method_cfg: dict) -> bool:
    """Return True for learned internal methods that should go through tuning."""
    return (
        "external" not in method_cfg
        and method_cfg.get("ground_metric", "learned") == "learned"
        and method_cfg.get("tune_enabled", True)
    )


def get_bench_hash(
    method_name: str,
    train_kwargs: dict,
    dataset_construction_params: dict,
) -> str:
    """Hash for benchmark output path — covers all result-relevant params.

    Excludes display/debug kwargs (verbose, print_table, plot_split, plot_iter).
    """
    _EXCLUDED = {"verbose", "print_table", "plot_split", "plot_iter"}
    relevant = {
        "method": method_name,
        "train": {k: v for k, v in train_kwargs.items() if k not in _EXCLUDED},
        "dataset": dataset_construction_params,
    }
    return _config_hash(relevant)


def get_tune_hash(
    method_name: str,
    tune_grid: dict,
    dataset_construction_params: dict,
) -> str:
    """Hash for tune output path."""
    _EXCLUDED = {"verbose", "print_latex", "plot_contour", "return_dataset"}
    relevant = {
        "method": method_name,
        "tune": {k: v for k, v in tune_grid.items() if k not in _EXCLUDED},
        "dataset": dataset_construction_params,
    }
    return _config_hash(relevant)


# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Configure file logging and return the root logger."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path),
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def resolve_use_rep(use_rep_str: str | None) -> str | None:
    """Normalise use_rep wildcard value to None when it means 'use .X'."""
    if use_rep_str in (None, "None", "X"):
        return None
    return use_rep_str


def get_dataset_source(ds_meta: dict) -> str:
    """Return the declared dataset source."""
    return ds_meta.get("source", "cellxgene")


def get_synthetic_gmm_source(ds_meta: dict) -> str | None:
    """Return how a synthetic dataset should provide its GMM artifact."""
    if get_dataset_source(ds_meta) != "synthetic_gmm":
        return None
    return ds_meta.get("synthetic", {}).get("gmm_source", "fit_gmm")


def get_method_representation_source(method_cfg: dict) -> str:
    """Resolve whether a method is built from the raw dataset or fitted GMM."""
    dataset_type = method_cfg["dataset_type"]
    if dataset_type not in {"empirical", "centroid", "gmm"}:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    source = method_cfg.get("representation_source")
    if source is None:
        return "fitted_gmm"
    if source not in {"fitted_gmm", "raw_dataset"}:
        raise ValueError(
            f"Unsupported representation_source={source!r}. Expected 'fitted_gmm' or 'raw_dataset'."
        )
    if dataset_type == "gmm" and source != "fitted_gmm":
        raise ValueError("dataset_type='gmm' requires representation_source='fitted_gmm'.")
    return source


def method_uses_fitted_gmm(method_cfg: dict) -> bool:
    """Return True when a method should load the fitted GMM-backed AnnData."""
    return get_method_representation_source(method_cfg) == "fitted_gmm"


def get_dataset_construction_params(config: dict, ds_meta: dict, method_cfg: dict, gmm_key: str) -> dict:
    """Return the dataset-construction metadata used for hashing/provenance."""
    dataset_type = method_cfg["dataset_type"]
    representation_source = get_method_representation_source(method_cfg)
    use_rep = get_dataset_use_rep(config, ds_meta)
    representation_label = get_dataset_representation_label(config, ds_meta)
    base = {
        "dataset_source": get_dataset_source(ds_meta),
        "representation_label": representation_label,
        "use_rep": use_rep,
        "dataset_type": dataset_type,
        "representation_source": representation_source,
        "from_anndata": config.get("from_anndata", {}),
        "patient_col": ds_meta["patient_col"],
        "label_col": ds_meta["label_col"],
    }
    if get_dataset_source(ds_meta) == "synthetic_gmm":
        base["synthetic"] = ds_meta.get("synthetic", {})
        base["synth"] = ds_meta.get("synth", {})
    if representation_source == "fitted_gmm":
        base["gmm_key"] = gmm_key
        if dataset_type == "empirical":
            base["gmm_postprocess"] = "sample_gmm"
        elif dataset_type == "centroid":
            base["gmm_postprocess"] = "drop_covariances"
        else:
            base["gmm_postprocess"] = "parametric_gmm"
    elif dataset_type == "centroid":
        base["group_by"] = ds_meta["cell_subtype_col"]
    return base


def _empty_dataset_tensor(dataset):
    """Return an empty tensor matching the dataset backend/device."""
    return dataset.supports.new_empty((0,))


def _drop_dataset_covariances(dataset):
    """Clear covariances using the package convention of empty tensors/arrays."""
    dataset.covariances = _empty_dataset_tensor(dataset)
    if hasattr(dataset, "adata"):
        preprocessed = dataset.adata.uns.setdefault("ggml_preprocessed", {})
        preprocessed["covariances"] = np.empty((0,), dtype=np.float32)
    return dataset


def slim_adata_after_build(dataset, gmm_key: str = "", drop_expression: bool = False, logger=None):
    """Drop large in-memory AnnData payloads that are unused after dataset build.

    Once :func:`build_dataset` has extracted the GMM into the dataset tensors
    (``supports``/``covariances``/``weights``), the fitted GMM arrays under
    ``uns[gmm_key]['model']`` are a redundant in-RAM copy, and ``raw``, the kNN
    graph (``uns['neighbors']``) and the antibody side-matrices are only used
    upstream (preprocessing / GMM fitting) or persisted to disk. Tuning and
    benchmarking operate purely on the dataset tensors, so removing these in
    place shrinks the resident AnnData — and every train/test subset copied from
    it during cross-validation. This mutates only the in-memory object, not the
    on-disk h5ad.

    Pass ``drop_expression=True`` only for fitted-GMM methods, whose supports come
    from the GMM and are independent of ``adata.X``/``layers``. Raw-dataset
    baselines build their supports from ``X`` and must keep it.
    """
    adata = getattr(dataset, "adata", None)
    if adata is None:
        return dataset

    dropped = []

    # Dense expression matrix: dead weight for fitted-GMM datasets (supports live
    # in the dataset tensors). Not touched for X-based baselines.
    if drop_expression:
        if adata.X is not None:
            adata.X = None
            dropped.append("X")
        if len(adata.layers) > 0:
            adata.layers.clear()
            dropped.append("layers")

    # Redundant fitted-GMM arrays (mu/pi/var already materialised in
    # dataset.supports / weights / covariances). Keep the rest of the schema
    # (distribution ids, weight-inference metadata).
    gmm_cfg = adata.uns.get(gmm_key) if gmm_key else None
    if isinstance(gmm_cfg, Mapping) and gmm_cfg.pop("model", None) is not None:
        dropped.append(f"uns/{gmm_key}/model")

    # Upstream-only / disk-only payloads not read during tuning or benchmarking.
    if adata.raw is not None:
        adata.raw = None
        dropped.append("raw")
    for key in [k for k in adata.uns if k == "neighbors" or k.startswith("antibody_")]:
        del adata.uns[key]
        dropped.append(f"uns/{key}")

    if logger is not None and dropped:
        logger.info(f"Slimmed in-memory AnnData; dropped: {', '.join(dropped)}")

    return dataset


def build_dataset(adata, config: dict, ds_meta: dict, method_cfg: dict, gmm_key: str):
    """Construct the ggml_ot dataset for one method/dataset combination."""
    import ggml_ot

    use_rep = get_dataset_use_rep(config, ds_meta)
    base_kwargs = {
        "patient_col": ds_meta["patient_col"],
        "label_col": ds_meta["label_col"],
        "use_rep": use_rep,
        **config.get("from_anndata", {}),
    }
    dataset_type = method_cfg["dataset_type"]
    representation_source = get_method_representation_source(method_cfg)
    if dataset_type == "empirical":
        if representation_source == "raw_dataset":
            return ggml_ot.from_anndata(adata, **base_kwargs)
        return ggml_ot.from_anndata(adata, **base_kwargs, gmm_key=gmm_key, sample_gmm=True)
    if dataset_type == "centroid":
        if representation_source == "raw_dataset":
            return ggml_ot.from_anndata(adata, **base_kwargs, group_by=ds_meta["cell_subtype_col"])
        dataset = ggml_ot.from_anndata(adata, **base_kwargs, gmm_key=gmm_key)
        return _drop_dataset_covariances(dataset)
    if dataset_type == "gmm":
        return ggml_ot.from_anndata(adata, **base_kwargs, gmm_key=gmm_key)
    raise ValueError(f"Unknown dataset_type: {dataset_type!r}")


def apply_runtime_settings(config: dict) -> None:
    """Apply runtime settings from the resolved Snakemake config."""
    import ggml_ot

    if config.get("training_device"):
        ggml_ot.settings.device = config["training_device"]
    if config.get("n_threads"):
        ggml_ot.settings.n_threads = config["n_threads"]
    if config.get("random_seed") is not None:
        ggml_ot.settings.random_seed = config["random_seed"]
    if config.get("init_strategy"):
        ggml_ot.settings.init_strategy = config["init_strategy"]


def get_benchmark_plot_config(config: dict) -> dict:
    """Normalise benchmark plot settings from config."""
    import ggml_ot

    plot_cfg = dict(config.get("benchmark_plots", {}))
    plot_types = plot_cfg.get("types", ["clustermap", "embedding"])
    if isinstance(plot_types, str):
        plot_types = [plot_types]
    plot_types = [str(plot_type) for plot_type in plot_types]
    valid_plot_types = {"clustermap", "embedding", "clustermap_embedding"}
    unknown = [plot_type for plot_type in plot_types if plot_type not in valid_plot_types]
    if unknown:
        raise ValueError(f"Unknown benchmark plot types: {unknown}")
    return {
        "enabled": bool(plot_cfg.get("enabled", False)) and bool(plot_types),
        "types": plot_types,
        "format": str(plot_cfg.get("format", "pdf")),
        "dpi": int(plot_cfg.get("dpi", ggml_ot.settings.figdpi)),
    }


def get_train_test_plot_kwargs(plot_cfg: dict, plot_root: str | Path, title: str) -> dict:
    """Translate benchmark plot config into ggml_ot.train_test kwargs."""
    if not plot_cfg["enabled"]:
        return {"plot_split": False, "plot_type": False}

    plot_type = plot_cfg["types"][0] if len(plot_cfg["types"]) == 1 else list(plot_cfg["types"])
    return {
        "plot_split": True,
        "plot_type": plot_type,
        "plot_split_dir": str(plot_root),
        "plot_title": title,
    }


def get_tune_plot_config(config: dict) -> dict:
    """Normalise tune contour settings from config."""
    import ggml_ot

    plot_cfg = dict(config.get("tune_plots", {}))
    value_col = plot_cfg.get("value_col", ["knn", "mean"])
    if isinstance(value_col, str):
        value_col = [value_col]
    if len(value_col) != 2:
        raise ValueError("tune_plots.value_col must have exactly two entries, e.g. ['knn_acc', 'Mean']")
    return {
        "enabled": bool(plot_cfg.get("enabled", False)),
        "x": plot_cfg.get("x"),
        "y": plot_cfg.get("y"),
        "fixed_params": dict(plot_cfg.get("fixed_params", {})),
        "value_col": tuple(value_col),
        "levels": int(plot_cfg.get("levels", 20)),
        "log_axis": plot_cfg.get("log_axis", "auto"),
        "format": str(plot_cfg.get("format", "pdf")),
        "dpi": int(plot_cfg.get("dpi", ggml_ot.settings.figdpi)),
    }


def infer_tune_contour_axes(scores_df, plot_cfg: dict) -> list[tuple[str, str]]:
    """Return contour axis pairs to plot.

    If ``plot_cfg`` specifies explicit ``x`` and ``y``, return that single pair.
    Otherwise, return all unordered pairs of hyperparameters that vary in the
    tuning grid (i.e. have more than one unique value).
    """
    from itertools import combinations

    x = plot_cfg.get("x")
    y = plot_cfg.get("y")
    if x is not None and y is not None:
        return [(str(x), str(y))]

    varied = [
        name
        for name in scores_df.index.names
        if name is not None
        and len(scores_df.index.get_level_values(name).unique()) > 1
    ]

    if len(varied) < 2:
        return []

    return list(combinations(varied, 2))


def resolve_tune_log_axis(scores_df, x: str, y: str, requested: bool | str) -> bool | str:
    """Resolve automatic log-axis selection for contour plots."""
    if requested != "auto":
        return requested

    def _is_log_candidate(name: str) -> bool:
        try:
            values = scores_df.index.get_level_values(name).unique()
        except KeyError:
            return False
        values = [float(value) for value in values]
        return bool(values) and all(value > 0 for value in values) and name in {"alpha", "reg", "mi_reg"}

    x_log = _is_log_candidate(x)
    y_log = _is_log_candidate(y)
    if x_log and y_log:
        return True
    if x_log:
        return "x"
    if y_log:
        return "y"
    return False


@contextmanager
def temporary_plot_settings(*, figdir=None, figformat=None, figdpi=None):
    """Temporarily override ggml_ot figure settings."""
    import ggml_ot

    previous = {
        "figdir": ggml_ot.settings.figdir,
        "figformat": ggml_ot.settings.figformat,
        "figdpi": ggml_ot.settings.figdpi,
    }
    try:
        if figdir is not None:
            ggml_ot.settings.figdir = figdir
        if figformat is not None:
            ggml_ot.settings.figformat = figformat
        if figdpi is not None:
            ggml_ot.settings.figdpi = figdpi
        yield
    finally:
        ggml_ot.settings.figdir = previous["figdir"]
        ggml_ot.settings.figformat = previous["figformat"]
        ggml_ot.settings.figdpi = previous["figdpi"]


def list_artifact_files(root: str | Path) -> list[str]:
    """Return all files below a directory in deterministic order."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(str(path) for path in root.rglob("*") if path.is_file())


def to_jsonable(obj):
    """Recursively convert an object to a JSON-serialisable form."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return str(obj)
