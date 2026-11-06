"""
Common helpers available to all rule modules.

Defines path-construction helpers and re-exports _utils functions so
that Snakefiles and rules can call them without extra imports.
"""

import sys
from pathlib import Path

# Make scripts/ importable inside rule modules
sys.path.insert(0, str(Path(workflow.basedir) / "scripts"))

from scripts._utils import (
    get_gmm_key,
    get_bench_hash,
    get_tune_hash,
    get_dataset_construction_params,
    get_fit_gmm_rule_config,
    get_rule_ds_meta,
    get_tune_rule_config,
    get_benchmark_rule_config,
    get_plot_rule_config,
    get_approx_error_bound_rule_config,
    get_dataset_table_rule_config,
    get_dataset_source,
    get_dataset_representation_label,
    get_method_train_test_kwargs,
    get_method_tune_kwargs,
    is_tunable_method,
    method_uses_fitted_gmm,
    _config_hash,
)


def _get_resources(rule_type: str, key: str, default: int) -> int:
    """Look up per-rule resource from config['resources'] or fall back to default."""
    return config.get("resources", {}).get(rule_type, {}).get(key, default)


# Dataset groups for separate aggregate runs. "all" is always present.
# Named groups (scRNA / synth) are auto-derived from the dataset `source` field
# and dropped if only one named group would be non-empty (since that would
# byte-equal the "all" aggregate).
def _build_dataset_groups(datasets: dict) -> dict:
    scRNA = [d for d in datasets if get_dataset_source(datasets[d]) != "synthetic_gmm"]
    synth = [d for d in datasets if get_dataset_source(datasets[d]) == "synthetic_gmm"]
    groups = {"all": list(datasets.keys())}
    if scRNA and synth:
        groups["scRNA"] = scRNA
        groups["synth"] = synth
    return groups


DATASET_GROUPS = _build_dataset_groups(config["datasets"])


def dataset_rep_label(dataset: str) -> str:
    return get_dataset_representation_label(config, config["datasets"][dataset])


def prepared_path(dataset: str) -> str:
    ds_meta = config["datasets"][dataset]
    subdir = "generated" if get_dataset_source(ds_meta) == "synthetic_gmm" else "prepared"
    return str(DATA_ROOT / dataset / subdir / f"{dataset_rep_label(dataset)}.h5ad")


def raw_path(dataset: str) -> str:
    return str(DATA_ROOT / dataset / "raw.h5ad")


def dataset_gmm_key(dataset: str) -> str:
    return get_gmm_key(config, config["datasets"][dataset])


def _uses_gmvae() -> bool:
    """GMMs come from PILOT-GM-VAE (rules/gmvae.smk) instead of EM (rules/gmm.smk)."""
    return config.get("gmm", {}).get("provider") == "gmvae"


def gmm_path(dataset: str) -> str:
    gmm_dir = "gmm_gmvae" if _uses_gmvae() else "gmm"
    return str(DATA_ROOT / dataset / gmm_dir / f"{dataset_gmm_key(dataset)}.h5ad")


def dataset_plot_path(dataset: str, suffix: str) -> str:
    # GM-VAE runs get their own subfolder so they never overwrite the EM run's tables
    base = DATA_ROOT / "plots" / dataset / dataset_rep_label(dataset)
    return str((base / "gmvae" if _uses_gmvae() else base) / suffix)


def aggregate_output_path(suffix: str) -> str:
    return str(DATA_ROOT / "plots" / "aggregate" / AGGREGATE_LABEL / suffix)


def approx_error_bound_output_path(suffix: str) -> str:
    return aggregate_output_path(f"approx_error_bound/{suffix}")


def approx_error_bound_dataset_output_path(suffix: str) -> str:
    return aggregate_output_path(f"approx_error_bound/{{dataset}}/{suffix}")


def method_h5ad_path(dataset: str, method_name: str) -> str:
    method = config["methods"][method_name]
    if method_uses_fitted_gmm(method):
        return gmm_path(dataset)
    return prepared_path(dataset)


def tune_path(dataset: str, method_name: str, suffix: str) -> str:
    """Return the tune output directory path + suffix for a method."""
    ds_params = _dataset_construction_params(dataset, method_name)
    tune_grid = _tune_grid(method_name)
    h = get_tune_hash(method_name, tune_grid, ds_params)
    return str(DATA_ROOT / dataset / "tune" / method_name / h / suffix)


def bench_path(dataset: str, method_name: str, suffix: str) -> str:
    """Return the benchmark output directory path + suffix for a method."""
    ds_params = _dataset_construction_params(dataset, method_name)
    train_kw = get_method_train_test_kwargs(config, method_name)
    h = get_bench_hash(method_name, train_kw, ds_params)
    return str(DATA_ROOT / dataset / "benchmark" / method_name / h / suffix)


def _dataset_construction_params(dataset: str, method_name: str) -> dict:
    """Params that go into the dataset hash (all construction-relevant fields)."""
    method = config["methods"][method_name]
    ds_meta = config["datasets"][dataset]
    return get_dataset_construction_params(config, ds_meta, method, dataset_gmm_key(dataset))


def _tune_grid(method_name: str) -> dict:
    """Build the tune grid dict for a given method (for hashing)."""
    return get_method_tune_kwargs(config, method_name)
