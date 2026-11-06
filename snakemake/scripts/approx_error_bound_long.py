"""
Map faithfulness table, standard normal rows: one setup trained longer and with larger weights.

A map from a standard normal initialization starts with ||A||_F^2 ~ dD, where the weights tuned
for the orthonormal start barely act within the benchmark's training. This script trains one
setup (``approx_error_bound_long_training.setups``) of one reference method on one dataset as the
tuned replay (scripts/approx_error_bound_tuned.py: same config, dataset construction, train_test
kwargs, splits and random state) with the given initialization, weights and max_iter.

Called via Snakemake ``script:`` directive (one invocation per dataset, reference and setup).
The ``snakemake`` object provides:
  snakemake.params.h5ad          — fitted GMM-backed AnnData of the reference run
  snakemake.params.manifest      — benchmark manifest of the reference method (train kwargs)
  (params, not inputs: the reference files come from the main benchmark's own workflow)
  snakemake.params.ref_config    — the keys of the reference config read here and its method
  snakemake.params.method_name   — reference method
  snakemake.params.ds_meta       — dataset metadata dict
  snakemake.params.gmm_key       — GMM key of the reference run
  snakemake.params.dataset_name  — dataset name (from wildcard)
  snakemake.params.setup         — setup name (from wildcard), as the Figure 5 variant names
  snakemake.params.init          — initialization
  snakemake.params.reg           — Frobenius weight
  snakemake.params.mi_reg        — MI weight
  snakemake.params.max_iter      — maximum number of epochs
  snakemake.params.error         — approximation error options (approx_error_bound.error)
  snakemake.output.trained_maps  — final map per split (npz, keys "<init>__<setup>__split<i>")
  snakemake.output.final_errors  — one row per split: weights, final approximation errors,
                                   ||A||_F, MI, benchmark scores
  snakemake.log[0]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _approx_error import flatten_gmm_model_arrays, gaussian_emd_approx_errors, map_metrics  # noqa: E402
from _utils import (  # noqa: E402
    apply_runtime_settings,
    build_dataset,
    method_uses_fitted_gmm,
    setup_logging,
    slim_adata_after_build,
)

import ggml_ot  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402

# Benchmark kwargs that only control plots and printing
OUTPUT_KWARGS = {"plot_split": False, "plot_split_dir": None, "plot_title": None, "plot_type": False,
                 "print_table": False, "verbose": False}


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    cfg = snakemake.params.ref_config
    method_name = snakemake.params.method_name
    method_cfg = cfg["methods"][method_name]
    gmm_key = str(snakemake.params.gmm_key)
    error_cfg = dict(snakemake.params.error or {})
    setup, init = str(snakemake.params.setup), str(snakemake.params.init)
    reg, mi_reg = float(snakemake.params.reg), float(snakemake.params.mi_reg)

    reference_files = {name: Path(getattr(snakemake.params, name)) for name in ("h5ad", "manifest")}
    missing = [str(path) for path in reference_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run the main benchmark first; missing: {missing}")

    # As in scripts/approx_error_bound_tuned.py, up to the call of train_test; the random state
    # after building the dataset is the one every replay setup starts from
    apply_runtime_settings(cfg)
    adata = sc.read_h5ad(reference_files["h5ad"])
    model = adata.uns[gmm_key]["model"]
    means, covariances, weights = flatten_gmm_model_arrays(model)
    dataset = build_dataset(adata, cfg, dict(snakemake.params.ds_meta), method_cfg, gmm_key)
    slim_adata_after_build(dataset, gmm_key, drop_expression=method_uses_fitted_gmm(method_cfg), logger=logger)

    manifest = json.loads(reference_files["manifest"].read_text())
    kwargs = {**manifest["train_kwargs"], **OUTPUT_KWARGS,
              "reg": reg, "mi_reg": mi_reg, "max_iter": int(snakemake.params.max_iter)}
    ggml_ot.settings.init_strategy = init
    logger.info(f"Reference {method_name}: init={init} setup={setup} reg={reg} mi_reg={mi_reg} "
                f"max_iter={kwargs['max_iter']}")
    metrics, scores = ggml_ot.train_test(dataset, **kwargs)
    per_split = scores.attrs["per_split"]

    maps, rows = {}, []
    for split_idx, linear_map in enumerate(metrics["Ws"]):
        linear_map = np.asarray(linear_map)
        maps[f"{init}__{setup}__split{split_idx}"] = linear_map
        errors = gaussian_emd_approx_errors(
            means, covariances, linear_map,
            weights=weights if error_cfg.get("weight_by_component", False) else None,
            min_distance_threshold=float(error_cfg.get("min_distance_threshold", 0.0)),
            min_active_fraction=float(error_cfg.get("min_active_fraction", 0.5)),
        )
        rows.append({
            "dataset": snakemake.params.dataset_name, "init": init, "variant_name": setup,
            "split": split_idx, "epoch": float("nan"), "reg": reg, "mi_reg": mi_reg,
            **errors, **map_metrics(covariances, weights, linear_map),
            **{f"benchmark_{name}": float(values[split_idx]) for name, values in per_split.items()},
        })
    logger.info(", ".join(f"{name}={np.mean(values):.3f}" for name, values in per_split.items()))

    np.savez(snakemake.output.trained_maps, **maps)
    pd.DataFrame(rows).to_csv(snakemake.output.final_errors, index=False)


run(snakemake)  # noqa: F821
