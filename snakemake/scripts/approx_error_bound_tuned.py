"""
Figure 5 (b2): the four regularization setups at the weights tuned for the main benchmark.

Replays the main benchmark's training of one reference method (an entry of
``approx_error_bound.tuned_references``, e.g. Ours (Diag.) or Ours (Exact)) on one dataset: same config, dataset construction, train_test kwargs
(30 epochs, early stopping, the 5 benchmark splits) and random state, with the tuned weights
(reg_t, mi_reg_t) swapped per setup: None (0, 0), Frobenius (reg_t, 0), MI (0, mi_reg_t) and
Both (reg_t, mi_reg_t), each with every initialization. Both with the benchmark's own
initialization reproduces the main benchmark run; the check is written to ``check``.

Called via Snakemake ``script:`` directive (one invocation per dataset).
The ``snakemake`` object provides:
  snakemake.params.h5ad             — fitted GMM-backed AnnData of the reference run
  snakemake.params.manifest         — benchmark manifest of the reference method (train kwargs,
                                      tuned params)
  snakemake.params.scores_per_split — per-split scores of the reference run (for the check)
  (params, not inputs: the reference files come from the main benchmark's own workflow)
  snakemake.output.trained_maps     — final map per init, setup and split (npz,
                                      keys "<init>__<setup>__split<i>")
  snakemake.output.final_errors     — one row per init, setup and split: weights, final
                                      approximation errors, ||A||_F, MI, benchmark scores
  snakemake.output.check            — JSON: reference files used (path, modification time) and
                                      per-split scores of the reproducing setup vs the reference
  snakemake.params.ref_config       — the keys of the reference config read here and its method
  snakemake.params.method_name      — reference method
  snakemake.params.ds_meta          — dataset metadata dict
  snakemake.params.gmm_key          — GMM key of the reference run
  snakemake.params.dataset_name     — dataset name (from wildcard)
  snakemake.params.inits            — initializations
  snakemake.params.error            — approximation error options (approx_error_bound.error)
  snakemake.log[0]
"""

import json
import random
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
import torch  # noqa: E402

# Benchmark kwargs that only control plots and printing
OUTPUT_KWARGS = {"plot_split": False, "plot_split_dir": None, "plot_title": None, "plot_type": False,
                 "print_table": False, "verbose": False}


def capture_random_state() -> dict:
    """Every random generator the training draws from (ggml-ot seeds torch's default generator)."""
    return {
        "torch": torch.get_rng_state(),
        "ggml_numpy": ggml_ot.settings.numpy_generator.bit_generator.state,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_random_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    ggml_ot.settings.numpy_generator.bit_generator.state = state["ggml_numpy"]
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def setups(tuned: dict) -> dict:
    """Regularization setups at the tuned weights, named as the Figure 5 variants."""
    reg, mi_reg = float(tuned.get("reg", 0.0)), float(tuned.get("mi_reg", 0.0))
    return {"NoReg": (0.0, 0.0), "Frobenius": (reg, 0.0), "MutualInformation": (0.0, mi_reg), "Both": (reg, mi_reg)}


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    cfg = snakemake.params.ref_config
    method_name = snakemake.params.method_name
    method_cfg = cfg["methods"][method_name]
    gmm_key = str(snakemake.params.gmm_key)
    error_cfg = dict(snakemake.params.error or {})

    reference_files = {name: Path(getattr(snakemake.params, name)) for name in ("h5ad", "manifest", "scores_per_split")}
    missing = [str(path) for path in reference_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run the main benchmark first; missing: {missing}")

    # As in scripts/benchmark.py, up to the call of train_test
    apply_runtime_settings(cfg)
    benchmark_init = ggml_ot.settings.init_strategy
    adata = sc.read_h5ad(reference_files["h5ad"])
    model = adata.uns[gmm_key]["model"]
    means, covariances, weights = flatten_gmm_model_arrays(model)
    dataset = build_dataset(adata, cfg, dict(snakemake.params.ds_meta), method_cfg, gmm_key)
    slim_adata_after_build(dataset, gmm_key, drop_expression=method_uses_fitted_gmm(method_cfg), logger=logger)
    state = capture_random_state()

    manifest = json.loads(reference_files["manifest"].read_text())
    tuned = manifest["tuned_params"]
    base_kwargs = {**manifest["train_kwargs"], **OUTPUT_KWARGS}
    logger.info(f"Reference {method_name}: tuned {tuned}, benchmark init {benchmark_init}")

    maps, rows, check = {}, [], {}
    for init in snakemake.params.inits:
        for setup, (reg, mi_reg) in setups(tuned).items():
            restore_random_state(state)
            ggml_ot.settings.init_strategy = init
            metrics, scores = ggml_ot.train_test(dataset, **{**base_kwargs, "reg": reg, "mi_reg": mi_reg})
            per_split = scores.attrs["per_split"]
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
            logger.info(f"init={init} setup={setup} reg={reg} mi_reg={mi_reg}: "
                        + ", ".join(f"{name}={np.mean(values):.3f}" for name, values in per_split.items()))
            if init == benchmark_init and setup == "Both":
                reference = pd.read_csv(reference_files["scores_per_split"])
                check = {name: {"replayed": [float(v) for v in values],
                                "reference": reference[name].tolist() if name in reference else None}
                         for name, values in per_split.items()}
                check["reproduces"] = all(
                    v["reference"] is not None and np.allclose(v["replayed"], v["reference"])
                    for name, v in check.items() if name in ("knn", "ari", "ari_complete", "silhouette"))
                logger.info(f"Reproduces the reference run: {check['reproduces']}")

    np.savez(snakemake.output.trained_maps, **maps)
    pd.DataFrame(rows).to_csv(snakemake.output.final_errors, index=False)
    check["reference_files"] = {name: {"path": str(path), "modified": path.stat().st_mtime}
                                for name, path in reference_files.items()}
    with open(snakemake.output.check, "w") as f:
        json.dump(check, f, indent=2)


run(snakemake)  # noqa: F821
