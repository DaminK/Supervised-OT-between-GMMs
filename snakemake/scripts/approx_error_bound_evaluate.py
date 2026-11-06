"""
Figure 5 downstream evaluation of the trained maps of one dataset.

For every initialization and variant in the trained maps, the final maps of all splits are evaluated with the
benchmark's evaluation (ggml_ot.test, one map per split, the benchmark splits), once with the
diagonal and once with the exact Bures term. Each row also gives the agreement of the two
patient distance matrices and the final errors of the map.

Called via Snakemake ``script:`` directive (one invocation per dataset).
The ``snakemake`` object provides:
  snakemake.input.h5ad            — fitted GMM-backed AnnData for this dataset
  snakemake.input.trained_maps    — final map per init, variant and split (npz,
                                    keys "<init>__<variant>__split<i>")
  snakemake.input.metrics_csv     — per-split and per-epoch errors (approx_error_epochs.csv)
  snakemake.output.downstream_csv — one row per init, variant and split: scores with diagonal
                                    distances, scores with exact distances (suffix _exact),
                                    agreement (rho, rho_top10, rho_within), final errors,
                                    ||A||_F and MI
  snakemake.params.config         — resolved experiment config subset
  snakemake.params.dataset_name   — dataset name (from wildcard)
  snakemake.params.ds_meta        — dataset metadata dict
  snakemake.params.gmm_key        — GMM key for this dataset
  snakemake.log[0]
"""

import sys

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _distance_agreement import (  # noqa: E402
    nearest_neighbour_spearman,
    patient_distances,
    spearman,
    within_class_spearman,
)
from _utils import apply_runtime_settings, build_dataset, setup_logging  # noqa: E402

import ggml_ot  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402

# Evaluation as in the main benchmark (config/paper/main_benchmark.yml, train_test)
DEFAULT_EVALUATION = {
    "scoring": ["knn", "ari", "silhouette"],
    "knn_k": 5,
    "knn_weights": "distance",
    "cluster_linkage": ["average", "complete"],
}
FINAL_COLUMNS = ["approx_error", "relative_approx_error", "fro_norm", "projected_mi_weighted"]


def final_errors(metrics_csv: str) -> pd.DataFrame:
    """Last-epoch errors per init, variant and split."""
    metrics = pd.read_csv(metrics_csv, keep_default_na=False, na_values=[""])
    metrics = metrics[metrics["split"].astype(str) != "mean"].copy()
    metrics["split"] = metrics["split"].astype(int)
    final = metrics.sort_values("epoch").groupby(["init", "variant_name", "split"]).tail(1)
    return final.set_index(["init", "variant_name", "split"])


def per_split_scores(dataset, split_maps: list, n_splits: int, train_size: float, evaluation: dict) -> dict:
    """Benchmark scores per split with the diagonal and with the exact Bures term."""
    scores = {}
    for suffix, diag_bures_approx in (("", True), ("_exact", False)):
        table = ggml_ot.test(
            dataset,
            ground_metric=split_maps,
            n_splits=n_splits,
            train_size=train_size,
            plot_split=False,
            print_table=False,
            diag_bures_approx=diag_bures_approx,
            **evaluation,
        )
        scores.update({f"{name}{suffix}": values for name, values in table.attrs["per_split"].items()})
    return scores


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    cfg = snakemake.params.config
    dataset_name = str(snakemake.params.dataset_name)
    fig_cfg = dict(cfg.get("approx_error_bound") or {})

    method_cfg = dict(fig_cfg.get("method") or {})
    method_cfg.setdefault("dataset_type", "gmm")
    method_cfg.setdefault("ground_metric", "learned")
    split_cfg = dict(fig_cfg.get("split") or {})
    n_splits = int(split_cfg.get("n_splits", 3))
    train_size = float(split_cfg.get("train_size", 0.6))
    evaluation = {**DEFAULT_EVALUATION, **dict(fig_cfg.get("evaluation") or {})}
    inits = list(fig_cfg.get("inits", ["orthonormal"]))
    variants = list(dict(fig_cfg.get("variants") or {}))

    apply_runtime_settings(cfg)
    adata = sc.read_h5ad(snakemake.input.h5ad)
    dataset = build_dataset(adata, cfg, dict(snakemake.params.ds_meta), method_cfg, str(snakemake.params.gmm_key))
    labels = np.asarray(dataset.distribution_labels)
    maps = np.load(snakemake.input.trained_maps)
    final = final_errors(snakemake.input.metrics_csv)
    logger.info(f"Evaluating dataset={dataset_name}, inits={inits}, variants={variants}, evaluation={evaluation}")

    rows = []
    for init in inits:
        for variant_name in variants:
            if f"{init}__{variant_name}__split0" not in maps.files:
                continue  # runs that train only some of them (rule approx_error_bound_long_dataset)
            split_maps = [maps[f"{init}__{variant_name}__split{i}"] for i in range(n_splits)]
            scores = per_split_scores(dataset, split_maps, n_splits, train_size, evaluation)
            for split_idx, split_map in enumerate(split_maps):
                diag = patient_distances(dataset, split_map, True)
                exact = patient_distances(dataset, split_map, False)
                errors = final.loc[(init, variant_name, split_idx)]
                rows.append({
                    "dataset": dataset_name,
                    "init": init,
                    "variant_name": variant_name,
                    "split": split_idx,
                    "reg": errors["reg"],
                    "mi_reg": errors["mi_reg"],
                    "epochs": errors["epoch"],  # NaN for runs with early stopping (tuned setups)
                    **{column: float(errors[column]) for column in FINAL_COLUMNS},
                    **{name: float(values[split_idx]) for name, values in scores.items()},
                    "rho": spearman(exact, diag),
                    "rho_top10": nearest_neighbour_spearman(exact, diag),
                    "rho_within": within_class_spearman(exact, diag, labels),
                })
            logger.info(f"init={init} variant={variant_name}: " + ", ".join(
                f"{name}={np.mean(values):.3f}" for name, values in scores.items()))

    pd.DataFrame(rows).to_csv(snakemake.output.downstream_csv, index=False)
    logger.info(f"Wrote {snakemake.output.downstream_csv}")


run(snakemake)  # noqa: F821
