"""Per-patient held-out kNN predictions for one (dataset, method) pair.

First step of the per-patient number-of-components check (appendix, GMM fitting;
target patient_component_stability_only). A post-hoc step that does not touch the
benchmark outputs. It runs
the same evaluation (``ggml_ot.train_test``) with the analysis split count and
reuses the returned per-split ground metrics and exact train/test indices to
reconstruct one held-out kNN outcome per patient and split. Each reconstructed
split accuracy is checked against the package's own kNN score before any row is
kept, so predictions cannot silently drift from the benchmark.

Inputs (snakemake):
  input.h5ad         — fitted-GMM AnnData
  input.tuned_params — best_params.json (tunable methods) or empty
  params.config      — benchmark-rule config subset for this method
  params.ds_meta     — dataset metadata
  params.method_name — method name
  params.gmm_key     — GMM key
  params.pcs         — patient_component_stability knobs (n_splits, seed, knn_*)
  output.predictions — per-patient/per-split CSV
"""


import json
import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import ggml_ot  # noqa: E402

from _utils import build_dataset, get_method_train_test_kwargs, setup_logging  # noqa: E402
from _split_scores import _dataset_patient_ids  # noqa: E402

# Kwargs consumed by train_test itself; strip them before reusing compute_OT.
_TRAIN_TEST_ONLY = {
    "cluster_linkage", "ground_metric", "knn_k", "knn_weights", "n_splits",
    "plot_split", "plot_split_dir", "plot_title", "plot_type", "print_latex",
    "print_table", "return_dataset", "scoring", "split_strategy", "test_size",
    "time_inference", "train_size", "measure_time", "symbols",
}


def _split_ground_metrics(metric_payload, n_splits):
    if isinstance(metric_payload, dict) and "Ws" in metric_payload:
        metrics = list(metric_payload["Ws"])
        if len(metrics) != n_splits:
            raise ValueError(f"{len(metrics)} ground metrics for {n_splits} splits.")
        return metrics
    if metric_payload is None:
        raise ValueError("No ground metric returned; cannot reconstruct predictions.")
    return [metric_payload] * n_splits


def _component_counts(dataset, gmm_key, n_dist):
    gmm = dataset.adata.uns[gmm_key]
    values = gmm.get("distribution_n_components")
    if values is None:
        values = np.full(n_dist, int(gmm["n_components"]))
    values = np.asarray(values, dtype=int)
    if values.shape != (n_dist,):
        raise ValueError(f"component counts {values.shape} != ({n_dist},)")
    return values


def run(snakemake):
    logger = setup_logging(snakemake.log[0])
    cfg = snakemake.params.config
    ds_meta = snakemake.params.ds_meta
    method_name = snakemake.params.method_name
    gmm_key = snakemake.params.gmm_key
    pcs = dict(snakemake.params.pcs)
    method = cfg["methods"][method_name]

    tuned = None
    tuned_input = snakemake.input.get("tuned_params", [])
    if tuned_input:
        tuned = json.loads(Path(tuned_input).read_text())

    adata = sc.read_h5ad(snakemake.input.h5ad)
    dataset = build_dataset(adata, cfg, ds_meta, method, gmm_key)

    knn_k = int(pcs.get("knn_k", 5))
    knn_weights = str(pcs.get("knn_weights", "distance"))
    kwargs = get_method_train_test_kwargs(cfg, method_name, tuned)
    kwargs.update(
        n_splits=int(pcs["n_splits"]), knn_k=knn_k, knn_weights=knn_weights,
        plot_type=False, plot_split=False, print_table=False, time_inference=False,
    )
    logger.info(f"train_test {method_name}/{ds_meta.get('id', '')} n_splits={kwargs['n_splits']}")
    metric_payload, scores_df = ggml_ot.train_test(dataset, **kwargs)

    attrs = scores_df.attrs
    per_split = dict(attrs["per_split"])
    train_splits, test_splits = list(attrs["split_train_idx"]), list(attrs["split_test_idx"])
    n_splits = int(attrs.get("n_splits", len(test_splits)))
    if "knn" not in per_split:
        raise ValueError("kNN scoring is required for this analysis.")

    n_dist = len(dataset.distribution_labels)
    patient_ids = _dataset_patient_ids(dataset, n_dist)
    comp_counts = _component_counts(dataset, gmm_key, n_dist)
    metrics = _split_ground_metrics(metric_payload, n_splits)
    ot_kwargs = {k: v for k, v in kwargs.items() if k not in _TRAIN_TEST_ONLY}

    from ggml_ot.benchmark._knn import knn_classification

    rows = []
    for s, (tr, te, W) in enumerate(zip(train_splits, test_splits, metrics)):
        tr, te = np.asarray(tr, dtype=int), np.asarray(te, dtype=int)
        subset = dataset._subset(np.concatenate((tr, te)))
        symbols = ["train"] * len(tr) + ["test"] * len(te)
        distances = subset.compute_OT(ground_metric=W, symbols=symbols, plot_type=False,
                                      measure_time=False, **ot_kwargs)
        if hasattr(distances, "detach"):
            distances = distances.detach().cpu().numpy()
        distances = np.asarray(distances)
        labels = np.asarray(subset.distribution_labels_str, dtype=str)
        loc_tr = np.arange(len(tr))
        loc_te = np.arange(len(te)) + len(tr)
        acc, preds = knn_classification(distances, labels, loc_tr, loc_te,
                                        n_neighbors=knn_k, weights=knn_weights)
        if not np.isclose(acc, float(per_split["knn"][s])):
            raise ValueError(
                f"split {s}: reconstructed kNN {acc} != benchmark {per_split['knn'][s]}"
            )
        true = labels[loc_te]
        for i, dist_idx in enumerate(te.tolist()):
            pred, truth = str(preds[i]), str(true[i])
            rows.append({
                "dataset": ds_meta.get("id", ds_meta.get("label_col", "")),
                "method": method_name,
                "split_idx": s,
                "patient_id": patient_ids[dist_idx],
                "true_label": truth,
                "predicted_label": pred,
                "correct": int(pred == truth),
                "selected_n_components": int(comp_counts[dist_idx]),
            })

    out = pd.DataFrame(rows)
    out["dataset"] = snakemake.wildcards.dataset if hasattr(snakemake, "wildcards") else out["dataset"]
    Path(snakemake.output.predictions).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(snakemake.output.predictions, index=False)
    logger.info(f"wrote {snakemake.output.predictions}: {len(out)} rows, "
                f"{out['patient_id'].nunique()} patients")


run(snakemake)  # noqa: F821
