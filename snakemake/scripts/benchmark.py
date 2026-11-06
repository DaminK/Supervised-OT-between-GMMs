"""
Run a single benchmark: train (or evaluate) one method on one dataset.

Handles both GGML methods (via ggml_ot.train_test) and external methods
(WDA/FDA via EXTERNAL_REGISTRY → ggml_ot.benchmark.train_test_external).

Called via Snakemake ``script:`` directive.  The ``snakemake`` object provides:
  snakemake.input.h5ad            — preprocessed or GMM-fitted adata
  snakemake.input.tuned_params    — path to best_params.json (may be empty list)
  snakemake.output.scores         — CSV of benchmark scores
  snakemake.output.scores_per_split — CSV retaining each evaluation split
  snakemake.output.metric         — pickle of the ground metrics: per-split dict for
                                    trained methods, metric name for fixed ones
  snakemake.output.manifest       — JSON manifest for provenance
  snakemake.params.config         — benchmark-relevant config subset for this method
  snakemake.params.ds_meta        — dataset metadata dict
  snakemake.params.method_name    — name of the method
  snakemake.params.gmm_key        — GMM key (empty string for non-GMM methods)
  snakemake.log[0]
"""

import json
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _utils import (  # noqa: E402
    apply_runtime_settings,
    build_dataset,
    get_benchmark_plot_config,
    get_method_train_test_kwargs,
    get_train_test_plot_kwargs,
    list_artifact_files,
    method_uses_fitted_gmm,
    setup_logging,
    slim_adata_after_build,
    temporary_plot_settings,
    to_jsonable,
)
from external_methods import EXTERNAL_REGISTRY  # noqa: E402
from _provenance import ggml_ot_provenance  # noqa: E402
from _split_scores import SPLIT_SCORE_SCHEMA, build_split_scores_frame  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import scanpy as sc  # noqa: E402
import ggml_ot  # noqa: E402
from ggml_ot.benchmark import train_test_external  # noqa: E402


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    cfg = snakemake.params.config
    ds_meta = snakemake.params.ds_meta
    method_name = snakemake.params.method_name
    gmm_key = snakemake.params.gmm_key

    method = cfg["methods"][method_name]

    out_dir = Path(snakemake.output.scores).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_root = Path(snakemake.output.plots)
    plot_root.mkdir(parents=True, exist_ok=True)

    apply_runtime_settings(cfg)

    logger.info(
        f"Benchmarking method={method_name}, dataset_type={method['dataset_type']}"
    )

    adata = sc.read_h5ad(snakemake.input.h5ad)
    dataset = build_dataset(adata, cfg, ds_meta, method, gmm_key)

    # Free large AnnData payloads now baked into the dataset tensors before
    # cross-validation deep-copies the AnnData per train/test split. Fitted-GMM
    # methods also drop the dense expression matrix (unused by their supports).
    slim_adata_after_build(
        dataset,
        gmm_key,
        drop_expression=method_uses_fitted_gmm(method),
        logger=logger,
    )

    # Load tuned params if provided
    tuned_params = None
    tuned_params_input = snakemake.input.get("tuned_params", [])
    if tuned_params_input:
        with open(tuned_params_input) as f:
            tuned_params = json.load(f)
        logger.info(f"Loaded tuned params: {tuned_params}")

    benchmark_plot_cfg = get_benchmark_plot_config(cfg)
    plot_kwargs = get_train_test_plot_kwargs(
        benchmark_plot_cfg,
        plot_root,
        method.get("display_name", method_name),
    )
    plot_context = (
        temporary_plot_settings(
            figdir=plot_root,
            figformat=benchmark_plot_cfg["format"],
            figdpi=benchmark_plot_cfg["dpi"],
        )
        if benchmark_plot_cfg["enabled"]
        else nullcontext()
    )

    with plot_context:
        if "external" in method:
            test_kwargs = get_method_train_test_kwargs(cfg, method_name)
            test_kwargs.update(plot_kwargs)
            logger.info(f"Benchmark kwargs: {test_kwargs}")

            # Iteration budget and latent dimension come from train_test, as for
            # GGML; the method's params can override them.
            train_kwargs = {key: test_kwargs[key] for key in ("max_iter", "n_comps") if key in test_kwargs}
            train_kwargs.update(method.get("params", {}))

            # Trained on the training distributions of each split, like GGML.
            ground_metrics, scores_df = train_test_external(
                dataset,
                train_fn=EXTERNAL_REGISTRY[method["external"]],
                train_kwargs=train_kwargs,
                **test_kwargs,
            )
            train_kw_for_manifest = {
                "external_params": train_kwargs,
                "train_test": dict(test_kwargs),
            }
        else:
            train_kwargs = get_method_train_test_kwargs(cfg, method_name, tuned_params)
            train_kwargs.update(plot_kwargs)
            logger.info(f"Benchmark kwargs: {train_kwargs}")

            ground_metrics, scores_df = ggml_ot.train_test(dataset, **train_kwargs)
            if ground_metrics is None:
                ground_metrics = train_kwargs.get("ground_metric")
            train_kw_for_manifest = dict(train_kwargs)

    # ── Save outputs ──────────────────────────────────────────────────────────

    split_config = (
        train_kw_for_manifest["train_test"]
        if "external" in method
        else train_kw_for_manifest
    )
    split_scores_df = build_split_scores_frame(
        scores_df,
        dataset,
        random_seed=cfg.get("random_seed"),
        split_config=split_config,
    )
    scores_df.to_csv(snakemake.output.scores)
    split_scores_df.to_csv(snakemake.output.scores_per_split, index=False)

    with open(snakemake.output.metric, "wb") as f:
        pickle.dump(ground_metrics, f)

    manifest = {
        "schema": "snakemake.benchmark.v2",
        "method_name": method_name,
        "dataset_type": method["dataset_type"],
        "train_kwargs": to_jsonable(train_kw_for_manifest),
        "tuned_params": to_jsonable(tuned_params),
        "plot_artifacts": to_jsonable(
            {
                "plot_root": str(plot_root),
                "plot_types": benchmark_plot_cfg["types"]
                if benchmark_plot_cfg["enabled"]
                else [],
                "files": list_artifact_files(plot_root),
            }
        ),
        "split_scores": {
            "schema": SPLIT_SCORE_SCHEMA,
            "path": str(snakemake.output.scores_per_split),
            "n_splits": len(split_scores_df),
            "ordered_patient_id_sha256": split_scores_df[
                "ordered_patient_id_sha256"
            ].iloc[0],
        },
        **ggml_ot_provenance(),
    }
    Path(snakemake.output.manifest).write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    logger.info(f"Wrote: {snakemake.output.scores}")
    logger.info(f"Wrote: {snakemake.output.scores_per_split}")


run(snakemake)  # noqa: F821
