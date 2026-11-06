"""Exact vs diagonal Bures patient distances, from the benchmark's trained metrics."""

DISTANCE_AGREEMENT_CONFIG = {
    "exact_method": "GMM_FullBures",
    "diag_method": "GMM_DiagBures",
    "datasets": list(config["datasets"].keys()),
    "dataset_labels": {},
    **dict(config.get("distance_agreement", {})),
}
DISTANCE_AGREEMENT_DATASETS = [
    dataset for dataset in DISTANCE_AGREEMENT_CONFIG["datasets"] if dataset in config["datasets"]
]
# Runs only when both trained methods are part of the run (not in, e.g., single-method overlays).
DISTANCE_AGREEMENT_ENABLED = bool(DISTANCE_AGREEMENT_DATASETS) and all(
    DISTANCE_AGREEMENT_CONFIG[key] in ALL_METHODS for key in ("exact_method", "diag_method")
)


def distance_agreement_path(suffix: str) -> str:
    return aggregate_output_path(f"distance_agreement/{suffix}")


rule distance_agreement_dataset:
    """Patient distance matrices under exact and diagonal Bures for one dataset."""
    wildcard_constraints:
        dataset="|".join(DISTANCE_AGREEMENT_DATASETS) or "^$"
    input:
        h5ad=lambda w: gmm_path(w.dataset),
        exact_metric=lambda w: bench_path(w.dataset, DISTANCE_AGREEMENT_CONFIG["exact_method"], "trained_metric.pkl"),
        diag_metric=lambda w: bench_path(w.dataset, DISTANCE_AGREEMENT_CONFIG["diag_method"], "trained_metric.pkl"),
    output:
        per_split=distance_agreement_path("{dataset}/spearman_per_split.csv"),
        matrices=distance_agreement_path("{dataset}/distance_matrices.npz"),
    params:
        config=lambda w: get_benchmark_rule_config(config, DISTANCE_AGREEMENT_CONFIG["exact_method"]),
        method_cfg=lambda w: config["methods"][DISTANCE_AGREEMENT_CONFIG["exact_method"]],
        ds_meta=lambda w: get_rule_ds_meta(config["datasets"][w.dataset]),
        gmm_key=lambda w: dataset_gmm_key(w.dataset),
    log:
        distance_agreement_path("{dataset}/distance_agreement.log")
    resources:
        mem_mb=lambda w: _get_resources("benchmark", "mem_mb", 32000),
    script:
        "../scripts/distance_agreement_dataset.py"


rule distance_agreement_table:
    """Spearman agreement table across datasets."""
    input:
        per_split=[distance_agreement_path(f"{dataset}/spearman_per_split.csv") for dataset in DISTANCE_AGREEMENT_DATASETS],
    output:
        csv=distance_agreement_path("distance_agreement.csv"),
        table=distance_agreement_path("distance_agreement.tex"),
        table_per_dataset=distance_agreement_path("distance_agreement_per_dataset.tex"),
    params:
        dataset_names=DISTANCE_AGREEMENT_DATASETS,
        dataset_labels=DISTANCE_AGREEMENT_CONFIG["dataset_labels"],
        n_splits=int(config.get("train_test", {}).get("n_splits", 5)),
    log:
        distance_agreement_path("distance_agreement_table.log")
    script:
        "../scripts/distance_agreement_table.py"
