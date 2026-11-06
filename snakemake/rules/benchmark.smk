"""Benchmark rule: train (learned metrics) and evaluate one method on one dataset."""


def _bench_input_h5ad(wildcards):
    """Select the AnnData backing the configured dataset representation."""
    return method_h5ad_path(wildcards.dataset, wildcards.method_name)


def _bench_input_tuned_params(wildcards):
    """Return path to best_params.json for tunable methods, else empty list."""
    method = config["methods"][wildcards.method_name]
    if not is_tunable_method(method):
        return []
    if not config.get("tune", {}).get("enabled", False):
        return []
    return tune_path(wildcards.dataset, wildcards.method_name, "best_params.json")


rule benchmark_method:
    """Train and evaluate one (dataset, method) pair."""
    input:
        h5ad=_bench_input_h5ad,
        tuned_params=_bench_input_tuned_params
    output:
        scores=str(DATA_ROOT / "{dataset}" / "benchmark" / "{method_name}" / "{bench_hash}" / "scores.csv"),
        scores_per_split=str(DATA_ROOT / "{dataset}" / "benchmark" / "{method_name}" / "{bench_hash}" / "scores_per_split.csv"),
        metric=str(DATA_ROOT / "{dataset}" / "benchmark" / "{method_name}" / "{bench_hash}" / "trained_metric.pkl"),
        manifest=str(DATA_ROOT / "{dataset}" / "benchmark" / "{method_name}" / "{bench_hash}" / "benchmark_manifest.json"),
        plots=directory(str(DATA_ROOT / "{dataset}" / "benchmark" / "{method_name}" / "{bench_hash}" / "train_test_plots"))
    params:
        config=lambda w: get_benchmark_rule_config(config, w.method_name),
        ds_meta=lambda w: get_rule_ds_meta(config["datasets"][w.dataset]),
        method_name=lambda w: w.method_name,
        gmm_key=lambda w: dataset_gmm_key(w.dataset)
    log:
        str(DATA_ROOT / "{dataset}" / "logs" / "benchmark_{method_name}_{bench_hash}.log")
    benchmark:
        # wall/CPU time and memory of the job, for the runtime table
        str(DATA_ROOT / "{dataset}" / "timings" / "benchmark_{method_name}_{bench_hash}.tsv")
    resources:
        mem_mb=lambda w: _get_resources("benchmark", "mem_mb", 32000),
        runtime=lambda w: _get_resources("benchmark", "runtime", 240),
        gpu=lambda w: 1 if config.get("training_device", "cpu") != "cpu" else 0
    script:
        "../scripts/benchmark.py"
