"""Tuning rule: hyperparameter tuning of the learned-metric methods."""


def _tune_input_h5ad(wildcards):
    """Select the AnnData backing the configured dataset representation."""
    return method_h5ad_path(wildcards.dataset, wildcards.method_name)


rule tune_method:
    """Tune hyperparameters for one (dataset, method) pair."""
    input:
        h5ad=_tune_input_h5ad
    output:
        best_params=str(DATA_ROOT / "{dataset}" / "tune" / "{method_name}" / "{tune_hash}" / "best_params.json"),
        scores=str(DATA_ROOT / "{dataset}" / "tune" / "{method_name}" / "{tune_hash}" / "scores.csv"),
        manifest=str(DATA_ROOT / "{dataset}" / "tune" / "{method_name}" / "{tune_hash}" / "tune_manifest.json"),
        plots=directory(str(DATA_ROOT / "{dataset}" / "tune" / "{method_name}" / "{tune_hash}" / "plots"))
    params:
        config=lambda w: get_tune_rule_config(config, w.method_name),
        ds_meta=lambda w: get_rule_ds_meta(config["datasets"][w.dataset]),
        method_name=lambda w: w.method_name,
        gmm_key=lambda w: dataset_gmm_key(w.dataset)
    log:
        str(DATA_ROOT / "{dataset}" / "logs" / "tune_{method_name}_{tune_hash}.log")
    benchmark:
        # wall/CPU time and memory of the job, for the runtime table
        str(DATA_ROOT / "{dataset}" / "timings" / "tune_{method_name}_{tune_hash}.tsv")
    resources:
        mem_mb=lambda w: _get_resources("tune", "mem_mb", 32000),
        runtime=lambda w: _get_resources("tune", "runtime", 480),
        gpu=lambda w: 1 if config.get("training_device", "cpu") != "cpu" else 0
    script:
        "../scripts/tune.py"
