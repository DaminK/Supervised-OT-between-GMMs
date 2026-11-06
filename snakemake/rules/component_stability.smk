"""Per-patient number-of-components check (appendix, GMM fitting).

Only reachable via the ``patient_component_stability_only`` target (overlay
config/ablation/_component_stability.yml). It adds no output to the benchmark and no global
defaults: the cohorts and methods are this config's own ``datasets`` and ``methods``.
Settings live under ``config['patient_component_stability']``.
"""

PCS_CONFIG = dict(config.get("patient_component_stability", {}))


def _pcs_predictions_path(dataset, method_name):
    return str(DATA_ROOT / dataset / "component_stability" / method_name / "patient_predictions.csv")


def pcs_output_path(suffix):
    """Dedicated output tree for the component-stability analysis.

    Deliberately NOT under plots/aggregate/: keeps every output isolated so this
    opt-in analysis can never overwrite benchmark/plot/export artifacts.
    """
    return str(DATA_ROOT / "component_stability_analysis" / AGGREGATE_LABEL / suffix)


def _pcs_tuned_params(wildcards):
    method = config["methods"][wildcards.method_name]
    if not is_tunable_method(method) or not config.get("tune", {}).get("enabled", False):
        return []
    return tune_path(wildcards.dataset, wildcards.method_name, "best_params.json")


rule patient_predictions:
    """Reconstruct held-out per-patient kNN outcomes for one (dataset, method)."""
    input:
        h5ad=lambda w: gmm_path(w.dataset),
        tuned_params=_pcs_tuned_params,
    output:
        predictions=str(DATA_ROOT / "{dataset}" / "component_stability" / "{method_name}" / "patient_predictions.csv"),
    params:
        config=lambda w: get_benchmark_rule_config(config, w.method_name),
        ds_meta=lambda w: config["datasets"][w.dataset],
        method_name=lambda w: w.method_name,
        gmm_key=lambda w: dataset_gmm_key(w.dataset),
        pcs=PCS_CONFIG,
    log:
        str(DATA_ROOT / "{dataset}" / "logs" / "patient_predictions_{method_name}.log"),
    resources:
        mem_mb=lambda w: _get_resources("benchmark", "mem_mb", 32000),
    script:
        "../scripts/patient_predictions.py"


rule patient_component_stability:
    """Relate per-patient held-out accuracy to selected GMM component counts."""
    input:
        predictions=[
            _pcs_predictions_path(dataset, method_name)
            for dataset in config["datasets"]
            for method_name in ALL_METHODS
        ],
    output:
        patients=pcs_output_path("component_count_patient_accuracy.csv"),
        associations=pcs_output_path("component_count_associations.csv"),
        plot=pcs_output_path("component_count_stability.pdf"),
        table=pcs_output_path("component_count_stability.tex"),
    params:
        display_names={
            m: config["methods"][m].get("display_name", m) for m in ALL_METHODS
        },
        pcs=PCS_CONFIG,
    log:
        pcs_output_path("component_count_stability.log"),
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 8000),
    script:
        "../scripts/patient_component_stability_agg.py"
