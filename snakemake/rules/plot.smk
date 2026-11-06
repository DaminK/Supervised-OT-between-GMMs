"""Plotting rules: per-dataset benchmark tables and cross-dataset aggregation."""

GMM_VALIDATION_DATASETS = [
    dataset
    for dataset in config["datasets"]
    if get_dataset_source(config["datasets"][dataset]) != "synthetic_gmm"
]

LATENT_DIM_ABLATION_PLOT_CFG = dict(config.get("latent_dim_ablation_plot", {}))

# Methods of Table 1; the per-dataset appendix tables show the same set.
MAIN_TABLE_METHODS = [m for m in ALL_METHODS if config["methods"][m].get("main_table", True)]


rule dataset_table:
    """Generate a dataset summary table for the active dataset config."""
    input:
        source_h5ads=[
            prepared_path(dataset) if get_dataset_source(config["datasets"][dataset]) == "synthetic_gmm" else raw_path(dataset)
            for dataset in config["datasets"]
        ],
        prepared_h5ads=[prepared_path(dataset) for dataset in config["datasets"]]
    output:
        table=aggregate_output_path("datasets_table.tex")
    params:
        config=get_dataset_table_rule_config(config),
        dataset_names=list(config["datasets"].keys()),
        dataset_labels=config.get("dataset_labels", {}),
    log:
        aggregate_output_path("dataset_table.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/dataset_table.py"


rule plot_dataset:
    """Generate notebook-style LaTeX tables and Pareto plot for one dataset."""
    input:
        scores=lambda w: [bench_path(w.dataset, m, "scores.csv") for m in MAIN_TABLE_METHODS],
    output:
        table=str(DATA_ROOT / "plots" / "{dataset}" / "{dataset_repr}" / "table.tex"),
        benchmark_table=str(DATA_ROOT / "plots" / "{dataset}" / "{dataset_repr}" / "benchmark_table.tex"),
        pareto=str(DATA_ROOT / "plots" / "{dataset}" / "{dataset_repr}" / "pareto_plot.pdf")
    params:
        config=get_plot_rule_config(config),
        method_names=MAIN_TABLE_METHODS,
    log:
        str(DATA_ROOT / "{dataset}" / "logs" / "plot_{dataset_repr}.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/plot.py"


rule gmm_holdout_validation_dataset:
    """Compute patient-wise train/validation NLL diagnostics for one fitted GMM."""
    input:
        h5ad=lambda w: gmm_path(w.dataset)
    output:
        csv=str(DATA_ROOT / "plots" / "{dataset}" / "{dataset_repr}" / "gmm_holdout_nll.csv")
    params:
        config={**get_fit_gmm_rule_config(config), "random_seed": config.get("random_seed")},
        ds_meta=lambda w: config["datasets"][w.dataset],
        dataset=lambda w: w.dataset,
        gmm_key=lambda w: dataset_gmm_key(w.dataset)
    log:
        str(DATA_ROOT / "{dataset}" / "logs" / "gmm_holdout_validation_{dataset_repr}.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/gmm_holdout_validation.py"


rule aggregate:
    """Cross-dataset aggregation for one dataset group: benchmark table + Pareto plot."""
    wildcard_constraints:
        group="|".join(DATASET_GROUPS.keys())
    input:
        scores=lambda w: [
            bench_path(dataset, method_name, "scores.csv")
            for dataset in DATASET_GROUPS[w.group]
            for method_name in ALL_METHODS
        ],
    output:
        table=aggregate_output_path("benchmark_table_{group}.tex"),
        pareto=aggregate_output_path("pareto_plot_{group}.pdf"),
    params:
        config=get_plot_rule_config(config),
        method_dataset_pairs=lambda w: [
            (method_name, dataset)
            for dataset in DATASET_GROUPS[w.group]
            for method_name in ALL_METHODS
        ],
    log:
        aggregate_output_path("aggregate_{group}.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/aggregate.py"


COMPLETE_LINKAGE_PAIRS = [
    (method, dataset)
    for dataset in DATASET_GROUPS["all"]
    for method in MAIN_TABLE_METHODS
]


rule complete_linkage_table:
    """Per-dataset average- and complete-linkage ARI for the main benchmark methods."""
    input:
        scores=[bench_path(dataset, method, "scores.csv") for method, dataset in COMPLETE_LINKAGE_PAIRS],
    output:
        table=aggregate_output_path("complete_linkage_table.tex"),
    params:
        config=get_plot_rule_config(config),
        method_dataset_pairs=COMPLETE_LINKAGE_PAIRS,
        dataset_labels=config.get("dataset_labels", {}),
    log:
        aggregate_output_path("complete_linkage_table.log")
    script:
        "../scripts/complete_linkage_table.py"


rule aggregate_gmm_holdout_validation:
    """Aggregate patient-wise GMM hold-out NLL diagnostics across datasets."""
    input:
        dataset_csvs=[dataset_plot_path(dataset, "gmm_holdout_nll.csv") for dataset in GMM_VALIDATION_DATASETS]
    output:
        csv=aggregate_output_path("gmm_holdout_nll.csv"),
        table=aggregate_output_path("gmm_holdout_nll_summary.tex"),
        plot=aggregate_output_path("gmm_holdout_nll_boxplot.pdf")
    params:
        config=get_dataset_table_rule_config(config),
        dataset_names=GMM_VALIDATION_DATASETS
    log:
        aggregate_output_path("gmm_holdout_nll.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/aggregate_gmm_holdout_validation.py"


rule aggregate_train_time_vs_patients:
    """Cross-dataset training/OT-time plots against patient count."""
    input:
        scores=[
            bench_path(dataset, method_name, "scores.csv")
            for dataset in config["datasets"]
            for method_name in ALL_METHODS
        ],
        source_h5ads=[
            prepared_path(dataset) if get_dataset_source(config["datasets"][dataset]) == "synthetic_gmm" else raw_path(dataset)
            for dataset in config["datasets"]
        ]
    output:
        plot=aggregate_output_path("train_time_vs_patients.pdf"),
        plot_log=aggregate_output_path("train_time_vs_patients_log.pdf"),
        ot_plot=aggregate_output_path("ot_time_vs_patients.pdf"),
        ot_plot_log=aggregate_output_path("ot_time_vs_patients_log.pdf"),
        panel=aggregate_output_path("time_vs_patients_panel.pdf"),
        panel_log=aggregate_output_path("time_vs_patients_panel_log.pdf")
    params:
        plot_config=get_plot_rule_config(config),
        table_config=get_dataset_table_rule_config(config),
        method_dataset_pairs=[
            (method_name, dataset)
            for dataset in config["datasets"]
            for method_name in ALL_METHODS
        ],
        method_names=ALL_METHODS,
        dataset_names=list(config["datasets"].keys())
    log:
        aggregate_output_path("train_time_vs_patients.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/plot_train_time_vs_patients.py"


if LATENT_DIM_ABLATION_PLOT_CFG.get("enabled", False):
    _latent_dataset = LATENT_DIM_ABLATION_PLOT_CFG.get("dataset")
    _latent_method = LATENT_DIM_ABLATION_PLOT_CFG.get("method")
    if not _latent_dataset or not _latent_method:
        raise ValueError(
            "latent_dim_ablation_plot requires both 'dataset' and 'method' when enabled."
        )

    _latent_output_dir = str(LATENT_DIM_ABLATION_PLOT_CFG.get("output_dir", "latent_dim_ablation")).strip("/")
    _latent_summary_csv = str(LATENT_DIM_ABLATION_PLOT_CFG.get("summary_csv", "best_by_n_comps.csv"))
    _latent_accuracy_plot = str(LATENT_DIM_ABLATION_PLOT_CFG.get("accuracy_plot", "knn_acc_vs_n_comps.pdf"))
    _latent_train_time_plot = str(LATENT_DIM_ABLATION_PLOT_CFG.get("train_time_plot", "train_time_vs_n_comps.pdf"))

    rule plot_latent_dim_ablation_summary:
        """Summarize best tuned performance and train time over latent dimension."""
        input:
            scores=tune_path(_latent_dataset, _latent_method, "scores.csv")
        output:
            summary_csv=aggregate_output_path(f"{_latent_output_dir}/{_latent_summary_csv}"),
            accuracy_plot=aggregate_output_path(f"{_latent_output_dir}/{_latent_accuracy_plot}"),
            train_time_plot=aggregate_output_path(f"{_latent_output_dir}/{_latent_train_time_plot}")
        params:
            plot_cfg=LATENT_DIM_ABLATION_PLOT_CFG
        log:
            aggregate_output_path(f"{_latent_output_dir}/latent_dim_ablation_summary.log")
        resources:
            mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
        script:
            "../scripts/plot_latent_dim_ablation.py"
