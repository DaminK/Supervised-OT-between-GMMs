"""Rules of the regularization ablation (Figure 5 and its appendix results).

The per-dataset rules train the maps of the ablation and record the approximation error over
training; the aggregate rules build the figures and tables across datasets. The tuned replay
retrains the setups at the weights tuned for the main benchmark (target approx_error_bound_tuned).
"""

from copy import deepcopy

import yaml

APPROX_ERROR_BOUND_INITS = list((config.get("approx_error_bound") or {}).get("inits", ["orthonormal"]))
# Every Figure 5 figure is written for the absolute and the relative error, each for every
# initialization alone and for all together ("combined"; as _error_plots.init_sets).
APPROX_ERROR_BOUND_ERRORS = ["approx_error", "relative_approx_error"]
APPROX_ERROR_BOUND_INIT_SETS = APPROX_ERROR_BOUND_INITS + (["combined"] if len(APPROX_ERROR_BOUND_INITS) > 1 else [])
# Figure 5 (b): per-epoch measures of dimensional collapse and redundancy of the latent space
# (_approx_error.map_metrics), one output folder each (rule approx_error_bound_latent_measures)
APPROX_ERROR_BOUND_LATENT_MEASURES = [
    "axis_redundancy", "gram_dependence", "effective_rank", "stable_rank",
    "latent_participation_ratio", "latent_correlation", "latent_correlation_within",
]
# Map measures not stored per epoch by the training: computed by the same rule from the trained
# maps and the regenerated initial maps (effective_dof: Figure 5 (b))
APPROX_ERROR_BOUND_MAP_MEASURES = ["effective_dof"]
# Reference config keys read by approx_error_bound_tuned.py, so other edits do not retrain it
APPROX_ERROR_BOUND_TUNED_CONFIG_KEYS = ["training_device", "n_threads", "random_seed", "init_strategy", "use_rep", "from_anndata"]


def _approx_error_bound_figure_paths(pattern: str) -> list[str]:
    """One path per error column and init set (error columns outer), *pattern* with {error} and {init}."""
    return [
        approx_error_bound_output_path(pattern.format(error=error, init=init))
        for error in APPROX_ERROR_BOUND_ERRORS
        for init in APPROX_ERROR_BOUND_INIT_SETS
    ]


def _deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


_APPROX_ERROR_BOUND_REFERENCE_CONFIG_CACHE: dict[str, dict] = {}


def _load_approx_error_bound_reference_config(config_relpath: str) -> dict:
    if config_relpath in _APPROX_ERROR_BOUND_REFERENCE_CONFIG_CACHE:
        return _APPROX_ERROR_BOUND_REFERENCE_CONFIG_CACHE[config_relpath]

    snakemake_dir = Path(workflow.basedir)
    defaults_path = snakemake_dir / "config" / "defaults.yml"
    config_path = snakemake_dir / "config" / config_relpath

    with open(defaults_path) as f:
        merged = yaml.safe_load(f) or {}
    with open(config_path) as f:
        overrides = yaml.safe_load(f) or {}
    _deep_update(merged, overrides)

    datasets_path = snakemake_dir / "config" / merged.get("datasets_file", "datasets.yml")
    with open(datasets_path) as f:
        merged["datasets"] = (yaml.safe_load(f) or {}).get("datasets", {})

    _APPROX_ERROR_BOUND_REFERENCE_CONFIG_CACHE[config_relpath] = merged
    return merged


_REFERENCE_KEYS = ("configfile", "method_name")


def _check_reference(ref_cfg: dict, where: str) -> dict:
    """A reference entry names its config (relative to config/) and method; there are no defaults."""
    missing = [key for key in _REFERENCE_KEYS if not ref_cfg.get(key)]
    if missing:
        raise ValueError(f"{where}: missing {', '.join(missing)}")
    return ref_cfg


def _reference_run(ref_cfg: dict, dataset: str) -> dict:
    """Config, method, GMM file and benchmark folder of a method in another run's config."""
    ref_config = _load_approx_error_bound_reference_config(str(ref_cfg["configfile"]))
    method_name = str(ref_cfg["method_name"])
    ds_meta = ref_config["datasets"][dataset]
    method_cfg = ref_config["methods"][method_name]

    data_root = Path(ref_config.get("data_root", "../data/snakemake"))
    if not data_root.is_absolute():
        data_root = (Path(workflow.basedir) / data_root).resolve()

    gmm_key = get_gmm_key(ref_config, ds_meta)
    ds_params = get_dataset_construction_params(ref_config, ds_meta, method_cfg, gmm_key)
    train_kw = get_method_train_test_kwargs(ref_config, method_name)
    bench_hash = get_bench_hash(method_name, train_kw, ds_params)
    return {
        "config": ref_config,
        "train_config": {
            **{key: ref_config[key] for key in APPROX_ERROR_BOUND_TUNED_CONFIG_KEYS if key in ref_config},
            "methods": {method_name: method_cfg},
        },
        "method_name": method_name,
        "ds_meta": ds_meta,
        "gmm_key": gmm_key,
        "h5ad": str(data_root / dataset / "gmm" / f"{gmm_key}.h5ad"),
        "benchmark_dir": data_root / dataset / "benchmark" / method_name / bench_hash,
    }


def _approx_error_bound_reference_benchmark_manifest(dataset: str):
    ref_cfg = dict((config.get("approx_error_bound") or {}).get("benchmark_reference") or {})
    if not ref_cfg.get("enabled", False):
        return []
    _check_reference(ref_cfg, "approx_error_bound.benchmark_reference")
    return str(_reference_run(ref_cfg, dataset)["benchmark_dir"] / "benchmark_manifest.json")


# Reference methods of the tuned replay, keyed by output folder name (approx_error_bound.tuned_references)
APPROX_ERROR_BOUND_TUNED_REFERENCES = {
    key: _check_reference(dict(ref or {}), f"approx_error_bound.tuned_references.{key}")
    for key, ref in ((config.get("approx_error_bound") or {}).get("tuned_references") or {}).items()
}


def _tuned_reference(dataset: str, reference: str) -> dict:
    """Run whose tuned weights the tuned replay uses for one reference method."""
    if reference not in APPROX_ERROR_BOUND_TUNED_REFERENCES:
        raise ValueError(f"Tuned replay: unknown reference {reference!r} in `approx_error_bound.tuned_references`.")
    return _reference_run(APPROX_ERROR_BOUND_TUNED_REFERENCES[reference], dataset)


def _reference_trains_diagonal(ref_cfg: dict) -> bool:
    """Whether a reference method is trained with the diagonal Bures term."""
    ref_config = _load_approx_error_bound_reference_config(str(ref_cfg["configfile"]))
    method_cfg = ref_config["methods"][str(ref_cfg["method_name"])]
    return bool(method_cfg["train_test"]["diag_bures_approx"])


rule approx_error_bound_per_dataset:
    """Run the Figure 5 diagonal-Bures regularization ablation for one dataset."""
    input:
        h5ad=lambda w: gmm_path(w.dataset),
        benchmark_manifest=lambda w: _approx_error_bound_reference_benchmark_manifest(w.dataset),
    output:
        metrics_csv=approx_error_bound_dataset_output_path("approx_error_epochs.csv"),
        metrics_json=approx_error_bound_dataset_output_path("approx_error_epochs.json"),
        figure_pdf=approx_error_bound_dataset_output_path("approx_error_epochs.pdf"),
        figure_log_pdf=approx_error_bound_dataset_output_path("approx_error_epochs_log.pdf"),
        relative_figure_pdf=approx_error_bound_dataset_output_path("approx_error_epochs_relative.pdf"),
        relative_figure_log_pdf=approx_error_bound_dataset_output_path("approx_error_epochs_relative_log.pdf"),
        bures_gap_figure_pdf=approx_error_bound_dataset_output_path("bures_gap.pdf"),
        bures_gap_figure_log_pdf=approx_error_bound_dataset_output_path("bures_gap_log.pdf"),
        bures_relative_figure_pdf=approx_error_bound_dataset_output_path("bures_relative_error.pdf"),
        bures_relative_figure_log_pdf=approx_error_bound_dataset_output_path("bures_relative_error_log.pdf"),
        bures_fraction_figure_pdf=approx_error_bound_dataset_output_path("bures_fraction.pdf"),
        fro_norm_figure_pdf=approx_error_bound_dataset_output_path("fro_norm.pdf"),
        projected_mi_figure_pdf=approx_error_bound_dataset_output_path("projected_mi.pdf"),
        trained_maps=approx_error_bound_dataset_output_path("trained_maps.npz"),
        manifest=approx_error_bound_dataset_output_path("approx_error_bound_manifest.json")
    params:
        config=get_approx_error_bound_rule_config(config),
        dataset_name=lambda w: w.dataset,
        ds_meta=lambda w: config["datasets"][w.dataset],
        gmm_key=lambda w: dataset_gmm_key(w.dataset),
    log:
        approx_error_bound_dataset_output_path("approx_error_bound.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/approx_error_bound_epochs.py"


rule approx_error_bound_evaluate_dataset:
    """Figure 5 downstream evaluation of one dataset's trained maps (ggml_ot.test, benchmark
    splits), with the diagonal and with the exact Bures term. Separate from training, so the
    evaluation can change without retraining."""
    input:
        h5ad=lambda w: gmm_path(w.dataset),
        trained_maps=approx_error_bound_dataset_output_path("trained_maps.npz"),
        metrics_csv=approx_error_bound_dataset_output_path("approx_error_epochs.csv"),
    output:
        downstream_csv=approx_error_bound_dataset_output_path("downstream.csv"),
    params:
        config=get_approx_error_bound_rule_config(config),
        dataset_name=lambda w: w.dataset,
        ds_meta=lambda w: config["datasets"][w.dataset],
        gmm_key=lambda w: dataset_gmm_key(w.dataset),
    log:
        approx_error_bound_dataset_output_path("approx_error_bound_evaluate.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/approx_error_bound_evaluate.py"


rule approx_error_bound_tuned_dataset:
    """Tuned replay: the four regularization setups at the weights tuned for the main benchmark,
    trained as in the main benchmark (scripts/approx_error_bound_tuned.py); Both with the
    benchmark's initialization reproduces the main benchmark run (output check). The reference
    files come from the main benchmark's own workflow, so they are params, not inputs: run the
    main benchmark first; the check records the files used. One run per reference method."""
    wildcard_constraints:
        reference="|".join(APPROX_ERROR_BOUND_TUNED_REFERENCES) or "^$"
    output:
        trained_maps=approx_error_bound_dataset_output_path("tuned/{reference}/trained_maps.npz"),
        final_errors=approx_error_bound_dataset_output_path("tuned/{reference}/final_errors.csv"),
        check=approx_error_bound_dataset_output_path("tuned/{reference}/reproduces_benchmark.json"),
    params:
        h5ad=lambda w: _tuned_reference(w.dataset, w.reference)["h5ad"],
        manifest=lambda w: str(_tuned_reference(w.dataset, w.reference)["benchmark_dir"] / "benchmark_manifest.json"),
        scores_per_split=lambda w: str(_tuned_reference(w.dataset, w.reference)["benchmark_dir"] / "scores_per_split.csv"),
        ref_config=lambda w: _tuned_reference(w.dataset, w.reference)["train_config"],
        method_name=lambda w: _tuned_reference(w.dataset, w.reference)["method_name"],
        ds_meta=lambda w: _tuned_reference(w.dataset, w.reference)["ds_meta"],
        gmm_key=lambda w: _tuned_reference(w.dataset, w.reference)["gmm_key"],
        dataset_name=lambda w: w.dataset,
        inits=APPROX_ERROR_BOUND_INITS,
        error=(config.get("approx_error_bound") or {}).get("error"),
    log:
        approx_error_bound_dataset_output_path("tuned/{reference}/approx_error_bound_tuned.log")
    resources:
        mem_mb=lambda w: _get_resources("benchmark", "mem_mb", 64000),
    script:
        "../scripts/approx_error_bound_tuned.py"


rule approx_error_bound_evaluate_tuned_dataset:
    """Tuned replay: downstream evaluation, as approx_error_bound_evaluate_dataset."""
    wildcard_constraints:
        reference="|".join(APPROX_ERROR_BOUND_TUNED_REFERENCES) or "^$"
    input:
        h5ad=lambda w: gmm_path(w.dataset),
        trained_maps=approx_error_bound_dataset_output_path("tuned/{reference}/trained_maps.npz"),
        metrics_csv=approx_error_bound_dataset_output_path("tuned/{reference}/final_errors.csv"),
    output:
        downstream_csv=approx_error_bound_dataset_output_path("tuned/{reference}/downstream.csv"),
    params:
        config=get_approx_error_bound_rule_config(config),
        dataset_name=lambda w: w.dataset,
        ds_meta=lambda w: config["datasets"][w.dataset],
        gmm_key=lambda w: dataset_gmm_key(w.dataset),
    log:
        approx_error_bound_dataset_output_path("tuned/{reference}/approx_error_bound_evaluate.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/approx_error_bound_evaluate.py"


# Map faithfulness table, standard normal rows (top-level approx_error_bound_long_training)
APPROX_ERROR_BOUND_LONG = dict(config.get("approx_error_bound_long_training") or {})
APPROX_ERROR_BOUND_LONG_SETUPS = list(dict(APPROX_ERROR_BOUND_LONG.get("setups") or {}))


def _long_weights(reference: str, setup: str) -> tuple[float, float]:
    """Weights of one long-training setup; no MI weight for a reference trained with the exact term."""
    weights = dict(APPROX_ERROR_BOUND_LONG["setups"][setup])
    trains_diagonal = _reference_trains_diagonal(APPROX_ERROR_BOUND_TUNED_REFERENCES[reference])
    return float(weights.get("reg", 0.0)), float(weights.get("mi_reg", 0.0)) if trains_diagonal else 0.0


def _long_downstream_csvs() -> list[str]:
    return [
        approx_error_bound_dataset_output_path("long/{reference}/{setup}/downstream.csv").format(
            dataset=ds, reference=reference, setup=setup)
        for reference in APPROX_ERROR_BOUND_TUNED_REFERENCES
        for setup in APPROX_ERROR_BOUND_LONG_SETUPS
        for ds in config["datasets"]
    ]


rule approx_error_bound_long_dataset:
    """Map faithfulness table, standard normal rows: one setup of one reference method trained as
    the tuned replay, with the initialization, weights and max_iter of
    approx_error_bound_long_training (scripts/approx_error_bound_long.py). Reference files are
    params, as in approx_error_bound_tuned_dataset."""
    wildcard_constraints:
        reference="|".join(APPROX_ERROR_BOUND_TUNED_REFERENCES) or "^$",
        setup="|".join(APPROX_ERROR_BOUND_LONG_SETUPS) or "^$",
    output:
        trained_maps=approx_error_bound_dataset_output_path("long/{reference}/{setup}/trained_maps.npz"),
        final_errors=approx_error_bound_dataset_output_path("long/{reference}/{setup}/final_errors.csv"),
    params:
        h5ad=lambda w: _tuned_reference(w.dataset, w.reference)["h5ad"],
        manifest=lambda w: str(_tuned_reference(w.dataset, w.reference)["benchmark_dir"] / "benchmark_manifest.json"),
        ref_config=lambda w: _tuned_reference(w.dataset, w.reference)["train_config"],
        method_name=lambda w: _tuned_reference(w.dataset, w.reference)["method_name"],
        ds_meta=lambda w: _tuned_reference(w.dataset, w.reference)["ds_meta"],
        gmm_key=lambda w: _tuned_reference(w.dataset, w.reference)["gmm_key"],
        dataset_name=lambda w: w.dataset,
        setup=lambda w: w.setup,
        init=APPROX_ERROR_BOUND_LONG.get("init", "random"),
        reg=lambda w: _long_weights(w.reference, w.setup)[0],
        mi_reg=lambda w: _long_weights(w.reference, w.setup)[1],
        max_iter=APPROX_ERROR_BOUND_LONG.get("max_iter"),
        error=(config.get("approx_error_bound") or {}).get("error"),
    log:
        approx_error_bound_dataset_output_path("long/{reference}/{setup}/approx_error_bound_long.log")
    resources:
        mem_mb=lambda w: _get_resources("benchmark", "mem_mb", 64000),
    script:
        "../scripts/approx_error_bound_long.py"


rule approx_error_bound_evaluate_long_dataset:
    """Map faithfulness table, standard normal rows: downstream evaluation, as
    approx_error_bound_evaluate_dataset."""
    wildcard_constraints:
        reference="|".join(APPROX_ERROR_BOUND_TUNED_REFERENCES) or "^$",
        setup="|".join(APPROX_ERROR_BOUND_LONG_SETUPS) or "^$",
    input:
        h5ad=lambda w: gmm_path(w.dataset),
        trained_maps=approx_error_bound_dataset_output_path("long/{reference}/{setup}/trained_maps.npz"),
        metrics_csv=approx_error_bound_dataset_output_path("long/{reference}/{setup}/final_errors.csv"),
    output:
        downstream_csv=approx_error_bound_dataset_output_path("long/{reference}/{setup}/downstream.csv"),
    params:
        config=get_approx_error_bound_rule_config(config),
        dataset_name=lambda w: w.dataset,
        ds_meta=lambda w: config["datasets"][w.dataset],
        gmm_key=lambda w: dataset_gmm_key(w.dataset),
    log:
        approx_error_bound_dataset_output_path("long/{reference}/{setup}/approx_error_bound_evaluate.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/approx_error_bound_evaluate.py"


rule approx_error_bound_tuned:
    """Target: tuned-replay data and figures for every dataset and reference method, and the
    map faithfulness table (needs the main benchmark run)."""
    input:
        [
            approx_error_bound_output_path(f"downstream_tuned/{reference}/downstream_table.tex")
            for reference in APPROX_ERROR_BOUND_TUNED_REFERENCES
        ],
        approx_error_bound_output_path("faithfulness/faithfulness_table.tex"),


rule approx_error_bound_faithfulness:
    """Faithfulness of the learned maps under both geometries: Spearman correlation of exact and
    diagonal patient distances of the same map, the relative approximation error, and KNN accuracy
    and ARI with the Bures term used in training, per training term (reference), regularization
    setup and initialization; the table shows with and without regularization from each
    initialization (scripts/approx_error_bound_faithfulness.py)."""
    input:
        downstream_csvs=[
            approx_error_bound_dataset_output_path("tuned/{reference}/downstream.csv").format(dataset=ds, reference=reference)
            for reference in APPROX_ERROR_BOUND_TUNED_REFERENCES
            for ds in config["datasets"]
        ] + _long_downstream_csvs(),
    output:
        csv=approx_error_bound_output_path("faithfulness/faithfulness.csv"),
        table=approx_error_bound_output_path("faithfulness/faithfulness_table.tex"),
    params:
        references={
            key: {"method_name": ref.get("method_name"), "display_name": ref.get("display_name", key),
                  "diag_bures_approx": _reference_trains_diagonal(ref)}
            for key, ref in APPROX_ERROR_BOUND_TUNED_REFERENCES.items()
        },
        # reference and training ("tuned" replay or "long") of each input, in input order
        reference_keys=[reference for reference in APPROX_ERROR_BOUND_TUNED_REFERENCES for ds in config["datasets"]]
        + [reference for reference in APPROX_ERROR_BOUND_TUNED_REFERENCES
           for setup in APPROX_ERROR_BOUND_LONG_SETUPS for ds in config["datasets"]],
        trainings=["tuned"] * (len(APPROX_ERROR_BOUND_TUNED_REFERENCES) * len(config["datasets"]))
        + ["long"] * (len(APPROX_ERROR_BOUND_TUNED_REFERENCES) * len(APPROX_ERROR_BOUND_LONG_SETUPS) * len(config["datasets"])),
        long_training=APPROX_ERROR_BOUND_LONG,
        n_datasets=len(config["datasets"]),
    log:
        approx_error_bound_output_path("faithfulness/approx_error_bound_faithfulness.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/approx_error_bound_faithfulness.py"


rule approx_error_bound_all:
    """Target: all Figure 5 results in the pipeline folder (the paper exports only the files it
    includes): aggregate, error_bound/, latent_measures/, downstream/ (not the tuned weights,
    which need the main benchmark run: target approx_error_bound_tuned)."""
    input:
        approx_error_bound_output_path("aggregate/approx_error_epochs.csv"),
        approx_error_bound_output_path("error_bound/bound_constants.csv"),
        approx_error_bound_output_path("latent_measures"),
        approx_error_bound_output_path("downstream/downstream.csv"),


rule approx_error_bound_aggregate:
    """Average relative approximation error across all datasets."""
    input:
        dataset_csvs=[
            approx_error_bound_dataset_output_path("approx_error_epochs.csv").format(dataset=ds)
            for ds in config["datasets"]
        ]
    output:
        csv=approx_error_bound_output_path("aggregate/approx_error_epochs.csv"),
        relative_pdf=approx_error_bound_output_path("aggregate/approx_error_epochs_relative.pdf"),
        relative_log_pdf=approx_error_bound_output_path("aggregate/approx_error_epochs_relative_log.pdf"),
        bures_rel_pdf=approx_error_bound_output_path("aggregate/bures_relative_error.pdf"),
        bures_rel_log_pdf=approx_error_bound_output_path("aggregate/bures_relative_error_log.pdf"),
        manifest=approx_error_bound_output_path("aggregate/approx_error_bound_aggregate_manifest.json"),
    params:
        config=get_approx_error_bound_rule_config(config),
        dataset_names=list(config["datasets"].keys())
    log:
        approx_error_bound_output_path("aggregate/approx_error_bound_aggregate.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/approx_error_bound_aggregate.py"


def _approx_error_bound_variant_headers() -> dict:
    variants = (config.get("approx_error_bound") or {}).get("variants") or {}
    return {name: str((cfg or {}).get("display_name", name)) for name, cfg in variants.items()}


rule approx_error_bound_figures:
    """Figure 5 (a): approximation error over training with the bound overlay, grouped and per
    dataset, one figure each per error column (absolute, relative) and init set (each
    initialization, combined); bound constants per error column and initialization."""
    input:
        dataset_csvs=[
            approx_error_bound_dataset_output_path("approx_error_epochs.csv").format(dataset=ds)
            for ds in config["datasets"]
        ],
        gmm_h5ads=[gmm_path(ds) for ds in config["datasets"]],
    output:
        grouped=_approx_error_bound_figure_paths("error_bound/grouped_{error}_geom_log_bound_{init}.pdf"),
        per_dataset=_approx_error_bound_figure_paths("error_bound/all_datasets_{error}_log_bound_{init}.pdf"),
        constants=approx_error_bound_output_path("error_bound/bound_constants.csv"),
    params:
        dataset_names=list(config["datasets"].keys()),
        synthetic=[ds for ds in config["datasets"] if get_dataset_source(config["datasets"][ds]) == "synthetic_gmm"],
        gmm_key={ds: dataset_gmm_key(ds) for ds in config["datasets"]},
        inits=APPROX_ERROR_BOUND_INITS,
        error_columns=APPROX_ERROR_BOUND_ERRORS,
    log:
        approx_error_bound_output_path("error_bound/approx_error_bound_figures.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/approx_error_bound_figures.py"


rule approx_error_bound_latent_measures:
    """Figure 5 (b): measures of dimensional collapse and redundancy of the latent space vs the
    approximation error, from epoch 0 to the trained map; one folder per measure with the joined,
    per-group mean, per-split and per-split-arrow figures, each per error column (absolute,
    relative) and init set (each initialization, combined)
    (scripts/approx_error_bound_latent_measures.py)."""
    input:
        dataset_csvs=[
            approx_error_bound_dataset_output_path("approx_error_epochs.csv").format(dataset=ds)
            for ds in config["datasets"]
        ],
        trained_maps=[
            approx_error_bound_dataset_output_path("trained_maps.npz").format(dataset=ds)
            for ds in config["datasets"]
        ],
        gmms=[gmm_path(ds) for ds in config["datasets"]],
    output:
        figures=directory(approx_error_bound_output_path("latent_measures")),
    params:
        measures=APPROX_ERROR_BOUND_LATENT_MEASURES,
        map_measures=APPROX_ERROR_BOUND_MAP_MEASURES,
        datasets=list(config["datasets"]),
        gmm_keys=[dataset_gmm_key(ds) for ds in config["datasets"]],
        random_seed=int(get_approx_error_bound_rule_config(config).get("random_seed", 42)),
        variants=_approx_error_bound_variant_headers(),
        inits=APPROX_ERROR_BOUND_INITS,
        error_columns=APPROX_ERROR_BOUND_ERRORS,
    log:
        approx_error_bound_output_path("approx_error_bound_latent_measures.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/approx_error_bound_latent_measures.py"


rule approx_error_bound_downstream:
    """Downstream scores of the Figure 5 maps with diagonal and with exact distances
    (figures per error column and init set, appendix table);
    scripts/approx_error_bound_downstream.py."""
    input:
        downstream_csvs=[
            approx_error_bound_dataset_output_path("downstream.csv").format(dataset=ds)
            for ds in config["datasets"]
        ],
    output:
        main=_approx_error_bound_figure_paths("downstream/main_{error}_{init}.pdf"),
        all_scores=_approx_error_bound_figure_paths("downstream/all_scores_{error}_{init}.pdf"),
        per_dataset=_approx_error_bound_figure_paths("downstream/per_dataset_{error}_{init}.pdf"),
        gap_scatter=_approx_error_bound_figure_paths("downstream/gap_scatter_{error}_{init}.pdf"),
        csv=approx_error_bound_output_path("downstream/downstream.csv"),
        table=approx_error_bound_output_path("downstream/downstream_table.tex"),
    params:
        dataset_names=list(config["datasets"].keys()),
        variants=_approx_error_bound_variant_headers(),
        inits=APPROX_ERROR_BOUND_INITS,
        error_columns=APPROX_ERROR_BOUND_ERRORS,
        caption="Maps of Figure~\\ref{fig:error_over_iter_vs_mi_reg}: diagonal Bures term, fixed regularization weights, 50 epochs.",
        label="tab:approx_error_downstream",
    log:
        approx_error_bound_output_path("downstream/approx_error_bound_downstream.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/approx_error_bound_downstream.py"


rule approx_error_bound_tuned_figures:
    """Tuned replay: as approx_error_bound_downstream, for the setups at the weights tuned for
    one reference method of the main benchmark (rule approx_error_bound_tuned_dataset)."""
    wildcard_constraints:
        reference="|".join(APPROX_ERROR_BOUND_TUNED_REFERENCES) or "^$"
    input:
        downstream_csvs=lambda w: [
            approx_error_bound_dataset_output_path("tuned/{reference}/downstream.csv").format(dataset=ds, reference=w.reference)
            for ds in config["datasets"]
        ],
    output:
        main=_approx_error_bound_figure_paths("downstream_tuned/{{reference}}/main_{error}_{init}.pdf"),
        all_scores=_approx_error_bound_figure_paths("downstream_tuned/{{reference}}/all_scores_{error}_{init}.pdf"),
        per_dataset=_approx_error_bound_figure_paths("downstream_tuned/{{reference}}/per_dataset_{error}_{init}.pdf"),
        gap_scatter=_approx_error_bound_figure_paths("downstream_tuned/{{reference}}/gap_scatter_{error}_{init}.pdf"),
        csv=approx_error_bound_output_path("downstream_tuned/{reference}/downstream.csv"),
        table=approx_error_bound_output_path("downstream_tuned/{reference}/downstream_table.tex"),
    params:
        dataset_names=list(config["datasets"].keys()),
        variants=_approx_error_bound_variant_headers(),
        inits=APPROX_ERROR_BOUND_INITS,
        error_columns=APPROX_ERROR_BOUND_ERRORS,
        caption=lambda w: (
            "Maps trained as in the main benchmark (30 epochs, early stopping) at the weights tuned for "
            f"{APPROX_ERROR_BOUND_TUNED_REFERENCES[w.reference].get('display_name', w.reference)}."
        ),
        label=lambda w: APPROX_ERROR_BOUND_TUNED_REFERENCES[w.reference].get(
            "label", f"tab:approx_error_downstream_tuned_{w.reference}"
        ),
    log:
        approx_error_bound_output_path("downstream_tuned/{reference}/approx_error_bound_tuned_figures.log")
    resources:
        mem_mb=lambda w: _get_resources("plot", "mem_mb", 16000),
    script:
        "../scripts/approx_error_bound_downstream.py"
