"""GMM fitting rule: fit GMMs for GMM-based methods."""


rule fit_gmm:
    """Fit a GMM to the prepared dataset artifact and store it in adata.uns."""
    input:
        h5ad=lambda w: prepared_path(w.dataset)
    output:
        h5ad=str(DATA_ROOT / "{dataset}" / "gmm" / "{gmm_key}.h5ad")
    params:
        config=get_fit_gmm_rule_config(config),
        ds_meta=lambda w: get_rule_ds_meta(config["datasets"][w.dataset]),
        gmm_key=lambda w: dataset_gmm_key(w.dataset)
    log:
        str(DATA_ROOT / "{dataset}" / "logs" / "gmm_{gmm_key}.log")
    benchmark:
        # wall/CPU time and memory of the job, for the runtime table
        str(DATA_ROOT / "{dataset}" / "timings" / "gmm_{gmm_key}.tsv")
    resources:
        mem_mb=lambda w: _get_resources("gmm", "mem_mb", 64000),
    script:
        "../scripts/fit_gmm.py"
