"""PILOT-GM-VAE rule: per-patient GMMs from the GM-VAE instead of EM (gmm.provider: gmvae)."""


rule fit_gmm_gmvae:
    """Fit per-patient GMMs with PILOT-GM-VAE (isolated environment) for the representation table."""
    input:
        h5ad=lambda w: prepared_path(w.dataset)
    output:
        h5ad=str(DATA_ROOT / "{dataset}" / "gmm_gmvae" / "{gmm_key}.h5ad")
    params:
        config={**get_fit_gmm_rule_config(config), "random_seed": config.get("random_seed")},
        ds_meta=lambda w: get_rule_ds_meta(config["datasets"][w.dataset]),
        gmm_key=lambda w: dataset_gmm_key(w.dataset)
    log:
        str(DATA_ROOT / "{dataset}" / "logs" / "gmm_gmvae_{gmm_key}.log")
    benchmark:
        # wall/CPU time and memory of the job, for the runtime table
        str(DATA_ROOT / "{dataset}" / "timings" / "gmm_gmvae_{gmm_key}.tsv")
    resources:
        mem_mb=lambda w: _get_resources("gmm", "mem_mb", 64000),
    script:
        "../scripts/fit_gmm_gmvae.py"
