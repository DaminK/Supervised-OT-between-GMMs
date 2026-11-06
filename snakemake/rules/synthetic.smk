"""Synthetic dataset generation rule."""


rule generate_synthetic:
    """Generate a synthetic dataset directly as the prepared artifact."""
    output:
        h5ad=str(DATA_ROOT / "{dataset}" / "generated" / "{dataset_repr}.h5ad")
    params:
        ds_meta=lambda w: config["datasets"][w.dataset],
        gmm_key=lambda w: dataset_gmm_key(w.dataset)
    log:
        str(DATA_ROOT / "{dataset}" / "logs" / "generate_synthetic_{dataset_repr}.log")
    resources:
        mem_mb=lambda w: _get_resources("download", "mem_mb", 8000),
    script:
        "../scripts/generate_synthetic.py"
