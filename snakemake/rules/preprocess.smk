"""AnnData preprocessing rule: PCA or HVG embedding."""


rule preprocess_anndata:
    """Preprocess raw AnnData with PCA or HVG selection."""
    input:
        h5ad=str(DATA_ROOT / "{dataset}" / "raw.h5ad")
    output:
        h5ad=str(DATA_ROOT / "{dataset}" / "prepared" / "{dataset_repr}.h5ad")
    log:
        str(DATA_ROOT / "{dataset}" / "logs" / "preprocess_anndata_{dataset_repr}.log")
    benchmark:
        # wall/CPU time and memory of the job, for the runtime table
        str(DATA_ROOT / "{dataset}" / "timings" / "preprocess_anndata_{dataset_repr}.tsv")
    resources:
        mem_mb=lambda w: _get_resources("preprocess", "mem_mb", 32000),
    script:
        "../scripts/preprocess_anndata.py"
