"""AnnData download rule: fetch raw datasets from CELLxGENE."""


rule download_anndata:
    """Download a raw AnnData dataset from CELLxGENE."""
    output:
        h5ad=str(DATA_ROOT / "{dataset}" / "raw.h5ad")
    params:
        ds_meta=lambda w: {
            "id": config["datasets"][w.dataset]["id"],
            "source": config["datasets"][w.dataset].get("source", "cellxgene"),
        }
    log:
        str(DATA_ROOT / "{dataset}" / "logs" / "download_anndata.log")
    resources:
        mem_mb=lambda w: _get_resources("download", "mem_mb", 8000),
    script:
        "../scripts/download_anndata.py"
