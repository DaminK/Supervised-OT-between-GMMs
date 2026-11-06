"""Runtime table (appendix): wall-clock time of the main benchmark's pipeline jobs.

Reads the ``benchmark:`` timing files of the preprocess, GMM and benchmark jobs. The paths are
built with the same helpers as those rules, so they point to the jobs of this config. They are
declared as inputs, so the table is rebuilt whenever one of these jobs reruns.
"""

RUNTIME_TABLE_CONFIG = dict(config.get("runtime_table") or {})
RUNTIME_TABLE_DATASETS = [ds for ds in RUNTIME_TABLE_CONFIG.get("datasets", []) if ds in config["datasets"]]
# EM GMMs only, and only when every listed method is part of the run (not in single-method overlays).
RUNTIME_TABLE_ENABLED = (
    bool(RUNTIME_TABLE_DATASETS)
    and not _uses_gmvae()
    and all(row["method"] in ALL_METHODS for row in RUNTIME_TABLE_CONFIG.get("methods", []))
)


def runtime_table_path(suffix: str) -> str:
    return aggregate_output_path(f"runtime_table/{suffix}")


def _timing_path(dataset: str, job: str) -> str:
    """Timing file of one job, named as in the rules' benchmark: directives."""
    return str(DATA_ROOT / dataset / "timings" / f"{job}.tsv")


def _runtime_table_timings() -> dict:
    """Per dataset: the timing file of the preprocess job, the GMM job and each listed method."""
    if not RUNTIME_TABLE_ENABLED:
        return {}
    timings = {}
    for dataset in RUNTIME_TABLE_DATASETS:
        files = {
            "preprocess": _timing_path(dataset, f"preprocess_anndata_{dataset_rep_label(dataset)}"),
            "gmm": _timing_path(dataset, f"gmm_{dataset_gmm_key(dataset)}"),
        }
        for row in RUNTIME_TABLE_CONFIG["methods"]:
            bench_hash = Path(bench_path(dataset, row["method"], "scores.csv")).parent.name
            files[row["method"]] = _timing_path(dataset, f"benchmark_{row['method']}_{bench_hash}")
        timings[dataset] = files
    return timings


rule runtime_table:
    """Appendix table: PCA, GMM fitting, training and evaluation, and end-to-end time per cohort."""
    input:
        timings=[path for files in _runtime_table_timings().values() for path in files.values()],
    output:
        csv=runtime_table_path("runtime_table.csv"),
        table=runtime_table_path("runtime_table.tex"),
    params:
        timings=_runtime_table_timings(),
        datasets=RUNTIME_TABLE_DATASETS,
        dataset_labels=RUNTIME_TABLE_CONFIG.get("dataset_labels", {}),
        preprocess_label=RUNTIME_TABLE_CONFIG.get("preprocess_label", "PCA"),
        gmm_label=RUNTIME_TABLE_CONFIG.get("gmm_label", "GMM fitting"),
        # GMM methods are charged for GMM fitting end to end (dataset_type gmm, as GMM_METHODS)
        rows=[
            {**row, "uses_gmm": config["methods"][row["method"]].get("dataset_type") == "gmm"}
            for row in RUNTIME_TABLE_CONFIG.get("methods", [])
        ],
        caption=RUNTIME_TABLE_CONFIG.get("caption", ""),
    log:
        runtime_table_path("runtime_table.log")
    script:
        "../scripts/runtime_table.py"
