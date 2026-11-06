"""scRNA baselines table: each scRNA-specific baseline next to our metric on its representation.

Rows with ``run: main`` come from this run. Rows with ``run: gmvae`` come from the run
with the ``gmvae_overlay`` config merged in; their result folders are computed with that
config and passed as params, not inputs. As inputs, Snakemake could fill a missing
GM-VAE folder with this run's (EM) GMMs, since the benchmark rule matches any hash.
"""

from copy import deepcopy

SCRNA_BASELINES_CONFIG = dict(config.get("scrna_baselines_table") or {})
# Built from the main config only: under the GM-VAE overlay the EM rows would be wrong.
SCRNA_BASELINES_ENABLED = bool(SCRNA_BASELINES_CONFIG) and not _uses_gmvae()


def _scrna_baselines_run_config(run: str) -> dict:
    """Config whose results a row reads: this run, or this run with the GM-VAE overlay."""
    if run == "main":
        return config
    if run != "gmvae":
        raise ValueError(f"scrna_baselines_table: unknown run {run!r} (main or gmvae)")
    overlay_path = Path(workflow.basedir) / "config" / SCRNA_BASELINES_CONFIG["gmvae_overlay"]
    with open(overlay_path) as f:
        return _deep_update(deepcopy(config), yaml.safe_load(f) or {})


def _scores_path_under(run_config: dict, dataset: str, method_name: str) -> str:
    """scores.csv of one method under a given config (same hashing as bench_path)."""
    ds_meta = run_config["datasets"][dataset]
    ds_params = get_dataset_construction_params(
        run_config, ds_meta, run_config["methods"][method_name], get_gmm_key(run_config, ds_meta)
    )
    train_kw = get_method_train_test_kwargs(run_config, method_name)
    bench_hash = get_bench_hash(method_name, train_kw, ds_params)
    return str(DATA_ROOT / dataset / "benchmark" / method_name / bench_hash / "scores.csv")


def _scrna_baselines_cells() -> list[dict]:
    """One entry per (row, role, dataset), role being baseline or ours, with the scores path."""
    if not SCRNA_BASELINES_ENABLED:
        return []
    datasets = [ds for ds in SCRNA_BASELINES_CONFIG["datasets"] if ds in config["datasets"]]
    run_configs = {}
    cells = []
    for row_index, row in enumerate(SCRNA_BASELINES_CONFIG["rows"]):
        run = row.get("run", "main")
        if run not in run_configs:
            run_configs[run] = _scrna_baselines_run_config(run)
        for role in ("baseline", "ours"):
            for dataset in datasets:
                cells.append(
                    {
                        "row": row_index,
                        "role": role,
                        "run": run,
                        "method": row[role]["method"],
                        "dataset": dataset,
                        "scores": _scores_path_under(run_configs[run], dataset, row[role]["method"]),
                    }
                )
    return cells


SCRNA_BASELINES_CELLS = _scrna_baselines_cells()


def scrna_baselines_path(suffix: str) -> str:
    return aggregate_output_path(f"scrna_baselines/{suffix}")


rule scrna_baselines_table:
    """Main-text table: scRNA baselines vs our metric on the same representation."""
    input:
        scores=sorted({cell["scores"] for cell in SCRNA_BASELINES_CELLS if cell["run"] == "main"}),
    output:
        csv=scrna_baselines_path("scrna_baselines_table.csv"),
        table=scrna_baselines_path("scrna_baselines_table.tex"),
        table_per_dataset=scrna_baselines_path("scrna_baselines_per_dataset.tex"),
    params:
        cells=SCRNA_BASELINES_CELLS,
        rows=SCRNA_BASELINES_CONFIG.get("rows", []),
        gmvae_overlay=SCRNA_BASELINES_CONFIG.get("gmvae_overlay"),
        n_splits=int(config.get("train_test", {}).get("n_splits", 5)),
        datasets=sorted({cell["dataset"] for cell in SCRNA_BASELINES_CELLS},
                        key=lambda ds: SCRNA_BASELINES_CONFIG["datasets"].index(ds)),
        dataset_labels=SCRNA_BASELINES_CONFIG.get("dataset_labels", {}),
    log:
        scrna_baselines_path("scrna_baselines_table.log")
    script:
        "../scripts/scrna_baselines_table.py"


SCRNA_BASELINES_EXPORT = _export_dest_path("tables", "scrna_baselines_table.tex")
SCRNA_BASELINES_PER_DATASET_EXPORT = _export_dest_path("tables", "appendix/scrna_baselines_per_dataset.tex")


rule scrna_baselines_table_export:
    """Copy the tables to the paper export tables root (not via export_paper_outputs, which
    belongs to all)."""
    input:
        table=scrna_baselines_path("scrna_baselines_table.tex"),
        table_per_dataset=scrna_baselines_path("scrna_baselines_per_dataset.tex"),
    output:
        table=SCRNA_BASELINES_EXPORT,
        table_per_dataset=SCRNA_BASELINES_PER_DATASET_EXPORT,
    shell:
        "cp {input.table} {output.table} && cp {input.table_per_dataset} {output.table_per_dataset}"

