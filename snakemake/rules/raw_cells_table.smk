"""Raw-cell table (appendix): point-cloud baselines on GMM-sampled points vs raw cells.

Enabled by the ``raw_cells_table`` block of config/ablation/_raw_cells.yml, which is
merged onto main_benchmark.yml. Each row pairs a main-benchmark point-cloud method with its
raw-cell variant; both are methods of this run, so their scores are regular inputs.
"""

RAW_CELLS_TABLE_CONFIG = dict(config.get("raw_cells_table") or {})
RAW_CELLS_TABLE_DATASETS = [ds for ds in RAW_CELLS_TABLE_CONFIG.get("datasets", []) if ds in config["datasets"]]
RAW_CELLS_TABLE_ENABLED = bool(RAW_CELLS_TABLE_DATASETS) and all(
    row[source] in ALL_METHODS
    for row in RAW_CELLS_TABLE_CONFIG.get("rows", [])
    for source in ("sampled", "raw")
)


def raw_cells_table_path(suffix: str) -> str:
    return aggregate_output_path(f"raw_cells/{suffix}")


def _raw_cells_table_cells() -> list[dict]:
    """One entry per (row, source, dataset) with the scores path."""
    if not RAW_CELLS_TABLE_ENABLED:
        return []
    return [
        {
            "row": row_index,
            "source": source,
            "method": row[source],
            "dataset": dataset,
            "scores": bench_path(dataset, row[source], "scores.csv"),
        }
        for row_index, row in enumerate(RAW_CELLS_TABLE_CONFIG["rows"])
        for source in ("sampled", "raw")
        for dataset in RAW_CELLS_TABLE_DATASETS
    ]


RAW_CELLS_TABLE_CELLS = _raw_cells_table_cells()


if RAW_CELLS_TABLE_ENABLED:

    rule raw_cells_table:
        """Appendix table: point-cloud baselines on GMM-sampled points and on raw cells."""
        input:
            scores=sorted({cell["scores"] for cell in RAW_CELLS_TABLE_CELLS}),
        output:
            csv=raw_cells_table_path("raw_cells_table.csv"),
            table=raw_cells_table_path("raw_cells_table.tex"),
        params:
            cells=RAW_CELLS_TABLE_CELLS,
            rows=RAW_CELLS_TABLE_CONFIG["rows"],
            n_splits=int(config.get("train_test", {}).get("n_splits", 5)),
        log:
            raw_cells_table_path("raw_cells_table.log")
        script:
            "../scripts/raw_cells_table.py"
