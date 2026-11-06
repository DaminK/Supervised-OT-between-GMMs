"""
Config-driven export/copy rules for paper-ready outputs.

This layer does not change upstream result locations. It only copies a curated
subset of existing Snakemake outputs into stable generated_plots/generated_tables
trees defined by the experiment YAML.
"""

from pathlib import Path


def _paper_export_cfg() -> dict:
    return dict(config.get("paper_exports", {}))


def _resolve_export_base(path_value, *, default_path: Path) -> Path:
    if path_value:
        root_path = Path(path_value)
        if not root_path.is_absolute():
            root_path = (Path(workflow.basedir) / root_path).resolve()
        return root_path
    return default_path.resolve()


def _paper_export_label() -> str:
    return str(_paper_export_cfg().get("label", AGGREGATE_LABEL))


def _paper_export_plots_root() -> Path:
    export_cfg = _paper_export_cfg()
    legacy_root = export_cfg.get("root")
    return _resolve_export_base(
        export_cfg.get("plots_root", legacy_root),
        default_path=REPO_ROOT / "data" / "generated_plots" / _paper_export_label(),
    )


def _paper_export_tables_root() -> Path:
    export_cfg = _paper_export_cfg()
    legacy_root = export_cfg.get("root")
    return _resolve_export_base(
        export_cfg.get("tables_root", legacy_root),
        default_path=REPO_ROOT / "data" / "generated_tables",
    )


def _paper_export_manifest_root(specs) -> Path:
    export_cfg = _paper_export_cfg()
    manifest_root = export_cfg.get("manifest_root")
    if manifest_root:
        return _resolve_export_base(manifest_root, default_path=Path(manifest_root))
    return (REPO_ROOT / "data" / "generated_plots" / "_export_manifests" / _paper_export_label()).resolve()


def _infer_export_group(source: str, explicit_group=None) -> str:
    if explicit_group is not None:
        if explicit_group not in {"plots", "tables"}:
            raise ValueError(f"Unsupported paper export group: {explicit_group!r}")
        return explicit_group
    if Path(source).suffix == ".tex":
        return "tables"
    return "plots"


def _export_dest_path(group: str, relative_dest: str) -> str:
    base = _paper_export_tables_root() if group == "tables" else _paper_export_plots_root()
    return str(base / relative_dest)


def _normalize_selection(value, *, default_values):
    if value is None:
        return list(default_values)
    if isinstance(value, str):
        if value in {"*", "all", "__all__"}:
            return list(default_values)
        return [value]
    return list(value)


def _normalize_splits(value, *, n_splits: int):
    if value is None:
        return list(range(n_splits))
    if isinstance(value, str):
        if value in {"*", "all", "__all__"}:
            return list(range(n_splits))
        return [int(value)]
    if isinstance(value, int):
        return [value]
    return [int(v) for v in value]


def _normalize_aggregate_items(items):
    """Aggregate exports: a file, or with ``directory: true`` a whole folder. ``source`` is a rule
    output; ``path`` optionally selects a file or folder inside a directory output."""
    normalized = []
    for item in items or []:
        if isinstance(item, str):
            normalized.append({"source": item, "dest": item, "group": _infer_export_group(item)})
        else:
            source = item["source"]
            path = item.get("path")
            normalized.append(
                {
                    "source": source,
                    "path": path,
                    "dest": item.get("dest", source if path is None else str(Path(source) / path)),
                    "group": _infer_export_group(path or source, item.get("group")),
                    "copy_mode": "directory" if item.get("directory", False) else "file",
                }
            )
    return normalized


def _build_paper_export_specs():
    export_cfg = _paper_export_cfg()
    if not export_cfg.get("enabled", False):
        return []

    dataset_names = list(config["datasets"].keys())
    n_splits = int(config.get("train_test", {}).get("n_splits", 5))
    specs = []

    for item in _normalize_aggregate_items(export_cfg.get("aggregate_files", [])):
        spec = {
            "kind": "aggregate_file",
            "group": item["group"],
            "input": aggregate_output_path(item["source"]),
            "dest": _export_dest_path(item["group"], item["dest"]),
        }
        if item.get("path"):
            spec["src_rel"] = item["path"]
        if item.get("copy_mode") == "directory":
            spec["copy_mode"] = "directory"
        specs.append(spec)

    for item in export_cfg.get("dataset_files", []):
        source = item["source"]
        dest_pattern = item["dest_pattern"]
        group = _infer_export_group(source, item.get("group"))
        datasets = _normalize_selection(item.get("datasets"), default_values=dataset_names)
        for dataset in datasets:
            specs.append(
                {
                    "kind": "dataset_file",
                    "group": group,
                    "input": dataset_plot_path(dataset, source),
                    "dest": _export_dest_path(
                        group,
                        dest_pattern.format(
                            dataset=dataset,
                            dataset_repr=dataset_rep_label(dataset),
                        ),
                    ),
                }
            )

    for item in export_cfg.get("benchmark_plot_files", []):
        datasets = _normalize_selection(item.get("datasets"), default_values=dataset_names)
        methods = _normalize_selection(item.get("methods"), default_values=ALL_METHODS)
        plot_types = _normalize_selection(
            item.get("plot_types"),
            default_values=["clustermap", "embedding"],
        )
        splits = _normalize_splits(item.get("splits"), n_splits=n_splits)
        dest_pattern = item.get(
            "dest_pattern",
            "benchmark_plots/{dataset}/{method}/split_{split:02d}/{plot_type}.pdf",
        )
        for dataset in datasets:
            for method_name in methods:
                for split_idx in splits:
                    for plot_type in plot_types:
                        specs.append(
                            {
                                "kind": "benchmark_plot_file",
                                "group": "plots",
                                "copy_mode": "file",
                                "input": bench_path(dataset, method_name, "train_test_plots"),
                                "src_rel": f"split_{split_idx:02d}/{plot_type}.pdf",
                                "dest": _export_dest_path(
                                    "plots",
                                    dest_pattern.format(
                                        dataset=dataset,
                                        dataset_repr=dataset_rep_label(dataset),
                                        method=method_name,
                                        split=split_idx,
                                        plot_type=plot_type,
                                    ),
                                ),
                            }
                        )

    for item in export_cfg.get("benchmark_plot_dirs", []):
        datasets = _normalize_selection(item.get("datasets"), default_values=dataset_names)
        # default: the methods shown in the main table (main_table: false ones are left out)
        table_methods = [m for m in ALL_METHODS if config["methods"][m].get("main_table", True)]
        methods = _normalize_selection(item.get("methods"), default_values=table_methods)
        dest_pattern = item.get("dest_pattern", "benchmark/{dataset}/{method}")
        for dataset in datasets:
            for method_name in methods:
                specs.append(
                    {
                        "kind": "benchmark_plot_dir",
                        "group": "plots",
                        "copy_mode": "directory",
                        "input": bench_path(dataset, method_name, "train_test_plots"),
                        "dest": _export_dest_path(
                            "plots",
                            dest_pattern.format(
                                dataset=dataset,
                                dataset_repr=dataset_rep_label(dataset),
                                method=method_name,
                            ),
                        ),
                    }
                )

    for item in export_cfg.get("tune_files", []):
        dataset = item["dataset"]
        method_name = item["method"]
        source = item["source"]
        specs.append(
            {
                "kind": "tune_file",
                "group": "plots",
                "copy_mode": "file",
                "input": tune_path(dataset, method_name, "plots"),
                "src_rel": source,
                "dest": _export_dest_path("plots", item.get("dest", source)),
            }
        )

    deduped = []
    seen_dests = set()
    for spec in specs:
        if spec["dest"] in seen_dests:
            raise ValueError(f"Duplicate paper export destination: {spec['dest']}")
        seen_dests.add(spec["dest"])
        deduped.append(spec)
    return deduped


PAPER_EXPORT_SPECS = _build_paper_export_specs()
PAPER_EXPORT_TARGETS = [spec["dest"] for spec in PAPER_EXPORT_SPECS]
if PAPER_EXPORT_SPECS:
    PAPER_EXPORT_MANIFEST_ROOT = _paper_export_manifest_root(PAPER_EXPORT_SPECS)
    PAPER_EXPORT_MANIFEST = str(PAPER_EXPORT_MANIFEST_ROOT / "export_manifest.json")
    PAPER_EXPORT_TARGETS = PAPER_EXPORT_TARGETS + [PAPER_EXPORT_MANIFEST]
    PAPER_EXPORT_INPUTS = sorted({spec["input"] for spec in PAPER_EXPORT_SPECS})
    PAPER_EXPORT_OUTPUTS = [
        directory(spec["dest"]) if spec.get("copy_mode") == "directory" else spec["dest"]
        for spec in PAPER_EXPORT_SPECS
    ]

    rule export_paper_outputs:
        """Copy configured paper-ready outputs into a stable export root."""
        input:
            deps=PAPER_EXPORT_INPUTS
        output:
            exported=PAPER_EXPORT_OUTPUTS,
            manifest=PAPER_EXPORT_MANIFEST
        params:
            export_specs=PAPER_EXPORT_SPECS,
            export_root=str(PAPER_EXPORT_MANIFEST_ROOT)
        log:
            str(PAPER_EXPORT_MANIFEST_ROOT / "export.log")
        resources:
            mem_mb=lambda w: _get_resources("plot", "mem_mb", 4000),
        script:
            "../scripts/export_outputs.py"
