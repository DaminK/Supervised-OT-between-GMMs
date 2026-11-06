"""
Copy curated Snakemake outputs into a stable export root.

Called via Snakemake ``script:`` directive. The ``snakemake`` object provides:
  snakemake.input.deps          — source artifact file/dir dependencies
  snakemake.output.files        — destination artifact paths
  snakemake.output.manifest     — JSON manifest for provenance
  snakemake.params.export_specs — list of export spec mappings
  snakemake.params.export_root  — export root path
  snakemake.log[0]
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, snakemake.scriptdir)  # noqa: F821

from _utils import setup_logging, to_jsonable  # noqa: E402


def _resolve_export_source(spec: dict) -> Path:
    base = Path(spec["input"])
    rel = spec.get("src_rel")
    if rel:
        return base / rel
    return base


def run(snakemake):
    logger = setup_logging(snakemake.log[0])

    export_specs = list(snakemake.params.export_specs)
    export_root = Path(snakemake.params.export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    copied = []
    for spec in export_specs:
        src = _resolve_export_source(spec)
        dest = Path(spec["dest"])
        if not src.exists():
            raise FileNotFoundError(f"Paper export source does not exist: {src}")
        if spec.get("copy_mode") == "directory":
            dest.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dest.resolve():
                shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
        copied.append({"kind": spec["kind"], "src": str(src), "dest": str(dest)})
        logger.info(f"Copied {src} -> {dest}")

    manifest = {
        "schema": "snakemake.paper_exports.v1",
        "export_root": str(export_root),
        "exports": to_jsonable(copied),
    }
    Path(snakemake.output.manifest).write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    logger.info(f"Wrote: {snakemake.output.manifest}")


run(snakemake)  # noqa: F821
