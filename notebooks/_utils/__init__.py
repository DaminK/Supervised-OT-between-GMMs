"""Notebook helper package.

This package intentionally exposes a small subset of Snakemake helper
functions used by exploratory notebooks.  The Snakemake module is loaded by
file path so this package does not collide with ``snakemake/scripts/_utils.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import matplotlib

# Same matplotlib settings as the pipeline (TrueType fonts in PDFs, see snakemake/config/matplotlibrc)
matplotlib.rc_file(
    Path(__file__).resolve().parents[2] / "snakemake" / "config" / "matplotlibrc",
    use_default_template=False,
)


def _find_repo_root(start: str | Path | None = None) -> Path:
    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "snakemake" / "config" / "defaults.yml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root from "
        f"{current}. Expected to find snakemake/config/defaults.yml in a parent directory."
    )


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pipeline_utils() -> ModuleType:
    repo = _find_repo_root()
    return _load_module("_notebook_snakemake_utils", repo / "snakemake" / "scripts" / "_utils.py")


def get_gmm_key(config: dict, ds_meta: dict) -> str:
    """Delegate to ``snakemake/scripts/_utils.py::get_gmm_key``."""
    return _pipeline_utils().get_gmm_key(config, ds_meta)


def get_aggregate_experiment_label(config: dict, datasets: dict[str, dict]) -> str:
    """Delegate to ``snakemake/scripts/_utils.py::get_aggregate_experiment_label``."""
    return _pipeline_utils().get_aggregate_experiment_label(config, datasets)


__all__ = ["get_aggregate_experiment_label", "get_gmm_key"]
