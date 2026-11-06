"""Record which ggml-ot code produced a result.

Written into benchmark manifests at runtime for documentation only. It is
deliberately not part of any rule params or output hashes, so recording it never
triggers reruns or invalidates cached results.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def ggml_ot_provenance() -> dict:
    """Version, commit, branch and dirty state of the imported ggml_ot package.

    The version comes from the installed package metadata, so it is recorded for
    pip installs and git checkouts alike; for an editable install it is the
    version at install time. Commit, branch and dirty state come from git and
    are ``"unknown"`` when ggml_ot is not a git checkout (e.g. installed from
    PyPI), where the version identifies the code. ``dirty`` covers uncommitted
    changes inside the package directory only.
    """
    import ggml_ot

    try:
        version = importlib.metadata.version("ggml-ot")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"

    info = {
        "ggml_ot_version": version,
        "ggml_ot_path": str(Path(ggml_ot.__file__).resolve().parent),
        "ggml_ot_commit": "unknown",
        "ggml_ot_branch": "unknown",
        "ggml_ot_dirty": "unknown",
    }
    repo = Path(ggml_ot.__file__).resolve().parent
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return info
    status = _git(repo, "status", "--porcelain", "--untracked-files=no", "--", ".")
    info.update(
        ggml_ot_commit=commit,
        ggml_ot_branch=_git(repo, "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        ggml_ot_dirty=bool(status) if status is not None else "unknown",
    )
    return info
