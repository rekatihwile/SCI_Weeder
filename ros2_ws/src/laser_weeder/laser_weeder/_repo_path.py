"""
_repo_path.py — walks up from this file to find the LaserWeeder repo root
and inserts it into sys.path so all repo modules are importable.

This is used by every node instead of installing the repo as a package.
"""
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "__init__.py").exists():
            return parent
    raise RuntimeError(
        "Could not locate LaserWeeder repo root. "
        "Expected to find config/__init__.py walking up from: "
        + str(Path(__file__).resolve())
    )


REPO_ROOT: Path = _find_repo_root()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
