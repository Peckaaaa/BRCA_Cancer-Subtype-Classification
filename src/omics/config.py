"""
Config loading. YAML files live in <project_root>/configs/.

Paths inside data.yaml may be relative to the project root; they are resolved
to absolute paths here so scripts work regardless of the current directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# .../src/omics/config.py -> project root is two levels above 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(name: str) -> dict[str, Any]:
    """Load a config file from configs/ by stem name (e.g. 'data')."""
    return load_yaml(CONFIGS_DIR / f"{name}.yaml")


def resolve_data_config(data_cfg: dict[str, Any]) -> tuple[dict[str, Path], Path, str]:
    """Resolve omics file paths and label path (relative to project root)."""
    base = Path(data_cfg.get("data_dir", "data"))
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    sep = data_cfg.get("sep", ",")
    omics_files = {k: base / v for k, v in data_cfg["omics_files"].items()}
    labels_file = base / data_cfg["labels_file"]
    return omics_files, labels_file, sep
