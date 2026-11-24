"""Experiment configuration: load a YAML file and keep a copy next to the results."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

CONFIG_DIR = Path("configs")
RESULTS_DIR = Path("results")


def load_config(path: Path) -> dict:
    """Read a YAML config. Only plain data is accepted (`safe_load`)."""
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "name" not in config:
        raise ValueError(f"{path}: a config must be a mapping with a 'name' key")
    return config


def results_dir(config: dict, config_path: Path, root: Path = RESULTS_DIR) -> Path:
    """Create `results/<name>/` and copy the config file into it."""
    out = root / config["name"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out / "config.yaml")
    return out
