"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Package root directory (where configs/ lives)
_PACKAGE_ROOT = Path(__file__).parent.parent


def get_config_dir() -> Path:
    """Get the configs directory path within the package."""
    return _PACKAGE_ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return data


def load_city_config(city: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load a single city configuration.

    Args:
        city: City name (e.g., "berlin", "leipzig")
        config_dir: Optional override for config directory.
                   Defaults to package's configs/cities/ directory.

    Returns:
        City configuration dictionary.
    """
    if config_dir is None:
        config_dir = get_config_dir() / "cities"
    path = config_dir / f"{city.lower()}.yaml"
    return load_yaml(path)


def load_city_configs(cities: list[str], config_dir: Path | None = None) -> dict[str, dict]:
    """Load multiple city configs by name."""
    return {city: load_city_config(city, config_dir=config_dir) for city in cities}
