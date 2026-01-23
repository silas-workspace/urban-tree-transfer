"""Tests for config loader."""

from urban_tree_transfer.config.loader import get_config_dir, load_city_config


def test_config_dir_exists():
    """Config directory should exist within the package."""
    config_dir = get_config_dir()
    assert config_dir.exists()
    assert (config_dir / "cities").exists()


def test_load_berlin_config():
    """Berlin config should load with required keys."""
    config = load_city_config("berlin")
    assert config["name"] == "Berlin"
    assert "boundaries" in config
    assert "trees" in config
    assert "elevation" in config


def test_load_leipzig_config():
    """Leipzig config should load with required keys."""
    config = load_city_config("leipzig")
    assert config["name"] == "Leipzig"
    assert "boundaries" in config
    assert "trees" in config
    assert "elevation" in config
