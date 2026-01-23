"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def berlin_config():
    """Load Berlin city config."""
    from urban_tree_transfer.config.loader import load_city_config

    return load_city_config("berlin")


@pytest.fixture
def leipzig_config():
    """Load Leipzig city config."""
    from urban_tree_transfer.config.loader import load_city_config

    return load_city_config("leipzig")
