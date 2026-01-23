"""Integration tests for external data endpoints.

These tests hit real external APIs to verify configurations are correct.
Run with: uv run nox -s test_integration

Each test fetches minimal data (1 feature or HEAD request) to verify:
- URL is reachable
- Layer/resource exists
- Filter syntax is correct
- Response schema matches expectations
"""

import pytest

# TODO: Implement integration tests as described below
