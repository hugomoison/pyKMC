"""Regression tests for KMC persistence paths."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

from pykmc.kmc import KMC


def test_save_honors_reference_table_output(config_Cu: Any, tmp_path: Path) -> None:
    """Save the reference table to the configured output path."""
    reference_output = tmp_path / "custom-reference-table.pickle"
    visited_output = tmp_path / "visited-environments.pickle"
    config_Cu.control.reference_table_output = str(reference_output)
    config_Cu.control.visited_environments_output = str(visited_output)

    kmc = KMC(config_Cu)
    reference_table = Mock()
    kmc.reference_table = reference_table
    kmc.visited_environments = {"crystal"}

    kmc._save()

    reference_table.save.assert_called_once_with(str(reference_output))
    assert visited_output.exists()


def test_save_falls_back_to_default_reference_table_name(
    config_Cu: Any, tmp_path: Path
) -> None:
    """Save the reference table to the historical name when no path is set."""
    visited_output = tmp_path / "visited-environments.pickle"
    config_Cu.control.reference_table_output = None
    config_Cu.control.visited_environments_output = str(visited_output)

    kmc = KMC(config_Cu)
    reference_table = Mock()
    kmc.reference_table = reference_table
    kmc.visited_environments = {"crystal"}

    kmc._save()

    reference_table.save.assert_called_once_with("reference_table.pickle")
