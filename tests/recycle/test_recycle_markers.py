"""Tests for the RT recycled-event marker in the ``refined`` column.

Rows carried over between KMC steps by a recycler are prefixed with 'R'
('T' -> 'RT') so the events log distinguishes recycled events from freshly
refined ones; unrefined rows are not carried. The marker is applied in
``ActiveEventTable.prune_for_recycling`` -- strategy-agnostic -- and is
idempotent for rows recycled over multiple steps.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd

from pykmc.event_table import ActiveEventTable

from .conftest import row


def _table_with_rows(rows: list[dict], recycler: Mock) -> ActiveEventTable:
    return ActiveEventTable(
        Mock(), event_dataframe=pd.DataFrame(rows), recycler=recycler
    )


def test_recycled_rows_get_r_prefix() -> None:
    """A refined row surviving the recycler's filter is marked RT; an unrefined one is dropped."""
    rows = [row(0), row(1)]
    rows[1]["refined"] = "F"
    survivors = pd.DataFrame(rows)

    recycler = Mock()
    recycler.select_recyclable.return_value = survivors

    table = _table_with_rows(rows, recycler)
    table.prune_for_recycling(0, Mock(), np.zeros((2, 3)))

    assert list(table.table["refined"]) == ["RT"]
    assert list(table.table["atom_index"]) == [0]


def test_marker_is_idempotent_across_steps() -> None:
    """A row recycled a second time stays RT, never RRT."""
    rows = [row(0)]
    rows[0]["refined"] = "RT"
    survivors = pd.DataFrame(rows)

    recycler = Mock()
    recycler.select_recyclable.return_value = survivors

    table = _table_with_rows(rows, recycler)
    table.prune_for_recycling(0, Mock(), np.zeros((1, 3)))

    assert list(table.table["refined"]) == ["RT"]
