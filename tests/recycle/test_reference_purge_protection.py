"""Regression tests for reference-purge protection of the executed event.

One generic reference event is instantiated as many active rows (IRA reuse maps
it onto every matching site). Several of those rows can be tried in a single
``reconstruction()`` loop: an earlier row may fail -- queuing its
``num_reference_event`` in ``err_reference`` -- while a later row under the
*same* reference reconstructs and becomes the executed event. Purging that
reference afterwards deletes a globally valid event and crashes the step on the
now-empty ``event_id`` lookup at the output log line
(``IndexError: index 0 is out of bounds for axis 0 with size 0``).

``_references_to_purge`` must drop the executed event's reference -- and its
forward/backward partner, since ``ReferenceEventTable.remove`` also removes the
backward sibling of anything it removes -- from the purge set.
"""

import pandas as pd

from pykmc.kmc import _references_to_purge


def _ref_table() -> pd.DataFrame:
    """Two forward/backward reference pairs: 0<->1 and 2<->3."""
    return pd.DataFrame(
        {
            "idx_ref": [0, 1, 2, 3],
            "idx_backward": [1, 0, 3, 2],
            "event_id": ["evt0", "evt1", "evt2", "evt3"],
        }
    )


def test_same_reference_is_protected() -> None:
    """A sibling under ref 0 failed; the executed event is also under ref 0."""
    refs, aes = _references_to_purge(_ref_table(), [0], ["topoA"], selected_ref=0)
    assert refs == []
    assert aes == set()


def test_backward_partner_is_protected() -> None:
    """A sibling under ref 1 failed; the executed event is under ref 0.

    ``remove([1])`` would also drop ``idx_backward(1) == 0`` -- the executed
    reference -- so ref 1 must be protected too.
    """
    refs, aes = _references_to_purge(_ref_table(), [1], ["topoB"], selected_ref=0)
    assert refs == []
    assert aes == set()


def test_unrelated_reference_is_purged() -> None:
    """Ref 2 is unrelated to the executed ref 0 and is genuinely purged."""
    refs, aes = _references_to_purge(_ref_table(), [2], ["topoC"], selected_ref=0)
    assert refs == [2]
    assert aes == {"topoC"}


def test_mixed_keeps_selected_purges_rest() -> None:
    """The executed ref survives while an unrelated failed ref is purged."""
    refs, aes = _references_to_purge(
        _ref_table(), [0, 2], ["topoA", "topoC"], selected_ref=0
    )
    assert refs == [2]
    assert aes == {"topoC"}


def test_selected_ref_absent_from_table_still_protects_itself() -> None:
    """Executed ref not in the table (no partner row): protect it, skip lookup."""
    refs, aes = _references_to_purge(
        _ref_table(), [5, 2], ["topoE", "topoC"], selected_ref=5
    )
    assert refs == [2]
    assert aes == {"topoC"}


def test_no_failures_returns_empty() -> None:
    """No failed references -> nothing to purge."""
    refs, aes = _references_to_purge(_ref_table(), [], [], selected_ref=0)
    assert refs == []
    assert aes == set()
