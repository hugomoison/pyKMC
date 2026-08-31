"""Tests for the reconstruction acceptance rule (movers + shell + containment).

The reconstruction match test used to reject an event when the single
most-displaced atom over the whole rcut neighbourhood exceeded
``psr.matching_score_thr``: a peripheral atom that merely relaxed during the
minimize could veto an otherwise-correct reconstruction, needlessly dropping
recyclable events. The acceptance rule is now focused on the atoms that
actually participate in the event (the movers), with a looser whole-shell
bound catching a peripheral atom that relaxed into a genuinely distinct site,
and a radius-containment guard rejecting events that reach the edge of the
stored rcut neighbourhood before the (expensive) minimize runs.
"""

from unittest.mock import Mock

import numpy as np

from pykmc.reconstruction import Reconstruction
from pykmc.result import ErrorType


def _config() -> Mock:
    config = Mock()
    config.reconstruction.push_fraction = 0.15
    config.reconstruction.n_movers = 3
    config.reconstruction.containment_margin = 1.0
    config.reconstruction.shell_tolerance = 1.0
    config.atomicenvironment.rcut = 6.5
    config.psr.matching_score_thr = 0.1
    return config


_SADDLE = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
_MIN1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
_MIN2 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
_CELL = np.diag([10.0, 10.0, 10.0])


def test_reconstruct_ok_path_still_works() -> None:
    """When both minimizes succeed and match the supposed minima -> Ok."""
    manager = Mock()
    manager.group_minimize_with_results.side_effect = [
        (_MIN1.copy(), 0.0),
        (_MIN2.copy(), -5.0),
    ]
    recon = Reconstruction(_config(), manager, types=["Ni", "Ni"])

    result = recon.reconstruct(_MIN1.copy(), _MIN2.copy(), _SADDLE.copy(), _CELL)

    assert result.is_ok()
    assert result.ok_value().min2_etot == -5.0


# Three-atom event: atom 1 is the mover (min1 [1,0,0] -> min2 [2,0,0]); atoms 0
# and 2 are static. atom 2 sits far out so a small reconstruction error on it
# must NOT veto the match once the acceptance focuses on the movers.
_MIN1_3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
_MIN2_3 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
_SADDLE_3 = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])


def test_peripheral_atom_offset_does_not_veto_when_movers_match() -> None:
    """A mislanded peripheral atom must not veto the reconstruction.

    A peripheral (non-event) atom reconstructed past the threshold must NOT
    reject an otherwise-correct reconstruction; only the event movers gate the
    tight acceptance check.
    """
    # atom 2 lands 0.5 A off (>> matching_score_thr=0.1), the mover atom 1 is exact.
    min1_ret = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    min2_ret = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    manager = Mock()
    manager.group_minimize_with_results.side_effect = [
        (min1_ret, 0.0),
        (min2_ret, -5.0),
    ]
    recon = Reconstruction(_config(), manager, types=["Ni", "Ni", "Ni"])

    result = recon.reconstruct(_MIN1_3.copy(), _MIN2_3.copy(), _SADDLE_3.copy(), _CELL)

    assert result.is_ok()


def test_peripheral_gross_misland_rejected_by_shell_bound() -> None:
    """A peripheral atom landing on a distinct site rejects via the shell bound.

    A peripheral (non-mover) atom that relaxes into a DISTINCT site (a large
    displacement, > shell_tolerance) must reject the reconstruction even though
    the event movers match -- the whole-shell loose bound catches a wrong
    overall state that the movers-only check would have accepted.
    """
    # mover atom 1 exact; peripheral atom 2 lands 1.5 A off (>> shell_tolerance=1.0).
    min1_ret = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
    manager = Mock()
    manager.group_minimize_with_results.side_effect = [(min1_ret, 0.0)]
    recon = Reconstruction(_config(), manager, types=["Ni", "Ni", "Ni"])

    result = recon.reconstruct(_MIN1_3.copy(), _MIN2_3.copy(), _SADDLE_3.copy(), _CELL)

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_INVALID_MIN1
    assert result.err_value().variables["delr_shell1"] > 1.0


def test_mover_offset_rejects_reconstruction() -> None:
    """If an event mover is reconstructed past the threshold, reject (INVALID_MIN1)."""
    # mover atom 1 lands 0.5 A off in min1.
    min1_ret = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    manager = Mock()
    manager.group_minimize_with_results.side_effect = [(min1_ret, 0.0)]
    recon = Reconstruction(_config(), manager, types=["Ni", "Ni", "Ni"])

    result = recon.reconstruct(_MIN1_3.copy(), _MIN2_3.copy(), _SADDLE_3.copy(), _CELL)

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_INVALID_MIN1


def test_event_not_contained_in_rcut_rejects_before_minimize() -> None:
    """An event reaching the outer rcut shell is rejected before the minimize.

    If a mover sits in the outer rcut shell, the event is not contained and
    reconstruction is rejected before the (expensive) minimize ever runs.
    """
    config = _config()
    config.atomicenvironment.rcut = 3.0  # limit = rcut - margin = 2.0
    # central atom 0 at origin; mover atom 1 at radius 2.5 (> 2.0) and it moves.
    min1 = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    min2 = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    saddle = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    manager = Mock()
    recon = Reconstruction(config, manager, types=["Ni", "Ni", "Ni"])

    result = recon.reconstruct(
        min1,
        min2,
        saddle,
        _CELL,
        neighbors=np.array([0, 1, 2]),
        central_atom=0,
    )

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_EVENT_NOT_CONTAINED
    manager.group_minimize_with_results.assert_not_called()


# 5-mover scenario: event_disp = [1.5, 1.4, 1.3, 1.2, 1.1] -- every atom is a
# genuine participant, more than the n_movers=3 floor. A fixed top-3 cap would
# tight-check only the first three and let the 4th mover land on a distinct
# nearby site, accepted against the loose shell bound alone.
_MIN1_5 = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.0, 1.4, 0.0],
        [0.0, 0.0, 1.3],
        [1.2, 0.0, 0.0],
    ]
)
_MIN2_5 = np.array(
    [
        [1.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
)
_SADDLE_5 = 0.5 * (_MIN1_5 + _MIN2_5)


def test_fourth_mover_offset_rejected() -> None:
    """Every genuine participant is tight-checked, not just a top-n cap.

    The min2 relaxation lands the 4th-largest mover 0.6 A off its supposed
    position: within shell_tolerance (1.0) but far past matching_score_thr
    (0.1). n_movers is a FLOOR, not a cap, so the 4th participant is still
    tight-checked and the reconstruction is rejected.
    """
    min2_off = _MIN2_5.copy() + np.array(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0.6, 0, 0], [0, 0, 0]]
    )
    manager = Mock()
    manager.group_minimize_with_results.side_effect = [
        (_MIN1_5.copy(), 0.0),
        (min2_off, -5.0),
    ]
    recon = Reconstruction(_config(), manager, types=["Ni"] * 5)

    result = recon.reconstruct(_MIN1_5.copy(), _MIN2_5.copy(), _SADDLE_5.copy(), _CELL)

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_INVALID_MIN2
    # the 4th mover is a genuine participant: 0.6 A exceeds matching_score_thr.
    assert result.err_value().variables["delr2"] > 0.1


def test_outward_event_not_contained_by_min2() -> None:
    """A mover inside rcut-margin at min1 but past it at min2 -> NOT_CONTAINED.

    The containment guard measures the whole path (min1, saddle, min2): an
    outward event can sit safely inside rcut at min1 yet reach the shell edge
    at the saddle or min2, where the frozen far field would truncate it.
    """
    config = _config()
    config.atomicenvironment.rcut = 6.0  # limit = 6.0 - 1.0 = 5.0
    # Large box so the radii are not minimum-image-wrapped.
    big_cell = np.diag([40.0, 40.0, 40.0])
    # central atom 0 at origin; mover atom 1 inside at min1 (4.8) but past at min2 (5.5).
    min1 = np.array([[0.0, 0.0, 0.0], [4.8, 0.0, 0.0]])
    saddle = np.array([[0.0, 0.0, 0.0], [5.15, 0.0, 0.0]])
    min2 = np.array([[0.0, 0.0, 0.0], [5.5, 0.0, 0.0]])
    manager = Mock()
    recon = Reconstruction(config, manager, types=["Ni", "Ni"])

    result = recon.reconstruct(
        min1,
        min2,
        saddle,
        big_cell,
        neighbors=np.array([0, 1]),
        central_atom=0,
    )

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_EVENT_NOT_CONTAINED
    manager.group_minimize_with_results.assert_not_called()


def test_missing_central_atom_rejects_as_not_contained() -> None:
    """A central id absent from the neighbours ordering must REJECT (fail closed).

    The containment guard is the only geometric sanity check on the
    reconstruction; a corrupted/permuted neighbours column that dropped the
    central id must fail closed rather than bypass it.
    """
    min1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    min2 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    saddle = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    manager = Mock()
    recon = Reconstruction(_config(), manager, types=["Ni", "Ni"])

    result = recon.reconstruct(
        min1,
        min2,
        saddle,
        _CELL,
        neighbors=np.array([0, 1]),
        central_atom=99,  # 99 not in neighbours
    )

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_EVENT_NOT_CONTAINED
    manager.group_minimize_with_results.assert_not_called()


def test_shell_edge_ripple_does_not_trip_containment() -> None:
    """A compact event with a small elastic ripple at the shell edge is accepted.

    Collective events (e.g. an SIA hop) displace their core atoms by ~1 A
    within a couple of Angstroms of the central atom, while dragging a small
    (just above matching_thr) elastic ripple that extends to the rcut shell
    edge. The containment guard is measured over the top-n_movers core only:
    counting the ripple atom at 5.8 A against rcut - margin = 5.5 A would
    reject an event that reconstructs exactly (observed on a 4000-atom Ni
    SIA system, where it purged the whole catalogue at step 1).
    """
    config = _config()  # rcut 6.5, margin 1.0 -> limit 5.5
    # central atom 0 and three core movers displacing ~1 A within 2.5 A of it
    # (they fill the top-n_movers=3 core); ripple atom 4 at radius 5.8
    # (> limit) displaces just 0.15 A and ranks below the core.
    min1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.5],
            [5.8, 0.0, 0.0],
        ]
    )
    min2 = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.5],
            [5.95, 0.0, 0.0],
        ]
    )
    saddle = 0.5 * (min1 + min2)
    big_cell = np.diag([40.0, 40.0, 40.0])
    manager = Mock()
    manager.group_minimize_with_results.side_effect = [
        (min1.copy(), 0.0),
        (min2.copy(), -5.0),
    ]
    recon = Reconstruction(config, manager, types=["Ni"] * 5)

    result = recon.reconstruct(
        min1.copy(),
        min2.copy(),
        saddle.copy(),
        big_cell,
        neighbors=np.array([0, 1, 2, 3, 4]),
        central_atom=0,
    )

    assert result.is_ok()


def test_degenerate_empty_event_rejected_gracefully() -> None:
    """An empty shell/displacement must return an Err, not crash on argmax."""
    empty = np.zeros((0, 3))
    manager = Mock()
    recon = Reconstruction(_config(), manager, types=[])

    result = recon.reconstruct(
        empty.copy(),
        empty.copy(),
        empty.copy(),
        _CELL,
        neighbors=np.zeros(0, dtype=int),
        central_atom=0,
    )

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA
    manager.group_minimize_with_results.assert_not_called()
