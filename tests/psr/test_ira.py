"""Tests for the IRA point set registration strategy."""

from dataclasses import dataclass

import numpy as np
import pytest

from pykmc.psr.result import PSRError
from pykmc.psr.strategies.ira import IRAStrategy
from pykmc.utils import geometry


@dataclass
class IRAConfigStub:
    kmax_factor: float = 1.8


def make_strategy(kmax_factor: float = 1.8) -> IRAStrategy:
    return IRAStrategy(IRAConfigStub(kmax_factor=kmax_factor))


def test_identical_point_sets_match_with_zero_score() -> None:
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 1.7, 0.0], [0.2, 0.1, 2.1]]
    )
    typ = ["X"] * 4

    result = make_strategy().match(4, typ, coords.copy(), 4, typ, coords.copy())

    assert result.is_ok()
    out = result.ok_value()
    assert out.matching_score == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(out.permutation_matrix, np.arange(4))


def test_recovers_known_rotation_translation_and_permutation() -> None:
    # coords2 = permutation(rotation(coords1) + translation) -- a rotated,
    # translated, relabeled copy of coords1.
    rng = np.random.default_rng(0)
    coords1 = rng.uniform(-2.0, 2.0, size=(6, 3))
    typ1 = ["X"] * 6

    theta = np.pi / 3
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([1.5, -0.7, 0.3])
    permutation = np.array([3, 0, 4, 1, 5, 2])

    coords2 = (coords1 @ rotation.T + translation)[permutation]
    typ2 = ["X"] * 6

    result = make_strategy().match(6, typ1, coords1.copy(), 6, typ2, coords2.copy())

    assert result.is_ok()
    out = result.ok_value()
    assert out.matching_score == pytest.approx(0.0, abs=1e-6)

    # The transformation found maps the *second* point set onto the first
    # (i.e. coords2 -- transform --> coords1), not the other way around: this
    # is the convention basin.py/refinement.py rely on when moving the
    # reference event's stored positions onto the real system.
    recovered = geometry.transform_positions(
        coords2, out.rotation_matrix, out.translation_matrix, out.permutation_matrix
    )
    np.testing.assert_allclose(recovered, coords1, atol=1e-6)


def test_no_atoms_returns_no_match_found_error() -> None:
    empty = np.zeros((0, 3))

    result = make_strategy().match(0, [], empty, 0, [], empty.copy())

    assert not result.is_ok()
    assert result.err_value().type is PSRError.NO_MATCH_FOUND
