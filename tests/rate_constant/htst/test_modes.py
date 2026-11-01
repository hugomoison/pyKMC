"""Tests for the Hessian spectrum and Vineyard kernels."""

import numpy as np
import pytest

from pykmc._core.constants import HBAR_OMEGA_EV
from pykmc.rate_constant.strategies.htst.modes import (
    _classify,
    minimum_error,
    saddle_error,
    spectrum,
    vineyard_thz,
)
from pykmc.rate_constant import PrefactorError
from pykmc.rate_constant.strategies.htst.thresholds import H_EV_PS


def test_vineyard_diagonal_spectrum_gives_known_nu0() -> None:
    a = np.array([0.5, 1.0, 2.0, 4.0])
    c = np.array([0.8, 1.5, 3.0])
    min_spec = spectrum(np.diag(a)).ok_value()
    sad_spec = spectrum(np.diag(np.concatenate(([-0.3], c)))).ok_value()

    assert minimum_error(min_spec) is None
    assert saddle_error(sad_spec) is None
    nu0 = vineyard_thz(min_spec, sad_spec).ok_value()

    expected = HBAR_OMEGA_EV * np.prod(np.sqrt(a)) / np.prod(np.sqrt(c)) / H_EV_PS
    assert isinstance(nu0, float)
    assert nu0 == pytest.approx(expected, rel=1e-12)


def test_saddle_with_two_negative_modes_is_rejected() -> None:
    spec = spectrum(np.diag([-0.3, -0.1, 1.0, 2.0])).ok_value()

    error = saddle_error(spec)

    assert error.type == PrefactorError.SADDLE_NOT_FIRST_ORDER
    assert error.message == "saddle has 2 negative modes (expected 1)"


def test_saddle_with_no_negative_mode_is_rejected() -> None:
    spec = spectrum(np.diag([0.5, 1.0, 2.0, 4.0])).ok_value()

    error = saddle_error(spec)

    assert error.type == PrefactorError.SADDLE_NOT_FIRST_ORDER
    assert error.message == "saddle has 0 negative modes (expected 1)"


def test_minimum_with_a_negative_mode_is_rejected() -> None:
    spec = spectrum(np.diag([-0.3, 0.5, 1.0, 2.0])).ok_value()

    error = minimum_error(spec)

    assert error.type == PrefactorError.UNSTABLE_MINIMUM
    assert error.message == "minimum has 1 negative modes (expected 0)"


def test_zero_mode_gives_nonfinite_prefactor() -> None:
    spec = spectrum(np.diag([0.0, 0.5, 1.0, 2.0]), n_zero_modes=0).ok_value()

    error = minimum_error(spec)

    assert error.type == PrefactorError.NONFINITE_PREFACTOR
    assert error.message == (
        "nu0 is not a finite positive number (1 zero modes at the minimum)"
    )


def test_nonfinite_hessian_gives_nonfinite_prefactor() -> None:
    h = np.diag([0.5, 1.0, 2.0])
    h[0, 1] = h[1, 0] = np.nan

    error = spectrum(h).err_value()

    assert error.type == PrefactorError.NONFINITE_PREFACTOR
    assert error.message == (
        "nu0 is not a finite positive number (non-finite Hessian entries)"
    )


def test_translations_are_projected_in_full_mode() -> None:
    # Chain of 5 atoms with isotropic springs: invariant under the 3 translations.
    n = 5
    laplacian = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    laplacian[0, 0] = laplacian[-1, -1] = 1.0
    sqrt_masses = np.sqrt(np.array([1.0, 12.0, 16.0, 58.69, 55.85]))
    inv = np.repeat(1.0 / sqrt_masses, 3)
    exact = np.kron(laplacian, np.eye(3)) * np.outer(inv, inv)
    # Finite-difference noise pushes the 3 translations below -ZERO_MODE_TOL.
    t = np.kron(sqrt_masses[:, None], np.eye(3)) / np.linalg.norm(sqrt_masses)
    h = exact - 1e-5 * t @ t.T

    unprojected = spectrum(h.copy()).ok_value()
    projected = spectrum(h.copy(), n_zero_modes=3, sqrt_masses=sqrt_masses).ok_value()
    reference = spectrum(exact.copy(), n_zero_modes=3, sqrt_masses=sqrt_masses)

    assert unprojected.n_negative == 3
    assert minimum_error(projected) is None
    assert projected.n_positive == 3 * n - 3
    # The translations are removed, and the other modes are those without noise.
    assert np.all(np.abs(projected.lowest[:3]) < 1e-12)
    assert projected.log_hbar_omega == pytest.approx(
        reference.ok_value().log_hbar_omega, rel=1e-12
    )


def test_classify_drops_projected_modes_below_zero() -> None:
    spec = _classify(np.array([-1e-15, 1e-16, 2e-16, 0.5, 1.0, 2.0]), 3)

    assert spec.n_negative == 0
    assert spec.n_zero == 0
    assert spec.n_positive == 3


def _pair_hessian(positions: np.ndarray, along_bond: bool) -> np.ndarray:
    """Hessian of unit springs between every pair of unit-mass atoms.

    With `along_bond` each spring acts along its bond, so the body can rotate
    freely. Otherwise it acts on the full relative displacement, and a rotation
    stretches it.
    """
    n = len(positions)
    h = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            bond = positions[j] - positions[i]
            bond /= np.linalg.norm(bond)
            block = np.outer(bond, bond) if along_bond else np.eye(3)
            a, b = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
            h[a, a] += block
            h[b, b] += block
            h[a, b] -= block
            h[b, a] -= block
    return h


def test_free_rotations_of_an_isolated_body_give_no_prefactor() -> None:
    positions = np.array(
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.3, 2.4, 0.0], [0.7, 0.9, 2.2]]
    )
    masses = np.ones(4)

    free = spectrum(_pair_hessian(positions, True), 3, masses, positions)
    error = free.err_value()
    assert error.type == PrefactorError.NONFINITE_PREFACTOR
    assert error.message == (
        "nu0 is not a finite positive number "
        "(3 rotational zero modes: the displaced atoms form an isolated body)"
    )

    held = spectrum(_pair_hessian(positions, False), 3, masses, positions)
    assert held.ok_value().n_zero == 0
    assert held.ok_value().n_positive == 9
    # Without the positions the rotations are not looked for.
    assert spectrum(_pair_hessian(positions, True), 3, masses).ok_value().n_zero == 3
