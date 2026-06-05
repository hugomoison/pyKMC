"""Tests for pykmc.htst.hessian: mass-weighted partial Hessian via finite differences."""

from __future__ import annotations

import numpy as np

from pykmc.htst.hessian import mass_weighted_partial_hessian


def test_recovers_diagonal_spring_constants() -> None:
    """2 atoms, decoupled isotropic springs k=2.0; mass=1 -> mass-weighted H == H."""
    # 2 atoms, decoupled isotropic springs k=2.0; mass=1 -> mass-weighted H == H.
    k = 2.0
    eq = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    def forces_fn(pos: np.ndarray) -> np.ndarray:
        return -k * (pos - eq)  # F = -k x, per component

    masses = np.array([1.0, 1.0])
    free = np.array([0, 1])
    H = mass_weighted_partial_hessian(forces_fn, eq, masses, free, dx=1e-3)
    assert H.shape == (6, 6)
    np.testing.assert_allclose(H, k * np.eye(6), atol=1e-4)


def test_mass_weighting_applied() -> None:
    """Heavy atom (mass=4) scales diagonal by 1/4: H_diag = k/4."""
    k = 2.0
    eq = np.zeros((1, 3))

    def forces_fn(pos: np.ndarray) -> np.ndarray:
        return -k * pos

    masses = np.array([4.0])  # H_ij / sqrt(m_i m_j) -> k/4 on the diagonal
    H = mass_weighted_partial_hessian(forces_fn, eq, masses, np.array([0]), dx=1e-3)
    np.testing.assert_allclose(H, (k / 4.0) * np.eye(3), atol=1e-4)
