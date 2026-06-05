"""Unit tests for pykmc.htst.normal_modes.

Copied from apps/PyKMC_Analysis/Analysis/tests/test_kappa_rpa.py
(only the tests that depend on normal_modes_from_hessian). Includes
the May-2026 v4 regression test for imaginary-mode-smaller-than-near-zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from pykmc.htst.normal_modes import normal_modes_from_hessian
from pykmc.htst.constants import HBAR_OMEGA_EV


# ---------------------------------------------------------------------------
# Helper: fabricate a synthetic mass-weighted Hessian for tests
# ---------------------------------------------------------------------------


def _toy_hessian(eigvals_natural: np.ndarray, n_zero: int = 3, seed: int = 42) -> tuple:  # type: ignore[type-arg]
    """Build a (M+n_zero, M+n_zero) symmetric matrix with the given eigenvalues.

    Appends n_zero zero modes. Random orthogonal eigenvectors.
    """
    full_eigs = np.concatenate([eigvals_natural, np.zeros(n_zero)])
    n = len(full_eigs)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    H = Q @ np.diag(full_eigs) @ Q.T
    H = 0.5 * (H + H.T)
    return H, Q


# ---------------------------------------------------------------------------
# normal_modes_from_hessian
# ---------------------------------------------------------------------------


def test_normal_modes_minimum() -> None:
    """All-positive eigenvalues + 3 zero modes → no negative mode, all retained."""
    eigvals = np.array([0.05, 0.1, 0.2, 0.3, 0.4])  # all positive
    H, _ = _toy_hessian(eigvals, n_zero=3)
    omega0, omegas, _, neg_idx = normal_modes_from_hessian(
        H, n_zero_modes=3, expect_saddle=False
    )
    assert omega0 is None
    assert neg_idx == -1
    assert len(omegas) == 5  # all 5 positive modes kept


def test_normal_modes_saddle() -> None:
    """1 negative + 4 positive + 3 zero → exactly 1 negative kept, 4 positive."""
    eigvals = np.array([-0.1, 0.05, 0.1, 0.2, 0.3])  # 1 neg, 4 pos
    H, _ = _toy_hessian(eigvals, n_zero=3)
    omega0, omegas, _, neg_idx = normal_modes_from_hessian(
        H, n_zero_modes=3, expect_saddle=True
    )
    assert omega0 is not None and omega0 > 0
    assert neg_idx >= 0
    assert len(omegas) == 4
    # ω₀ should equal ℏω of the imag mode: HBAR_OMEGA_EV · sqrt(0.1)
    assert abs(omega0 - HBAR_OMEGA_EV * np.sqrt(0.1)) < 1e-9


def test_normal_modes_saddle_raises_on_two_neg() -> None:
    """Two negative eigenvalues → not a first-order saddle → raises."""
    eigvals = np.array([-0.1, -0.05, 0.1, 0.2, 0.3])
    H, _ = _toy_hessian(eigvals, n_zero=3)
    with pytest.raises(ValueError, match="exactly 1 negative"):
        normal_modes_from_hessian(H, n_zero_modes=3, expect_saddle=True)


def test_normal_modes_saddle_imag_mode_smaller_than_near_zero() -> None:
    """Regression: saddle with |imaginary mode| < |smallest positive mode|.

    Before this fix, `normal_modes_from_hessian` projected out the 3
    smallest-|λ| modes as zero modes, which on a sub-converged NEB saddle
    can include the genuine imaginary mode (when it has |λ| smaller than
    some near-zero positive translational modes that haven't fully
    relaxed to 0 numerically).

    This case mirrors what hit the canonical_kappa.py surface_1NN run on
    May 5: real saddle Hessian had eigenvalues [-0.028, 0.015, 0.025,
    0.029, 0.030, ...]. The legacy algorithm picked indices for
    {0.015, 0.025, 0.028} as zero modes — silently absorbing the
    imaginary mode and reporting "0 negative eigenvalues."

    After the fix, negative modes are identified BEFORE picking zero
    modes, so the imaginary mode is preserved.
    """
    # Imaginary mode at -0.028 is SMALLER in |λ| than 0.029, 0.030, 0.04 (positive)
    # but larger than 0.015 and 0.025 (the genuine zero-modes-in-disguise).
    real_eigs = np.array([-0.028, 0.029, 0.030, 0.04])
    near_zero = np.array([0.015, 0.025, 0.018])  # 3 "zero modes" with |λ| > 0
    H, _ = _toy_hessian(np.concatenate([real_eigs, near_zero]), n_zero=0, seed=99)
    omega0, omegas, _, neg_idx = normal_modes_from_hessian(
        H,
        n_zero_modes=3,
        expect_saddle=True,
    )
    assert omega0 is not None and omega0 > 0
    # ω₀ should match the imaginary mode at -0.028 (the largest neg eigenvalue)
    assert abs(omega0 - HBAR_OMEGA_EV * np.sqrt(0.028)) < 1e-9, (
        f"omega0={omega0}, expected={HBAR_OMEGA_EV * np.sqrt(0.028)}"
    )
    # 4 positive modes remain after dropping 1 neg + 3 zero: but real_eigs has
    # 3 positive modes (0.029, 0.030, 0.04) so we expect len(omegas) == 3.
    assert len(omegas) == 3
