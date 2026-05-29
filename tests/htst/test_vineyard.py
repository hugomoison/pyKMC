"""Unit tests for pykmc.htst.vineyard.

Adapted from apps/PyKMC_Analysis/Analysis/tests/test_kappa_rpa.py —
vineyard_prefactor tests only; κ recrossing tests are excluded (out of scope).
"""

from __future__ import annotations

import numpy as np
import pytest

from pykmc.htst.vineyard import vineyard_prefactor
from pykmc.htst.constants import HBAR_EV_S, HBAR_OMEGA_EV


# ---------------------------------------------------------------------------
# Helper: fabricate a synthetic mass-weighted Hessian for tests
# ---------------------------------------------------------------------------


def _toy_hessian(
    eigvals_natural: np.ndarray,
    n_zero: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (M+n_zero, M+n_zero) symmetric matrix with the given eigenvalues plus n_zero zero modes.

    Random orthogonal eigenvectors.
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
# Vineyard prefactor
# ---------------------------------------------------------------------------


def test_vineyard_one_extra_mode_at_init() -> None:
    """ν₀ should be roughly the leftover mode ω in Hz when sad has one fewer mode."""
    init_eigs = np.array([0.05, 0.1, 0.2, 0.3, 0.4])  # 5 positive
    sad_eigs = np.array([-0.1, 0.05, 0.1, 0.2, 0.3])  # 1 neg + 4 pos = 4 kept
    H_init, _ = _toy_hessian(init_eigs, n_zero=3)
    H_sad, _ = _toy_hessian(sad_eigs, n_zero=3, seed=43)
    nu0 = vineyard_prefactor(H_init, H_sad, n_zero_modes=3)
    assert nu0 > 0, "ν₀ must be positive"
    # Sanity: should be in the THz range for these toy eigenvalues
    assert 1e10 < nu0 < 1e15, f"ν₀={nu0} Hz outside reasonable range"


def test_vineyard_identity_when_init_equals_sad_plus_one_mode() -> None:
    """Analytical check: if sad eigvals are init eigvals minus one (replaced by a negative).

    ν₀ should equal HBAR_OMEGA_EV · sqrt(missing) / (2π·ℏ).
    """
    init_eigs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # 5 identical pos modes
    sad_eigs = np.array([-0.1, 0.5, 0.5, 0.5, 0.5])  # 1 neg, 4 pos identical
    H_init, _ = _toy_hessian(init_eigs, n_zero=3)
    H_sad, _ = _toy_hessian(sad_eigs, n_zero=3, seed=44)
    nu0 = vineyard_prefactor(H_init, H_sad, n_zero_modes=3)
    # Expected: prod ratio = ω_extra (the 5th 0.5 mode that has no sad counterpart)
    omega_extra_eV = HBAR_OMEGA_EV * np.sqrt(0.5)
    nu0_expected = omega_extra_eV / (2 * np.pi * HBAR_EV_S)
    rel_err = abs(nu0 - nu0_expected) / nu0_expected
    assert rel_err < 1e-9, f"ν₀={nu0}, expected={nu0_expected}"


def test_vineyard_mode_count_mismatch_raises() -> None:
    """vineyard_prefactor raises ValueError when N(init) != N(sad) + 1.

    Force mismatch by using n_zero_modes=0 on tiny matrices with equal positive-mode counts.
    init: 2 positive → keeps 2; sad: 1 neg + 2 pos → keeps 2. 2 != 2+1.
    """
    # init: 2 pos → n_zero=0 → keeps 2
    init_eigs2 = np.array([0.1, 0.2])
    # sad: 1 neg + 2 pos → keeps 2
    sad_eigs2 = np.array([-0.05, 0.1, 0.2])
    H_init2 = _toy_hessian(init_eigs2, n_zero=0)[0]
    H_sad2 = _toy_hessian(sad_eigs2, n_zero=0)[0]
    with pytest.raises(ValueError, match="N\\(init\\) = N\\(sad\\) \\+ 1"):
        vineyard_prefactor(H_init2, H_sad2, n_zero_modes=0)


def test_vineyard_no_zero_modes() -> None:
    """Smoke test with n_zero_modes=0: a minimal valid min→saddle pair."""
    # init: 2 positive modes (a minimum in 2D)
    # sad:  1 negative + 1 positive (a first-order saddle in 2D)
    # N(init)=2, N(sad)=1 → 2 = 1+1 ✓
    init_eigs = np.array([0.3, 0.5])
    sad_eigs = np.array([-0.2, 0.4])
    H_init = _toy_hessian(init_eigs, n_zero=0, seed=7)[0]
    H_sad = _toy_hessian(sad_eigs, n_zero=0, seed=8)[0]
    nu0 = vineyard_prefactor(H_init, H_sad, n_zero_modes=0)
    assert nu0 > 0, "ν₀ must be positive"
    # Expected: (ω_init_0 · ω_init_1) / ω_sad_1  (ω_sad_0 is the imaginary mode)
    # Both ωs are in eV; the ratio has one leftover ω → Hz via /2πℏ
    omega_init_0 = HBAR_OMEGA_EV * np.sqrt(0.3)
    omega_init_1 = HBAR_OMEGA_EV * np.sqrt(0.5)
    omega_sad_1 = HBAR_OMEGA_EV * np.sqrt(0.4)
    nu0_expected = (omega_init_0 * omega_init_1) / (
        omega_sad_1 * 2.0 * np.pi * HBAR_EV_S
    )
    rel_err = abs(nu0 - nu0_expected) / nu0_expected
    assert rel_err < 1e-9, f"ν₀={nu0}, expected={nu0_expected}"
