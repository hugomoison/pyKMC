"""Normal-mode analysis from a mass-weighted Hessian.

PROVENANCE: vendored from
apps/PyKMC_Analysis/Analysis/htst/kappa_rpa.py (the proven analysis-side
implementation, 15 unit tests, 6 debug rounds, May-2026 "v4" correctness fix).
Duplicated here to keep pyKMC self-contained; the analysis package is not a
pyKMC runtime dependency. Keep the two copies in sync if either changes.

Units convention (DO NOT MIX)
------------------------------
- Mass-weighted Hessian H_mw : eV / (amu · Å²)
- Eigenvalues of H_mw         : same; equal to (ωᵢ_rad/s)² in atomic-style natural units
- ℏω frequencies              : eV (matches kᵇ = 8.617333e-5 eV/K)
- Conversion factor           : ℏω_eV = HBAR_OMEGA_EV * sqrt(eigenvalue_in_natural)
                                with HBAR_OMEGA_EV ≈ 0.06466 eV / sqrt(eV/(amu·Å²))
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

from .constants import HBAR_OMEGA_EV, ZERO_MODE_TOL_EV2


def normal_modes_from_hessian(
    H_mw: np.ndarray,
    n_zero_modes: int = 3,
    expect_saddle: bool = True,
) -> Tuple[Optional[float], np.ndarray, np.ndarray, int]:
    """Diagonalise a mass-weighted Hessian and return frequency data.

    Parameters
    ----------
    H_mw : (M, M) ndarray
        Mass-weighted Hessian over M = 3 × n_free_atoms degrees of freedom.
    n_zero_modes : int
        Number of translational/rotational zero modes to project out
        (3 for periodic slabs, 6 for free clusters). Identified as the
        smallest-|eigenvalue| modes after diagonalisation.
    expect_saddle : bool
        If True, exactly ONE negative eigenvalue is expected (first-order
        saddle). If False (e.g. minimum), no negative mode required.

    Returns
    -------
    omega0_eV : float or None
        Positive magnitude of ℏω at the imaginary mode. None if no saddle
        (when expect_saddle=False and no negative eigenvalue found).
    omegas_eV : (N,) ndarray
        Positive ℏω of the real modes (after dropping zero modes and the
        negative mode). N = M - n_zero_modes - 1 (saddle) or M - n_zero_modes (min).
    R : (M, M) ndarray
        Eigenvectors as columns, in the same order as the SORTED eigenvalues
        (most-negative first). Used by `coupling_matrix` to transform the
        Hessian derivative into normal-mode basis.
    neg_idx : int
        Column index in R of the negative-mode eigenvector (-1 if none).

    Raises
    ------
    ValueError
        If H_mw is not symmetric, or if `expect_saddle=True` and the
        spectrum has !=1 negative eigenvalue (after zero-mode removal).

    """
    if H_mw.ndim != 2 or H_mw.shape[0] != H_mw.shape[1]:
        raise ValueError(f"H_mw must be square, got shape {H_mw.shape}")
    if not np.allclose(H_mw, H_mw.T, atol=1e-8):
        raise ValueError("H_mw must be symmetric")

    eigvals, eigvecs = np.linalg.eigh(H_mw)  # ascending order

    # Identify negative mode(s) FIRST. For a first-order saddle, exactly
    # one eigenvalue is below -ZERO_MODE_TOL_EV2; we must NOT confuse it
    # with a "zero mode". Without this step, a saddle whose imaginary mode
    # has |λ| smaller than some near-zero positive modes (common for
    # NEB-only saddles where translations haven't fully relaxed to 0) gets
    # the imaginary mode misclassified as a zero mode.
    neg_full_indices = np.where(eigvals < -ZERO_MODE_TOL_EV2)[0]

    # Pick zero modes from the POSITIVE-or-near-zero eigenvalue subset,
    # excluding any negative modes found above. We sort the remaining by
    # |λ| ascending and take the n_zero_modes smallest as zero modes.
    keep_mask_for_zero_picking = np.ones(len(eigvals), dtype=bool)
    keep_mask_for_zero_picking[neg_full_indices] = False
    candidate_indices = np.where(keep_mask_for_zero_picking)[0]
    abs_candidate_eigs = np.abs(eigvals[candidate_indices])
    sort_order = np.argsort(abs_candidate_eigs)
    zero_mode_indices = candidate_indices[sort_order[:n_zero_modes]]

    keep_mask = np.ones(len(eigvals), dtype=bool)
    keep_mask[zero_mode_indices] = False

    eigvals_kept = eigvals[keep_mask]

    # Identify negative mode(s) among the kept (now reliably the
    # imaginary mode if expect_saddle).
    neg_indices = np.where(eigvals_kept < -ZERO_MODE_TOL_EV2)[0]

    if expect_saddle:
        if len(neg_indices) != 1:
            raise ValueError(
                f"Expected exactly 1 negative eigenvalue at saddle, got "
                f"{len(neg_indices)} (eigenvalues: {eigvals_kept[neg_indices]})"
            )
        neg_idx = int(neg_indices[0])
        omega0_natural = float(np.sqrt(-eigvals_kept[neg_idx]))
        omega0_eV = HBAR_OMEGA_EV * omega0_natural

        # Drop the negative mode for the positive-frequency list
        keep_pos = np.ones(len(eigvals_kept), dtype=bool)
        keep_pos[neg_idx] = False
        positive_eigvals = eigvals_kept[keep_pos]
    else:
        if len(neg_indices) > 0:
            warnings.warn(
                f"Expected minimum (no negative mode) but found {len(neg_indices)} "
                f"negative eigenvalues. Treating as warning.",
                RuntimeWarning,
                stacklevel=2,
            )
        neg_idx = -1
        omega0_eV = None
        positive_eigvals = eigvals_kept[eigvals_kept > 0]

    # Convert positive eigenvalues to ℏω in eV
    if (positive_eigvals < 0).any():
        # After saddle removal, all should be positive. Anything still negative
        # is a soft mode that should have been projected; clamp to 0.
        positive_eigvals = np.maximum(positive_eigvals, 0.0)
    omegas_eV = HBAR_OMEGA_EV * np.sqrt(positive_eigvals)

    # Reconstruct full eigenvector matrix in original sorted order
    # (with zero modes still included so neg_idx maps consistently)
    return (
        omega0_eV,
        omegas_eV,
        eigvecs,
        _absolute_neg_idx(eigvals, neg_idx, zero_mode_indices),
    )


def _absolute_neg_idx(
    eigvals_full: np.ndarray,
    neg_idx_kept: int,
    zero_mode_indices: np.ndarray,
) -> int:
    """Map neg_idx from the post-zero-mode-removal ordering back to the full eigenvalue ordering returned by eigh."""
    if neg_idx_kept < 0:
        return -1
    keep_mask = np.ones(len(eigvals_full), dtype=bool)
    keep_mask[zero_mode_indices] = False
    full_indices = np.where(keep_mask)[0]
    return int(full_indices[neg_idx_kept])
