"""Eigenvalue → vibrational-frequency conversions for HTST.

PROVENANCE: vendored from
apps/PyKMC_Analysis/Analysis/htst/kappa_rpa.py (the proven analysis-side
implementation, 15 unit tests). Duplicated here to keep pyKMC self-contained;
the analysis package is not a pyKMC runtime dependency. Keep the two copies in
sync if either changes.
"""

import math

HBAR_OMEGA_EV = 0.06466  # eV / sqrt(eV/(amu·Å²)); see kappa_rpa.py provenance
HBAR_EV_S = 6.582119569e-16  # ℏ in eV·s (needed by the vendored vineyard_prefactor)
ZERO_MODE_TOL_EV2 = 1.0e-6  # |λ| below this is a projected-out zero mode
KB_EV_PER_K = 8.617333e-5  # eV/K (matches pyKMC PhysicalConstants.kb)

# LAMMPS ``dynamical_matrix ... eskm`` (metal units) writes the mass-weighted
# Hessian scaled by conv_energy = 9648.5 so its eigenvalues are (rad/ps)^2.
# Divide the matrix by this to recover eV/(amu·Å²) — the convention used by
# ``normal_modes_from_hessian`` / ``vineyard_prefactor`` (via HBAR_OMEGA_EV).
ESKM_DIV_EV_AMU_A2 = 9648.5


def eigval_to_omega_eV(lmbda: float) -> float:
    """Return ℏω in eV (angular) from a positive mass-weighted-Hessian eigenvalue.

    Parameters
    ----------
    lmbda : float
        Eigenvalue of the mass-weighted Hessian, in eV/(amu·Å²). Must be >= 0.

    Returns
    -------
    float
        ℏω in eV.

    """
    return HBAR_OMEGA_EV * math.sqrt(lmbda)


def omega_eV_to_hz(omega_eV: float) -> float:
    """Convert ℏω [eV] (angular) to a LINEAR frequency ν [Hz] via ν = ℏω / (2π·ℏ).

    Equivalent to dividing by h = 2π·ℏ = 4.135667e-15 eV·s. This matches the
    conversion inside the vendored ``vineyard_prefactor`` (÷ 2π·HBAR_EV_S), so ν₀
    comes out in Hz and the KMC rate k = ν₀·exp(-Ea/kT) is in s⁻¹.

    Parameters
    ----------
    omega_eV : float
        ℏω in eV.

    Returns
    -------
    float
        Linear frequency ν in Hz.

    """
    return omega_eV / (2.0 * math.pi * HBAR_EV_S)


def hz_to_thz(f_hz: float) -> float:
    """Convert a frequency from Hz to THz."""
    return f_hz * 1.0e-12


def thz_to_hz(f_thz: float) -> float:
    """Convert a frequency from THz to Hz."""
    return f_thz * 1.0e12
