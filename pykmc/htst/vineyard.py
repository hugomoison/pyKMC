"""Vineyard harmonic prefactor ν₀ for HTST rate constants.

PROVENANCE: vendored from
apps/PyKMC_Analysis/Analysis/htst/kappa_rpa.py (the proven analysis-side
implementation, 15 unit tests). Duplicated here to keep pyKMC self-contained;
the analysis package is not a pyKMC runtime dependency. Keep the two copies in
sync if either changes.

Units convention (DO NOT MIX)
------------------------------
- Mass-weighted Hessian H_mw : eV / (amu · Å²)
- Eigenvalues of H_mw         : same; equal to (ωᵢ_rad/s)² in atomic-style natural units
- ℏω frequencies              : eV (matches kᵇ = 8.617333e-5 eV/K)
- ν₀ output                   : Hz  (linear frequency, NOT angular)

# NOTE: the Sharia & Henkelman RPA recrossing correction kappa (coupling_matrix,
# build_G, kappa_rpa) is deferred to a future version and intentionally NOT
# vendored here. See apps/PyKMC_Analysis/Analysis/htst/kappa_rpa.py for that code.
"""

from __future__ import annotations

import numpy as np

from .constants import HBAR_EV_S
from .normal_modes import normal_modes_from_hessian


def vineyard_prefactor(
    H_mw_init: np.ndarray,
    H_mw_sad: np.ndarray,
    n_zero_modes: int = 3,
) -> float:
    """Compute the Vineyard harmonic prefactor ν₀ in Hz.

        ν₀ = ∏ωᵢ_init / ∏ωᵢ_sad  (sad excludes the imaginary mode)

    Parameters
    ----------
    H_mw_init : (M, M) ndarray
        Mass-weighted Hessian at the initial-state minimum (free atoms only).
    H_mw_sad : (M, M) ndarray
        Mass-weighted Hessian at the saddle (free atoms only, same dimension).
    n_zero_modes : int
        Translational zero modes to project out. Default 3 (slab geometry).

    Returns
    -------
    nu0_Hz : float
        Harmonic prefactor in Hz. Typical for FCC Ni: 1–10 THz.

    Notes
    -----
    Both Hessians must use the same free-atom subset for the products to be
    physically meaningful. The unit cancellation works because the prefactor
    is a ratio: each ω is in (eV/(amu·Å²))^0.5, so the natural-unit ratio
    is dimensionless. We convert one ω to Hz at the end via ω_Hz = ω_eV / (2π·ℏ_eV·s).

    """
    _, omegas_init_eV, _, _ = normal_modes_from_hessian(
        H_mw_init,
        n_zero_modes=n_zero_modes,
        expect_saddle=False,
    )
    _, omegas_sad_eV, _, _ = normal_modes_from_hessian(
        H_mw_sad,
        n_zero_modes=n_zero_modes,
        expect_saddle=True,
    )

    if len(omegas_init_eV) != len(omegas_sad_eV) + 1:
        raise ValueError(
            f"Vineyard prefactor expects N(init) = N(sad) + 1 positive modes; "
            f"got init={len(omegas_init_eV)}, sad={len(omegas_sad_eV)}."
        )

    # The ratio of ωs is the same in any unit; pick eV.
    log_ratio = np.sum(np.log(omegas_init_eV)) - np.sum(np.log(omegas_sad_eV))
    nu0_eV = np.exp(log_ratio)  # this is ν in eV (i.e. ℏν), still needs conversion
    # Actually log(prod_init/prod_sad) gives ratio of ω products. The result
    # has units of ω (since #init = #sad + 1, one ω is left over). So nu0_eV
    # is an ℏω in eV, which we now convert to Hz.
    nu0_Hz = nu0_eV / (2.0 * np.pi * HBAR_EV_S)
    return float(nu0_Hz)
