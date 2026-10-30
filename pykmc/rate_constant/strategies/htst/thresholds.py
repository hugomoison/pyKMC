"""Tolerances and warning thresholds of HTST prefactors.

The physical constants live in `pykmc._core.constants`.

Units
-----
- Frequency: THz, i.e. ps^-1, `hbar * omega / H_EV_PS` (linear, not angular).
"""

from pykmc._core.constants import PhysicalConstants

H_EV_PS = PhysicalConstants.h
"""Planck constant (eV.ps): hbar*omega (eV) / H_EV_PS is a frequency in THz."""

ZERO_MODE_TOL = 1e-6
"""Eigenvalues within this of zero (eV / (amu A^2)) count as zero modes."""

ROTATION_MODE_TOL = 1e-4
"""A rigid rotation whose curvature (eV / (amu A^2)) is below this is a zero mode."""

NU0_WARN_BAND_THZ = (0.1, 1000.0)
"""A nu0 outside this band (THz) is used and logged."""

RMOV_WARN_A = 8.0
"""Active-volume `rmov` (A) below which nu0 is not converged."""

AV_BUFFER_WARN_A = 4.0
"""Active-volume buffer `ract - rmov` (A) below which the crop error grows."""
