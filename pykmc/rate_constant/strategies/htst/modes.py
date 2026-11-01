"""Normal-mode spectrum of a mass-weighted Hessian and the Vineyard prefactor."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pykmc._core.result import Err, ErrorInfo, Ok, Result

from pykmc._core.constants import HBAR_OMEGA_EV

from .thresholds import H_EV_PS, ROTATION_MODE_TOL, ZERO_MODE_TOL
from ...result import PrefactorError


@dataclass(frozen=True)
class Spectrum:
    """Mode counts and the log-product of the positive modes of one Hessian.

    Attributes
    ----------
    n_negative : int
        Number of eigenvalues below `-ZERO_MODE_TOL`.
    n_zero : int
        Number of eigenvalues within `ZERO_MODE_TOL` of zero, once the projected
        zero modes are dropped.
    n_positive : int
        Number of eigenvalues above `ZERO_MODE_TOL`.
    log_hbar_omega : float
        Sum of `log(hbar * omega)` over the positive modes (`hbar * omega` in eV).
    lowest : np.ndarray
        The four smallest eigenvalues (eV / (amu A^2)).

    """

    n_negative: int
    n_zero: int
    n_positive: int
    log_hbar_omega: float
    lowest: np.ndarray


def _nonfinite(detail: str) -> ErrorInfo:
    return ErrorInfo(
        type=PrefactorError.NONFINITE_PREFACTOR,
        message=f"nu0 is not a finite positive number ({detail})",
    )


def _project_translations(h: np.ndarray, sqrt_masses: np.ndarray) -> None:
    """Remove the three mass-weighted translations from `h`, in place.

    It computes P h P with P = 1 - T T^t, where the columns of T are the
    normalised translations, without forming P.
    """
    n = len(sqrt_masses)
    t = np.zeros((3 * n, 3))
    for axis in range(3):
        t[axis::3, axis] = sqrt_masses
    t /= np.sqrt(np.sum(np.square(sqrt_masses)))
    ht = h @ t
    c = t.T @ ht
    for start in range(0, 3 * n, 512):
        r = slice(start, start + 512)
        h[r] -= t[r] @ ht.T + ht[r] @ t.T - t[r] @ c @ t.T


def _free_rotations(
    h: np.ndarray, positions: np.ndarray, sqrt_masses: np.ndarray
) -> int:
    """Count the rigid rotations of the atoms that are zero modes of `h`.

    `h` is the mass-weighted Hessian of every atom, with the translations already
    removed, and `positions` are the atoms as one body (not split by a periodic
    boundary). In a periodic crystal a rotation strains the cell and none is
    counted; an isolated body (a cluster in vacuum) has three, a linear one two.
    """
    centre = np.average(positions, axis=0, weights=np.square(sqrt_masses))
    arms = positions - centre
    rotations = np.column_stack(
        [(np.cross(axis, arms) * sqrt_masses[:, None]).ravel() for axis in np.eye(3)]
    )
    u, s, _ = np.linalg.svd(rotations, full_matrices=False)
    u = u[:, s > 1e-8 * s[0]]
    curvatures = np.linalg.eigvalsh(u.T @ (h @ u))
    return int(np.count_nonzero(np.abs(curvatures) < ROTATION_MODE_TOL))


def _classify(eigenvalues: np.ndarray, n_zero_modes: int) -> Spectrum:
    """Count the modes of an ascending eigenvalue array.

    The `n_zero_modes` non-negative eigenvalues closest to zero are dropped
    before the zero and positive modes are counted.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    negative = eigenvalues < -ZERO_MODE_TOL
    candidates = np.flatnonzero(~negative)
    order = np.argsort(np.abs(eigenvalues[candidates]), kind="stable")
    kept = ~negative
    kept[candidates[order[:n_zero_modes]]] = False
    zero = np.abs(eigenvalues[kept]) <= ZERO_MODE_TOL
    positive = eigenvalues[kept][~zero]
    return Spectrum(
        n_negative=int(np.count_nonzero(negative)),
        n_zero=int(np.count_nonzero(zero)),
        n_positive=int(positive.size),
        log_hbar_omega=float(np.sum(np.log(HBAR_OMEGA_EV * np.sqrt(positive)))),
        lowest=eigenvalues[:4].copy(),
    )


def spectrum(
    h: np.ndarray,
    n_zero_modes: int = 0,
    sqrt_masses: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
) -> Result[Spectrum, ErrorInfo]:
    """Compute the mode spectrum of a mass-weighted Hessian.

    Parameters
    ----------
    h : np.ndarray
        Symmetric mass-weighted Hessian, shape (3n, 3n), in eV / (amu A^2). It is
        modified in place.
    n_zero_modes : int
        Number of zero modes to drop: 3 (the translations) when every atom of the
        system is displaced, else 0.
    sqrt_masses : np.ndarray, optional
        Square root of the mass of each displaced atom, shape (n,). Required when
        `n_zero_modes` is 3: the translations are then projected out of `h`.
    positions : np.ndarray, optional
        Positions of the displaced atoms as one body, shape (n, 3). When given
        with `n_zero_modes` 3, rotations that are zero modes are detected: they
        are not removed, so no prefactor is returned.

    Returns
    -------
    Result[Spectrum, ErrorInfo]
        `Ok` with the spectrum, or `Err` (`NONFINITE_PREFACTOR`) when `h` holds
        non-finite entries or the atoms can rotate freely.

    """
    if not np.all(np.isfinite(h)):
        return Err(_nonfinite("non-finite Hessian entries"))
    if n_zero_modes == 3:
        _project_translations(h, sqrt_masses)
        if positions is not None:
            n_free = _free_rotations(h, positions, sqrt_masses)
            if n_free > 0:
                return Err(
                    _nonfinite(
                        f"{n_free} rotational zero modes: the displaced atoms form "
                        "an isolated body"
                    )
                )
    return Ok(_classify(np.linalg.eigvalsh(h), n_zero_modes))


def saddle_error(spec: Spectrum) -> Optional[ErrorInfo]:
    """Return why `spec` is not a first-order saddle, or `None` if it is one."""
    if spec.n_negative != 1:
        return ErrorInfo(
            type=PrefactorError.SADDLE_NOT_FIRST_ORDER,
            message=f"saddle has {spec.n_negative} negative modes (expected 1)",
        )
    if spec.n_zero > 0:
        return _nonfinite(f"{spec.n_zero} zero modes at the saddle")
    return None


def minimum_error(spec: Spectrum) -> Optional[ErrorInfo]:
    """Return why `spec` is not a minimum, or `None` if it is one."""
    if spec.n_negative != 0:
        return ErrorInfo(
            type=PrefactorError.UNSTABLE_MINIMUM,
            message=f"minimum has {spec.n_negative} negative modes (expected 0)",
        )
    if spec.n_zero > 0:
        return _nonfinite(f"{spec.n_zero} zero modes at the minimum")
    return None


def vineyard_thz(min_spec: Spectrum, sad_spec: Spectrum) -> Result[float, ErrorInfo]:
    r"""Compute the Vineyard prefactor of a minimum and a saddle.

    It uses the following equation :
    $$
    \nu_{0} = \frac{\prod_{i=1}^{3n} \nu_{i}^{min}}{\prod_{i=1}^{3n-1} \nu_{i}^{saddle}}
    $$

    Both spectra must come from the same displaced atoms and have passed
    `minimum_error` and `saddle_error`.

    Returns
    -------
    Result[float, ErrorInfo]
        `Ok` with nu0 in THz (ps^-1), or `Err` (`NONFINITE_PREFACTOR`) when the value is
        not finite and positive.

    """
    with np.errstate(over="ignore", under="ignore"):
        nu0 = np.exp(min_spec.log_hbar_omega - sad_spec.log_hbar_omega) / H_EV_PS
    if not np.isfinite(nu0) or nu0 <= 0.0:
        return Err(_nonfinite(f"value {float(nu0)!r} THz"))
    return Ok(float(nu0))
